/*! Buffer Trackers
 *
 * Buffers are represented by a single state for the whole resource,
 * a 16 bit bitflag of buffer usages. Because there is only ever
 * one subresource, they have no selector.
!*/

use std::{borrow::Cow, marker::PhantomData, sync::Arc};

use super::{PendingTransition, ResourceTracker};
use crate::{
    hal_api::HalApi,
    id::{BufferId, TypedId},
    resource::{Buffer, Resource},
    snatch::SnatchGuard,
    storage::Storage,
    track::{
        invalid_resource_state, skip_barrier, ResourceMetadata, ResourceMetadataProvider,
        ResourceUses, UsageConflict,
    },
};
use hal::{BufferBarrier, BufferUses};
use parking_lot::Mutex;
use wgt::{strict_assert, strict_assert_eq};

impl ResourceUses for BufferUses {
    const EXCLUSIVE: Self = Self::EXCLUSIVE;

    type Id = BufferId;
    type Selector = ();

    fn bits(self) -> u16 {
        Self::bits(&self)
    }

    fn all_ordered(self) -> bool {
        Self::ORDERED.contains(self)
    }

    fn any_exclusive(self) -> bool {
        self.intersects(Self::EXCLUSIVE)
    }
}

/// Stores all the buffers that a bind group stores.
#[derive(Debug)]
pub(crate) struct BufferBindGroupState<A: HalApi> {
    buffers: Mutex<Vec<(Arc<Buffer<A>>, BufferUses)>>,

    _phantom: PhantomData<A>,
}
impl<A: HalApi> BufferBindGroupState<A> {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(Vec::new()),

            _phantom: PhantomData,
        }
    }

    /// Optimize the buffer bind group state by sorting it by ID.
    ///
    /// When this list of states is merged into a tracker, the memory
    /// accesses will be in a constant assending order.
    #[allow(clippy::pattern_type_mismatch)]
    pub(crate) fn optimize(&self) {
        let mut buffers = self.buffers.lock();
        buffers.sort_unstable_by_key(|(b, _)| b.as_info().id().unzip().0);
    }

    /// Returns a list of all buffers tracked. May contain duplicates.
    #[allow(clippy::pattern_type_mismatch)]
    pub fn used_ids(&self) -> impl Iterator<Item = BufferId> + '_ {
        let buffers = self.buffers.lock();
        buffers
            .iter()
            .map(|(ref b, _)| b.as_info().id())
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Returns a list of all buffers tracked. May contain duplicates.
    pub fn drain_resources(&self) -> impl Iterator<Item = Arc<Buffer<A>>> + '_ {
        let mut buffers = self.buffers.lock();
        buffers
            .drain(..)
            .map(|(buffer, _u)| buffer)
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Adds the given resource with the given state.
    pub fn add_single<'a>(
        &self,
        storage: &'a Storage<Buffer<A>, BufferId>,
        id: BufferId,
        state: BufferUses,
    ) -> Option<&'a Arc<Buffer<A>>> {
        let buffer = storage.get(id).ok()?;

        let mut buffers = self.buffers.lock();
        buffers.push((buffer.clone(), state));

        Some(buffer)
    }
}

/// Stores all buffer state within a single usage scope.
#[derive(Debug)]
pub(crate) struct BufferUsageScope<A: HalApi> {
    state: Vec<BufferUses>,

    metadata: ResourceMetadata<A, BufferId, Buffer<A>>,
}

impl<A: HalApi> BufferUsageScope<A> {
    pub fn new() -> Self {
        Self {
            state: Vec::new(),

            metadata: ResourceMetadata::new(),
        }
    }

    fn tracker_assert_in_bounds(&self, index: usize) {
        strict_assert!(index < self.state.len());
        self.metadata.tracker_assert_in_bounds(index);
    }

    /// Sets the size of all the vectors inside the tracker.
    ///
    /// Must be called with the highest possible Buffer ID before
    /// all unsafe functions are called.
    pub fn set_size(&mut self, size: usize) {
        self.state.resize(size, BufferUses::empty());
        self.metadata.set_size(size);
    }

    /// Extend the vectors to let the given index be valid.
    fn allow_index(&mut self, index: usize) {
        if index >= self.state.len() {
            self.set_size(index + 1);
        }
    }

    /// Drains all buffers tracked.
    pub fn drain_resources(&mut self) -> impl Iterator<Item = Arc<Buffer<A>>> + '_ {
        let resources = self.metadata.drain_resources();
        self.state.clear();
        resources.into_iter()
    }

    pub fn get(&self, id: BufferId) -> Option<&Arc<Buffer<A>>> {
        let index = id.unzip().0 as usize;
        if index > self.metadata.size() {
            return None;
        }
        self.tracker_assert_in_bounds(index);
        unsafe {
            if self.metadata.contains_unchecked(index) {
                return Some(self.metadata.get_resource_unchecked(index));
            }
        }
        None
    }

    /// Merge the list of buffer states in the given bind group into this usage scope.
    ///
    /// If any of the resulting states is invalid, stops the merge and returns a usage
    /// conflict with the details of the invalid state.
    ///
    /// Because bind groups do not check if the union of all their states is valid,
    /// this method is allowed to return Err on the first bind group bound.
    ///
    /// # Safety
    ///
    /// [`Self::set_size`] must be called with the maximum possible Buffer ID before this
    /// method is called.
    pub unsafe fn merge_bind_group(
        &mut self,
        bind_group: &BufferBindGroupState<A>,
    ) -> Result<(), UsageConflict> {
        let buffers = bind_group.buffers.lock();
        for &(ref resource, state) in &*buffers {
            let index = resource.as_info().id().unzip().0 as usize;

            unsafe {
                insert_or_merge(
                    None,
                    &mut self.state,
                    &mut self.metadata,
                    index as _,
                    index,
                    BufferStateProvider::Direct { state },
                    ResourceMetadataProvider::Direct {
                        resource: Cow::Borrowed(resource),
                    },
                )?
            };
        }

        Ok(())
    }

    /// Merge the list of buffer states in the given usage scope into this UsageScope.
    ///
    /// If any of the resulting states is invalid, stops the merge and returns a usage
    /// conflict with the details of the invalid state.
    ///
    /// If the given tracker uses IDs higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn merge_usage_scope(&mut self, scope: &Self) -> Result<(), UsageConflict> {
        let incoming_size = scope.state.len();
        if incoming_size > self.state.len() {
            self.set_size(incoming_size);
        }

        for index in scope.metadata.owned_indices() {
            self.tracker_assert_in_bounds(index);
            scope.tracker_assert_in_bounds(index);

            unsafe {
                insert_or_merge(
                    None,
                    &mut self.state,
                    &mut self.metadata,
                    index as u32,
                    index,
                    BufferStateProvider::Indirect {
                        state: &scope.state,
                    },
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                )?;
            };
        }

        Ok(())
    }

    /// Merge a single state into the UsageScope.
    ///
    /// If the resulting state is invalid, returns a usage
    /// conflict with the details of the invalid state.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn merge_single<'a>(
        &mut self,
        storage: &'a Storage<Buffer<A>, BufferId>,
        id: BufferId,
        new_state: BufferUses,
    ) -> Result<&'a Arc<Buffer<A>>, UsageConflict> {
        let buffer = storage
            .get(id)
            .map_err(|_| UsageConflict::BufferInvalid { id })?;

        let index = id.unzip().0 as usize;

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            insert_or_merge(
                None,
                &mut self.state,
                &mut self.metadata,
                index as _,
                index,
                BufferStateProvider::Direct { state: new_state },
                ResourceMetadataProvider::Direct {
                    resource: Cow::Owned(buffer.clone()),
                },
            )?;
        }

        Ok(buffer)
    }
}

pub(crate) type SetSingleResult<A> =
    Option<(Arc<Buffer<A>>, Option<PendingTransition<BufferUses>>)>;

/// Stores all buffer state within a command buffer or device.
pub(crate) struct BufferTracker<A: HalApi> {
    start: Vec<BufferUses>,
    end: Vec<BufferUses>,

    metadata: ResourceMetadata<A, BufferId, Buffer<A>>,

    temp: Vec<PendingTransition<BufferUses>>,
}

impl<A: HalApi> ResourceTracker<BufferId, Buffer<A>> for BufferTracker<A> {
    /// Try to remove the buffer `id` from this tracker if it is otherwise unused.
    ///
    /// A buffer is 'otherwise unused' when the only references to it are:
    ///
    /// 1) the `Arc` that our caller, `LifetimeTracker::triage_suspected`, has just
    ///    drained from `LifetimeTracker::suspected_resources`,
    ///
    /// 2) its `Arc` in [`self.metadata`] (owned by [`Device::trackers`]), and
    ///
    /// 3) its `Arc` in the [`Hub::buffers`] registry.
    ///
    /// If the buffer is indeed unused, this function removes 2), and
    /// `triage_suspected` will remove 3), leaving 1) as the sole
    /// remaining reference.
    ///
    /// Returns true if the resource was removed or if not existing in metadata.
    ///
    /// [`Device::trackers`]: crate::device::Device
    /// [`self.metadata`]: BufferTracker::metadata
    /// [`Hub::buffers`]: crate::hub::Hub::buffers
    fn remove_abandoned(&mut self, id: BufferId) -> bool {
        let index = id.unzip().0 as usize;

        if index > self.metadata.size() {
            return false;
        }

        self.tracker_assert_in_bounds(index);

        unsafe {
            if self.metadata.contains_unchecked(index) {
                let existing_ref_count = self.metadata.get_ref_count_unchecked(index);
                //RefCount 2 means that resource is hold just by DeviceTracker and this suspected resource itself
                //so it's already been released from user and so it's not inside Registry\Storage
                if existing_ref_count <= 2 {
                    self.metadata.remove(index);
                    log::trace!("Buffer {:?} is not tracked anymore", id,);
                    return true;
                } else {
                    log::trace!(
                        "Buffer {:?} is still referenced from {}",
                        id,
                        existing_ref_count
                    );
                    return false;
                }
            }
        }
        true
    }
}

impl<A: HalApi> BufferTracker<A> {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),

            metadata: ResourceMetadata::new(),

            temp: Vec::new(),
        }
    }

    fn tracker_assert_in_bounds(&self, index: usize) {
        strict_assert!(index < self.start.len());
        strict_assert!(index < self.end.len());
        self.metadata.tracker_assert_in_bounds(index);
    }

    /// Sets the size of all the vectors inside the tracker.
    ///
    /// Must be called with the highest possible Buffer ID before
    /// all unsafe functions are called.
    pub fn set_size(&mut self, size: usize) {
        self.start.resize(size, BufferUses::empty());
        self.end.resize(size, BufferUses::empty());

        self.metadata.set_size(size);
    }

    /// Extend the vectors to let the given index be valid.
    fn allow_index(&mut self, index: usize) {
        if index >= self.start.len() {
            self.set_size(index + 1);
        }
    }

    /// Returns a list of all buffers tracked.
    pub fn used_resources(&self) -> impl Iterator<Item = Arc<Buffer<A>>> + '_ {
        self.metadata.owned_resources()
    }

    /// Drains all currently pending transitions.
    pub fn drain_transitions<'a, 'b: 'a>(
        &'b mut self,
        snatch_guard: &'a SnatchGuard<'a>,
    ) -> impl Iterator<Item = BufferBarrier<'a, A>> {
        let buffer_barriers = self.temp.drain(..).map(|pending| {
            let buf = unsafe { self.metadata.get_resource_unchecked(pending.id as _) };
            pending.into_hal(buf, snatch_guard)
        });
        buffer_barriers
    }

    /// Inserts a single buffer and its state into the resource tracker.
    ///
    /// If the resource already exists in the tracker, this will panic.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn insert_single(&mut self, id: BufferId, resource: Arc<Buffer<A>>, state: BufferUses) {
        let index = id.unzip().0 as usize;

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            let currently_owned = self.metadata.contains_unchecked(index);

            if currently_owned {
                panic!("Tried to insert buffer already tracked");
            }

            insert(
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index,
                BufferStateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Direct {
                    resource: Cow::Owned(resource),
                },
            )
        }
    }

    /// Sets the state of a single buffer.
    ///
    /// If a transition is needed to get the buffer into the given state, that transition
    /// is returned. No more than one transition is needed.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn set_single(&mut self, buffer: &Arc<Buffer<A>>, state: BufferUses) -> SetSingleResult<A> {
        let index: usize = buffer.as_info().id().unzip().0 as usize;

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            insert_or_barrier_update(
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index,
                BufferStateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Direct {
                    resource: Cow::Owned(buffer.clone()),
                },
                &mut self.temp,
            )
        };

        strict_assert!(self.temp.len() <= 1);

        Some((buffer.clone(), self.temp.pop()))
    }

    /// Sets the given state for all buffers in the given tracker.
    ///
    /// If a transition is needed to get the buffers into the needed state,
    /// those transitions are stored within the tracker. A subsequent
    /// call to [`Self::drain_transitions`] is needed to get those transitions.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn set_from_tracker(&mut self, tracker: &Self) {
        let incoming_size = tracker.start.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in tracker.metadata.owned_indices() {
            self.tracker_assert_in_bounds(index);
            tracker.tracker_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index,
                    BufferStateProvider::Indirect {
                        state: &tracker.start,
                    },
                    Some(BufferStateProvider::Indirect {
                        state: &tracker.end,
                    }),
                    ResourceMetadataProvider::Indirect {
                        metadata: &tracker.metadata,
                    },
                    &mut self.temp,
                )
            }
        }
    }

    /// Sets the given state for all buffers in the given UsageScope.
    ///
    /// If a transition is needed to get the buffers into the needed state,
    /// those transitions are stored within the tracker. A subsequent
    /// call to [`Self::drain_transitions`] is needed to get those transitions.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn set_from_usage_scope(&mut self, scope: &BufferUsageScope<A>) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in scope.metadata.owned_indices() {
            self.tracker_assert_in_bounds(index);
            scope.tracker_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index,
                    BufferStateProvider::Indirect {
                        state: &scope.state,
                    },
                    None,
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    &mut self.temp,
                )
            }
        }
    }

    /// Iterates through all buffers in the given bind group and adopts
    /// the state given for those buffers in the UsageScope. It also
    /// removes all touched buffers from the usage scope.
    ///
    /// If a transition is needed to get the buffers into the needed state,
    /// those transitions are stored within the tracker. A subsequent
    /// call to [`Self::drain_transitions`] is needed to get those transitions.
    ///
    /// This is a really funky method used by Compute Passes to generate
    /// barriers after a call to dispatch without needing to iterate
    /// over all elements in the usage scope. We use each the
    /// a given iterator of ids as a source of which IDs to look at.
    /// All the IDs must have first been added to the usage scope.
    ///
    /// # Safety
    ///
    /// [`Self::set_size`] must be called with the maximum possible Buffer ID before this
    /// method is called.
    pub unsafe fn set_and_remove_from_usage_scope_sparse(
        &mut self,
        scope: &mut BufferUsageScope<A>,
        id_source: impl IntoIterator<Item = BufferId>,
    ) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for id in id_source {
            let (index32, _, _) = id.unzip();
            let index = index32 as usize;

            scope.tracker_assert_in_bounds(index);

            if unsafe { !scope.metadata.contains_unchecked(index) } {
                continue;
            }
            unsafe {
                insert_or_barrier_update(
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index,
                    BufferStateProvider::Indirect {
                        state: &scope.state,
                    },
                    None,
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    &mut self.temp,
                )
            };

            unsafe { scope.metadata.remove(index) };
        }
    }

    #[allow(dead_code)]
    pub fn get(&self, id: BufferId) -> Option<&Arc<Buffer<A>>> {
        let index = id.unzip().0 as usize;
        if index > self.metadata.size() {
            return None;
        }
        self.tracker_assert_in_bounds(index);
        unsafe {
            if self.metadata.contains_unchecked(index) {
                return Some(self.metadata.get_resource_unchecked(index));
            }
        }
        None
    }
}

/// Source of Buffer State.
#[derive(Debug, Clone)]
enum BufferStateProvider<'a> {
    /// Get a state that was provided directly.
    Direct { state: BufferUses },
    /// Get a state from an an array of states.
    Indirect { state: &'a [BufferUses] },
}
impl BufferStateProvider<'_> {
    /// Gets the state from the provider, given a resource ID index.
    ///
    /// # Safety
    ///
    /// Index must be in bounds for the indirect source iff this is in the indirect state.
    #[inline(always)]
    unsafe fn get_state(&self, index: usize) -> BufferUses {
        match *self {
            BufferStateProvider::Direct { state } => state,
            BufferStateProvider::Indirect { state } => {
                strict_assert!(index < state.len());
                *unsafe { state.get_unchecked(index) }
            }
        }
    }
}

/// Does an insertion operation if the index isn't tracked
/// in the current metadata, otherwise merges the given state
/// with the current state. If the merging would cause
/// a conflict, returns that usage conflict.
///
/// # Safety
///
/// Indexes must be valid indexes into all arrays passed in
/// to this function, either directly or via metadata or provider structs.
#[inline(always)]
unsafe fn insert_or_merge<A: HalApi>(
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A, BufferId, Buffer<A>>,
    index32: u32,
    index: usize,
    state_provider: BufferStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A, BufferId, Buffer<A>>,
) -> Result<(), UsageConflict> {
    let currently_owned = unsafe { resource_metadata.contains_unchecked(index) };

    if !currently_owned {
        unsafe {
            insert(
                start_states,
                current_states,
                resource_metadata,
                index,
                state_provider,
                None,
                metadata_provider,
            )
        };
        return Ok(());
    }

    unsafe {
        merge(
            current_states,
            index32,
            index,
            state_provider,
            metadata_provider,
        )
    }
}

/// If the resource isn't tracked
/// - Inserts the given resource.
/// - Uses the `start_state_provider` to populate `start_states`
/// - Uses either `end_state_provider` or `start_state_provider`
///   to populate `current_states`.
/// If the resource is tracked
/// - Inserts barriers from the state in `current_states`
///   to the state provided by `start_state_provider`.
/// - Updates the `current_states` with either the state from
///   `end_state_provider` or `start_state_provider`.
///
/// Any barriers are added to the barrier vector.
///
/// # Safety
///
/// Indexes must be valid indexes into all arrays passed in
/// to this function, either directly or via metadata or provider structs.
#[inline(always)]
unsafe fn insert_or_barrier_update<A: HalApi>(
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A, BufferId, Buffer<A>>,
    index: usize,
    start_state_provider: BufferStateProvider<'_>,
    end_state_provider: Option<BufferStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A, BufferId, Buffer<A>>,
    barriers: &mut Vec<PendingTransition<BufferUses>>,
) {
    let currently_owned = unsafe { resource_metadata.contains_unchecked(index) };

    if !currently_owned {
        unsafe {
            insert(
                start_states,
                current_states,
                resource_metadata,
                index,
                start_state_provider,
                end_state_provider,
                metadata_provider,
            )
        };
        return;
    }

    let update_state_provider = end_state_provider.unwrap_or_else(|| start_state_provider.clone());
    unsafe { barrier(current_states, index, start_state_provider, barriers) };

    unsafe { update(current_states, index, update_state_provider) };
}

#[inline(always)]
unsafe fn insert<A: HalApi>(
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A, BufferId, Buffer<A>>,
    index: usize,
    start_state_provider: BufferStateProvider<'_>,
    end_state_provider: Option<BufferStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A, BufferId, Buffer<A>>,
) {
    let new_start_state = unsafe { start_state_provider.get_state(index) };
    let new_end_state =
        end_state_provider.map_or(new_start_state, |p| unsafe { p.get_state(index) });

    // This should only ever happen with a wgpu bug, but let's just double
    // check that resource states don't have any conflicts.
    strict_assert_eq!(invalid_resource_state(new_start_state), false);
    strict_assert_eq!(invalid_resource_state(new_end_state), false);

    log::trace!("\tbuf {index}: insert {new_start_state:?}..{new_end_state:?}");

    unsafe {
        if let Some(&mut ref mut start_state) = start_states {
            *start_state.get_unchecked_mut(index) = new_start_state;
        }
        *current_states.get_unchecked_mut(index) = new_end_state;

        let resource = metadata_provider.get_own(index);
        resource_metadata.insert(index, resource);
    }
}

#[inline(always)]
unsafe fn merge<A: HalApi>(
    current_states: &mut [BufferUses],
    index32: u32,
    index: usize,
    state_provider: BufferStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A, BufferId, Buffer<A>>,
) -> Result<(), UsageConflict> {
    let current_state = unsafe { current_states.get_unchecked_mut(index) };
    let new_state = unsafe { state_provider.get_state(index) };

    let merged_state = *current_state | new_state;

    if invalid_resource_state(merged_state) {
        return Err(UsageConflict::from_buffer(
            BufferId::zip(
                index32,
                unsafe { metadata_provider.get_epoch(index) },
                A::VARIANT,
            ),
            *current_state,
            new_state,
        ));
    }

    log::trace!("\tbuf {index32}: merge {current_state:?} + {new_state:?}");

    *current_state = merged_state;

    Ok(())
}

#[inline(always)]
unsafe fn barrier(
    current_states: &mut [BufferUses],
    index: usize,
    state_provider: BufferStateProvider<'_>,
    barriers: &mut Vec<PendingTransition<BufferUses>>,
) {
    let current_state = unsafe { *current_states.get_unchecked(index) };
    let new_state = unsafe { state_provider.get_state(index) };

    if skip_barrier(current_state, new_state) {
        return;
    }

    barriers.push(PendingTransition {
        id: index as _,
        selector: (),
        usage: current_state..new_state,
    });

    log::trace!("\tbuf {index}: transition {current_state:?} -> {new_state:?}");
}

#[inline(always)]
unsafe fn update(
    current_states: &mut [BufferUses],
    index: usize,
    state_provider: BufferStateProvider<'_>,
) {
    let current_state = unsafe { current_states.get_unchecked_mut(index) };
    let new_state = unsafe { state_provider.get_state(index) };

    *current_state = new_state;
}
