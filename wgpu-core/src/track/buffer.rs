/*! Buffer Trackers
 *
 * Buffers are represented by a single state for the whole resource,
 * a 16 bit bitflag of buffer usages. Because there is only ever
 * one subresource, they have no selector.
!*/

use std::{borrow::Cow, marker::PhantomData, vec::Drain};

use super::PendingTransition;
use crate::{
    hub,
    id::{BufferId, TypedId, Valid},
    resource::Buffer,
    track::{
        invalid_resource_state, iterate_bitvec_indices, skip_barrier, ResourceMetadata,
        ResourceMetadataProvider, ResourceUses, UsageConflict,
    },
    LifeGuard, RefCount,
};
use hal::BufferUses;

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
pub(crate) struct BufferBindGroupState<A: hub::HalApi> {
    buffers: Vec<(Valid<BufferId>, RefCount, BufferUses)>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> BufferBindGroupState<A> {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),

            _phantom: PhantomData,
        }
    }

    /// Optimize the buffer bind group state by sorting it by ID.
    ///
    /// When this list of states is merged into a tracker, the memory
    /// accesses will be in a constant assending order.
    pub(crate) fn optimize(&mut self) {
        self.buffers
            .sort_unstable_by_key(|&(id, _, _)| id.0.unzip().0);
    }

    /// Returns a list of all buffers tracked. May contain duplicates.
    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.buffers.iter().map(|&(id, _, _)| id)
    }

    /// Adds the given resource with the given state.
    pub fn add_single<'a>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        state: BufferUses,
    ) -> Option<&'a Buffer<A>> {
        let buffer = storage.get(id).ok()?;

        self.buffers
            .push((Valid(id), buffer.life_guard.add_ref(), state));

        Some(buffer)
    }
}

/// Stores all buffer state within a single usage scope.
#[derive(Debug)]
pub(crate) struct BufferUsageScope<A: hub::HalApi> {
    state: Vec<BufferUses>,

    metadata: ResourceMetadata<A>,
}

impl<A: hub::HalApi> BufferUsageScope<A> {
    pub fn new() -> Self {
        Self {
            state: Vec::new(),

            metadata: ResourceMetadata::new(),
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.state.len());
        self.metadata.debug_assert_in_bounds(index);
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

    /// Returns a list of all buffers tracked.
    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.metadata.used()
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
        for &(id, ref ref_count, state) in &bind_group.buffers {
            let (index32, epoch, _) = id.0.unzip();
            let index = index32 as usize;

            insert_or_merge(
                None,
                None,
                &mut self.state,
                &mut self.metadata,
                index32,
                index,
                BufferStateProvider::Direct { state },
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Borrowed(ref_count),
                },
            )?;
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

        for index in iterate_bitvec_indices(&scope.metadata.owned) {
            self.debug_assert_in_bounds(index);
            scope.debug_assert_in_bounds(index);

            unsafe {
                insert_or_merge(
                    None,
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
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        new_state: BufferUses,
    ) -> Result<&'a Buffer<A>, UsageConflict> {
        let buffer = storage
            .get(id)
            .map_err(|_| UsageConflict::BufferInvalid { id })?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            insert_or_merge(
                Some(&buffer.life_guard),
                None,
                &mut self.state,
                &mut self.metadata,
                index32,
                index,
                BufferStateProvider::Direct { state: new_state },
                ResourceMetadataProvider::Resource { epoch },
            )?;
        }

        Ok(buffer)
    }
}

/// Stores all buffer state within a command buffer or device.
pub(crate) struct BufferTracker<A: hub::HalApi> {
    start: Vec<BufferUses>,
    end: Vec<BufferUses>,

    metadata: ResourceMetadata<A>,

    temp: Vec<PendingTransition<BufferUses>>,
}
impl<A: hub::HalApi> BufferTracker<A> {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),

            metadata: ResourceMetadata::new(),

            temp: Vec::new(),
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.start.len());
        debug_assert!(index < self.end.len());
        self.metadata.debug_assert_in_bounds(index);
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
    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.metadata.used()
    }

    /// Drains all currently pending transitions.
    pub fn drain(&mut self) -> Drain<'_, PendingTransition<BufferUses>> {
        self.temp.drain(..)
    }

    /// Inserts a single buffer and its state into the resource tracker.
    ///
    /// If the resource already exists in the tracker, this will panic.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn insert_single(&mut self, id: Valid<BufferId>, ref_count: RefCount, state: BufferUses) {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            let currently_owned = self.metadata.owned.get(index).unwrap_unchecked();

            if currently_owned {
                panic!("Tried to insert buffer already tracked");
            }

            insert(
                None,
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index,
                BufferStateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Owned(ref_count),
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
    pub fn set_single<'a>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        state: BufferUses,
    ) -> Option<(&'a Buffer<A>, Option<PendingTransition<BufferUses>>)> {
        let value = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            insert_or_barrier_update(
                Some(&value.life_guard),
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index32,
                index,
                BufferStateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Resource { epoch },
                &mut self.temp,
            )
        };

        debug_assert!(self.temp.len() <= 1);

        Some((value, self.temp.pop()))
    }

    /// Sets the given state for all buffers in the given tracker.
    ///
    /// If a transition is needed to get the buffers into the needed state,
    /// those transitions are stored within the tracker. A subsequent
    /// call to [`Self::drain`] is needed to get those transitions.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn set_from_tracker(&mut self, tracker: &Self) {
        let incoming_size = tracker.start.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&tracker.metadata.owned) {
            self.debug_assert_in_bounds(index);
            tracker.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    None,
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index as u32,
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
    /// call to [`Self::drain`] is needed to get those transitions.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn set_from_usage_scope(&mut self, scope: &BufferUsageScope<A>) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.metadata.owned) {
            self.debug_assert_in_bounds(index);
            scope.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    None,
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index as u32,
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
    /// call to [`Self::drain`] is needed to get those transitions.
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
        id_source: impl IntoIterator<Item = Valid<BufferId>>,
    ) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for id in id_source {
            let (index32, _, _) = id.0.unzip();
            let index = index32 as usize;

            scope.debug_assert_in_bounds(index);

            if !scope.metadata.owned.get(index).unwrap_unchecked() {
                continue;
            }
            insert_or_barrier_update(
                None,
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index as u32,
                index,
                BufferStateProvider::Indirect {
                    state: &scope.state,
                },
                None,
                ResourceMetadataProvider::Indirect {
                    metadata: &scope.metadata,
                },
                &mut self.temp,
            );

            scope.metadata.reset(index);
        }
    }

    /// Removes the given resource from the tracker iff we have the last reference to the
    /// resource and the epoch matches.
    ///
    /// Returns true if the resource was removed.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// false will be returned.
    pub fn remove_abandoned(&mut self, id: Valid<BufferId>) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.metadata.owned.len() {
            return false;
        }

        self.debug_assert_in_bounds(index);

        unsafe {
            if self.metadata.owned.get(index).unwrap_unchecked() {
                let existing_epoch = self.metadata.epochs.get_unchecked_mut(index);
                let existing_ref_count = self.metadata.ref_counts.get_unchecked_mut(index);

                if *existing_epoch == epoch
                    && existing_ref_count.as_mut().unwrap_unchecked().load() == 1
                {
                    self.metadata.reset(index);

                    return true;
                }
            }
        }

        false
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
                debug_assert!(index < state.len());
                *state.get_unchecked(index)
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
unsafe fn insert_or_merge<A: hub::HalApi>(
    life_guard: Option<&LifeGuard>,
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    state_provider: BufferStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) -> Result<(), UsageConflict> {
    let currently_owned = resource_metadata.owned.get(index).unwrap_unchecked();

    if !currently_owned {
        insert(
            life_guard,
            start_states,
            current_states,
            resource_metadata,
            index,
            state_provider,
            None,
            metadata_provider,
        );
        return Ok(());
    }

    merge(
        current_states,
        index32,
        index,
        state_provider,
        metadata_provider,
    )
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
unsafe fn insert_or_barrier_update<A: hub::HalApi>(
    life_guard: Option<&LifeGuard>,
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    start_state_provider: BufferStateProvider<'_>,
    end_state_provider: Option<BufferStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
    barriers: &mut Vec<PendingTransition<BufferUses>>,
) {
    let currently_owned = resource_metadata.owned.get(index).unwrap_unchecked();

    if !currently_owned {
        insert(
            life_guard,
            start_states,
            current_states,
            resource_metadata,
            index,
            start_state_provider,
            end_state_provider,
            metadata_provider,
        );
        return;
    }

    let update_state_provider = end_state_provider.unwrap_or_else(|| start_state_provider.clone());
    barrier(
        current_states,
        index32,
        index,
        start_state_provider,
        barriers,
    );

    update(current_states, index, update_state_provider);
}

#[inline(always)]
unsafe fn insert<A: hub::HalApi>(
    life_guard: Option<&LifeGuard>,
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A>,
    index: usize,
    start_state_provider: BufferStateProvider<'_>,
    end_state_provider: Option<BufferStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) {
    let new_start_state = start_state_provider.get_state(index);
    let new_end_state = end_state_provider.map_or(new_start_state, |p| p.get_state(index));

    // This should only ever happen with a wgpu bug, but let's just double
    // check that resource states don't have any conflicts.
    debug_assert_eq!(invalid_resource_state(new_start_state), false);
    debug_assert_eq!(invalid_resource_state(new_end_state), false);

    log::trace!("\tbuf {index}: insert {new_start_state:?}..{new_end_state:?}");

    if let Some(&mut ref mut start_state) = start_states {
        *start_state.get_unchecked_mut(index) = new_start_state;
    }
    *current_states.get_unchecked_mut(index) = new_end_state;

    let (epoch, ref_count) = metadata_provider.get_own(life_guard, index);

    resource_metadata.owned.set(index, true);
    *resource_metadata.epochs.get_unchecked_mut(index) = epoch;
    *resource_metadata.ref_counts.get_unchecked_mut(index) = Some(ref_count);
}

#[inline(always)]
unsafe fn merge<A: hub::HalApi>(
    current_states: &mut [BufferUses],
    index32: u32,
    index: usize,
    state_provider: BufferStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) -> Result<(), UsageConflict> {
    let current_state = current_states.get_unchecked_mut(index);
    let new_state = state_provider.get_state(index);

    let merged_state = *current_state | new_state;

    if invalid_resource_state(merged_state) {
        return Err(UsageConflict::from_buffer(
            BufferId::zip(index32, metadata_provider.get_epoch(index), A::VARIANT),
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
    index32: u32,
    index: usize,
    state_provider: BufferStateProvider<'_>,
    barriers: &mut Vec<PendingTransition<BufferUses>>,
) {
    let current_state = *current_states.get_unchecked(index);
    let new_state = state_provider.get_state(index);

    if skip_barrier(current_state, new_state) {
        return;
    }

    barriers.push(PendingTransition {
        id: index32,
        selector: (),
        usage: current_state..new_state,
    });

    log::trace!("\tbuf {index32}: transition {current_state:?} -> {new_state:?}");
}

#[inline(always)]
unsafe fn update(
    current_states: &mut [BufferUses],
    index: usize,
    state_provider: BufferStateProvider<'_>,
) {
    let current_state = current_states.get_unchecked_mut(index);
    let new_state = state_provider.get_state(index);

    *current_state = new_state;
}
