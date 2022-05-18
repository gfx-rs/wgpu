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

    pub(crate) fn optimize(&mut self) {
        self.buffers
            .sort_unstable_by_key(|&(id, _, _)| id.0.unzip().0);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.buffers.iter().map(|&(id, _, _)| id)
    }

    pub fn extend<'a>(
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

    pub fn set_size(&mut self, size: usize) {
        self.state.resize(size, BufferUses::empty());
        self.metadata.set_size(size);
    }

    fn allow_index(&mut self, index: usize) {
        if index >= self.state.len() {
            self.set_size(index + 1);
        }
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.metadata.used()
    }

    pub unsafe fn extend_from_bind_group(
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
                StateProvider::Direct { state },
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Borrowed(ref_count),
                },
            )?;
        }

        Ok(())
    }

    pub fn extend_from_scope(&mut self, scope: &Self) -> Result<(), UsageConflict> {
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
                    StateProvider::Indirect {
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

    pub fn extend<'a>(
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
                StateProvider::Direct { state: new_state },
                ResourceMetadataProvider::Resource { epoch },
            )?;
        }

        Ok(buffer)
    }
}

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

    pub fn set_size(&mut self, size: usize) {
        self.start.resize(size, BufferUses::empty());
        self.end.resize(size, BufferUses::empty());

        self.metadata.set_size(size);
    }

    fn allow_index(&mut self, index: usize) {
        if index >= self.start.len() {
            self.set_size(index + 1);
        }
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.metadata.used()
    }

    pub fn drain(&mut self) -> Drain<PendingTransition<BufferUses>> {
        self.temp.drain(..)
    }

    pub fn init(&mut self, id: Valid<BufferId>, ref_count: RefCount, state: BufferUses) {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            insert(
                None,
                Some(&mut self.start),
                &mut self.end,
                &mut self.metadata,
                index,
                StateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Owned(ref_count),
                },
            )
        }
    }

    pub fn change_state<'a>(
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
                StateProvider::Direct { state },
                None,
                ResourceMetadataProvider::Resource { epoch },
                &mut self.temp,
            )
        };

        debug_assert!(self.temp.len() <= 1);

        Some((value, self.temp.pop()))
    }

    pub fn change_states_tracker(&mut self, tracker: &Self) {
        let incoming_size = tracker.start.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&tracker.metadata.owned) {
            tracker.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    None,
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index as u32,
                    index,
                    StateProvider::Indirect {
                        state: &tracker.start,
                    },
                    Some(StateProvider::Indirect {
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

    pub fn change_states_scope(&mut self, scope: &BufferUsageScope<A>) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.metadata.owned) {
            scope.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    None,
                    Some(&mut self.start),
                    &mut self.end,
                    &mut self.metadata,
                    index as u32,
                    index,
                    StateProvider::Indirect {
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

    pub unsafe fn change_states_bind_group(
        &mut self,
        scope: &mut BufferUsageScope<A>,
        bind_group_state: &BufferBindGroupState<A>,
    ) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for &(id, ref ref_count, _) in bind_group_state.buffers.iter() {
            let (index32, epoch, _) = id.0.unzip();
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
                StateProvider::Indirect {
                    state: &scope.state,
                },
                None,
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Borrowed(ref_count),
                },
                &mut self.temp,
            );

            scope.metadata.reset(index);
        }
    }

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

#[derive(Debug, Clone)]
enum StateProvider<'a> {
    Direct { state: BufferUses },
    Indirect { state: &'a [BufferUses] },
}
impl StateProvider<'_> {
    #[inline(always)]
    unsafe fn get_state(&self, index: usize) -> BufferUses {
        match *self {
            StateProvider::Direct { state } => state,
            StateProvider::Indirect { state: other } => other[index],
        }
    }
}

#[inline(always)]
unsafe fn insert_or_merge<A: hub::HalApi>(
    life_guard: Option<&LifeGuard>,
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    state_provider: StateProvider<'_>,
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

#[inline(always)]
unsafe fn insert_or_barrier_update<A: hub::HalApi>(
    life_guard: Option<&LifeGuard>,
    start_states: Option<&mut [BufferUses]>,
    current_states: &mut [BufferUses],
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    start_state_provider: StateProvider<'_>,
    end_state_provider: Option<StateProvider<'_>>,
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
    start_state_provider: StateProvider<'_>,
    end_state_provider: Option<StateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) {
    let new_start_state = start_state_provider.get_state(index);
    let new_end_state = end_state_provider.map_or(new_start_state, |p| p.get_state(index));

    // This should only ever happen with a wgpu bug, but let's just double
    // check that resource states don't have any conflicts.
    debug_assert_eq!(invalid_resource_state(new_start_state), false);
    debug_assert_eq!(invalid_resource_state(new_end_state), false);

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
    state_provider: StateProvider<'_>,
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

    *current_state = merged_state;

    Ok(())
}

#[inline(always)]
unsafe fn barrier(
    current_states: &mut [BufferUses],
    index32: u32,
    index: usize,
    state_provider: StateProvider<'_>,
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
    })
}

#[inline(always)]
unsafe fn update(
    current_states: &mut [BufferUses],
    index: usize,
    state_provider: StateProvider<'_>,
) {
    let current_state = current_states.get_unchecked_mut(index);
    let new_state = state_provider.get_state(index);

    *current_state = new_state;
}
