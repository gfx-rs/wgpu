use std::{marker::PhantomData, vec::Drain};

use super::PendingTransition;
use crate::{
    hub,
    id::{BufferId, TypedId, Valid},
    resource::Buffer,
    track::{
        invalid_resource_state, iterate_bitvec_indices, resize_bitvec, skip_barrier, ResourceUses,
        UsageConflict,
    },
    Epoch, RefCount,
};
use bit_vec::BitVec;
use hal::BufferUses;

impl ResourceUses for BufferUses {
    const EXCLUSIVE: Self = Self::EXCLUSIVE;

    type Id = BufferId;
    type Selector = ();

    fn bits(self) -> u16 {
        Self::bits(&self)
    }

    fn all_ordered(self) -> bool {
        self.contains(Self::ORDERED)
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

    ref_counts: Vec<Option<RefCount>>,
    epochs: Vec<Epoch>,

    owned: BitVec<usize>,

    _phantom: PhantomData<A>,
}

impl<A: hub::HalApi> BufferUsageScope<A> {
    pub fn new() -> Self {
        Self {
            state: Vec::new(),

            ref_counts: Vec::new(),
            epochs: Vec::new(),

            owned: BitVec::default(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.state.len());
        debug_assert!(index < self.ref_counts.len());
        debug_assert!(index < self.epochs.len());
        debug_assert!(index < self.owned.len());

        debug_assert!(if self.owned.get(index).unwrap() {
            self.ref_counts[index].is_some()
        } else {
            true
        });
    }

    pub fn set_size(&mut self, size: usize) {
        self.state.resize(size, BufferUses::empty());
        self.ref_counts.resize(size, None);
        self.epochs.resize(size, u32::MAX);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.debug_assert_in_bounds(self.owned.len() - 1);
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            Valid(BufferId::zip(index as u32, epoch, A::VARIANT))
        })
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        bind_group: &BufferBindGroupState<A>,
    ) -> Result<(), UsageConflict> {
        for (id, ref_count, state) in &bind_group.buffers {
            let (index32, epoch, _) = id.0.unzip();
            let index = index32 as usize;

            self.extend_inner(*id, index, epoch, ref_count, *state)?;
        }

        Ok(())
    }

    pub fn extend_from_scope(&mut self, scope: &Self) -> Result<(), UsageConflict> {
        let incoming_size = scope.state.len();
        if incoming_size > self.state.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.owned) {
            self.debug_assert_in_bounds(index);
            scope.debug_assert_in_bounds(index);

            unsafe {
                let ref_count = scope
                    .ref_counts
                    .get_unchecked(index)
                    .as_ref()
                    .unwrap_unchecked();
                let epoch = *scope.epochs.get_unchecked(index);
                let new_state = *scope.state.get_unchecked(index);

                self.extend_inner(
                    Valid(BufferId::zip(index as u32, epoch, A::VARIANT)),
                    index,
                    epoch,
                    ref_count,
                    new_state,
                )?;
            };
        }

        Ok(())
    }

    pub unsafe fn extend<'a>(
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

        self.extend_inner(
            Valid(id),
            index,
            epoch,
            buffer.life_guard.ref_count.as_ref().unwrap(),
            new_state,
        )?;

        Ok(buffer)
    }

    unsafe fn extend_inner<'a>(
        &mut self,
        id: Valid<BufferId>,
        index: usize,
        epoch: u32,
        ref_count: &RefCount,
        new_state: BufferUses,
    ) -> Result<(), UsageConflict> {
        self.debug_assert_in_bounds(index);

        let currently_active = self.owned.get(index).unwrap_unchecked();
        if currently_active {
            let current_state = *self.state.get_unchecked(index);

            let merged_state = current_state | new_state;

            if invalid_resource_state(merged_state) {
                return Err(UsageConflict::from_buffer(id.0, current_state, new_state));
            }

            *self.state.get_unchecked_mut(index) = merged_state;
        }

        // We're the first to use this resource, let's add it.
        *self.epochs.get_unchecked_mut(index) = epoch;
        *self.ref_counts.get_unchecked_mut(index) = Some(ref_count.clone());
        self.owned.set(index, true);

        *self.state.get_unchecked_mut(index) = new_state;

        Ok(())
    }
}

pub(crate) struct BufferTracker<A: hub::HalApi> {
    start: Vec<BufferUses>,
    end: Vec<BufferUses>,

    epochs: Vec<Epoch>,
    ref_counts: Vec<Option<RefCount>>,
    owned: BitVec<usize>,

    temp: Vec<PendingTransition<BufferUses>>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> BufferTracker<A> {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),

            epochs: Vec::new(),
            ref_counts: Vec::new(),
            owned: BitVec::default(),

            temp: Vec::new(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.start.len());
        debug_assert!(index < self.end.len());
        debug_assert!(index < self.ref_counts.len());
        debug_assert!(index < self.epochs.len());
        debug_assert!(index < self.owned.len());

        debug_assert!(if self.owned.get(index).unwrap() {
            self.ref_counts[index].is_some()
        } else {
            true
        });
    }

    pub fn set_size(&mut self, size: usize) {
        self.start.resize(size, BufferUses::empty());
        self.end.resize(size, BufferUses::empty());

        self.epochs.resize(size, u32::MAX);
        self.ref_counts.resize(size, None);

        resize_bitvec(&mut self.owned, size);
    }

    fn allow_index(&mut self, index: usize) {
        if index >= self.start.len() {
            self.set_size(index + 1);
        }
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<BufferId>> + '_ {
        self.debug_assert_in_bounds(self.owned.len() - 1);
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            Valid(BufferId::zip(index as u32, epoch, A::VARIANT))
        })
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
            *self.start.get_unchecked_mut(index) = state;
            *self.end.get_unchecked_mut(index) = state;

            *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);
            *self.epochs.get_unchecked_mut(index) = epoch;

            self.owned.set(index, true);
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

        unsafe {
            self.transition_inner(
                index,
                epoch,
                value.life_guard.ref_count.as_ref().unwrap(),
                state,
            )
        };

        Some((value, self.temp.pop()))
    }

    pub fn change_states_tracker(&mut self, tracker: &Self) {
        let incoming_size = tracker.start.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&tracker.owned) {
            tracker.debug_assert_in_bounds(index);
            unsafe {
                let ref_count = tracker
                    .ref_counts
                    .get_unchecked(index)
                    .as_ref()
                    .unwrap_unchecked();

                let epoch = *tracker.epochs.get_unchecked(index);

                self.transition(&tracker.start, ref_count, index, epoch);
            }
        }
    }

    pub fn change_states_scope(&mut self, scope: &BufferUsageScope<A>) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.owned) {
            scope.debug_assert_in_bounds(index);
            unsafe {
                let ref_count = scope
                    .ref_counts
                    .get_unchecked(index)
                    .as_ref()
                    .unwrap_unchecked();

                let epoch = *scope.epochs.get_unchecked(index);

                self.transition(&scope.state, ref_count, index, epoch);
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

        for (id, ref_count, _) in bind_group_state.buffers.iter() {
            let (index32, epoch, _) = id.0.unzip();
            let index = index32 as usize;

            scope.debug_assert_in_bounds(index);

            if !scope.owned.get(index).unwrap_unchecked() {
                continue;
            }
            self.transition(&scope.state, ref_count, index, epoch);

            *scope.ref_counts.get_unchecked_mut(index) = None;
            *scope.epochs.get_unchecked_mut(index) = u32::MAX;
            scope.owned.set(index, false);
        }
    }

    unsafe fn transition(
        &mut self,
        incoming_set: &Vec<BufferUses>,
        ref_count: &RefCount,
        index: usize,
        epoch: u32,
    ) {
        let new_state = *incoming_set.get_unchecked(index);

        self.transition_inner(index, epoch, ref_count, new_state);
    }

    unsafe fn transition_inner(
        &mut self,
        index: usize,
        epoch: u32,
        ref_count: &RefCount,
        new_state: BufferUses,
    ) {
        self.debug_assert_in_bounds(index);

        let old_tracked = self.owned.get(index).unwrap_unchecked();
        let old_state = *self.end.get_unchecked(index);

        if old_tracked {
            if skip_barrier(old_state, new_state) {
                return;
            }

            self.temp.push(PendingTransition {
                id: index as u32,
                selector: (),
                usage: old_state..new_state,
            });

            *self.end.get_unchecked_mut(index) = new_state;
        } else {
            *self.start.get_unchecked_mut(index) = new_state;
            *self.end.get_unchecked_mut(index) = new_state;

            *self.ref_counts.get_unchecked_mut(index) = Some(ref_count.clone());
            *self.epochs.get_unchecked_mut(index) = epoch;

            self.owned.set(index, true);
        }
    }

    pub fn remove_abandoned(&mut self, id: Valid<BufferId>) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.owned.len() {
            return false;
        }

        self.debug_assert_in_bounds(index);

        unsafe {
            if self.owned.get(index).unwrap_unchecked() {
                let existing_epoch = self.epochs.get_unchecked_mut(index);
                let existing_ref_count = self.ref_counts.get_unchecked_mut(index);

                if *existing_epoch == epoch
                    && existing_ref_count.as_mut().unwrap_unchecked().load() == 1
                {
                    self.owned.set(index, false);
                    *existing_epoch = u32::MAX;
                    *existing_ref_count = None;

                    return true;
                }
            }
        }

        false
    }
}
