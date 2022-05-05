use std::vec::Drain;

use super::PendingTransition;
use crate::{
    hub,
    id::{BufferId, TypedId},
    resource::Buffer,
    track::{
        invalid_resource_state, iterate_bitvec, resize_bitvec, skip_barrier, ResourceUses,
        UsageConflict,
    },
    RefCount,
};
use bit_vec::BitVec;
use hal::BufferUses;

impl ResourceUses for BufferUses {
    const EXCLUSIVE: Self = Self::EXCLUSIVE;

    type Id = BufferId;
    type Selector = ();

    fn bits(self) -> u16 {
        self.bits()
    }

    fn all_ordered(self) -> bool {
        self.contains(Self::ORDERED)
    }

    fn any_exclusive(self) -> bool {
        self.intersects(Self::EXCLUSIVE)
    }
}

pub struct BufferBindGroupState {
    buffers: Vec<(BufferId, RefCount, BufferUses)>,
}
impl BufferBindGroupState {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }

    pub fn extend<'a, A: hal::Api>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        state: BufferUses,
    ) -> Option<&'a Buffer<A>> {
        let buffer = storage.get(id).ok()?;

        self.buffers.push((id, buffer.life_guard.add_ref(), state));

        Some(buffer)
    }
}

#[derive(Debug)]
pub(crate) struct BufferUsageScope {
    state: Vec<BufferUses>,
    owned: BitVec<usize>,
    ref_count: Vec<Option<RefCount>>,
}

impl BufferUsageScope {
    pub fn new() -> Self {
        Self {
            state: Vec::new(),
            owned: BitVec::default(),
            ref_count: Vec::new(),
        }
    }

    pub fn set_max_index(&mut self, size: usize) {
        self.state.resize(size, BufferUses::empty());
        self.ref_count.resize(size, None);

        resize_bitvec(&mut self.owned, size);
    }

    pub unsafe fn extend_from_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        bind_group: &BufferBindGroupState,
    ) -> Result<(), UsageConflict> {
        for (id, ref_count, state) in &bind_group.buffers {
            self.extend_inner(storage, id.unzip().0, ref_count, *state)?;
        }

        Ok(())
    }

    pub unsafe fn extend_from_scope<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        scope: &Self,
    ) -> Result<(), UsageConflict> {
        let incoming_size = scope.state.len();
        if incoming_size > self.state.len() {
            self.set_max_index(incoming_size);
        }

        iterate_bitvec(&scope.owned, |index| {
            unsafe {
                self.extend_inner(
                    storage,
                    index as u32,
                    scope
                        .ref_count
                        .get_unchecked(index)
                        .as_ref()
                        .unwrap_unchecked(),
                    *scope.state.get_unchecked(index),
                )
            };
        });

        Ok(())
    }

    pub unsafe fn extend<'a, A: hal::Api>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        new_state: BufferUses,
    ) -> Result<&'a Buffer<A>, UsageConflict> {
        let buffer = storage
            .get(id)
            .map_err(|_| UsageConflict::BufferInvalid { id: id.unzip().0 })?;

        self.extend_inner(
            storage,
            id.unzip().0,
            buffer.life_guard.ref_count.as_ref().unwrap(),
            new_state,
        )?;

        Ok(buffer)
    }

    unsafe fn extend_inner<'a, A: hal::Api>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        index: u32,
        ref_count: &RefCount,
        new_state: BufferUses,
    ) -> Result<(), UsageConflict> {
        let index = index as usize;

        let currently_active = self.owned.get(index).unwrap_unchecked();
        if currently_active {
            let current_state = *self.state.get_unchecked(index);

            let merged_state = current_state | new_state;

            if invalid_resource_state(merged_state) {
                return Err(UsageConflict::from_buffer(
                    index as u32,
                    current_state,
                    new_state,
                ));
            }

            *self.state.get_unchecked_mut(index) = merged_state;
        }

        // We're the first to use this resource, let's add it.
        self.owned.set(index, true);
        *self.ref_count.get_unchecked_mut(index) = Some(ref_count.clone());

        *self.state.get_unchecked_mut(index) = new_state;

        Ok(())
    }
}

pub(crate) struct BufferTracker {
    start: Vec<BufferUses>,
    end: Vec<BufferUses>,
    temp: Vec<PendingTransition<BufferUses>>,
    owned: BitVec<usize>,
    ref_count: Vec<Option<RefCount>>,
}
impl BufferTracker {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),
            temp: Vec::new(),
            owned: BitVec::default(),
            ref_count: Vec::new(),
        }
    }

    fn set_max_index(&mut self, size: usize) {
        self.start.resize(size, BufferUses::empty());
        self.end.resize(size, BufferUses::empty());
        self.ref_count.resize(size, None);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn drain(&mut self) -> Drain<PendingTransition<BufferUses>> {
        self.temp.drain(..)
    }

    pub unsafe fn init(&mut self, id: BufferId, state: BufferUses) {
        let index = id.unzip().0 as usize;
        *self.start.get_unchecked_mut(index) = state;
        *self.end.get_unchecked_mut(index) = state;

        self.owned.set(index, true);
    }

    pub unsafe fn change_state<'a, A: hal::Api>(
        &mut self,
        storage: &'a hub::Storage<Buffer<A>, BufferId>,
        id: BufferId,
        state: BufferUses,
    ) -> Option<(&'a Buffer<A>, Option<PendingTransition<BufferUses>>)> {
        let value = storage.get(id).ok()?;

        self.transition_inner(
            storage,
            id.unzip().0 as usize,
            value.life_guard.ref_count.as_ref().unwrap(),
            state,
        );

        Some((value, self.temp.pop()))
    }

    pub fn change_states_scope<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        scope: &BufferUsageScope,
    ) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_max_index(incoming_size);
        }

        iterate_bitvec(&scope.owned, |index| unsafe {
            let ref_count = scope
                .ref_count
                .get_unchecked(index)
                .as_ref()
                .unwrap_unchecked();

            self.transition(storage, &scope.state, ref_count, index);
        });
    }

    pub unsafe fn change_states_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        scope: &mut BufferUsageScope,
        bind_group_state: &BufferBindGroupState,
    ) {
        let incoming_size = scope.state.len();
        if incoming_size > self.start.len() {
            self.set_max_index(incoming_size);
        }

        for (index, ref_count, _) in bind_group_state.buffers.iter() {
            let index = index.unzip().0 as usize;
            if !scope.owned.get(index).unwrap_unchecked() {
                continue;
            }
            self.transition(storage, &scope.state, ref_count, index);
            scope.owned.set(index, false);
        }
    }

    unsafe fn transition<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        incoming_set: &Vec<BufferUses>,
        ref_count: &RefCount,
        index: usize,
    ) {
        let new_state = *incoming_set.get_unchecked(index);

        self.transition_inner(storage, index, ref_count, new_state);
    }

    unsafe fn transition_inner<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        index: usize,
        ref_count: &RefCount,
        new_state: BufferUses,
    ) {
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

            self.owned.set(index, true);
        }
    }
}
