use std::vec::Drain;

use super::PendingTransition;
use crate::{
    hub,
    id::{BufferId, TypedId, Valid},
    resource::Buffer,
    track::{invalid_resource_state, resize_bitvec, skip_barrier, ResourceUses, UsageConflict},
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
    buffers: Vec<(Valid<BufferId>, BufferUses)>,
}

pub(crate) struct BufferUsageScope {
    state: Vec<BufferUses>,
    owned: BitVec<usize>,
}

impl BufferUsageScope {
    pub fn new() -> Self {
        Self {
            state: Vec::new(),
            owned: BitVec::default(),
        }
    }

    pub fn set_max_index(&mut self, size: usize) {
        self.state.resize(size, BufferUses::empty());

        resize_bitvec(&mut self.owned, size);
    }

    pub unsafe fn extend_from_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        bind_group: &BufferBindGroupState,
    ) -> Result<(), UsageConflict> {
        for &(id, state) in &bind_group.buffers {
            self.extend(storage, id, state)?;
        }

        Ok(())
    }

    pub unsafe fn extend<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        id: Valid<BufferId>,
        new_state: BufferUses,
    ) -> Result<(), UsageConflict> {
        let (index, _, _) = id.0.unzip();
        let index = index as usize;

        let currently_active = self.owned.get(index).unwrap_unchecked();
        if currently_active {
            let current_state = *self.state.get_unchecked(index);

            let merged_state = current_state | new_state;

            if invalid_resource_state(merged_state) {
                return Err(UsageConflict::from_buffer(id, current_state, new_state));
            }

            *self.state.get_unchecked_mut(index) = merged_state;
        }

        // We're the first to use this resource, let's add it.
        self.owned.set(index, true);

        *self.state.get_unchecked_mut(index) = new_state;

        Ok(())
    }
}

pub(crate) struct BufferTracker {
    start: Vec<BufferUses>,
    end: Vec<BufferUses>,
    temp: Vec<PendingTransition<BufferUses>>,
    owned: BitVec<usize>,
}
impl BufferTracker {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),
            temp: Vec::new(),
            owned: BitVec::default(),
        }
    }

    fn set_max_index(&mut self, size: usize) {
        self.start.resize(size, BufferUses::empty());
        self.end.resize(size, BufferUses::empty());

        resize_bitvec(&mut self.owned, size);
    }

    pub fn change_states<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        incoming_set: &Vec<BufferUses>,
        incoming_ownership: &BitVec<usize>,
    ) -> Drain<PendingTransition<BufferUses>> {
        let incoming_size = incoming_set.len();
        if incoming_size > self.start.len() {
            self.set_max_index(incoming_size);
        }

        for (word_index, mut word) in incoming_ownership.blocks().enumerate() {
            if word == 0 {
                continue;
            }

            let bit_start = word_index * 64;
            let bit_end = ((word_index + 1) * 64).min(incoming_size);

            for index in bit_start..bit_end {
                if word & 0b1 == 0 {
                    continue;
                }
                word >>= 1;

                unsafe { self.transition(storage, incoming_set, index) };
            }
        }

        self.temp.drain(..)
    }

    pub unsafe fn change_states_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        incoming_set: &Vec<BufferUses>,
        incoming_ownership: &mut BitVec<usize>,
        bind_group_state: &BufferBindGroupState,
    ) -> Drain<PendingTransition<BufferUses>> {
        let incoming_size = incoming_set.len();
        if incoming_size > self.start.len() {
            self.set_max_index(incoming_size);
        }

        for &(index, _) in bind_group_state.buffers.iter() {
            let index = index.0.unzip().0 as usize;
            if !incoming_ownership.get(index).unwrap_unchecked() {
                continue;
            }
            self.transition(storage, incoming_set, index);
            incoming_ownership.set(index, false);
        }

        self.temp.drain(..)
    }

    unsafe fn transition<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Buffer<A>, BufferId>,
        incoming_set: &Vec<BufferUses>,
        index: usize,
    ) {
        let old_tracked = self.owned.get(index).unwrap_unchecked();
        let old_state = *self.end.get_unchecked(index);
        let new_state = *incoming_set.get_unchecked(index);

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::id::Id;

    #[test]
    fn change_extend() {
        let mut bs = Unit {
            first: None,
            last: BufferUses::INDEX,
        };
        let id = Id::dummy();
        assert_eq!(
            bs.change(id, (), BufferUses::STORAGE_WRITE, None),
            Err(PendingTransition {
                id,
                selector: (),
                usage: BufferUses::INDEX..BufferUses::STORAGE_WRITE,
            }),
        );
        bs.change(id, (), BufferUses::VERTEX, None).unwrap();
        bs.change(id, (), BufferUses::INDEX, None).unwrap();
        assert_eq!(bs, Unit::new(BufferUses::VERTEX | BufferUses::INDEX));
    }

    #[test]
    fn change_replace() {
        let mut bs = Unit {
            first: None,
            last: BufferUses::STORAGE_WRITE,
        };
        let id = Id::dummy();
        let mut list = Vec::new();
        bs.change(id, (), BufferUses::VERTEX, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::STORAGE_WRITE..BufferUses::VERTEX,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::STORAGE_WRITE),
                last: BufferUses::VERTEX,
            }
        );

        list.clear();
        bs.change(id, (), BufferUses::STORAGE_WRITE, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::VERTEX..BufferUses::STORAGE_WRITE,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::STORAGE_WRITE),
                last: BufferUses::STORAGE_WRITE,
            }
        );
    }

    #[test]
    fn merge_replace() {
        let mut bs = Unit {
            first: None,
            last: BufferUses::empty(),
        };
        let other_smooth = Unit {
            first: Some(BufferUses::empty()),
            last: BufferUses::COPY_DST,
        };
        let id = Id::dummy();
        let mut list = Vec::new();
        bs.merge(id, &other_smooth, Some(&mut list)).unwrap();
        assert!(list.is_empty());
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::empty()),
                last: BufferUses::COPY_DST,
            }
        );

        let other_rough = Unit {
            first: Some(BufferUses::empty()),
            last: BufferUses::UNIFORM,
        };
        bs.merge(id, &other_rough, Some(&mut list)).unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::COPY_DST..BufferUses::empty(),
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::empty()),
                last: BufferUses::UNIFORM,
            }
        );
    }
}
