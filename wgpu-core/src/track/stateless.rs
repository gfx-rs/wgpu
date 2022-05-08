use std::marker::PhantomData;

use bit_vec::BitVec;

use crate::{
    hub,
    id::{TypedId, Valid},
    track::{iterate_bitvec_indices, resize_bitvec},
    Epoch, RefCount,
};

pub(crate) struct StatelessBindGroupSate<T, Id: TypedId> {
    resources: Vec<(Valid<Id>, RefCount)>,

    _phantom: PhantomData<T>,
}

impl<T: hub::Resource, Id: TypedId> StatelessBindGroupSate<T, Id> {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),

            _phantom: PhantomData,
        }
    }

    pub(crate) fn optimize(&mut self) {
        self.resources
            .sort_unstable_by_key(|&(id, _)| id.0.unzip().0);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<Id>> + '_ {
        self.resources.iter().map(|&(id, _)| id)
    }

    pub fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        let resource = storage.get(id).ok()?;

        self.resources
            .push((Valid(id), resource.life_guard().add_ref()));

        Some(resource)
    }
}

pub(crate) struct StatelessTracker<A: hub::HalApi, T, Id: TypedId> {
    ref_counts: Vec<Option<RefCount>>,
    epochs: Vec<Epoch>,

    owned: BitVec<usize>,

    _phantom: PhantomData<(A, T, Id)>,
}

impl<A: hub::HalApi, T: hub::Resource, Id: TypedId> StatelessTracker<A, T, Id> {
    pub fn new() -> Self {
        Self {
            ref_counts: Vec::new(),
            epochs: Vec::new(),

            owned: BitVec::default(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
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
        self.epochs.resize(size, u32::MAX);
        self.ref_counts.resize(size, None);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<Id>> + '_ {
        self.debug_assert_in_bounds(self.owned.len() - 1);
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            Valid(Id::zip(index as u32, epoch, A::VARIANT))
        })
    }

    pub unsafe fn init(&mut self, id: Valid<Id>, ref_count: RefCount) {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.debug_assert_in_bounds(index);

        *self.epochs.get_unchecked_mut(index) = epoch;
        *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);
        self.owned.set(index, true);
    }

    /// Requires set_size to be called
    pub unsafe fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        let item = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.debug_assert_in_bounds(index);

        *self.epochs.get_unchecked_mut(index) = epoch;
        *self.ref_counts.get_unchecked_mut(index) = Some(item.life_guard().add_ref());
        self.owned.set(index, true);

        Some(item)
    }

    pub unsafe fn extend_from_bind_group(&mut self, bind_group: &StatelessBindGroupSate<T, Id>) {
        for (id, ref_count) in &bind_group.resources {
            let (index32, epoch, _) = id.0.unzip();
            let index = index32 as usize;
            self.debug_assert_in_bounds(index);

            let previously_owned = self.owned.get(index).unwrap_unchecked();
            if !previously_owned {
                *self.epochs.get_unchecked_mut(index) = epoch;
                *self.ref_counts.get_unchecked_mut(index) = Some(ref_count.clone());
                self.owned.set(index, true);
            }
        }
    }

    pub fn extend_from_tracker(&mut self, other: &Self) {
        let incoming_size = other.owned.len();
        if incoming_size > self.owned.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&other.owned) {
            self.debug_assert_in_bounds(index);
            unsafe {
                let previously_owned = self.owned.get(index).unwrap_unchecked();

                if !previously_owned {
                    self.owned.set(index, true);

                    let other_ref_count = other
                        .ref_counts
                        .get_unchecked(index)
                        .clone()
                        .unwrap_unchecked();
                    *self.ref_counts.get_unchecked_mut(index) = Some(other_ref_count);

                    let epoch = *other.epochs.get_unchecked(index);
                    *self.epochs.get_unchecked_mut(index) = epoch;
                }
            }
        }
    }

    pub fn remove_abandoned(&mut self, id: Valid<Id>) -> bool {
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
