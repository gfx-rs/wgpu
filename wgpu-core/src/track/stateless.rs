use std::marker::PhantomData;

use crate::{
    hub,
    id::{TypedId, Valid},
    track::{iterate_bitvec_indices, ResourceMetadata},
    RefCount,
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
    metadata: ResourceMetadata<A>,

    _phantom: PhantomData<(T, Id)>,
}

impl<A: hub::HalApi, T: hub::Resource, Id: TypedId> StatelessTracker<A, T, Id> {
    pub fn new() -> Self {
        Self {
            metadata: ResourceMetadata::new(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        self.metadata.debug_assert_in_bounds(index);
    }

    pub fn set_size(&mut self, size: usize) {
        self.metadata.set_size(size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<Id>> + '_ {
        self.metadata.used()
    }

    fn allow_index(&mut self, index: usize) {
        if index >= self.metadata.owned.len() {
            self.set_size(index + 1);
        }
    }

    pub fn init(&mut self, id: Valid<Id>, ref_count: RefCount) {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            *self.metadata.epochs.get_unchecked_mut(index) = epoch;
            *self.metadata.ref_counts.get_unchecked_mut(index) = Some(ref_count);
            self.metadata.owned.set(index, true);
        }
    }

    /// Requires set_size to be called
    pub fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        let item = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            *self.metadata.epochs.get_unchecked_mut(index) = epoch;
            *self.metadata.ref_counts.get_unchecked_mut(index) = Some(item.life_guard().add_ref());
            self.metadata.owned.set(index, true);
        }

        Some(item)
    }

    pub fn extend_from_tracker(&mut self, other: &Self) {
        let incoming_size = other.metadata.owned.len();
        if incoming_size > self.metadata.owned.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&other.metadata.owned) {
            self.debug_assert_in_bounds(index);
            other.debug_assert_in_bounds(index);
            unsafe {
                let previously_owned = self.metadata.owned.get(index).unwrap_unchecked();

                if !previously_owned {
                    self.metadata.owned.set(index, true);

                    let other_ref_count = other
                        .metadata
                        .ref_counts
                        .get_unchecked(index)
                        .clone()
                        .unwrap_unchecked();
                    *self.metadata.ref_counts.get_unchecked_mut(index) = Some(other_ref_count);

                    let epoch = *other.metadata.epochs.get_unchecked(index);
                    *self.metadata.epochs.get_unchecked_mut(index) = epoch;
                }
            }
        }
    }

    pub fn remove_abandoned(&mut self, id: Valid<Id>) -> bool {
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
