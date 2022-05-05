use std::marker::PhantomData;

use bit_vec::BitVec;

use crate::{
    hub,
    id::TypedId,
    track::{iterate_bitvec, resize_bitvec},
    RefCount,
};

pub struct StatelessBindGroupSate<T, Id: TypedId> {
    resource: Vec<(Id, RefCount)>,

    _phantom: PhantomData<T>,
}

impl<T: hub::Resource, Id: TypedId> StatelessBindGroupSate<T, Id> {
    pub fn new() -> Self {
        Self {
            resource: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn extend<'a>(
        &mut self,
        storage: &'a hub::Storage<T, Id>,
        id: Id,
    ) -> Option<&'a T> {
        let resource = storage.get(id).ok()?;

        self.resource.push((id, resource.life_guard().add_ref()));

        Some(resource)
    }
}

pub struct StatelessTracker<T, Id: TypedId> {
    owned: BitVec<usize>,
    ref_counts: Vec<Option<RefCount>>,

    _phantom: PhantomData<(T, Id)>,
}

impl<T: hub::Resource, Id: TypedId> StatelessTracker<T, Id> {
    pub fn new() -> Self {
        Self {
            owned: BitVec::default(),
            ref_counts: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn set_max_index(&mut self, size: usize) {
        self.ref_counts.resize(size, None);
        resize_bitvec(&mut self.owned, size);
    }

    pub unsafe fn init(&mut self, id: Id, ref_count: RefCount) {
        let index = id.unzip().0 as usize;

        self.owned.set(index, true);
        *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);
    }

    pub unsafe fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        let item = storage.get(id).ok()?;
        let index = id.unzip().0 as usize;

        self.owned.set(index, true);
        *self.ref_counts.get_unchecked_mut(index) = Some(item.life_guard().add_ref());

        Some(item)
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        storage: &hub::Storage<T, Id>,
        bind_group: &StatelessBindGroupSate<T, Id>,
    ) {
        for (id, ref_count) in &bind_group.resource {
            let index = id.unzip().0 as usize;
            let previously_owned = self.owned.get(index).unwrap_unchecked();
            if !previously_owned {
                self.owned.set(id.unzip().0 as usize, true);
                *self.ref_counts.get_unchecked_mut(index) = Some(ref_count.clone());
            }
        }
    }

    pub fn extend_from_tracker(&mut self, storage: &hub::Storage<T, Id>, other: &Self) {
        let incoming_size = other.owned.len();
        if incoming_size > self.owned.len() {
            self.set_max_index(incoming_size);
        }

        iterate_bitvec(&other.owned, |index| unsafe {
            let previously_owned = self.owned.get(index).unwrap_unchecked();

            if !previously_owned {
                self.owned.set(index, true);

                let other_ref_count = other
                    .ref_counts
                    .get_unchecked(index)
                    .unwrap_unchecked()
                    .clone();
                *self.ref_counts.get_unchecked_mut(index) = Some(other_ref_count);
            }
        })
    }
}
