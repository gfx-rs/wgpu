use std::marker::PhantomData;

use bit_vec::BitVec;

use crate::{hub, id::TypedId, track::resize_bitvec};

pub struct StatelessBindGroupSate<T, Id: TypedId> {
    resource: Vec<Id>,

    _phantom: PhantomData<T>,
}

impl<T, Id: TypedId> StatelessBindGroupSate<T, Id> {
    pub fn new() -> Self {
        Self {
            resource: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        self.resource.push(id);

        storage.get(id).ok()
    }
}

pub struct StatelessTracker<T, Id: TypedId> {
    owned: BitVec<usize>,

    _phantom: PhantomData<(T, Id)>,
}

impl<T, Id: TypedId> StatelessTracker<T, Id> {
    pub fn new() -> Self {
        Self {
            owned: BitVec::default(),
            _phantom: PhantomData,
        }
    }

    pub fn set_max_index(&mut self, size: usize) {
        resize_bitvec(&mut self.owned, size);
    }

    pub unsafe fn extend<'a>(&mut self, storage: &'a hub::Storage<T, Id>, id: Id) -> Option<&'a T> {
        self.owned.set(id.unzip().0 as usize, true);

        storage.get(id).ok()
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        storage: &hub::Storage<T, Id>,
        bind_group: &StatelessBindGroupSate<T, Id>,
    ) {
        for &id in &bind_group.resource {
            self.owned.set(id.unzip().0 as usize, true);
        }
    }

    pub fn extend_from_tracker(&mut self, storage: &hub::Storage<T, Id>, other: &Self) {
        let incoming_size = other.owned.len();
        if incoming_size > self.owned.len() {
            self.set_max_index(incoming_size);
        }

        for (left, &right) in
            unsafe { self.owned.storage_mut().iter_mut() }.zip(other.owned.storage())
        {
            *left |= right;
        }
    }
}
