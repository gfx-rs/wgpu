use std::marker::PhantomData;

use bit_vec::BitVec;

use crate::{
    hub,
    id::{TypedId, Valid},
    track::resize_bitvec,
};

pub struct StatelessBindGroupSate<T, Id: TypedId> {
    resource: Vec<Valid<Id>>,

    _phantom: PhantomData<T>,
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
            self.owned.set(id.0.unzip().0 as usize, true);
        }
    }
}
