use core::cell::{Ref, RefCell};

use crate::{
    id::Id,
    storage::{Storage, StorageItem},
};

/// Registry is the primary holder of each resource type
/// Every resource is now arcanized so the last arc released
/// will in the end free the memory and release the inner raw resource
///
/// Registry act as the main entry point to keep resource alive
/// when created and released from user land code
///
/// A resource may still be alive when released from user land code
/// if it's used in active submission or anyway kept alive from
/// any other dependent resource
///
#[derive(Debug)]
pub(crate) struct Registry<T: StorageItem> {
    storage: RefCell<Storage<T>>,
}

impl<T: StorageItem> Registry<T> {
    pub(crate) fn new() -> Self {
        Self {
            storage: RefCell::new(Storage::new()),
        }
    }
}

#[must_use]
pub(crate) struct FutureId<'a, T: StorageItem> {
    id: Id<T::Marker>,
    data: &'a RefCell<Storage<T>>,
}

impl<T: StorageItem> FutureId<'_, T> {
    /// Assign a new resource to this ID.
    ///
    /// Registers it with the registry.
    pub fn assign(self, value: T) -> Id<T::Marker> {
        let mut data = self.data.borrow_mut();
        data.insert(self.id, value);
        self.id
    }
}

impl<T: StorageItem> Registry<T> {
    pub(crate) fn prepare(&self, id_in: Id<T::Marker>) -> FutureId<'_, T> {
        FutureId {
            id: id_in,
            data: &self.storage,
        }
    }

    #[track_caller]
    pub(crate) fn read<'a>(&'a self) -> Ref<'a, Storage<T>> {
        self.storage.borrow()
    }

    pub(crate) fn remove(&self, id: Id<T::Marker>) -> T {
        let value = self.storage.borrow_mut().remove(id);
        value
    }
}

impl<T: StorageItem + Clone> Registry<T> {
    pub(crate) fn get(&self, id: Id<T::Marker>) -> T {
        self.read().get(id)
    }
}
