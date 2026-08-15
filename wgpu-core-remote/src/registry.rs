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
    storage: Storage<T>,
}

impl<T: StorageItem> Registry<T> {
    pub(crate) fn new() -> Self {
        Self {
            storage: Storage::new(),
        }
    }
}

#[must_use]
pub(crate) struct FutureId<'a, T: StorageItem> {
    id: Id<T::Marker>,
    data: &'a mut Storage<T>,
}

impl<T: StorageItem> FutureId<'_, T> {
    /// Assign a new resource to this ID.
    ///
    /// Registers it with the registry.
    pub fn assign(self, value: T) -> Id<T::Marker> {
        self.data.insert(self.id, value);
        self.id
    }
}

impl<T: StorageItem> Registry<T> {
    pub(crate) fn prepare(&mut self, id_in: Id<T::Marker>) -> FutureId<'_, T> {
        FutureId {
            id: id_in,
            data: &mut self.storage,
        }
    }

    pub(crate) fn remove(&mut self, id: Id<T::Marker>) -> T {
        self.storage.remove(id)
    }
}

impl<T: StorageItem + Clone> Registry<T> {
    pub(crate) fn get(&self, id: Id<T::Marker>) -> T {
        self.storage.get(id)
    }
}

impl<T: StorageItem> Registry<T> {
    pub(crate) fn get_mut(&mut self, id: Id<T::Marker>) -> &mut T {
        self.storage.get_mut(id)
    }
}
