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

impl<T: StorageItem> Registry<T> {
    pub(crate) fn assign(&mut self, id: Id<T::Marker>, value: T) -> Id<T::Marker> {
        self.storage.insert(id, value);
        id
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
