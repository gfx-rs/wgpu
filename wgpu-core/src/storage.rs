use alloc::{sync::Arc, vec::Vec};
use core::mem;

use crate::id::{Id, Marker};
use crate::resource::ResourceType;
use crate::{Epoch, Index};
use parking_lot::Mutex;

/// An entry in a `Storage::map` table.
#[derive(Debug)]
pub(crate) enum Element<T>
where
    T: StorageItem,
{
    /// There are no live ids with this index.
    Vacant,

    /// There is one live id with this index, allocated at the given
    /// epoch.
    Occupied(T, Epoch),
}

/// Not a public API. For use only by `player`.
#[doc(hidden)]
pub trait StorageItem: ResourceType {
    type Marker: Marker;
}

impl<T: ResourceType> ResourceType for Arc<T> {
    const TYPE: &'static str = T::TYPE;
}

impl<T: StorageItem> StorageItem for Arc<T> {
    type Marker = T::Marker;
}

impl<T: ResourceType> ResourceType for Mutex<T> {
    const TYPE: &'static str = T::TYPE;
}

impl<T: StorageItem> StorageItem for Mutex<T> {
    type Marker = T::Marker;
}

#[macro_export]
macro_rules! impl_storage_item {
    ($ty:ident) => {
        impl $crate::storage::StorageItem for $ty {
            type Marker = $crate::id::markers::$ty;
        }
    };
}

/// A table of `T` values indexed by the id type `I`.
///
/// `Storage` implements [`core::ops::Index`], accepting `Id` values as
/// indices.
///
/// The table is represented as a vector indexed by the ids' index
/// values, so you should use an id allocator like `IdentityManager`
/// that keeps the index values dense and close to zero.
#[derive(Debug)]
pub(crate) struct Storage<T>
where
    T: StorageItem,
{
    pub(crate) map: Vec<Element<T>>,
}

impl<T> Storage<T>
where
    T: StorageItem,
{
    pub(crate) fn new() -> Self {
        Self { map: Vec::new() }
    }
}

impl<T> Storage<T>
where
    T: StorageItem,
{
    pub(crate) fn insert(&mut self, id: Id<T::Marker>, value: T) {
        let (index, epoch) = id.unzip();
        let index = index as usize;
        if index >= self.map.len() {
            self.map.resize_with(index + 1, || Element::Vacant);
        }
        match mem::replace(&mut self.map[index], Element::Occupied(value, epoch)) {
            Element::Vacant => {}
            Element::Occupied(_, storage_epoch) => {
                panic!(
                    "Cannot insert {id:?}, found existing resource {other:?}",
                    other = Id::<T::Marker>::zip(index as Index, storage_epoch),
                );
            }
        }
    }

    pub(crate) fn remove(&mut self, id: Id<T::Marker>) -> T {
        let (index, epoch) = id.unzip();
        let stored = self.map.get_mut(index as usize);
        match stored.map(|stored| mem::replace(stored, Element::Vacant)) {
            Some(Element::Occupied(value, storage_epoch)) => {
                assert_eq!(
                    epoch,
                    storage_epoch,
                    "Cannot remove {id:?}, found other resource {other:?}",
                    other = Id::<T::Marker>::zip(index, storage_epoch),
                );
                value
            }
            None | Some(Element::Vacant) => {
                panic!("Cannot remove non-existent resource {id:?}");
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Id<T::Marker>, &T)> {
        self.map
            .iter()
            .enumerate()
            .filter_map(move |(index, x)| match *x {
                Element::Occupied(ref value, storage_epoch) => {
                    Some((Id::zip(index as Index, storage_epoch), value))
                }
                _ => None,
            })
    }
}

impl<T> Storage<T>
where
    T: StorageItem + Clone,
{
    /// Get an owned reference to an item.
    /// Panics if there is an epoch mismatch, the entry is empty or in error.
    pub(crate) fn get(&self, id: Id<T::Marker>) -> T {
        let (index, epoch) = id.unzip();
        let (result, storage_epoch) = match self.map.get(index as usize) {
            Some(&Element::Occupied(ref v, epoch)) => (v.clone(), epoch),
            None | Some(&Element::Vacant) => {
                panic!("Cannot get non-existent resource {id:?}");
            }
        };
        assert_eq!(
            epoch,
            storage_epoch,
            "Cannot get {id:?}, found other resource {other:?}",
            other = Id::<T::Marker>::zip(index, storage_epoch),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct TestItem;

    impl ResourceType for TestItem {
        const TYPE: &'static str = "TestItem";
    }

    impl StorageItem for TestItem {
        type Marker = ();
    }

    fn id(index: Index, epoch: Epoch) -> Id<()> {
        Id::zip(index, epoch)
    }

    #[test]
    #[should_panic(
        expected = "Cannot insert UntypedId(0,1), found existing resource UntypedId(0,1)"
    )]
    fn insert_occupied_same_epoch() {
        let mut storage = Storage::new();
        storage.insert(id(0, 1), TestItem);
        storage.insert(id(0, 1), TestItem);
    }

    #[test]
    #[should_panic(
        expected = "Cannot insert UntypedId(0,2), found existing resource UntypedId(0,1)"
    )]
    fn insert_occupied_different_epoch() {
        let mut storage = Storage::new();
        storage.insert(id(0, 1), TestItem);
        storage.insert(id(0, 2), TestItem);
    }

    #[test]
    #[should_panic(expected = "Cannot remove UntypedId(0,2), found other resource UntypedId(0,1)")]
    fn remove_epoch_mismatch() {
        let mut storage = Storage::new();
        storage.insert(id(0, 1), TestItem);
        storage.remove(id(0, 2));
    }

    #[test]
    #[should_panic(expected = "Cannot remove non-existent resource UntypedId(0,1)")]
    fn remove_vacant() {
        let mut storage = Storage::<TestItem>::new();
        storage.remove(id(0, 1));
    }

    #[test]
    #[should_panic(expected = "Cannot get non-existent resource UntypedId(0,1)")]
    fn get_vacant() {
        let storage = Storage::<TestItem>::new();
        storage.get(id(0, 1));
    }

    #[test]
    #[should_panic(expected = "Cannot get UntypedId(0,2), found other resource UntypedId(0,1)")]
    fn get_epoch_mismatch() {
        let mut storage = Storage::new();
        storage.insert(id(0, 1), TestItem);
        storage.get(id(0, 2));
    }
}
