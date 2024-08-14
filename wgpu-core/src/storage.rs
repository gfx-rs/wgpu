use std::sync::Arc;

use wgt::Backend;

use crate::id::{Id, Marker};
use crate::resource::ResourceType;
use crate::{Epoch, Index};

/// An entry in a `Storage::map` table.
#[derive(Debug)]
pub(crate) enum Element<T> {
    /// There are no live ids with this index.
    Vacant,

    /// There is one live id with this index, allocated at the given
    /// epoch.
    Occupied(Arc<T>, Epoch),

    /// Like `Occupied`, but an error occurred when creating the
    /// resource.
    Error(Epoch),
}

#[derive(Clone, Debug)]
pub(crate) struct InvalidId;

pub(crate) trait StorageItem: ResourceType {
    type Marker: Marker;
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
/// `Storage` implements [`std::ops::Index`], accepting `Id` values as
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
    kind: &'static str,
}

impl<T> Storage<T>
where
    T: StorageItem,
{
    pub(crate) fn new() -> Self {
        Self {
            map: Vec::new(),
            kind: T::TYPE,
        }
    }
}

impl<T> Storage<T>
where
    T: StorageItem,
{
    /// Get a reference to an item behind a potentially invalid ID.
    /// Panics if there is an epoch mismatch, or the entry is empty.
    pub(crate) fn get(&self, id: Id<T::Marker>) -> Result<&Arc<T>, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map.get(index as usize) {
            Some(&Element::Occupied(ref v, epoch)) => (Ok(v), epoch),
            None | Some(&Element::Vacant) => panic!("{}[{:?}] does not exist", self.kind, id),
            Some(&Element::Error(epoch)) => (Err(InvalidId), epoch),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{:?}] is no longer alive",
            self.kind, id
        );
        result
    }

    /// Get an owned reference to an item behind a potentially invalid ID.
    /// Panics if there is an epoch mismatch, or the entry is empty.
    pub(crate) fn get_owned(&self, id: Id<T::Marker>) -> Result<Arc<T>, InvalidId> {
        Ok(Arc::clone(self.get(id)?))
    }

    fn insert_impl(&mut self, index: usize, epoch: Epoch, element: Element<T>) {
        if index >= self.map.len() {
            self.map.resize_with(index + 1, || Element::Vacant);
        }
        match std::mem::replace(&mut self.map[index], element) {
            Element::Vacant => {}
            Element::Occupied(_, storage_epoch) => {
                assert_ne!(
                    epoch,
                    storage_epoch,
                    "Index {index:?} of {} is already occupied",
                    T::TYPE
                );
            }
            Element::Error(storage_epoch) => {
                assert_ne!(
                    epoch,
                    storage_epoch,
                    "Index {index:?} of {} is already occupied with Error",
                    T::TYPE
                );
            }
        }
    }

    pub(crate) fn insert(&mut self, id: Id<T::Marker>, value: Arc<T>) {
        let (index, epoch, _backend) = id.unzip();
        self.insert_impl(index as usize, epoch, Element::Occupied(value, epoch))
    }

    pub(crate) fn insert_error(&mut self, id: Id<T::Marker>) {
        let (index, epoch, _) = id.unzip();
        self.insert_impl(index as usize, epoch, Element::Error(epoch))
    }

    pub(crate) fn replace_with_error(&mut self, id: Id<T::Marker>) -> Result<Arc<T>, InvalidId> {
        let (index, epoch, _) = id.unzip();
        match std::mem::replace(&mut self.map[index as usize], Element::Error(epoch)) {
            Element::Vacant => panic!("Cannot access vacant resource"),
            Element::Occupied(value, storage_epoch) => {
                assert_eq!(epoch, storage_epoch);
                Ok(value)
            }
            _ => Err(InvalidId),
        }
    }

    pub(crate) fn remove(&mut self, id: Id<T::Marker>) -> Option<Arc<T>> {
        let (index, epoch, _) = id.unzip();
        match std::mem::replace(&mut self.map[index as usize], Element::Vacant) {
            Element::Occupied(value, storage_epoch) => {
                assert_eq!(epoch, storage_epoch);
                Some(value)
            }
            Element::Error(_) => None,
            Element::Vacant => panic!("Cannot remove a vacant resource"),
        }
    }

    pub(crate) fn iter(&self, backend: Backend) -> impl Iterator<Item = (Id<T::Marker>, &Arc<T>)> {
        self.map
            .iter()
            .enumerate()
            .filter_map(move |(index, x)| match *x {
                Element::Occupied(ref value, storage_epoch) => {
                    Some((Id::zip(index as Index, storage_epoch, backend), value))
                }
                _ => None,
            })
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}
