use std::{marker::PhantomData, mem, ops, sync::Arc};

use wgt::Backend;

use crate::{id, resource::Resource, Epoch, Index};

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
    ///
    /// The given `String` is the resource's descriptor label.
    Error(Epoch, String),
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StorageReport {
    pub num_occupied: usize,
    pub num_vacant: usize,
    pub num_error: usize,
    pub element_size: usize,
}

impl StorageReport {
    pub fn is_empty(&self) -> bool {
        self.num_occupied + self.num_vacant + self.num_error == 0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InvalidId;

/// A table of `T` values indexed by the id type `I`.
///
/// The table is represented as a vector indexed by the ids' index
/// values, so you should use an id allocator like `IdentityManager`
/// that keeps the index values dense and close to zero.
#[derive(Debug)]
pub struct Storage<T, I>
where
    T: Resource<I>,
    I: id::TypedId,
{
    pub(crate) map: Vec<Element<T>>,
    kind: &'static str,
    _phantom: PhantomData<I>,
}

impl<T, I> ops::Index<id::Valid<I>> for Storage<T, I>
where
    T: Resource<I>,
    I: id::TypedId,
{
    type Output = Arc<T>;
    fn index(&self, id: id::Valid<I>) -> &Arc<T> {
        self.get(id.0).unwrap()
    }
}
impl<T, I> Storage<T, I>
where
    T: Resource<I>,
    I: id::TypedId,
{
    pub(crate) fn new() -> Self {
        Self {
            map: Vec::new(),
            kind: T::TYPE,
            _phantom: PhantomData,
        }
    }
}

impl<T, I> Storage<T, I>
where
    T: Resource<I>,
    I: id::TypedId,
{
    pub(crate) fn contains(&self, id: I) -> bool {
        let (index, epoch, _) = id.unzip();
        match self.map.get(index as usize) {
            Some(&Element::Vacant) => false,
            Some(&Element::Occupied(_, storage_epoch) | &Element::Error(storage_epoch, _)) => {
                storage_epoch == epoch
            }
            None => false,
        }
    }

    /// Attempts to get a reference to an item behind a potentially invalid ID.
    ///
    /// Returns [`None`] if there is an epoch mismatch, or the entry is empty.
    ///
    /// This function is primarily intended for the `as_hal` family of functions
    /// where you may need to fallibly get a object backed by an id that could
    /// be in a different hub.
    pub(crate) fn try_get(&self, id: I) -> Result<Option<&Arc<T>>, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map.get(index as usize) {
            Some(&Element::Occupied(ref v, epoch)) => (Ok(Some(v)), epoch),
            Some(&Element::Vacant) => return Ok(None),
            Some(&Element::Error(epoch, ..)) => (Err(InvalidId), epoch),
            None => return Err(InvalidId),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{}] is no longer alive",
            self.kind, index
        );
        result
    }

    /// Get refcount of an item with specified ID
    /// And return true if it's 1 or false otherwise
    pub(crate) fn is_unique(&self, id: I) -> Result<bool, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map.get(index as usize) {
            Some(&Element::Occupied(ref v, epoch)) => (Ok(v.is_unique()), epoch),
            Some(&Element::Vacant) => panic!("{}[{}] does not exist", self.kind, index),
            Some(&Element::Error(epoch, ..)) => (Err(InvalidId), epoch),
            None => return Err(InvalidId),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{}] is no longer alive",
            self.kind, index
        );
        result
    }

    /// Get a reference to an item behind a potentially invalid ID.
    /// Panics if there is an epoch mismatch, or the entry is empty.
    pub(crate) fn get(&self, id: I) -> Result<&Arc<T>, InvalidId> {
        let (index, epoch, _) = id.unzip();
        let (result, storage_epoch) = match self.map.get(index as usize) {
            Some(&Element::Occupied(ref v, epoch)) => (Ok(v), epoch),
            Some(&Element::Vacant) => panic!("{}[{}] does not exist", self.kind, index),
            Some(&Element::Error(epoch, ..)) => (Err(InvalidId), epoch),
            None => return Err(InvalidId),
        };
        assert_eq!(
            epoch, storage_epoch,
            "{}[{}] is no longer alive",
            self.kind, index
        );
        result
    }

    pub(crate) unsafe fn get_unchecked(&self, id: u32) -> &Arc<T> {
        match self.map[id as usize] {
            Element::Occupied(ref v, _) => v,
            Element::Vacant => panic!("{}[{}] does not exist", self.kind, id),
            Element::Error(_, _) => panic!(""),
        }
    }

    pub(crate) fn label_for_invalid_id(&self, id: I) -> &str {
        let (index, _, _) = id.unzip();
        match self.map.get(index as usize) {
            Some(&Element::Error(_, ref label)) => label,
            _ => "",
        }
    }

    fn insert_impl(&mut self, index: usize, element: Element<T>) {
        if index >= self.map.len() {
            self.map.resize_with(index + 1, || Element::Vacant);
        }
        match std::mem::replace(&mut self.map[index], element) {
            Element::Vacant => {}
            _ => panic!("Index {index:?} is already occupied"),
        }
    }

    pub(crate) fn insert(&mut self, id: I, value: Arc<T>) {
        let (index, epoch, _backend) = id.unzip();
        self.insert_impl(index as usize, Element::Occupied(value, epoch))
    }

    pub(crate) fn insert_error(&mut self, id: I, label: &str) {
        let (index, epoch, _) = id.unzip();
        self.insert_impl(index as usize, Element::Error(epoch, label.to_string()))
    }

    pub(crate) fn force_replace(&mut self, id: I, mut value: T) {
        let (index, epoch, _) = id.unzip();
        value.as_info_mut().set_id(id);
        self.map[index as usize] = Element::Occupied(Arc::new(value), epoch);
    }

    pub(crate) fn remove(&mut self, id: I) -> Option<Arc<T>> {
        let (index, epoch, _) = id.unzip();
        match std::mem::replace(&mut self.map[index as usize], Element::Vacant) {
            Element::Occupied(value, storage_epoch) => {
                assert_eq!(epoch, storage_epoch);
                Some(value)
            }
            Element::Error(..) => None,
            Element::Vacant => panic!("Cannot remove a vacant resource"),
        }
    }

    // Prevents panic on out of range access, allows Vacant elements.
    pub(crate) fn _try_remove(&mut self, id: I) -> Option<Arc<T>> {
        let (index, epoch, _) = id.unzip();
        if index as usize >= self.map.len() {
            None
        } else if let Element::Occupied(value, storage_epoch) =
            std::mem::replace(&mut self.map[index as usize], Element::Vacant)
        {
            assert_eq!(epoch, storage_epoch);
            Some(value)
        } else {
            None
        }
    }

    pub(crate) fn iter(&self, backend: Backend) -> impl Iterator<Item = (I, &Arc<T>)> {
        self.map
            .iter()
            .enumerate()
            .filter_map(move |(index, x)| match *x {
                Element::Occupied(ref value, storage_epoch) => {
                    Some((I::zip(index as Index, storage_epoch, backend), value))
                }
                _ => None,
            })
    }

    pub(crate) fn kind(&self) -> &str {
        self.kind
    }

    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    pub(crate) fn generate_report(&self) -> StorageReport {
        let mut report = StorageReport {
            element_size: mem::size_of::<T>(),
            ..Default::default()
        };
        for element in self.map.iter() {
            match *element {
                Element::Occupied(..) => report.num_occupied += 1,
                Element::Vacant => report.num_vacant += 1,
                Element::Error(..) => report.num_error += 1,
            }
        }
        report
    }
}
