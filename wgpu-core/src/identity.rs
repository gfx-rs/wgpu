use parking_lot::Mutex;
use wgt::Backend;

use crate::{
    id::{Id, Marker},
    Epoch, FastHashMap, Index,
};
use std::{fmt::Debug, marker::PhantomData};

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`free`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// Use `IdentityManager::default` to construct new instances.
///
/// `IdentityManager` returns `Id`s whose index values are suitable for use as
/// indices into a `Storage<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Each live id's index selects a
///   distinct element in the vector.
///
/// - `IdentityManager` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityManager` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// See the module-level documentation for an overview of how this
/// fits together.
///
/// [`Id`]: crate::id::Id
/// [`Backend`]: wgt::Backend;
/// [`alloc`]: IdentityManager::alloc
/// [`free`]: IdentityManager::free
#[derive(Debug, Default)]
pub(super) struct IdentityValues {
    free: Vec<(Index, Epoch)>,
    //sorted by Index
    used: FastHashMap<Epoch, Vec<Index>>,
    count: usize,
}

impl IdentityValues {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<T: Marker>(&mut self, backend: Backend) -> Id<T> {
        self.count += 1;
        match self.free.pop() {
            Some((index, epoch)) => Id::zip(index, epoch + 1, backend),
            None => {
                let epoch = 1;
                let used = self.used.entry(epoch).or_insert_with(Default::default);
                let index = if let Some(i) = used.iter().max_by_key(|v| *v) {
                    i + 1
                } else {
                    0
                };
                used.push(index);
                Id::zip(index, epoch, backend)
            }
        }
    }

    pub fn mark_as_used<T: Marker>(&mut self, id: Id<T>) -> Id<T> {
        self.count += 1;
        let (index, epoch, _backend) = id.unzip();
        let used = self.used.entry(epoch).or_insert_with(Default::default);
        used.push(index);
        id
    }

    /// Free `id`. It will never be returned from `alloc` again.
    pub fn release<T: Marker>(&mut self, id: Id<T>) {
        let (index, epoch, _backend) = id.unzip();
        self.free.push((index, epoch));
        self.count -= 1;
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug)]
pub struct IdentityManager<T: Marker> {
    pub(super) values: Mutex<IdentityValues>,
    _phantom: PhantomData<T>,
}

impl<T: Marker> IdentityManager<T> {
    pub fn process(&self, backend: Backend) -> Id<T> {
        self.values.lock().alloc(backend)
    }
    pub fn mark_as_used(&self, id: Id<T>) -> Id<T> {
        self.values.lock().mark_as_used(id)
    }
    pub fn free(&self, id: Id<T>) {
        self.values.lock().release(id)
    }
}

impl<T: Marker> IdentityManager<T> {
    pub fn new() -> Self {
        Self {
            values: Mutex::new(IdentityValues::default()),
            _phantom: PhantomData,
        }
    }
}

#[test]
fn test_epoch_end_of_life() {
    use crate::id;

    let man = IdentityManager::<id::markers::Buffer>::new();
    let forced_id = man.mark_as_used(id::BufferId::zip(0, 1, Backend::Empty));
    assert_eq!(forced_id.unzip().0, 0);
    let id1 = man.process(Backend::Empty);
    assert_eq!(id1.unzip().0, 1);
    man.free(id1);
    let id2 = man.process(Backend::Empty);
    // confirm that the epoch 1 is no longer re-used
    assert_eq!(id2.unzip().0, 1);
    assert_eq!(id2.unzip().1, 2);
}
