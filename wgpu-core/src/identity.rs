use wgt::Backend;

use crate::{
    id::{Id, Marker},
    lock::{rank, Mutex},
    Epoch, Index,
};
use std::{fmt::Debug, marker::PhantomData};

#[derive(Copy, Clone, Debug, PartialEq)]
enum IdSource {
    External,
    Allocated,
    None,
}

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`release`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// `IdentityValues` returns `Id`s whose index values are suitable for use as
/// indices into a `Vec<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Every live id's index
///   selects a distinct element in the vector.
///
/// - `IdentityValues` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityValues` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// [`Id`]: crate::id::Id
/// [`Backend`]: wgt::Backend;
/// [`alloc`]: IdentityValues::alloc
/// [`release`]: IdentityValues::release
#[derive(Debug)]
pub(super) struct IdentityValues {
    free: Vec<(Index, Epoch)>,
    next_index: Index,
    count: usize,
    // Sanity check: The allocation logic works under the assumption that we don't
    // do a mix of allocating ids from here and providing ids manually for the same
    // storage container.
    id_source: IdSource,
}

impl IdentityValues {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<T: Marker>(&mut self, backend: Backend) -> Id<T> {
        assert!(
            self.id_source != IdSource::External,
            "Mix of internally allocated and externally provided IDs"
        );
        self.id_source = IdSource::Allocated;

        self.count += 1;
        match self.free.pop() {
            Some((index, epoch)) => Id::zip(index, epoch + 1, backend),
            None => {
                let index = self.next_index;
                self.next_index += 1;
                let epoch = 1;
                Id::zip(index, epoch, backend)
            }
        }
    }

    pub fn mark_as_used<T: Marker>(&mut self, id: Id<T>) -> Id<T> {
        assert!(
            self.id_source != IdSource::Allocated,
            "Mix of internally allocated and externally provided IDs"
        );
        self.id_source = IdSource::External;

        self.count += 1;
        id
    }

    /// Free `id`. It will never be returned from `alloc` again.
    pub fn release<T: Marker>(&mut self, id: Id<T>) {
        if let IdSource::Allocated = self.id_source {
            let (index, epoch, _backend) = id.unzip();
            self.free.push((index, epoch));
        }
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
            values: Mutex::new(
                rank::IDENTITY_MANAGER_VALUES,
                IdentityValues {
                    free: Vec::new(),
                    next_index: 0,
                    count: 0,
                    id_source: IdSource::None,
                },
            ),
            _phantom: PhantomData,
        }
    }
}

#[test]
fn test_epoch_end_of_life() {
    use crate::id;
    let man = IdentityManager::<id::markers::Buffer>::new();
    let id1 = man.process(Backend::Empty);
    assert_eq!(id1.unzip(), (0, 1, Backend::Empty));
    man.free(id1);
    let id2 = man.process(Backend::Empty);
    // confirm that the epoch 1 is no longer re-used
    assert_eq!(id2.unzip(), (0, 2, Backend::Empty));
}
