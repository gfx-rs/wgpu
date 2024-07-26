use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU64};

use crate::context::ObjectId;

/// Opaque globally-unique identifier
#[repr(transparent)]
pub struct Id<T>(NonZeroU64, PhantomData<*mut T>);

impl<T> Id<T> {
    /// Create a new `Id` from a ObjectID.
    pub(crate) fn new(id: ObjectId) -> Self {
        Id(id.global_id(), PhantomData)
    }

    /// For testing use only. We provide no guarantees about the actual value of the ids.
    #[doc(hidden)]
    pub fn inner(&self) -> u64 {
        self.0.get()
    }
}

// SAFETY: `Id` is a bare `NonZeroU64`, the type parameter is a marker purely to avoid confusing Ids
// returned for different types , so `Id` can safely implement Send and Sync.
unsafe impl<T> Send for Id<T> {}

// SAFETY: See the implementation for `Send`.
unsafe impl<T> Sync for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Id").field(&self.0).finish()
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Id<T>) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Id<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Id<T>) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
