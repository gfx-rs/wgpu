use alloc::{sync::Arc, vec::Vec};
use core::slice::Iter;

use crate::FastHashMap;

/// A tracker that holds strong references to resources, deduplicated by pointer
/// identity.
///
/// Deduplication ensures each unique resource is stored and checked at most
/// once at submit time, avoiding redundant liveness checks when the same
/// resource is recorded multiple times in a command buffer.
#[derive(Debug)]
pub(crate) struct StatelessTracker<T> {
    resources: Vec<Arc<T>>,
    ptr_to_index: FastHashMap<usize, usize>,
}

impl<T> StatelessTracker<T> {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            ptr_to_index: FastHashMap::default(),
        }
    }

    /// Inserts a resource into the tracker.
    ///
    /// If an `Arc` pointing to the same allocation is already present, the
    /// existing entry is reused. Returns a reference to the stored `Arc`.
    pub fn insert_single(&mut self, resource: Arc<T>) -> &Arc<T> {
        let ptr = Arc::as_ptr(&resource) as usize;
        let index = match self.ptr_to_index.get(&ptr) {
            Some(&index) => index,
            None => {
                let index = self.resources.len();
                self.ptr_to_index.insert(ptr, index);
                self.resources.push(resource);
                index
            }
        };
        &self.resources[index]
    }
}

impl<'a, T> IntoIterator for &'a StatelessTracker<T> {
    type Item = &'a Arc<T>;
    type IntoIter = Iter<'a, Arc<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.resources.as_slice().iter()
    }
}
