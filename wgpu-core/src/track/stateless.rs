use alloc::{sync::Arc, vec::Vec};
use core::slice::Iter;

use crate::resource::Trackable;

/// A tracker that holds strong references to resources.
///
/// This is only used to keep resources alive.
#[derive(Debug)]
pub(crate) struct StatelessTracker<T> {
    resources: Vec<Arc<T>>,
}

impl<T> StatelessTracker<T> {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
        }
    }

    /// Inserts a single resource into the resource tracker.
    ///
    /// Returns a reference to the newly inserted resource.
    /// (This allows avoiding a clone/reference count increase in many cases.)
    pub fn insert_single(&mut self, resource: Arc<T>) -> &Arc<T> {
        self.resources.push(resource);
        unsafe { self.resources.last().unwrap_unchecked() }
    }
}

impl<T: Trackable> StatelessTracker<T> {
    /// Returns true if the tracker holds a reference to the given resource.
    pub fn contains(&self, resource: &T) -> bool {
        self.resources
            .iter()
            .any(|r| r.tracker_index() == resource.tracker_index())
    }

    /// Iterates over the resources held by the tracker.
    pub fn used_resources(&self) -> impl Iterator<Item = &Arc<T>> + '_ {
        self.resources.iter()
    }
}

impl<'a, T> IntoIterator for &'a StatelessTracker<T> {
    type Item = &'a Arc<T>;
    type IntoIter = Iter<'a, Arc<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.resources.as_slice().iter()
    }
}
