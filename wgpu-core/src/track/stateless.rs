use std::sync::Arc;

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
