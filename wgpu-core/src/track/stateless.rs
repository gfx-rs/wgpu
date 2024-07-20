//! Stateless Trackers
//!
//! Stateless trackers don't have any state, so make no
//! distinction between a usage scope and a full tracker.

use std::sync::Arc;

use crate::{
    lock::{rank, Mutex},
    resource::Trackable,
    track::ResourceMetadata,
};

/// Stores all the resources that a bind group stores.
#[derive(Debug)]
pub(crate) struct StatelessBindGroupState<T: Trackable> {
    resources: Mutex<Vec<Arc<T>>>,
}

impl<T: Trackable> StatelessBindGroupState<T> {
    pub fn new() -> Self {
        Self {
            resources: Mutex::new(rank::STATELESS_BIND_GROUP_STATE_RESOURCES, Vec::new()),
        }
    }

    /// Optimize the buffer bind group state by sorting it by ID.
    ///
    /// When this list of states is merged into a tracker, the memory
    /// accesses will be in a constant ascending order.
    pub(crate) fn optimize(&self) {
        let mut resources = self.resources.lock();
        resources.sort_unstable_by_key(|resource| resource.tracker_index());
    }

    /// Adds the given resource.
    pub fn add_single(&self, resource: &Arc<T>) {
        let mut resources = self.resources.lock();
        resources.push(resource.clone());
    }
}

/// Stores all resource state within a command buffer or device.
#[derive(Debug)]
pub(crate) struct StatelessTracker<T: Trackable> {
    metadata: ResourceMetadata<Arc<T>>,
}

impl<T: Trackable> StatelessTracker<T> {
    pub fn new() -> Self {
        Self {
            metadata: ResourceMetadata::new(),
        }
    }

    fn tracker_assert_in_bounds(&self, index: usize) {
        self.metadata.tracker_assert_in_bounds(index);
    }

    /// Sets the size of all the vectors inside the tracker.
    ///
    /// Must be called with the highest possible Resource ID of this type
    /// before all unsafe functions are called.
    pub fn set_size(&mut self, size: usize) {
        self.metadata.set_size(size);
    }

    /// Extend the vectors to let the given index be valid.
    fn allow_index(&mut self, index: usize) {
        if index >= self.metadata.size() {
            self.set_size(index + 1);
        }
    }

    /// Inserts a single resource into the resource tracker.
    ///
    /// If the resource already exists in the tracker, it will be overwritten.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    ///
    /// Returns a reference to the newly inserted resource.
    /// (This allows avoiding a clone/reference count increase in many cases.)
    pub fn insert_single(&mut self, resource: Arc<T>) -> &Arc<T> {
        let index = resource.tracker_index().as_usize();

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe { self.metadata.insert(index, resource) }
    }

    /// Adds the given resources from the given tracker.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn add_from_tracker(&mut self, other: &Self) {
        let incoming_size = other.metadata.size();
        if incoming_size > self.metadata.size() {
            self.set_size(incoming_size);
        }

        for index in other.metadata.owned_indices() {
            self.tracker_assert_in_bounds(index);
            other.tracker_assert_in_bounds(index);
            unsafe {
                let previously_owned = self.metadata.contains_unchecked(index);

                if !previously_owned {
                    let other_resource = other.metadata.get_resource_unchecked(index);
                    self.metadata.insert(index, other_resource.clone());
                }
            }
        }
    }
}
