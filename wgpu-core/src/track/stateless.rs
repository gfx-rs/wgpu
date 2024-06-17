/*! Stateless Trackers
 *
 * Stateless trackers don't have any state, so make no
 * distinction between a usage scope and a full tracker.
!*/

use std::sync::Arc;

use crate::{
    id::Id,
    lock::{rank, Mutex},
    resource::Resource,
    resource_log,
    storage::Storage,
    track::ResourceMetadata,
};

use super::{ResourceTracker, TrackerIndex};

/// Satisfy clippy.
type Pair<T> = (Id<<T as Resource>::Marker>, Arc<T>);

/// Stores all the resources that a bind group stores.
#[derive(Debug)]
pub(crate) struct StatelessBindGroupState<T: Resource> {
    resources: Mutex<Vec<Pair<T>>>,
}

impl<T: Resource> StatelessBindGroupState<T> {
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
        resources.sort_unstable_by_key(|&(id, _)| id.unzip().0);
    }

    /// Returns a list of all resources tracked. May contain duplicates.
    pub fn used_resources(&self) -> impl Iterator<Item = Arc<T>> + '_ {
        let resources = self.resources.lock();
        resources
            .iter()
            .map(|(_, resource)| resource.clone())
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Returns a list of all resources tracked. May contain duplicates.
    pub fn drain_resources(&self) -> impl Iterator<Item = Arc<T>> + '_ {
        let mut resources = self.resources.lock();
        resources
            .drain(..)
            .map(|(_, r)| r)
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Adds the given resource.
    pub fn add_single<'a>(&self, storage: &'a Storage<T>, id: Id<T::Marker>) -> Option<&'a T> {
        let resource = storage.get(id).ok()?;

        let mut resources = self.resources.lock();
        resources.push((id, resource.clone()));

        Some(resource)
    }
}

/// Stores all resource state within a command buffer or device.
#[derive(Debug)]
pub(crate) struct StatelessTracker<T: Resource> {
    metadata: ResourceMetadata<T>,
}

impl<T: Resource> ResourceTracker for StatelessTracker<T> {
    /// Try to remove the given resource from the tracker iff we have the last reference to the
    /// resource and the epoch matches.
    ///
    /// Returns true if the resource was removed or if not existing in metadata.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// false will be returned.
    fn remove_abandoned(&mut self, index: TrackerIndex) -> bool {
        let index = index.as_usize();

        if index >= self.metadata.size() {
            return false;
        }

        resource_log!("StatelessTracker::remove_abandoned {index:?}");

        self.tracker_assert_in_bounds(index);

        unsafe {
            if self.metadata.contains_unchecked(index) {
                let existing_ref_count = self.metadata.get_ref_count_unchecked(index);
                //RefCount 2 means that resource is hold just by DeviceTracker and this suspected resource itself
                //so it's already been released from user and so it's not inside Registry\Storage
                if existing_ref_count <= 2 {
                    self.metadata.remove(index);
                    return true;
                }

                return false;
            }
        }
        true
    }
}

impl<T: Resource> StatelessTracker<T> {
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

    /// Returns a list of all resources tracked.
    pub fn used_resources(&self) -> impl Iterator<Item = Arc<T>> + '_ {
        self.metadata.owned_resources()
    }

    /// Returns a list of all resources tracked.
    pub fn drain_resources(&mut self) -> impl Iterator<Item = Arc<T>> + '_ {
        let resources = self.metadata.drain_resources();
        resources.into_iter()
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
        let index = resource.as_info().tracker_index().as_usize();

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe { self.metadata.insert(index, resource) }
    }

    /// Adds the given resource to the tracker.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn add_single<'a>(
        &mut self,
        storage: &'a Storage<T>,
        id: Id<T::Marker>,
    ) -> Option<&'a Arc<T>> {
        let resource = storage.get(id).ok()?;

        let index = resource.as_info().tracker_index().as_usize();

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            self.metadata.insert(index, resource.clone());
        }

        Some(resource)
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
