/*! Stateless Trackers
 *
 * Stateless trackers don't have any state, so make no
 * distinction between a usage scope and a full tracker.
!*/

use std::marker::PhantomData;

use crate::{
    hal_api::HalApi,
    id::{TypedId, Valid},
    resource, storage,
    track::ResourceMetadata,
    RefCount,
};

/// Stores all the resources that a bind group stores.
pub(crate) struct StatelessBindGroupSate<T, Id: TypedId> {
    resources: Vec<(Valid<Id>, RefCount)>,

    _phantom: PhantomData<T>,
}

impl<T: resource::Resource, Id: TypedId> StatelessBindGroupSate<T, Id> {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),

            _phantom: PhantomData,
        }
    }

    /// Optimize the buffer bind group state by sorting it by ID.
    ///
    /// When this list of states is merged into a tracker, the memory
    /// accesses will be in a constant assending order.
    pub(crate) fn optimize(&mut self) {
        self.resources
            .sort_unstable_by_key(|&(id, _)| id.0.unzip().0);
    }

    /// Returns a list of all resources tracked. May contain duplicates.
    pub fn used(&self) -> impl Iterator<Item = Valid<Id>> + '_ {
        self.resources.iter().map(|&(id, _)| id)
    }

    /// Adds the given resource.
    pub fn add_single<'a>(
        &mut self,
        storage: &'a storage::Storage<T, Id>,
        id: Id,
    ) -> Option<&'a T> {
        let resource = storage.get(id).ok()?;

        self.resources
            .push((Valid(id), resource.life_guard().add_ref()));

        Some(resource)
    }
}

/// Stores all resource state within a command buffer or device.
pub(crate) struct StatelessTracker<A: HalApi, T, Id: TypedId> {
    metadata: ResourceMetadata<A>,

    _phantom: PhantomData<(T, Id)>,
}

impl<A: HalApi, T: resource::Resource, Id: TypedId> StatelessTracker<A, T, Id> {
    pub fn new() -> Self {
        Self {
            metadata: ResourceMetadata::new(),

            _phantom: PhantomData,
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
    pub fn used(&self) -> impl Iterator<Item = Valid<Id>> + '_ {
        self.metadata.owned_ids()
    }

    /// Inserts a single resource into the resource tracker.
    ///
    /// If the resource already exists in the tracker, it will be overwritten.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn insert_single(&mut self, id: Valid<Id>, ref_count: RefCount) {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            self.metadata.insert(index, epoch, ref_count);
        }
    }

    /// Adds the given resource to the tracker.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// the vectors will be extended. A call to set_size is not needed.
    pub fn add_single<'a>(
        &mut self,
        storage: &'a storage::Storage<T, Id>,
        id: Id,
    ) -> Option<&'a T> {
        let item = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);

        unsafe {
            self.metadata
                .insert(index, epoch, item.life_guard().add_ref());
        }

        Some(item)
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
                    let epoch = other.metadata.get_epoch_unchecked(index);
                    let other_ref_count = other.metadata.get_ref_count_unchecked(index);
                    self.metadata.insert(index, epoch, other_ref_count.clone());
                }
            }
        }
    }

    /// Removes the given resource from the tracker iff we have the last reference to the
    /// resource and the epoch matches.
    ///
    /// Returns true if the resource was removed.
    ///
    /// If the ID is higher than the length of internal vectors,
    /// false will be returned.
    pub fn remove_abandoned(&mut self, id: Valid<Id>) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.metadata.size() {
            return false;
        }

        self.tracker_assert_in_bounds(index);

        unsafe {
            if self.metadata.contains_unchecked(index) {
                let existing_epoch = self.metadata.get_epoch_unchecked(index);
                let existing_ref_count = self.metadata.get_ref_count_unchecked(index);

                if existing_epoch == epoch && existing_ref_count.load() == 1 {
                    self.metadata.remove(index);
                    return true;
                }
            }
        }

        false
    }
}
