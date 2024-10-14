use crate::resource::AccelerationStructure;
use crate::track::metadata::ResourceMetadata;
use crate::track::ResourceUses;
use hal::AccelerationStructureUses;
use std::sync::Arc;
use wgt::strict_assert;

pub(crate) struct AccelerationStructureTracker<T: AccelerationStructure> {
    start: Vec<AccelerationStructureUses>,
    end: Vec<AccelerationStructureUses>,

    metadata: ResourceMetadata<Arc<T>>,
}

impl<T: AccelerationStructure> AccelerationStructureTracker<T> {
    pub fn new() -> Self {
        Self {
            start: Vec::new(),
            end: Vec::new(),

            metadata: ResourceMetadata::new(),
        }
    }

    fn tracker_assert_in_bounds(&self, index: usize) {
        strict_assert!(index < self.start.len());
        strict_assert!(index < self.end.len());
        self.metadata.tracker_assert_in_bounds(index);
    }

    /// Sets the size of all the vectors inside the tracker.
    ///
    /// Must be called with the highest possible Buffer ID before
    /// all unsafe functions are called.
    pub fn set_size(&mut self, size: usize) {
        self.start.resize(size, AccelerationStructureUses::empty());
        self.end.resize(size, AccelerationStructureUses::empty());

        self.metadata.set_size(size);
    }

    /// Extend the vectors to let the given index be valid.
    fn allow_index(&mut self, index: usize) {
        if index >= self.start.len() {
            self.set_size(index + 1);
        }
    }

    /// Returns true if the given buffer is tracked.
    pub fn contains(&self, acceleration_structure: &T) -> bool {
        self.metadata
            .contains(acceleration_structure.tracker_index().as_usize())
    }

    /// Inserts a single resource into the resource tracker.
    pub fn set_single(&mut self, resource: Arc<T>) {
        let index: usize = resource.tracker_index().as_usize();

        self.allow_index(index);

        self.tracker_assert_in_bounds(index);
    }
}

impl ResourceUses for AccelerationStructureUses {
    const EXCLUSIVE: Self = Self::empty();

    type Selector = ();

    fn bits(self) -> u16 {
        Self::bits(&self) as u16
    }

    fn all_ordered(self) -> bool {
        true
    }

    fn any_exclusive(self) -> bool {
        self.intersects(Self::EXCLUSIVE)
    }
}
