use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) enum MemoryInitKind {
    // The memory range is going to be written by an already initialized source, thus doesn't need extra attention other than marking as initialized.
    ImplicitlyInitialized,
    // The memory range is going to be read, therefore needs to ensure prior initialization.
    NeedsInitializedMemory,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryInitTrackerAction<ResourceId> {
    pub(crate) id: ResourceId,
    pub(crate) range: Range<wgt::BufferAddress>,
    pub(crate) kind: MemoryInitKind,
}

/// Tracks initialization status of a linear range from 0..size
#[derive(Debug)]
pub(crate) struct MemoryInitTracker {
    uninitialized_ranges: Vec<Range<wgt::BufferAddress>>,
}

pub(crate) struct MemoryInitTrackerDrain<'a> {
    uninitialized_ranges: &'a mut Vec<Range<wgt::BufferAddress>>,
    drain_range: Range<wgt::BufferAddress>,
    next_index: usize,
}

impl<'a> Iterator for MemoryInitTrackerDrain<'a> {
    type Item = Range<wgt::BufferAddress>;

    fn next(&mut self) -> Option<Self::Item> {
        let uninitialized_range = match self.uninitialized_ranges.get_mut(self.next_index) {
            Some(range) => range,
            None => return None,
        };
        if uninitialized_range.start >= self.drain_range.end {
            // No more cuts possible (we're going left to right!)
            None
        } else if uninitialized_range.end > self.drain_range.end {
            // cut-out / split
            if uninitialized_range.start < self.drain_range.start {
                let old_start = uninitialized_range.start;
                uninitialized_range.start = self.drain_range.end;
                self.uninitialized_ranges
                    .insert(self.next_index, old_start..self.drain_range.start);
                self.next_index = std::usize::MAX;
                Some(self.drain_range.clone())
            }
            // right cut
            else {
                let result = uninitialized_range.start..self.drain_range.end;
                self.next_index = std::usize::MAX;
                uninitialized_range.start = self.drain_range.end;
                Some(result)
            }
        } else {
            // left cut
            if uninitialized_range.start < self.drain_range.start {
                let result = self.drain_range.start..uninitialized_range.end;
                uninitialized_range.end = self.drain_range.start;
                self.next_index = self.next_index + 1;
                Some(result)
            }
            // fully contained.
            else {
                let result = uninitialized_range.clone();
                self.uninitialized_ranges.remove(self.next_index);
                Some(result)
            }
        }
    }
}

impl MemoryInitTracker {
    pub(crate) fn new(size: wgt::BufferAddress) -> Self {
        Self {
            uninitialized_ranges: vec![0..size],
        }
    }

    pub(crate) fn is_initialized(&self, query_range: &Range<wgt::BufferAddress>) -> bool {
        match self
            .uninitialized_ranges
            .iter()
            .find(|r| r.end > query_range.start)
        {
            Some(r) => r.start >= query_range.end,
            None => true,
        }
    }

    // Drains uninitialized ranges in a query range.
    #[must_use]
    pub(crate) fn drain<'a>(
        &'a mut self,
        drain_range: Range<wgt::BufferAddress>,
    ) -> MemoryInitTrackerDrain<'a> {
        let next_index = self
            .uninitialized_ranges
            .iter()
            .position(|r| r.end > drain_range.start)
            .unwrap_or(std::usize::MAX);

        MemoryInitTrackerDrain {
            next_index,
            drain_range,
            uninitialized_ranges: &mut self.uninitialized_ranges,
        }
    }

    // Clears uninitialized ranges in a query range.
    pub(crate) fn clear(&mut self, drain_range: Range<wgt::BufferAddress>) {
        self.drain(drain_range).for_each(drop);
    }
}

#[cfg(test)]
mod test {
    use super::MemoryInitTracker;
    use std::ops::Range;

    #[test]
    fn is_initialized_for_empty_tracker() {
        let tracker = MemoryInitTracker::new(10);
        assert!(!tracker.is_initialized(&(0..10)));
        assert!(!tracker.is_initialized(&(0..3)));
        assert!(!tracker.is_initialized(&(3..4)));
        assert!(!tracker.is_initialized(&(4..10)));
    }

    #[test]
    fn is_initialized_for_filled_tracker() {
        let mut tracker = MemoryInitTracker::new(10);
        tracker.clear(0..10);
        assert!(tracker.is_initialized(&(0..10)));
        assert!(tracker.is_initialized(&(0..3)));
        assert!(tracker.is_initialized(&(3..4)));
        assert!(tracker.is_initialized(&(4..10)));
    }

    #[test]
    fn is_initialized_for_partially_filled_tracker() {
        let mut tracker = MemoryInitTracker::new(10);
        tracker.clear(4..6);
        assert!(!tracker.is_initialized(&(0..10))); // entire range
        assert!(!tracker.is_initialized(&(0..4))); // left non-overlapping
        assert!(!tracker.is_initialized(&(3..5))); // left overlapping
        assert!(tracker.is_initialized(&(4..6))); // entire initialized range
        assert!(tracker.is_initialized(&(4..5))); // left part
        assert!(tracker.is_initialized(&(5..6))); // right part
        assert!(!tracker.is_initialized(&(5..7))); // right overlapping
        assert!(!tracker.is_initialized(&(7..10))); // right non-overlapping
    }

    #[test]
    fn drain_never_returns_ranges_twice_for_same_range() {
        let mut tracker = MemoryInitTracker::new(19);
        assert_eq!(tracker.drain(0..19).count(), 1);
        assert_eq!(tracker.drain(0..19).count(), 0);

        let mut tracker = MemoryInitTracker::new(17);
        assert_eq!(tracker.drain(5..8).count(), 1);
        assert_eq!(tracker.drain(5..8).count(), 0);
        assert_eq!(tracker.drain(1..3).count(), 1);
        assert_eq!(tracker.drain(1..3).count(), 0);
        assert_eq!(tracker.drain(7..13).count(), 1);
        assert_eq!(tracker.drain(7..13).count(), 0);
    }

    #[test]
    fn drain_splits_ranges_correctly() {
        let mut tracker = MemoryInitTracker::new(1337);
        assert_eq!(
            tracker
                .drain(21..42)
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![21..42]
        );
        assert_eq!(
            tracker
                .drain(900..1000)
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![900..1000]
        );

        // Splitted ranges.
        assert_eq!(
            tracker
                .drain(5..1003)
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![5..21, 42..900, 1000..1003]
        );
        assert_eq!(
            tracker
                .drain(0..1337)
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![0..5, 1003..1337]
        );
    }
}
