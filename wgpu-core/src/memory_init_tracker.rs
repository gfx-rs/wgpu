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
    // TODO: Use a more fitting data structure!
    // An allocated range in this allocator means that the range in question is NOT yet initialized.
    uninitialized_ranges: range_alloc::RangeAllocator<wgt::BufferAddress>,
}

impl MemoryInitTracker {
    pub(crate) fn new(size: wgt::BufferAddress) -> Self {
        let mut uninitialized_ranges =
            range_alloc::RangeAllocator::<wgt::BufferAddress>::new(0..size);
        let _ = uninitialized_ranges.allocate_range(size);

        Self {
            uninitialized_ranges,
        }
    }

    pub(crate) fn is_initialized(&self, range: &Range<wgt::BufferAddress>) -> bool {
        self.uninitialized_ranges
            .allocated_ranges()
            .all(|r: Range<wgt::BufferAddress>| r.start >= range.end || r.end <= range.start)
    }

    #[must_use]
    pub(crate) fn drain_uninitialized_ranges<'a>(
        &'a mut self,
        range: &Range<wgt::BufferAddress>,
    ) -> impl Iterator<Item = Range<wgt::BufferAddress>> + 'a {
        let mut uninitialized_ranges: Vec<Range<wgt::BufferAddress>> = self
            .uninitialized_ranges
            .allocated_ranges()
            .filter_map(|r: Range<wgt::BufferAddress>| {
                if r.end > range.start && r.start < range.end {
                    Some(Range {
                        start: range.start.max(r.start),
                        end: range.end.min(r.end),
                    })
                } else {
                    None
                }
            })
            .collect();

        std::iter::from_fn(move || {
            let range: Option<Range<wgt::BufferAddress>> =
                uninitialized_ranges.last().map(|r| r.clone());
            match range {
                Some(range) => {
                    uninitialized_ranges.pop();
                    let result = range.clone();
                    self.uninitialized_ranges.free_range(range);
                    Some(result)
                }
                None => None,
            }
        })
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use super::MemoryInitTracker;

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
        tracker.drain_uninitialized_ranges(&(0..10)).for_each(drop);
        assert!(tracker.is_initialized(&(0..10)));
        assert!(tracker.is_initialized(&(0..3)));
        assert!(tracker.is_initialized(&(3..4)));
        assert!(tracker.is_initialized(&(4..10)));
    }

    #[test]
    fn is_initialized_for_partially_filled_tracker() {
        let mut tracker = MemoryInitTracker::new(10);
        tracker.drain_uninitialized_ranges(&(4..6)).for_each(drop);
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
    fn drain_uninitialized_ranges_never_returns_ranges_twice_for_same_range() {
        let mut tracker = MemoryInitTracker::new(19);
        assert_eq!(tracker.drain_uninitialized_ranges(&(0..19)).count(), 1);
        assert_eq!(tracker.drain_uninitialized_ranges(&(0..19)).count(), 0);

        let mut tracker = MemoryInitTracker::new(17);
        assert_eq!(tracker.drain_uninitialized_ranges(&(5..8)).count(), 1);
        assert_eq!(tracker.drain_uninitialized_ranges(&(5..8)).count(), 0);
        assert_eq!(tracker.drain_uninitialized_ranges(&(1..3)).count(), 1);
        assert_eq!(tracker.drain_uninitialized_ranges(&(1..3)).count(), 0);
        assert_eq!(tracker.drain_uninitialized_ranges(&(7..13)).count(), 1);
        assert_eq!(tracker.drain_uninitialized_ranges(&(7..13)).count(), 0);
    }

    #[test]
    fn drain_uninitialized_ranges_splits_ranges_correctly() {
        let mut tracker = MemoryInitTracker::new(1337);
        assert_eq!(
            tracker
                .drain_uninitialized_ranges(&(21..42))
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![21..42]
        );
        assert_eq!(
            tracker
                .drain_uninitialized_ranges(&(900..1000))
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![900..1000]
        );

        // Splitted ranges.
        assert_eq!(
            tracker
                .drain_uninitialized_ranges(&(5..1003))
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![1000..1003, 42..900, 5..21]
        );
        assert_eq!(
            tracker
                .drain_uninitialized_ranges(&(0..1337))
                .collect::<Vec<Range<wgt::BufferAddress>>>(),
            vec![1003..1337, 0..5]
        );
    }
}
