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
    ) -> Option<impl Iterator<Item = Range<wgt::BufferAddress>> + 'a> {
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

        if uninitialized_ranges.is_empty() {
            return None;
        }

        Some(std::iter::from_fn(move || {
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
        }))
    }
}

// TODO: Add some unit tests for this construct
