use hal::memory::Segment;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) enum MemoryInitTrackerAction {
    // The memory range is going to be written by an already initialized source, thus doesn't need extra attention.
    ImplicitlyInitialized(Range<wgt::BufferAddress>),
    // The memory range is going to be read, therefore needs to ensure prior initialization.
    NeedsInitializedMemory(Range<wgt::BufferAddress>),
}

#[derive(Debug, Clone)]
pub(crate) struct ResourceMemoryInitTrackerAction<ResourceId> {
    pub(crate) id: ResourceId,
    pub(crate) action: MemoryInitTrackerAction,
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

    pub(crate) fn drain_uninitialized_segments<'a>(
        &'a mut self,
        segment: Segment,
    ) -> impl Iterator<Item = Segment> + 'a {
        let range = match segment.size {
            Some(size) => segment.offset..(segment.offset + size),
            None => segment.offset..self.uninitialized_ranges.initial_range().end,
        };

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
                    let result = Some(Segment {
                        offset: range.start,
                        size: Some(range.end - range.start),
                    });
                    self.uninitialized_ranges.free_range(range);
                    result
                }
                None => None,
            }
        })
    }
}

// TODO: Add some unit tests for this construct
