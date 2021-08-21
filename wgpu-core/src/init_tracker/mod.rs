// WebGPU specification requires all texture & buffer memory to be zero initialized on first read.
// To avoid unnecessary inits, we track the initialization status of every resource and perform inits lazily.
//
// The granularity is different for buffers and textures:
// * Buffer: Byte granularity to support usecases with large, partially bound buffers well.
// * Texture: Mip-level per layer. I.e. a 2D surface is either completely initialized or not, subrects are not tracked.
//
// Every use of a buffer/texture generates a InitTrackerAction which are recorded and later resolved at queue submit by merging them with the current state and each other in execution order.
// It is important to note that from the point of view of the memory init system there are two kind of writes:
// * Full writes:
//      Any kind of memcpy operation. These cause a `MemoryInitKind.ImplicitlyInitialized` action.
// * (Potentially) partial writes:
//      E.g. write use in a Shader. The system is not able to determine if a resource is fully initialized afterwards but is no longer allowed to perform any clears,
//      therefore this leads to a `MemoryInitKind.ImplicitlyInitialized` action, exactly like a read would.

use smallvec::SmallVec;
use std::{iter, ops::Range};

mod buffer;
//mod texture;

pub(crate) use buffer::{BufferInitTracker, BufferInitTrackerAction};
//pub(crate) use texture::{TextureInitRange, TextureInitTracker, TextureInitTrackerAction};

#[derive(Debug, Clone, Copy)]
pub(crate) enum MemoryInitKind {
    // The memory range is going to be written by an already initialized source, thus doesn't need extra attention other than marking as initialized.
    ImplicitlyInitialized,
    // The memory range is going to be read, therefore needs to ensure prior initialization.
    NeedsInitializedMemory,
    // The memory going to be discarded and regarded therefore regarded uninitialized.
    // TODO: This is tricky to implement: Discards needs to be resolved within AND between command buffers!
    //       Being able to do this would be quite nice because then we could mark any resource uninitialized at any point in time.
    //       Practically speaking however, discard can only ever happen for single rendertarget surfaces.
    //       Considering this, this could potentially be implemented differently since we don't really need to care about ranges of memory.
    //DiscardMemory,
}

// Most of the time a resource is either fully uninitialized (one element) or initialized (zero elements).
type UninitializedRangeVec<Idx> = SmallVec<[Range<Idx>; 1]>;

/// Tracks initialization status of a linear range from 0..size
#[derive(Debug, Clone)]
pub(crate) struct InitTracker<Idx: Ord + Copy + Default> {
    // Ordered, non overlapping list of all uninitialized ranges.
    uninitialized_ranges: UninitializedRangeVec<Idx>,
}

pub(crate) struct InitTrackerDrain<'a, Idx: Ord + Copy> {
    uninitialized_ranges: &'a mut UninitializedRangeVec<Idx>,
    drain_range: Range<Idx>,
    first_index: usize,
    next_index: usize,
}

impl<'a, Idx> Iterator for InitTrackerDrain<'a, Idx>
where
    Idx: Ord + Copy,
{
    type Item = Range<Idx>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(r) = self
            .uninitialized_ranges
            .get(self.next_index)
            .and_then(|range| {
                if range.start < self.drain_range.end {
                    Some(range.clone())
                } else {
                    None
                }
            })
        {
            self.next_index += 1;
            Some(r.start.max(self.drain_range.start)..r.end.min(self.drain_range.end))
        } else {
            let num_affected = self.next_index - self.first_index;
            if num_affected == 0 {
                return None;
            }

            let first_range = &mut self.uninitialized_ranges[self.first_index];

            // Split one "big" uninitialized range?
            if num_affected == 1
                && first_range.start < self.drain_range.start
                && first_range.end > self.drain_range.end
            {
                let old_start = first_range.start;
                first_range.start = self.drain_range.end;
                self.uninitialized_ranges
                    .insert(self.first_index, old_start..self.drain_range.start);
            }
            // Adjust border ranges and delete everything in-between.
            else {
                let remove_start = if first_range.start >= self.drain_range.start {
                    self.first_index
                } else {
                    first_range.end = self.drain_range.start;
                    self.first_index + 1
                };

                let last_range = &mut self.uninitialized_ranges[self.next_index - 1];
                let remove_end = if last_range.end <= self.drain_range.end {
                    self.next_index
                } else {
                    last_range.start = self.drain_range.end;
                    self.next_index - 1
                };

                self.uninitialized_ranges.drain(remove_start..remove_end);
            }

            None
        }
    }
}

impl<Idx> InitTracker<Idx>
where
    Idx: Ord + Copy + Default,
{
    pub(crate) fn new(size: Idx) -> Self {
        Self {
            uninitialized_ranges: iter::once(Idx::default()..size).collect(),
        }
    }

    // Search smallest range.end which is bigger than bound in O(log n) (with n being number of uninitialized ranges)
    fn lower_bound(&self, bound: Idx) -> usize {
        // This is equivalent to, except that it may return an out of bounds index instead of
        //self.uninitialized_ranges.iter().position(|r| r.end > bound)

        // In future Rust versions this operation can be done with partition_point
        // See https://github.com/rust-lang/rust/pull/73577/
        let mut left = 0;
        let mut right = self.uninitialized_ranges.len();

        while left != right {
            let mid = left + (right - left) / 2;
            let value = unsafe { self.uninitialized_ranges.get_unchecked(mid) };

            if value.end <= bound {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    // Checks if there's any uninitialized ranges within a query.
    // If there are any, the range returned a the subrange of the query_range that contains all these uninitialized regions.
    // Returned range may be larger than necessary (tradeoff for making this function O(log n))
    pub(crate) fn check(&self, query_range: Range<Idx>) -> Option<Range<Idx>> {
        let index = self.lower_bound(query_range.start);
        self.uninitialized_ranges
            .get(index)
            .map(|start_range| {
                if start_range.start < query_range.end {
                    let start = start_range.start.max(query_range.start);
                    match self.uninitialized_ranges.get(index + 1) {
                        Some(next_range) => {
                            if next_range.start < query_range.end {
                                // Would need to keep iterating for more accurate upper bound. Don't do that here.
                                Some(start..query_range.end)
                            } else {
                                Some(start..start_range.end.min(query_range.end))
                            }
                        }
                        None => Some(start..start_range.end.min(query_range.end)),
                    }
                } else {
                    None
                }
            })
            .flatten()
    }

    // Drains uninitialized ranges in a query range.
    #[must_use]
    pub(crate) fn drain(&mut self, drain_range: Range<Idx>) -> InitTrackerDrain<Idx> {
        let index = self.lower_bound(drain_range.start);
        InitTrackerDrain {
            drain_range,
            uninitialized_ranges: &mut self.uninitialized_ranges,
            first_index: index,
            next_index: index,
        }
    }

    // Clears uninitialized ranges in a query range.
    pub(crate) fn clear(&mut self, range: Range<Idx>) {
        self.drain(range).for_each(drop);
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    type Tracker = super::InitTracker<usize>;

    #[test]
    fn check_for_newly_created_tracker() {
        let tracker = Tracker::new(10);
        assert_eq!(tracker.check(0..10), Some(0..10));
        assert_eq!(tracker.check(0..3), Some(0..3));
        assert_eq!(tracker.check(3..4), Some(3..4));
        assert_eq!(tracker.check(4..10), Some(4..10));
    }

    #[test]
    fn check_for_cleared_tracker() {
        let mut tracker = Tracker::new(10);
        tracker.clear(0..10);
        assert_eq!(tracker.check(0..10), None);
        assert_eq!(tracker.check(0..3), None);
        assert_eq!(tracker.check(3..4), None);
        assert_eq!(tracker.check(4..10), None);
    }

    #[test]
    fn check_for_partially_filled_tracker() {
        let mut tracker = Tracker::new(25);
        // Two regions of uninitialized memory
        tracker.clear(0..5);
        tracker.clear(10..15);
        tracker.clear(20..25);

        assert_eq!(tracker.check(0..25), Some(5..25)); // entire range

        assert_eq!(tracker.check(0..5), None); // left non-overlapping
        assert_eq!(tracker.check(3..8), Some(5..8)); // left overlapping region
        assert_eq!(tracker.check(3..17), Some(5..17)); // left overlapping region + contained region

        assert_eq!(tracker.check(8..22), Some(8..22)); // right overlapping region + contained region (yes, doesn't fix range end!)
        assert_eq!(tracker.check(17..22), Some(17..20)); // right overlapping region
        assert_eq!(tracker.check(20..25), None); // right non-overlapping
    }

    #[test]
    fn clear_already_cleared() {
        let mut tracker = Tracker::new(30);
        tracker.clear(10..20);

        // Overlapping with non-cleared
        tracker.clear(5..15); // Left overlap
        tracker.clear(15..25); // Right overlap
        tracker.clear(0..30); // Inner overlap

        // Clear fully cleared
        tracker.clear(0..30);

        assert_eq!(tracker.check(0..30), None);
    }

    #[test]
    fn drain_never_returns_ranges_twice_for_same_range() {
        let mut tracker = Tracker::new(19);
        assert_eq!(tracker.drain(0..19).count(), 1);
        assert_eq!(tracker.drain(0..19).count(), 0);

        let mut tracker = Tracker::new(17);
        assert_eq!(tracker.drain(5..8).count(), 1);
        assert_eq!(tracker.drain(5..8).count(), 0);
        assert_eq!(tracker.drain(1..3).count(), 1);
        assert_eq!(tracker.drain(1..3).count(), 0);
        assert_eq!(tracker.drain(7..13).count(), 1);
        assert_eq!(tracker.drain(7..13).count(), 0);
    }

    #[test]
    fn drain_splits_ranges_correctly() {
        let mut tracker = Tracker::new(1337);
        assert_eq!(
            tracker.drain(21..42).collect::<Vec<Range<usize>>>(),
            vec![21..42]
        );
        assert_eq!(
            tracker.drain(900..1000).collect::<Vec<Range<usize>>>(),
            vec![900..1000]
        );

        // Splitted ranges.
        assert_eq!(
            tracker.drain(5..1003).collect::<Vec<Range<usize>>>(),
            vec![5..21, 42..900, 1000..1003]
        );
        assert_eq!(
            tracker.drain(0..1337).collect::<Vec<Range<usize>>>(),
            vec![0..5, 1003..1337]
        );
    }
}
