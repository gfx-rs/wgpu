/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::ops::Range;

#[derive(Debug, Clone, Copy)]
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
    // Ordered, non overlapping list of all uninitialized ranges.
    uninitialized_ranges: Vec<Range<wgt::BufferAddress>>,
}

pub(crate) struct MemoryInitTrackerDrain<'a> {
    uninitialized_ranges: &'a mut Vec<Range<wgt::BufferAddress>>,
    drain_range: Range<wgt::BufferAddress>,
    first_index: usize,
    next_index: usize,
}

impl<'a> Iterator for MemoryInitTrackerDrain<'a> {
    type Item = Range<wgt::BufferAddress>;

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

impl MemoryInitTracker {
    pub(crate) fn new(size: wgt::BufferAddress) -> Self {
        Self {
            uninitialized_ranges: vec![0..size],
        }
    }

    // Search smallest range.end which is bigger than bound in O(log n) (with n being number of uninitialized ranges)
    fn lower_bound(&self, bound: wgt::BufferAddress) -> usize {
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
    pub(crate) fn check(
        &self,
        query_range: Range<wgt::BufferAddress>,
    ) -> Option<Range<wgt::BufferAddress>> {
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
    pub(crate) fn drain(
        &mut self,
        drain_range: Range<wgt::BufferAddress>,
    ) -> MemoryInitTrackerDrain {
        let index = self.lower_bound(drain_range.start);
        MemoryInitTrackerDrain {
            drain_range,
            uninitialized_ranges: &mut self.uninitialized_ranges,
            first_index: index,
            next_index: index,
        }
    }

    // Clears uninitialized ranges in a query range.
    pub(crate) fn clear(&mut self, range: Range<wgt::BufferAddress>) {
        self.drain(range).for_each(drop);
    }
}

#[cfg(test)]
mod test {
    use super::MemoryInitTracker;
    use std::ops::Range;

    #[test]
    fn check_for_newly_created_tracker() {
        let tracker = MemoryInitTracker::new(10);
        assert_eq!(tracker.check(0..10), Some(0..10));
        assert_eq!(tracker.check(0..3), Some(0..3));
        assert_eq!(tracker.check(3..4), Some(3..4));
        assert_eq!(tracker.check(4..10), Some(4..10));
    }

    #[test]
    fn check_for_cleared_tracker() {
        let mut tracker = MemoryInitTracker::new(10);
        tracker.clear(0..10);
        assert_eq!(tracker.check(0..10), None);
        assert_eq!(tracker.check(0..3), None);
        assert_eq!(tracker.check(3..4), None);
        assert_eq!(tracker.check(4..10), None);
    }

    #[test]
    fn check_for_partially_filled_tracker() {
        let mut tracker = MemoryInitTracker::new(25);
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
        let mut tracker = MemoryInitTracker::new(30);
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
