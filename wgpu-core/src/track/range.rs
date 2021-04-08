/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use smallvec::SmallVec;

use std::{cmp::Ordering, fmt::Debug, iter, ops::Range, slice::Iter};

/// Structure that keeps track of a I -> T mapping,
/// optimized for a case where keys of the same values
/// are often grouped together linearly.
#[derive(Clone, Debug, PartialEq)]
pub struct RangedStates<I, T> {
    /// List of ranges, each associated with a singe value.
    /// Ranges of keys have to be non-intersecting and ordered.
    ranges: SmallVec<[(Range<I>, T); 1]>,
}

impl<I: Copy + PartialOrd, T: Copy + PartialEq> RangedStates<I, T> {
    pub fn empty() -> Self {
        Self {
            ranges: SmallVec::new(),
        }
    }

    pub fn from_range(range: Range<I>, value: T) -> Self {
        Self {
            ranges: iter::once((range, value)).collect(),
        }
    }

    /// Construct a new instance from a slice of ranges.
    #[cfg(test)]
    pub fn from_slice(values: &[(Range<I>, T)]) -> Self {
        Self {
            ranges: values.iter().cloned().collect(),
        }
    }

    /// Clear all the ranges.
    pub fn clear(&mut self) {
        self.ranges.clear();
    }

    /// Append a range.
    ///
    /// Assumes that the object is being constructed from a set of
    /// ranges, and they are given in the ascending order of their keys.
    pub fn append(&mut self, index: Range<I>, value: T) {
        if let Some(last) = self.ranges.last() {
            debug_assert!(last.0.end <= index.start);
        }
        self.ranges.push((index, value));
    }

    /// Check that all the ranges are non-intersecting and ordered.
    /// Panics otherwise.
    #[cfg(test)]
    fn check_sanity(&self) {
        for a in self.ranges.iter() {
            assert!(a.0.start < a.0.end);
        }
        for (a, b) in self.ranges.iter().zip(self.ranges[1..].iter()) {
            assert!(a.0.end <= b.0.start);
        }
    }

    /// Merge the neighboring ranges together, where possible.
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn coalesce(&mut self) {
        let mut num_removed = 0;
        let mut iter = self.ranges.iter_mut();
        let mut cur = match iter.next() {
            Some(elem) => elem,
            None => return,
        };
        for next in iter {
            if cur.0.end == next.0.start && cur.1 == next.1 {
                num_removed += 1;
                cur.0.end = next.0.end;
                next.0.end = next.0.start;
            } else {
                cur = next;
            }
        }
        if num_removed != 0 {
            self.ranges.retain(|pair| pair.0.start != pair.0.end);
        }
    }

    /// Check if all intersecting ranges have the same value, which is returned.
    ///
    /// Returns `None` if no intersections are detected.
    /// Returns `Some(Err)` if the intersected values are inconsistent.
    pub fn query<U: PartialEq>(
        &self,
        index: &Range<I>,
        fun: impl Fn(&T) -> U,
    ) -> Option<Result<U, ()>> {
        let mut result = None;
        for &(ref range, ref value) in self.ranges.iter() {
            if range.end > index.start && range.start < index.end {
                let old = result.replace(fun(value));
                if old.is_some() && old != result {
                    return Some(Err(()));
                }
            }
        }
        result.map(Ok)
    }

    /// Split the storage ranges in such a way that there is a linear subset of
    /// them occupying exactly `index` range, which is returned mutably.
    ///
    /// Gaps in the ranges are filled with `default` value.
    pub fn isolate(&mut self, index: &Range<I>, default: T) -> &mut [(Range<I>, T)] {
        //TODO: implement this in 2 passes:
        // 1. scan the ranges to figure out how many extra ones need to be inserted
        // 2. go through the ranges by moving them them to the right and inserting the missing ones

        let mut start_pos = match self.ranges.iter().position(|pair| pair.0.end > index.start) {
            Some(pos) => pos,
            None => {
                let pos = self.ranges.len();
                self.ranges.push((index.clone(), default));
                return &mut self.ranges[pos..];
            }
        };

        {
            let (range, value) = self.ranges[start_pos].clone();
            if range.start < index.start {
                self.ranges[start_pos].0.start = index.start;
                self.ranges
                    .insert(start_pos, (range.start..index.start, value));
                start_pos += 1;
            }
        }
        let mut pos = start_pos;
        let mut range_pos = index.start;
        loop {
            let (range, value) = self.ranges[pos].clone();
            if range.start >= index.end {
                self.ranges.insert(pos, (range_pos..index.end, default));
                pos += 1;
                break;
            }
            if range.start > range_pos {
                self.ranges.insert(pos, (range_pos..range.start, default));
                pos += 1;
                range_pos = range.start;
            }
            if range.end >= index.end {
                if range.end != index.end {
                    self.ranges[pos].0.start = index.end;
                    self.ranges.insert(pos, (range_pos..index.end, value));
                }
                pos += 1;
                break;
            }
            pos += 1;
            range_pos = range.end;
            if pos == self.ranges.len() {
                self.ranges.push((range_pos..index.end, default));
                pos += 1;
                break;
            }
        }

        &mut self.ranges[start_pos..pos]
    }

    /// Helper method for isolation that checks the sanity of the results.
    #[cfg(test)]
    pub fn sanely_isolated(&self, index: Range<I>, default: T) -> Vec<(Range<I>, T)> {
        let mut clone = self.clone();
        let result = clone.isolate(&index, default).to_vec();
        clone.check_sanity();
        result
    }

    /// Produce an iterator that merges two instances together.
    ///
    /// Each range in the returned iterator is a subset of a range in either
    /// `self` or `other`, and the value returned as a `Range` from `self` to `other`.
    pub fn merge<'a>(&'a self, other: &'a Self, base: I) -> Merge<'a, I, T> {
        Merge {
            base,
            sa: self.ranges.iter().peekable(),
            sb: other.ranges.iter().peekable(),
        }
    }
}

/// A custom iterator that goes through two `RangedStates` and process a merge.
#[derive(Debug)]
pub struct Merge<'a, I, T> {
    base: I,
    sa: iter::Peekable<Iter<'a, (Range<I>, T)>>,
    sb: iter::Peekable<Iter<'a, (Range<I>, T)>>,
}

impl<'a, I: Copy + Debug + Ord, T: Copy + Debug> Iterator for Merge<'a, I, T> {
    type Item = (Range<I>, Range<Option<T>>);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.sa.peek(), self.sb.peek()) {
            // we have both streams
            (Some(&&(ref ra, va)), Some(&&(ref rb, vb))) => {
                let (range, usage) = if ra.start < self.base {
                    // in the middle of the left stream
                    let (end, end_value) = if self.base == rb.start {
                        // right stream is starting
                        debug_assert!(self.base < ra.end);
                        (rb.end, Some(vb))
                    } else {
                        // right hasn't started yet
                        debug_assert!(self.base < rb.start);
                        (rb.start, None)
                    };
                    (self.base..ra.end.min(end), Some(va)..end_value)
                } else if rb.start < self.base {
                    // in the middle of the right stream
                    let (end, start_value) = if self.base == ra.start {
                        // left stream is starting
                        debug_assert!(self.base < rb.end);
                        (ra.end, Some(va))
                    } else {
                        // left hasn't started yet
                        debug_assert!(self.base < ra.start);
                        (ra.start, None)
                    };
                    (self.base..rb.end.min(end), start_value..Some(vb))
                } else {
                    // no active streams
                    match ra.start.cmp(&rb.start) {
                        // both are starting
                        Ordering::Equal => (ra.start..ra.end.min(rb.end), Some(va)..Some(vb)),
                        // only left is starting
                        Ordering::Less => (ra.start..rb.start.min(ra.end), Some(va)..None),
                        // only right is starting
                        Ordering::Greater => (rb.start..ra.start.min(rb.end), None..Some(vb)),
                    }
                };
                self.base = range.end;
                if ra.end == range.end {
                    let _ = self.sa.next();
                }
                if rb.end == range.end {
                    let _ = self.sb.next();
                }
                Some((range, usage))
            }
            // only right stream
            (None, Some(&&(ref rb, vb))) => {
                let range = self.base.max(rb.start)..rb.end;
                self.base = rb.end;
                let _ = self.sb.next();
                Some((range, None..Some(vb)))
            }
            // only left stream
            (Some(&&(ref ra, va)), None) => {
                let range = self.base.max(ra.start)..ra.end;
                self.base = ra.end;
                let _ = self.sa.next();
                Some((range, Some(va)..None))
            }
            // done
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod test {
    //TODO: randomized/fuzzy testing
    use super::RangedStates;
    use std::{fmt::Debug, ops::Range};

    fn easy_merge<T: PartialEq + Copy + Debug>(
        ra: &[(Range<usize>, T)],
        rb: &[(Range<usize>, T)],
    ) -> Vec<(Range<usize>, Range<Option<T>>)> {
        RangedStates::from_slice(ra)
            .merge(&RangedStates::from_slice(rb), 0)
            .collect()
    }

    #[test]
    fn sane_good() {
        let rs = RangedStates::from_slice(&[(1..4, 9u8), (4..5, 9)]);
        rs.check_sanity();
    }

    #[test]
    #[should_panic]
    fn sane_empty() {
        let rs = RangedStates::from_slice(&[(1..4, 9u8), (5..5, 9)]);
        rs.check_sanity();
    }

    #[test]
    #[should_panic]
    fn sane_intersect() {
        let rs = RangedStates::from_slice(&[(1..4, 9u8), (3..5, 9)]);
        rs.check_sanity();
    }

    #[test]
    fn coalesce() {
        let mut rs = RangedStates::from_slice(&[(1..4, 9u8), (4..5, 9), (5..7, 1), (8..9, 1)]);
        rs.coalesce();
        rs.check_sanity();
        assert_eq!(rs.ranges.as_slice(), &[(1..5, 9), (5..7, 1), (8..9, 1),]);
    }

    #[test]
    fn query() {
        let rs = RangedStates::from_slice(&[(1..4, 1u8), (5..7, 2)]);
        assert_eq!(rs.query(&(0..1), |v| *v), None);
        assert_eq!(rs.query(&(1..3), |v| *v), Some(Ok(1)));
        assert_eq!(rs.query(&(1..6), |v| *v), Some(Err(())));
    }

    #[test]
    fn isolate() {
        let rs = RangedStates::from_slice(&[(1..4, 9u8), (4..5, 9), (5..7, 1), (8..9, 1)]);
        assert_eq!(&rs.sanely_isolated(4..5, 0), &[(4..5, 9u8),]);
        assert_eq!(
            &rs.sanely_isolated(0..6, 0),
            &[(0..1, 0), (1..4, 9u8), (4..5, 9), (5..6, 1),]
        );
        assert_eq!(&rs.sanely_isolated(8..10, 1), &[(8..9, 1), (9..10, 1),]);
        assert_eq!(
            &rs.sanely_isolated(6..9, 0),
            &[(6..7, 1), (7..8, 0), (8..9, 1),]
        );
    }

    #[test]
    fn merge_same() {
        assert_eq!(
            &easy_merge(&[(1..4, 0u8),], &[(1..4, 2u8),],),
            &[(1..4, Some(0)..Some(2)),]
        );
    }

    #[test]
    fn merge_empty() {
        assert_eq!(
            &easy_merge(&[(1..2, 0u8),], &[],),
            &[(1..2, Some(0)..None),]
        );
        assert_eq!(
            &easy_merge(&[], &[(3..4, 1u8),],),
            &[(3..4, None..Some(1)),]
        );
    }

    #[test]
    fn merge_separate() {
        assert_eq!(
            &easy_merge(&[(1..2, 0u8), (5..6, 1u8),], &[(2..4, 2u8),],),
            &[
                (1..2, Some(0)..None),
                (2..4, None..Some(2)),
                (5..6, Some(1)..None),
            ]
        );
    }

    #[test]
    fn merge_subset() {
        assert_eq!(
            &easy_merge(&[(1..6, 0u8),], &[(2..4, 2u8),],),
            &[
                (1..2, Some(0)..None),
                (2..4, Some(0)..Some(2)),
                (4..6, Some(0)..None),
            ]
        );
        assert_eq!(
            &easy_merge(&[(2..4, 0u8),], &[(1..4, 2u8),],),
            &[(1..2, None..Some(2)), (2..4, Some(0)..Some(2)),]
        );
    }

    #[test]
    fn merge_all() {
        assert_eq!(
            &easy_merge(&[(1..4, 0u8), (5..8, 1u8),], &[(2..6, 2u8), (7..9, 3u8),],),
            &[
                (1..2, Some(0)..None),
                (2..4, Some(0)..Some(2)),
                (4..5, None..Some(2)),
                (5..6, Some(1)..Some(2)),
                (6..7, Some(1)..None),
                (7..8, Some(1)..Some(3)),
                (8..9, None..Some(3)),
            ]
        );
    }

    #[test]
    fn merge_complex() {
        assert_eq!(
            &easy_merge(
                &[
                    (0..8, 0u8),
                    (8..9, 1),
                    (9..16, 2),
                    (16..17, 3),
                    (17..118, 4),
                    (118..119, 5),
                    (119..124, 6),
                    (124..125, 7),
                    (125..512, 8),
                ],
                &[(15..16, 10u8), (51..52, 11), (126..127, 12),],
            ),
            &[
                (0..8, Some(0)..None),
                (8..9, Some(1)..None),
                (9..15, Some(2)..None),
                (15..16, Some(2)..Some(10)),
                (16..17, Some(3)..None),
                (17..51, Some(4)..None),
                (51..52, Some(4)..Some(11)),
                (52..118, Some(4)..None),
                (118..119, Some(5)..None),
                (119..124, Some(6)..None),
                (124..125, Some(7)..None),
                (125..126, Some(8)..None),
                (126..127, Some(8)..Some(12)),
                (127..512, Some(8)..None),
            ]
        );
    }
}
