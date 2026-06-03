//Note: this could be the only place where we need `SmallVec`.
//TODO: consider getting rid of it.
use smallvec::SmallVec;

use core::{fmt::Debug, iter, ops::Range};

/// Structure that keeps track of a I -> T mapping,
/// optimized for a case where keys of the same values
/// are often grouped together linearly.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RangedStates<I, T> {
    /// List of ranges, each associated with a singe value.
    /// Ranges of keys have to be non-intersecting and ordered.
    ranges: SmallVec<[(Range<I>, T); 1]>,
}

impl<I: Copy + Ord, T: Copy + PartialEq> RangedStates<I, T> {
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

    pub fn iter(&self) -> impl Iterator<Item = &(Range<I>, T)> + Clone {
        self.ranges.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut (Range<I>, T)> {
        self.ranges.iter_mut()
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

    pub fn iter_filter<'a>(
        &'a self,
        range: &'a Range<I>,
    ) -> impl Iterator<Item = (Range<I>, &'a T)> + 'a {
        self.ranges
            .iter()
            .filter(move |&(inner, ..)| inner.end > range.start && inner.start < range.end)
            .map(move |(inner, v)| {
                let new_range = inner.start.max(range.start)..inner.end.min(range.end);

                (new_range, v)
            })
    }

    /// Split the storage ranges in such a way that there is a linear subset of
    /// them occupying exactly `index` range, which is returned mutably.
    ///
    /// Gaps in the ranges are filled with `default` value.
    pub fn isolate(&mut self, index: &Range<I>, default: T) -> &mut [(Range<I>, T)] {
        let start_pos = match self.ranges.iter().position(|pair| pair.0.end > index.start) {
            Some(pos) => pos,
            None => {
                let pos = self.ranges.len();
                self.ranges.push((index.clone(), default));
                return &mut self.ranges[pos..];
            }
        };

        // pass 1: count how many extra slots are needed and find end_pos.
        // doing this before any mutation lets us reserve exactly once and avoid
        // repeated O(n) shifts from incremental SmallVec::insert calls.
        let mut extra = 0usize;
        let mut end_pos = start_pos;
        let mut has_prefix_split = false;
        let mut has_suffix_split = false;
        {
            let mut scan_cursor = index.start;
            let mut i = start_pos;
            loop {
                if i >= self.ranges.len() || self.ranges[i].0.start >= index.end {
                    if scan_cursor < index.end {
                        extra += 1;
                    }
                    end_pos = i.min(self.ranges.len());
                    break;
                }
                let (ref range, _) = self.ranges[i];
                if i == start_pos && range.start < index.start {
                    extra += 1;
                    has_prefix_split = true;
                } else if range.start > scan_cursor {
                    extra += 1;
                }
                if range.end >= index.end {
                    if range.end > index.end {
                        extra += 1;
                        has_suffix_split = true;
                    }
                    end_pos = i + 1;
                    break;
                }
                scan_cursor = range.end;
                i += 1;
            }
        }

        if extra == 0 {
            return &mut self.ranges[start_pos..end_pos];
        }

        // pass 2: extend once, shift the tail right by `extra`, then fill the
        // affected window backwards. write_pos always leads read_pos, so reads
        // are never clobbered before use.
        let original_length = self.ranges.len();
        let filler = self.ranges[start_pos].clone();
        for _ in 0..extra {
            self.ranges.push(filler.clone());
        }
        for i in (end_pos..original_length).rev() {
            let val = self.ranges[i].clone();
            self.ranges[i + extra] = val;
        }

        let mut write_pos = (end_pos + extra) as isize - 1;
        let mut fill_cursor_end = index.end;

        let mut read_pos = end_pos as isize - 1;
        while read_pos >= start_pos as isize {
            let read_index = read_pos as usize;
            let (range, value) = self.ranges[read_index].clone();
            let range_start = range.start;
            let range_end = range.end;

            if has_suffix_split && read_index == end_pos - 1 {
                self.ranges[write_pos as usize] = (index.end..range_end, value);
                write_pos -= 1;
                fill_cursor_end = index.end;
            }

            let effective_end = range_end.min(index.end);
            if effective_end < fill_cursor_end {
                self.ranges[write_pos as usize] = (effective_end..fill_cursor_end, default);
                write_pos -= 1;
                fill_cursor_end = effective_end;
            }

            if has_prefix_split && read_index == start_pos {
                self.ranges[write_pos as usize] = (index.start..effective_end, value);
                write_pos -= 1;
                fill_cursor_end = index.start;
                self.ranges[write_pos as usize] = (range_start..index.start, value);
                write_pos -= 1;
            } else {
                self.ranges[write_pos as usize] = (range_start..effective_end, value);
                write_pos -= 1;
                fill_cursor_end = range_start;
            }

            read_pos -= 1;
        }

        if fill_cursor_end > index.start {
            self.ranges[write_pos as usize] = (index.start..fill_cursor_end, default);
        }

        let out_start = start_pos + has_prefix_split as usize;
        let out_end = end_pos + extra - has_suffix_split as usize;
        &mut self.ranges[out_start..out_end]
    }

    /// Helper method for isolation that checks the sanity of the results.
    #[cfg(test)]
    pub fn sanely_isolated(&self, index: Range<I>, default: T) -> alloc::vec::Vec<(Range<I>, T)> {
        let mut clone = self.clone();
        let result = clone.isolate(&index, default).to_vec();
        clone.check_sanity();
        result
    }
}

#[cfg(test)]
mod test {
    //TODO: randomized/fuzzy testing
    use super::RangedStates;

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
}
