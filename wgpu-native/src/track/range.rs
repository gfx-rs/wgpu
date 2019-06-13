use std::{
    cmp::Ordering,
    iter::Peekable,
    ops::Range,
    slice::Iter,
};

#[derive(Clone, Debug)]
pub struct RangedStates<I, T> {
    ranges: Vec<(Range<I>, T)>,
}

impl<I, T> Default for RangedStates<I, T> {
    fn default() -> Self {
        RangedStates {
            ranges: Vec::new(),
        }
    }
}

impl<I: Copy + PartialOrd, T: Copy + PartialEq> RangedStates<I, T> {
    pub fn clear(&mut self) {
        self.ranges.clear();
    }

    pub fn append(&mut self, index: Range<I>, value: T) {
        self.ranges.push((index, value));
    }

    pub fn iter(&self) -> Iter<(Range<I>, T)> {
        self.ranges.iter()
    }

    #[cfg(test)]
    fn check_sanity(&self) {
        for a in self.ranges.iter() {
            assert!(a.0.start < a.0.end);
        }
        for (a, b) in self.ranges.iter().zip(self.ranges[1..].iter()) {
            assert!(a.0.end <= b.0.start);
        }
    }

    #[cfg(test)]
    fn coalesce(&mut self) {
        let mut num_removed = 0;
        let mut iter = self.ranges.iter_mut();
        let mut cur = match iter.next() {
            Some(elem) => elem,
            None => return,
        };
        while let Some(next) = iter.next() {
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

    pub fn isolate(&mut self, index: &Range<I>, default: T) -> &mut [(Range<I>, T)] {
        //TODO: implement this in 2 passes:
        // 1. scan the ranges to figure out how many extra ones need to be inserted
        // 2. go through the ranges by moving them them to the right and inserting the missing ones

        let mut start_pos = match self.ranges
            .iter()
            .position(|pair| pair.0.end > index.start)
        {
            Some(pos) => pos,
            None => {
                let pos = self.ranges.len();
                self.ranges.push((index.clone(), default));
                return &mut self.ranges[pos ..];
            }
        };

        {
            let (range, value) = self.ranges[start_pos].clone();
            if range.start < index.start {
                self.ranges[start_pos].0.start = index.start;
                self.ranges.insert(start_pos, (range.start .. index.start, value));
                start_pos += 1;
            }
        }
        let mut pos = start_pos;
        let mut range_pos = index.start;
        loop {
            let (range, value) = self.ranges[pos].clone();
            if range.start >= index.end {
                self.ranges.insert(pos, (range_pos .. index.end, default));
                pos += 1;
                break;
            }
            if range.start > range_pos {
                self.ranges.insert(pos, (range_pos .. range.start, default));
                pos += 1;
                range_pos = range.start;
            }
            if range.end >= index.end {
                if range.end != index.end {
                    self.ranges[pos].0.start = index.end;
                    self.ranges.insert(pos, (range_pos .. index.end, value));
                }
                pos += 1;
                break;
            }
            pos += 1;
            range_pos = range.end;
            if pos == self.ranges.len() {
                self.ranges.push((range_pos .. index.end, default));
                pos += 1;
                break;
            }
        }

        &mut self.ranges[start_pos .. pos]
    }

    pub fn merge<'a>(&'a self, other: &'a Self, base: I) -> Merge<'a, I, T> {
        Merge {
            base,
            sa: self.ranges.iter().peekable(),
            sb: other.ranges.iter().peekable(),
        }
    }
}

pub struct Merge<'a, I, T> {
    base: I,
    sa: Peekable<Iter<'a, (Range<I>, T)>>,
    sb: Peekable<Iter<'a, (Range<I>, T)>>,
}

impl<'a, I: Copy + Ord, T: Copy> Iterator for Merge<'a, I, T> {
    type Item = (Range<I>, Range<T>);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.sa.peek(), self.sb.peek()) {
            // we have both streams
            (Some(&(ref ra, va)), Some(&(ref rb, vb))) => {
                let (range, usage) = if ra.start < self.base { // in the middle of the left stream
                    if self.base == rb.start { // right stream is starting
                        debug_assert!(self.base < ra.end);
                        (self.base .. ra.end.min(rb.end), *va .. *vb)
                    } else { // right hasn't started yet
                        debug_assert!(self.base < rb.start);
                        (self.base .. rb.start, *va .. *va)
                    }
                } else if rb.start < self.base { // in the middle of the right stream
                    if self.base == ra.start { // left stream is starting
                        debug_assert!(self.base < rb.end);
                        (self.base .. ra.end.min(rb.end), *va .. *vb)
                    } else { // left hasn't started yet
                        debug_assert!(self.base < ra.start);
                        (self.base .. ra.start, *vb .. *vb)
                    }
                } else { // no active streams
                    match ra.start.cmp(&rb.start) {
                        // both are starting
                        Ordering::Equal => (ra.start .. ra.end.min(rb.end), *va .. *vb),
                        // only left is starting
                        Ordering::Less => (ra.start .. rb.start, *va .. *va),
                        // only right is starting
                        Ordering::Greater => (rb.start .. ra.start, *vb .. *vb),
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
            (None, Some(&(ref rb, vb))) => {
                let range = self.base.max(rb.start) .. rb.end;
                self.base = rb.end;
                let _ = self.sb.next();
                Some((range, *vb .. *vb))
            }
            // only left stream
            (Some(&(ref ra, va)), None) => {
                let range = self.base.max(ra.start) .. ra.end;
                self.base = ra.end;
                let _ = self.sa.next();
                Some((range, *va .. *va))
            }
            // done
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::RangedStates;

    #[test]
    fn test_sane0() {
        let rs = RangedStates { ranges: vec![
            (1..4, 9u8),
            (4..5, 9),
        ]};
        rs.check_sanity();
    }

    #[test]
    #[should_panic]
    fn test_sane1() {
        let rs = RangedStates { ranges: vec![
            (1..4, 9u8),
            (5..5, 9),
        ]};
        rs.check_sanity();
    }

    #[test]
    #[should_panic]
    fn test_sane2() {
        let rs = RangedStates { ranges: vec![
            (1..4, 9u8),
            (3..5, 9),
        ]};
        rs.check_sanity();
    }

    #[test]
    fn test_coalesce() {
        let mut rs = RangedStates { ranges: vec![
            (1..4, 9u8),
            (4..5, 9),
            (5..7, 1),
            (8..9, 1),
        ]};
        rs.coalesce();
        rs.check_sanity();
        assert_eq!(rs.ranges, vec![
            (1..5, 9),
            (5..7, 1),
            (8..9, 1),
        ]);
    }

    #[test]
    fn test_isolate() {
        let rs = RangedStates { ranges: vec![
            (1..4, 9u8),
            (4..5, 9),
            (5..7, 1),
            (8..9, 1),
        ]};
        assert_eq!(rs.clone().isolate(&(4..5), 0), [
            (4..5, 9u8),
        ]);
        assert_eq!(rs.clone().isolate(&(0..6), 0), [
            (0..1, 0),
            (1..4, 9u8),
            (4..5, 9),
            (5..6, 1),
        ]);
        assert_eq!(rs.clone().isolate(&(8..10), 1), [
            (8..9, 1),
            (9..10, 1),
        ]);
        assert_eq!(rs.clone().isolate(&(6..9), 0), [
            (6..7, 1),
            (7..8, 0),
            (8..9, 1),
        ]);
    }
}
