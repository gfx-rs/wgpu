use std::ops::Range;

/// A source code span, used for error reporting.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Span {
    start: u32,
    end: u32,
}

impl Span {
    /// Creates a new `Span` from a range of byte indices
    ///
    /// Note: end is exclusive, it doesn't belong to the `Span`
    pub fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }

    /// Modifies `self` to contain the smallest `Span` possible that
    /// contains both `self` and `other`
    pub fn subsume(&mut self, other: Self) {
        *self = if !self.is_defined() {
            // self isn't defined so use other
            other
        } else if !other.is_defined() {
            // other isn't defined so don't try to subsume
            *self
        } else {
            // Both self and other are defined so calculate the span that contains them both
            Span {
                start: self.start.min(other.start),
                end: self.end.max(other.end),
            }
        }
    }

    /// Returns the smallest `Span` possible that contains all the `Span`s
    /// defined in the `from` iterator
    pub fn total_span<T: Iterator<Item = Self>>(from: T) -> Self {
        let mut span: Self = Default::default();
        for other in from {
            span.subsume(other);
        }
        span
    }

    /// Converts `self` to a range if the span is not unknown
    pub fn to_range(self) -> Option<Range<usize>> {
        if self.is_defined() {
            Some(self.start as usize..self.end as usize)
        } else {
            None
        }
    }

    /// Check wether `self` was defined or is a default/unknown span
    fn is_defined(&self) -> bool {
        *self != Self::default()
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Span {
            start: range.start as u32,
            end: range.end as u32,
        }
    }
}
