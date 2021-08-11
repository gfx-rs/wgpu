use std::ops::Range;

// A source code span, used for error reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum Span {
    // Span is unknown - no source information.
    Unknown,
    // Byte range.
    ByteRange(Range<usize>),
}

impl Default for Span {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Span {
    pub fn subsume(&mut self, other: &Self) {
        match *self {
            Self::Unknown => self.clone_from(other),
            Self::ByteRange(ref mut self_range) => {
                if let Self::ByteRange(ref other_range) = *other {
                    self_range.start = self_range.start.min(other_range.start);
                    self_range.end = self_range.end.max(other_range.end);
                }
            }
        }
    }

    pub fn total_span<'a, T: Iterator<Item = &'a Self>>(from: T) -> Self {
        let mut span: Self = Default::default();
        for other in from {
            span.subsume(other);
        }
        span
    }
}
