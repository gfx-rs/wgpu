use std::ops::Range;

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct TokenMetadata {
    pub line: usize,
    pub chars: Range<usize>,
}
