use std::ops::Range;

#[derive(Debug, Clone)]
pub struct TokenMetadata {
    pub line: usize,
    pub chars: Range<usize>,
}
