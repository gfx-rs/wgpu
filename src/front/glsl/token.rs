pub use pp_rs::token::{Float, Integer, PreprocessorError};

use crate::{Interpolation, Sampling, Type};
use std::{fmt, ops::Range};

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SourceMetadata {
    /// Byte offset into the source where the first char starts
    pub start: usize,
    /// Byte offset into the source where the first char not belonging to this
    /// source metadata starts
    pub end: usize,
}

impl SourceMetadata {
    pub fn union(&self, other: &Self) -> Self {
        SourceMetadata {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl From<SourceMetadata> for Range<usize> {
    fn from(meta: SourceMetadata) -> Self {
        meta.start..meta.end
    }
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Token {
    pub value: TokenValue,
    pub meta: SourceMetadata,
}

#[derive(Debug, PartialEq)]
pub enum TokenValue {
    Unknown(PreprocessorError),
    Identifier(String),

    Extension,
    Version,
    Pragma,

    FloatConstant(Float),
    IntConstant(Integer),
    BoolConstant(bool),

    Layout,
    In,
    Out,
    InOut,
    Uniform,
    Buffer,
    Const,
    Interpolation(Interpolation),
    Sampling(Sampling),

    Continue,
    Break,
    Return,
    Discard,

    If,
    Else,
    Switch,
    Case,
    Default,
    While,
    Do,
    For,

    Void,
    Struct,
    TypeName(Type),

    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    LeftShiftAssign,
    RightShiftAssign,
    AndAssign,
    XorAssign,
    OrAssign,

    Increment,
    Decrement,

    LogicalOr,
    LogicalAnd,
    LogicalXor,

    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,

    LeftShift,
    RightShift,

    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftAngle,
    RightAngle,

    Comma,
    Semicolon,
    Colon,
    Dot,
    Bang,
    Dash,
    Tilde,
    Plus,
    Star,
    Slash,
    Percent,
    VerticalBar,
    Caret,
    Ampersand,
    Question,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.value)
    }
}
