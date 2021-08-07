use pp_rs::token::Location;
pub use pp_rs::token::{Float, Integer, PreprocessorError, Token as PPToken};

use super::ast::Precision;
use crate::{Interpolation, Sampling, Type};
use std::ops::Range;

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

    pub fn none() -> Self {
        SourceMetadata::default()
    }
}

impl From<Location> for SourceMetadata {
    fn from(loc: Location) -> Self {
        SourceMetadata {
            start: loc.start as usize,
            end: loc.end as usize,
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
    Identifier(String),

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

    Restrict,
    StorageAccess(crate::StorageAccess),

    Interpolation(Interpolation),
    Sampling(Sampling),
    Precision,
    PrecisionQualifier(Precision),

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

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Directive {
    pub kind: DirectiveKind,
    pub tokens: Vec<PPToken>,
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum DirectiveKind {
    Version { is_first_directive: bool },
    Extension,
    Pragma,
}
