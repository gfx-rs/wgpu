use super::{
    constants::ConstantSolvingError,
    token::{SourceMetadata, Token, TokenValue},
};
use std::borrow::Cow;
use thiserror::Error;

fn join_with_comma(list: &[ExpectedToken]) -> String {
    let mut string = "".to_string();
    for (i, val) in list.iter().enumerate() {
        string.push_str(&val.to_string());
        match i {
            i if i == list.len() - 1 => {}
            i if i == list.len() - 2 => string.push_str(" or "),
            _ => string.push_str(", "),
        }
    }
    string
}

#[derive(Debug, PartialEq)]
pub enum ExpectedToken {
    Token(TokenValue),
    TypeName,
    Identifier,
    IntLiteral,
    FloatLiteral,
    BoolLiteral,
    Eof,
}
impl From<TokenValue> for ExpectedToken {
    fn from(token: TokenValue) -> Self {
        ExpectedToken::Token(token)
    }
}
impl std::fmt::Display for ExpectedToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ExpectedToken::Token(ref token) => write!(f, "{:?}", token),
            ExpectedToken::TypeName => write!(f, "a type"),
            ExpectedToken::Identifier => write!(f, "identifier"),
            ExpectedToken::IntLiteral => write!(f, "integer literal"),
            ExpectedToken::FloatLiteral => write!(f, "float literal"),
            ExpectedToken::BoolLiteral => write!(f, "bool literal"),
            ExpectedToken::Eof => write!(f, "end of file"),
        }
    }
}

#[derive(Debug, Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ErrorKind {
    #[error("Unexpected end of file")]
    EndOfFile,
    #[error("Invalid profile: {1}")]
    InvalidProfile(SourceMetadata, String),
    #[error("Invalid version: {1}")]
    InvalidVersion(SourceMetadata, u64),
    #[error("Expected {}, found {0}", join_with_comma(.1))]
    InvalidToken(Token, Vec<ExpectedToken>),
    #[error("Not implemented {0}")]
    NotImplemented(&'static str),
    #[error("Unknown variable: {1}")]
    UnknownVariable(SourceMetadata, String),
    #[error("Unknown type: {1}")]
    UnknownType(SourceMetadata, String),
    #[error("Unknown field: {1}")]
    UnknownField(SourceMetadata, String),
    #[error("Unknown layout qualifier: {1}")]
    UnknownLayoutQualifier(SourceMetadata, String),
    #[cfg(feature = "glsl-validate")]
    #[error("Variable already declared: {1}")]
    VariableAlreadyDeclared(SourceMetadata, String),
    #[error("{1}")]
    SemanticError(SourceMetadata, Cow<'static, str>),
}

impl ErrorKind {
    /// Returns the TokenMetadata if available
    pub fn metadata(&self) -> Option<SourceMetadata> {
        match *self {
            ErrorKind::UnknownVariable(metadata, _)
            | ErrorKind::InvalidProfile(metadata, _)
            | ErrorKind::InvalidVersion(metadata, _)
            | ErrorKind::UnknownLayoutQualifier(metadata, _)
            | ErrorKind::SemanticError(metadata, _)
            | ErrorKind::UnknownField(metadata, _) => Some(metadata),
            #[cfg(feature = "glsl-validate")]
            ErrorKind::VariableAlreadyDeclared(metadata, _) => Some(metadata),
            ErrorKind::InvalidToken(ref token, _) => Some(token.meta),
            _ => None,
        }
    }

    pub(crate) fn wrong_function_args(
        name: String,
        expected: usize,
        got: usize,
        meta: SourceMetadata,
    ) -> Self {
        let msg = format!(
            "Function \"{}\" expects {} arguments, got {}",
            name, expected, got
        );

        ErrorKind::SemanticError(meta, msg.into())
    }
}

impl From<(SourceMetadata, ConstantSolvingError)> for ErrorKind {
    fn from((meta, err): (SourceMetadata, ConstantSolvingError)) -> Self {
        ErrorKind::SemanticError(meta, err.to_string().into())
    }
}

#[derive(Debug, Error)]
#[error("{kind}")]
pub struct ParseError {
    pub kind: ErrorKind,
}

impl From<ErrorKind> for ParseError {
    fn from(kind: ErrorKind) -> Self {
        ParseError { kind }
    }
}
