use super::{
    constants::ConstantSolvingError,
    token::{SourceMetadata, Token},
};
use std::borrow::Cow;
use thiserror::Error;

#[derive(Debug, Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ErrorKind {
    #[error("Unexpected end of file")]
    EndOfFile,
    #[error("Invalid profile: {1}")]
    InvalidProfile(SourceMetadata, String),
    #[error("Invalid version: {1}")]
    InvalidVersion(SourceMetadata, u64),
    #[error("Unexpected token: {0}")]
    InvalidToken(Token),
    #[error("Not implemented {0}")]
    NotImplemented(&'static str),
    #[error("Unknown variable: {1}")]
    UnknownVariable(SourceMetadata, String),
    #[error("Unknown field: {1}")]
    UnknownField(SourceMetadata, String),
    #[error("Unknown layout qualifier: {1}")]
    UnknownLayoutQualifier(SourceMetadata, String),
    #[cfg(feature = "glsl-validate")]
    #[error("Variable already declared: {0}")]
    VariableAlreadyDeclared(String),
    #[error("{1}")]
    SemanticError(SourceMetadata, Cow<'static, str>),
}

impl ErrorKind {
    /// Returns the TokenMetadata if available
    pub fn metadata(&self) -> Option<&SourceMetadata> {
        match *self {
            ErrorKind::UnknownVariable(ref metadata, _)
            | ErrorKind::InvalidProfile(ref metadata, _)
            | ErrorKind::InvalidVersion(ref metadata, _)
            | ErrorKind::UnknownLayoutQualifier(ref metadata, _)
            | ErrorKind::SemanticError(ref metadata, _)
            | ErrorKind::UnknownField(ref metadata, _) => Some(metadata),
            ErrorKind::InvalidToken(ref token) => Some(&token.meta),
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
