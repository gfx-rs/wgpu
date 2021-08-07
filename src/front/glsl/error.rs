use super::{
    constants::ConstantSolvingError,
    token::{SourceMetadata, TokenValue},
};
use pp_rs::token::PreprocessorError;
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
    #[error("Invalid profile: {0}")]
    InvalidProfile(String),
    #[error("Invalid version: {0}")]
    InvalidVersion(u64),
    #[error("Expected {}, found {0:?}", join_with_comma(.1))]
    InvalidToken(TokenValue, Vec<ExpectedToken>),
    #[error("Not implemented: {0}")]
    NotImplemented(&'static str),
    #[error("Unknown variable: {0}")]
    UnknownVariable(String),
    #[error("Unknown type: {0}")]
    UnknownType(String),
    #[error("Unknown field: {0}")]
    UnknownField(String),
    #[error("Unknown layout qualifier: {0}")]
    UnknownLayoutQualifier(String),
    #[cfg(feature = "glsl-validate")]
    #[error("Variable already declared: {0}")]
    VariableAlreadyDeclared(String),
    #[error("{0}")]
    SemanticError(Cow<'static, str>),
    #[error("{0:?}")]
    PreprocessorError(PreprocessorError),
}

impl From<ConstantSolvingError> for ErrorKind {
    fn from(err: ConstantSolvingError) -> Self {
        ErrorKind::SemanticError(err.to_string().into())
    }
}

#[derive(Debug, Error)]
#[error("{kind}")]
#[cfg_attr(test, derive(PartialEq))]
pub struct Error {
    pub kind: ErrorKind,
    pub meta: SourceMetadata,
}

impl Error {
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

        Error {
            kind: ErrorKind::SemanticError(msg.into()),
            meta,
        }
    }
}
