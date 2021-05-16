use super::token::{SourceMetadata, Token};
use std::{borrow::Cow, fmt};

//TODO: use `thiserror`
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ErrorKind {
    EndOfFile,
    InvalidInput,
    InvalidProfile(SourceMetadata, String),
    InvalidToken(Token),
    InvalidVersion(SourceMetadata, u64),
    ParserFail,
    ParserStackOverflow,
    NotImplemented(&'static str),
    UnknownVariable(SourceMetadata, String),
    UnknownField(SourceMetadata, String),
    UnknownLayoutQualifier(SourceMetadata, String),
    #[cfg(feature = "glsl-validate")]
    VariableAlreadyDeclared(String),
    ExpectedConstant,
    SemanticError(Cow<'static, str>),
    PreprocessorError(String),
    WrongNumberArgs(String, usize, usize),
}

impl ErrorKind {
    // Returns the TokenMetadata if available
    pub fn metadata(&self) -> Option<&SourceMetadata> {
        match *self {
            ErrorKind::UnknownVariable(ref metadata, _)
            | ErrorKind::InvalidProfile(ref metadata, _)
            | ErrorKind::InvalidVersion(ref metadata, _)
            | ErrorKind::UnknownLayoutQualifier(ref metadata, _)
            | ErrorKind::UnknownField(ref metadata, _) => Some(metadata),
            ErrorKind::InvalidToken(ref token) => Some(&token.meta),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ErrorKind::EndOfFile => write!(f, "Unexpected end of file"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::InvalidProfile(_, ref val) => {
                write!(f, "Invalid profile {}", val)
            }
            ErrorKind::InvalidToken(ref token) => write!(f, "Invalid Token {:?}", token),
            ErrorKind::InvalidVersion(_, ref val) => {
                write!(f, "Invalid version {}", val)
            }
            ErrorKind::ParserFail => write!(f, "Parser failed"),
            ErrorKind::ParserStackOverflow => write!(f, "Parser stack overflow"),
            ErrorKind::NotImplemented(ref msg) => write!(f, "Not implemented: {}", msg),
            ErrorKind::UnknownVariable(_, ref val) => {
                write!(f, "Unknown variable {}", val)
            }
            ErrorKind::UnknownField(_, ref val) => {
                write!(f, "Unknown field {}", val)
            }
            ErrorKind::UnknownLayoutQualifier(_, ref val) => {
                write!(f, "Unknown layout qualifier name {}", val)
            }
            #[cfg(feature = "glsl-validate")]
            ErrorKind::VariableAlreadyDeclared(ref val) => {
                write!(f, "Variable {} already declared in current scope", val)
            }
            ErrorKind::ExpectedConstant => write!(f, "Expected constant"),
            ErrorKind::SemanticError(ref msg) => write!(f, "Semantic error: {}", msg),
            ErrorKind::PreprocessorError(ref val) => write!(f, "Preprocessor error: {}", val),
            ErrorKind::WrongNumberArgs(ref fun, expected, actual) => {
                write!(f, "{} requires {} args, got {}", fun, expected, actual)
            }
        }
    }
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: ErrorKind,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<ErrorKind> for ParseError {
    fn from(kind: ErrorKind) -> Self {
        ParseError { kind }
    }
}

impl std::error::Error for ParseError {}
