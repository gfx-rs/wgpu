use super::parser::Token;
use super::token::TokenMetadata;
use std::{fmt, io};

#[derive(Debug)]
pub enum ErrorKind {
    EndOfFile,
    InvalidInput,
    InvalidProfile(TokenMetadata, String),
    InvalidToken(Token),
    InvalidVersion(TokenMetadata, i64),
    IoError(io::Error),
    ParserFail,
    ParserStackOverflow,
    NotImplemented(&'static str),
    UnknownVariable(TokenMetadata, String),
    UnknownField(TokenMetadata, String),
    #[cfg(feature = "glsl-validate")]
    VariableAlreadyDeclared(String),
    #[cfg(feature = "glsl-validate")]
    VariableNotAvailable(String),
    ExpectedConstant,
    SemanticError(&'static str),
    PreprocessorError(String),
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::EndOfFile => write!(f, "Unexpected end of file"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::InvalidProfile(meta, val) => {
                write!(f, "Invalid profile {} at {:?}", val, meta)
            }
            ErrorKind::InvalidToken(token) => write!(f, "Invalid Token {:?}", token),
            ErrorKind::InvalidVersion(meta, val) => {
                write!(f, "Invalid version {} at {:?}", val, meta)
            }
            ErrorKind::IoError(error) => write!(f, "IO Error {}", error),
            ErrorKind::ParserFail => write!(f, "Parser failed"),
            ErrorKind::ParserStackOverflow => write!(f, "Parser stack overflow"),
            ErrorKind::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            ErrorKind::UnknownVariable(meta, val) => {
                write!(f, "Unknown variable {} at {:?}", val, meta)
            }
            ErrorKind::UnknownField(meta, val) => write!(f, "Unknown field {} at {:?}", val, meta),
            #[cfg(feature = "glsl-validate")]
            ErrorKind::VariableAlreadyDeclared(val) => {
                write!(f, "Variable {} already decalred in current scope", val)
            }
            #[cfg(feature = "glsl-validate")]
            ErrorKind::VariableNotAvailable(val) => {
                write!(f, "Variable {} not available in this stage", val)
            }
            ErrorKind::ExpectedConstant => write!(f, "Expected constant"),
            ErrorKind::SemanticError(msg) => write!(f, "Semantic error: {}", msg),
            ErrorKind::PreprocessorError(val) => write!(f, "Preprocessor error: {}", val),
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

impl From<io::Error> for ParseError {
    fn from(error: io::Error) -> Self {
        ParseError {
            kind: ErrorKind::IoError(error),
        }
    }
}

impl From<ErrorKind> for ParseError {
    fn from(kind: ErrorKind) -> Self {
        ParseError { kind }
    }
}

impl std::error::Error for ParseError {}
