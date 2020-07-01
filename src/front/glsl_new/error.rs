use super::parser::Token;
use std::{fmt, io};

#[derive(Debug)]
pub enum ErrorKind {
    InvalidInput,
    IoError(io::Error),
    InvalidToken(Token),
    EndOfFile,
    ParserFail,
    ParserStackOverflow,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::IoError(error) => write!(f, "IO Error {}", error),
            ErrorKind::InvalidToken(token) => write!(f, "Invalid Token {:?}", token),
            ErrorKind::EndOfFile => write!(f, "Unexpected end of file"),
            ErrorKind::ParserFail => write!(f, "Parser failed"),
            ErrorKind::ParserStackOverflow => write!(f, "Parser stack overflow"),
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
