use crate::proc::ResolveError;
use std::{
    fmt::{self, Error as FmtError},
    io::Error as IoError,
};

pub type BackendResult = std::result::Result<(), Error>;

#[derive(Debug)]
pub enum Error {
    FormatError(FmtError),
    IoError(IoError),
    Type(ResolveError),
    Custom(String),
}

impl From<FmtError> for Error {
    fn from(err: FmtError) -> Self {
        Error::FormatError(err)
    }
}

impl From<IoError> for Error {
    fn from(err: IoError) -> Self {
        Error::IoError(err)
    }
}

impl From<ResolveError> for Error {
    fn from(err: ResolveError) -> Self {
        Error::Type(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FormatError(err) => write!(f, "Formatting error {}", err),
            Error::IoError(err) => write!(f, "Io error: {}", err),
            Error::Type(err) => write!(f, "Type error: {:?}", err),
            Error::Custom(err) => write!(f, "{}", err),
        }
    }
}
