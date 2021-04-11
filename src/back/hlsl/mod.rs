mod keywords;
mod writer;

use std::io::Error as IoError;
use std::string::FromUtf8Error;
use thiserror::Error;

pub use writer::Writer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IoError(#[from] IoError),
    #[error(transparent)]
    Utf8(#[from] FromUtf8Error),
}

pub fn write_string(module: &crate::Module) -> Result<String, Error> {
    let mut w = Writer::new(Vec::new());
    w.write(module)?;
    let output = String::from_utf8(w.finish())?;
    Ok(output)
}
