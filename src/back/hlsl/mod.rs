mod keywords;
mod writer;

use std::fmt::Error as FmtError;
use thiserror::Error;

pub use writer::Writer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IoError(#[from] FmtError),
}

pub fn write_string(module: &crate::Module) -> Result<String, Error> {
    let mut w = Writer::new(String::new());
    w.write(module)?;
    let output = w.finish();
    Ok(output)
}
