mod keywords;
mod writer;

use thiserror::Error;

pub use writer::Writer;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FmtError(#[from] std::fmt::Error),
    #[error("{0}")]
    Custom(String),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
}

pub fn write_string(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new());
    w.write(module, info)?;
    let output = w.finish();
    Ok(output)
}
