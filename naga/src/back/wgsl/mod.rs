/*!
Backend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod polyfill;
mod writer;

use thiserror::Error;

pub use writer::{Writer, WriterFlags};

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FmtError(#[from] std::fmt::Error),
    #[error("{0}")]
    Custom(String),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
    #[error("Unsupported math function: {0:?}")]
    UnsupportedMathFunction(crate::MathFunction),
    #[error("Unsupported relational function: {0:?}")]
    UnsupportedRelationalFunction(crate::RelationalFunction),
}

pub fn write_string(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
    flags: WriterFlags,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), flags);
    w.write(module, info)?;
    let output = w.finish();
    Ok(output)
}

impl crate::AtomicFunction {
    const fn to_wgsl(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Subtract => "Sub",
            Self::And => "And",
            Self::InclusiveOr => "Or",
            Self::ExclusiveOr => "Xor",
            Self::Min => "Min",
            Self::Max => "Max",
            Self::Exchange { compare: None } => "Exchange",
            Self::Exchange { .. } => "CompareExchangeWeak",
        }
    }
}
