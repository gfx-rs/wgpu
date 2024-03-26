use prettyplease;
pub mod writer;
pub use writer::{Writer, WriterError, WriterFlags};

#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub enum Target {
    Cpu,
    #[default]
    Gpu,
}

pub fn write_string(
    module: &crate::Module,
    info: &crate::valid::ModuleInfo,
    target: Target,
    flags: WriterFlags,
) -> Result<String, WriterError> {
    let mut w = Writer::new(target, flags);
    let file = w.write_module(module, info)?;
    Ok(prettyplease::unparse(&file))
}
