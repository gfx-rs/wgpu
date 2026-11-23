/*!
Backend for [WGSL][wgsl] (WebGPU Shading Language).

[wgsl]: https://gpuweb.github.io/gpuweb/wgsl.html
*/

mod polyfill;
mod writer;

use alloc::format;
use alloc::string::String;

use thiserror::Error;

pub use writer::{Writer, WriterFlags};

use crate::common::wgsl;

#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    FmtError(#[from] core::fmt::Error),
    #[error("{0}")]
    Custom(String),
    #[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
    #[error("Unsupported relational function: {0:?}")]
    UnsupportedRelationalFunction(crate::RelationalFunction),
    #[error(transparent)]
    Pass(#[from] crate::back::passes::PassError),
    #[error("function `{0}` has pointer parameters or arguments that must be inlined by `naga::inline::process`")]
    NeedsInline(String),
    #[error("Unsupported {kind}: {value}")]
    Unsupported {
        /// What kind of unsupported thing this is: interpolation, builtin, etc.
        kind: &'static str,

        /// The debug form of the Naga IR value that this backend can't express.
        value: String,
    },
}

impl Error {
    /// Produce an [`Unsupported`] error for `value`.
    ///
    /// [`Unsupported`]: Error::Unsupported
    fn unsupported<T: core::fmt::Debug>(kind: &'static str, value: T) -> Error {
        Error::Unsupported {
            kind,
            value: format!("{value:?}"),
        }
    }
}

trait ToWgslIfImplemented {
    fn to_wgsl_if_implemented(self) -> Result<&'static str, Error>;
}

impl<T> ToWgslIfImplemented for T
where
    T: wgsl::TryToWgsl + core::fmt::Debug + Copy,
{
    fn to_wgsl_if_implemented(self) -> Result<&'static str, Error> {
        self.try_to_wgsl()
            .ok_or_else(|| Error::unsupported(T::DESCRIPTION, self))
    }
}

/// Configuration used for the [`Writer`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct Options {
    /// Configuration flags for the writer.
    pub flags: WriterFlags,
    /// Whether the consumer of the output supports the WGSL
    /// `unrestricted_pointer_parameters` language extension.
    ///
    /// If `true`, the output may pass pointers in the `storage`, `uniform`,
    /// `workgroup`, `immediate` and `task_payload` address spaces to
    /// functions, and pointers to part of a variable like `&a[i]`.
    ///
    /// If `false`, the backend inlines the calls that need the extension.
    pub unrestricted_pointer_parameters: bool,
    /// Options for the IR passes the backend runs before writing the module.
    pub passes: crate::back::PassOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            flags: WriterFlags::empty(),
            unrestricted_pointer_parameters: true,
            passes: Default::default(),
        }
    }
}

pub fn write_string<'m>(
    module: impl Into<alloc::borrow::Cow<'m, crate::Module>>,
    options: &Options,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), options);
    w.write(module)?;
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

pub const fn supported_capabilities() -> crate::valid::Capabilities {
    // WGSL regurgitation supports almost everything, though browser webgpu can't parse most of these.
    use crate::valid::Capabilities as Caps;
    Caps::all()
}
