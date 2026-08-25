//! Shared types for WebGPU errors. See also:
//! <https://gpuweb.github.io/gpuweb/#errors-and-debugging>

use alloc::boxed::Box;
use alloc::string::String;
use core::{error, fmt};

/// A classification of WebGPU error for implementers of the WebGPU API to use in their own error
/// layer(s).
///
/// Strongly correlates to the [`GPUError`] and [`GPUErrorFilter`] types in the WebGPU API, with an
/// additional [`Self::DeviceLost`] variant.
///
/// [`GPUError`]: https://gpuweb.github.io/gpuweb/#gpuerror
/// [`GPUErrorFilter`]: https://gpuweb.github.io/gpuweb/#enumdef-gpuerrorfilter
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ErrorType {
    /// A [`GPUInternalError`].
    ///
    /// [`GPUInternalError`]: https://gpuweb.github.io/gpuweb/#gpuinternalerror
    Internal,
    /// A [`GPUOutOfMemoryError`].
    ///
    /// [`GPUOutOfMemoryError`]: https://gpuweb.github.io/gpuweb/#gpuoutofmemoryerror
    OutOfMemory,
    /// A [`GPUValidationError`].
    ///
    /// [`GPUValidationError`]: https://gpuweb.github.io/gpuweb/#gpuvalidationerror
    Validation,
    /// Indicates that device loss occurred. In JavaScript, this means the [`GPUDevice.lost`]
    /// property should be `resolve`d.
    ///
    /// [`GPUDevice.lost`]: https://www.w3.org/TR/webgpu/#dom-gpudevice-lost
    DeviceLost,
}

/// A trait for querying the [`ErrorType`] classification of an error.
///
/// This is intended to be used as a convenience by implementations of WebGPU to classify errors
/// returned by [`wgpu_core`](crate).
pub trait WebGpuError: error::Error + 'static {
    /// Determine the classification of this error as a WebGPU [`ErrorType`].
    fn webgpu_error_type(&self) -> ErrorType;
}

/// The callback of [`uncaptured_error`](https://gpuweb.github.io/gpuweb/#eventdef-gpudevice-uncapturederror)
///
/// It must be a function with this signature.
pub trait UncapturedErrorHandler: Fn(Error) + Send + Sync + 'static {}
impl<T> UncapturedErrorHandler for T where T: Fn(Error) + Send + Sync + 'static {}

/// Kinds of [`Error`]s a [`push_error_scope`](https://gpuweb.github.io/gpuweb/#dom-gpudevice-pusherrorscope) may be configured to catch.
///
/// Corresponds to the [`GPUErrorFilter`] type in the WebGPU API.
///
/// [`GPUErrorFilter`]: https://gpuweb.github.io/gpuweb/#enumdef-gpuerrorfilter
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum ErrorFilter {
    /// Catch only out-of-memory errors.
    OutOfMemory,
    /// Catch only validation errors.
    Validation,
    /// Catch only internal errors.
    Internal,
}
static_assertions::assert_impl_all!(ErrorFilter: Send, Sync);

/// Lower level source of the error.
///
/// `Send + Sync` varies depending on configuration.
#[cfg(any(
    not(target_family = "wasm"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
#[cfg_attr(docsrs, doc(cfg(all())))]
pub type ErrorSource = Box<dyn error::Error + Send + Sync + 'static>;
/// Lower level source of the error.
///
/// `Send + Sync` varies depending on configuration.
#[cfg(not(any(
    not(target_family = "wasm"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
)))]
#[cfg_attr(docsrs, doc(cfg(all())))]
pub type ErrorSource = Box<dyn error::Error + 'static>;

/// Errors resulting from usage of GPU APIs.
#[derive(Debug)]
pub enum Error {
    /// Out of memory.
    OutOfMemory {
        /// Lower level source of the error.
        source: ErrorSource,
    },
    /// Validation error, signifying a bug in code or data provided to `wgpu`.
    Validation {
        /// Lower level source of the error.
        source: ErrorSource,
        /// Description of the validation error.
        description: String,
    },
    /// Internal error. Used for signalling any failures not explicitly expected by WebGPU.
    ///
    /// These could be due to internal implementation or system limits being reached.
    Internal {
        /// Lower level source of the error.
        source: ErrorSource,
        /// Description of the internal GPU error.
        description: String,
    },
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::OutOfMemory { source } => Some(source.as_ref()),
            Error::Validation { source, .. } => Some(source.as_ref()),
            Error::Internal { source, .. } => Some(source.as_ref()),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory { .. } => f.write_str("Out of Memory"),
            Error::Validation { description, .. } => f.write_str(description),
            Error::Internal { description, .. } => f.write_str(description),
        }
    }
}

impl WebGpuError for Error {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Error::OutOfMemory { .. } => ErrorType::OutOfMemory,
            Error::Validation { .. } => ErrorType::Validation,
            Error::Internal { .. } => ErrorType::Internal,
        }
    }
}
