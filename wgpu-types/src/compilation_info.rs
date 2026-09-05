use alloc::string::String;
use alloc::vec::Vec;

/// Compilation information for a shader module.
///
/// Corresponds to [WebGPU `GPUCompilationInfo`](https://gpuweb.github.io/gpuweb/#gpucompilationinfo).
/// The source locations use bytes, and index a UTF-8 encoded string.
#[derive(Debug, Clone, Default)]
pub struct CompilationInfo {
    /// The messages from the shader compilation process.
    pub messages: Vec<CompilationMessage>,
}

/// A single message from the shader compilation process.
///
/// Roughly corresponds to [`GPUCompilationMessage`](https://www.w3.org/TR/webgpu/#gpucompilationmessage),
/// except that the location uses UTF-8 for all positions.
#[derive(Debug, Clone)]
pub struct CompilationMessage {
    /// The text of the message.
    pub message: String,
    /// The type of the message.
    pub message_type: CompilationMessageType,
    /// Where in the source code the message points at.
    pub location: Option<SourceLocation>,
}

/// The type of a compilation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationMessageType {
    /// An error message.
    Error,
    /// A warning message.
    Warning,
    /// An informational message.
    Info,
}

/// A human-readable representation for a span, tailored for text source.
///
/// Roughly corresponds to the positional members of [`GPUCompilationMessage`][gcm] from
/// the WebGPU specification, except
/// - `offset` and `length` are in bytes (UTF-8 code units), instead of UTF-16 code units.
/// - `line_position` is in bytes (UTF-8 code units), and is usually not directly intended for humans.
///
/// [gcm]: https://www.w3.org/TR/webgpu/#gpucompilationmessage
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    /// 1-based line number.
    pub line_number: u32,
    /// 1-based column in code units (in bytes) of the start of the span.
    /// Remember to convert accordingly when displaying to the user.
    pub line_position: u32,
    /// 0-based Offset in code units (in bytes) of the start of the span.
    pub offset: u32,
    /// Length in code units (in bytes) of the span.
    pub length: u32,
}
