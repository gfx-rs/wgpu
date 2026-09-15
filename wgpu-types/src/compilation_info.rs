use alloc::string::String;
use alloc::vec::Vec;

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

/// Compilation information for a shader module.
///
/// Corresponds to [WebGPU `GPUCompilationInfo`](https://gpuweb.github.io/gpuweb/#gpucompilationinfo).
/// The source locations use bytes, and index a UTF-8 or UTF-16 encoded string.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompilationInfo<SL = SourceLocation> {
    /// The messages from the shader compilation process.
    pub messages: Vec<CompilationMessage<SL>>,
}

/// A single message from the shader compilation process.
///
/// Roughly corresponds to [`GPUCompilationMessage`](https://www.w3.org/TR/webgpu/#gpucompilationmessage),
/// except that the location may use UTF-8 for all positions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompilationMessage<SL = SourceLocation> {
    /// The text of the message.
    pub message: String,
    /// The type of the message.
    pub message_type: CompilationMessageType,
    /// Where in the source code the message points at.
    pub location: Option<SL>,
}

/// The type of a compilation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(u8)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

/// A human-readable representation for a span, tailored for text source.
///
/// Corresponds to the positional members of [`GPUCompilationMessage`][gcm] from
/// the WebGPU specification.
///
/// Instead of using UTF-8 code units, this uses UTF-16 code units,
/// which is what the WebGPU specification uses.
///
/// [gcm]: https://www.w3.org/TR/webgpu/#gpucompilationmessage
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Utf16SourceLocation {
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

impl SourceLocation {
    /// Converts a `SourceLocation` in UTF-8 code units to UTF-16 code units.
    pub fn to_utf16(&self, source: &str) -> Utf16SourceLocation {
        let len_utf16 = |s: &str| s.chars().map(|c| c.len_utf16() as u32).sum::<u32>();
        let start = self.offset as usize;
        let end = start + self.length as usize;
        let utf16_offset = len_utf16(&source[..start]);
        let utf16_length = len_utf16(&source[start..end]);

        let line_start = source[..start].rfind('\n').map_or(0, |pos| pos + 1);
        let utf16_line_position = len_utf16(&source[line_start..start]) + 1;

        Utf16SourceLocation {
            line_number: self.line_number,
            line_position: utf16_line_position,
            offset: utf16_offset,
            length: utf16_length,
        }
    }
}

impl Utf16SourceLocation {
    /// Converts a `SourceLocation` in UTF-16 code units to UTF-8 code units.
    pub fn to_utf8(&self, source: &str) -> SourceLocation {
        fn map_utf16_to_utf8_offset(utf16_offset: u32, text: &str) -> u32 {
            let mut utf16_i = 0;
            for (utf8_index, c) in text.char_indices() {
                if utf16_i >= utf16_offset {
                    return utf8_index as u32;
                }
                utf16_i += c.len_utf16() as u32;
            }
            if utf16_i >= utf16_offset {
                text.len() as u32
            } else {
                log::error!("UTF16 offset {utf16_offset} is out of bounds for string {text}");
                u32::MAX
            }
        }
        let utf8_offset = map_utf16_to_utf8_offset(self.offset, source);
        let utf8_length = map_utf16_to_utf8_offset(self.length, &source[utf8_offset as usize..]);

        let prefix = &source[..utf8_offset as usize];
        let line_start = prefix.rfind('\n').map(|pos| pos + 1).unwrap_or(0) as u32;
        let utf8_line_position = utf8_offset - line_start + 1; // Counting UTF-8 bytes

        SourceLocation {
            line_number: self.line_number,
            line_position: utf8_line_position,
            offset: utf8_offset,
            length: utf8_length,
        }
    }
}
