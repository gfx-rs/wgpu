//! Utility structures and functions.

mod belt;

#[cfg(all(not(target_arch = "wasm32"), feature = "subscriber"))]
pub use wgc::logging::subscriber::{initialize_default_subscriber, ChromeTracingLayer};

pub use belt::StagingBelt;

/// Wrapper aligning contents to at least 4.
#[repr(align(4))]
pub struct WordAligned<Bytes: ?Sized>(pub Bytes);

/// Treat the given byte slice as a SPIR-V module.
///
/// # Panic
///
/// This function panics if:
///
/// - Input isn't aligned to 4 bytes
/// - Input length isn't multiple of 4
/// - Input is longer than [`usize::max_value`]
/// - SPIR-V magic number is missing from beginning of stream
pub fn make_spirv<'a>(data: &'a [u8]) -> super::ShaderModuleSource<'a> {
    const MAGIC_NUMBER: u32 = 0x0723_0203;

    let (pre, words, post) = unsafe { data.align_to::<u32>() };
    assert_eq!(pre, &[], "data offset is not aligned to words!");
    assert_eq!(post, &[], "data size is not aligned to words!");
    assert_eq!(
        words[0], MAGIC_NUMBER,
        "wrong magic word {:x}. Make sure you are using a binary SPIRV file.",
        words[0]
    );
    super::ShaderModuleSource::SpirV(words)
}
