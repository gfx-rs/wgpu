//! Utility structures and functions.

mod belt;
mod device;
mod encoder;

use std::{
    borrow::Cow,
    mem::{align_of, size_of},
    ptr::copy_nonoverlapping,
};

pub use belt::StagingBelt;
pub use device::{BufferInitDescriptor, DeviceExt};
pub use encoder::RenderEncoder;

/// Treat the given byte slice as a SPIR-V module.
///
/// # Panic
///
/// This function panics if:
///
/// - Input length isn't multiple of 4
/// - Input is longer than [`usize::max_value`]
/// - SPIR-V magic number is missing from beginning of stream
pub fn make_spirv<'a>(data: &'a [u8]) -> super::ShaderSource<'a> {
    const MAGIC_NUMBER: u32 = 0x0723_0203;

    assert_eq!(
        data.len() % size_of::<u32>(),
        0,
        "data size is not a multiple of 4"
    );

    //If the data happens to be aligned, directly use the byte array,
    // otherwise copy the byte array in an owned vector and use that instead.
    let words = if data.as_ptr().align_offset(align_of::<u32>()) == 0 {
        let (pre, words, post) = unsafe { data.align_to::<u32>() };
        debug_assert!(pre.is_empty());
        debug_assert!(post.is_empty());
        Cow::from(words)
    } else {
        let mut words = vec![0u32; data.len() / size_of::<u32>()];
        unsafe {
            copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr() as *mut u8, data.len());
        }
        Cow::from(words)
    };

    assert_eq!(
        words[0], MAGIC_NUMBER,
        "wrong magic word {:x}. Make sure you are using a binary SPIRV file.",
        words[0]
    );
    super::ShaderSource::SpirV(words)
}

struct ReadBuffer(super::BufferMappedRange);
impl std::ops::Deref for ReadBuffer{
  type Target = [u8];
  fn deref(&self) -> &[u8] {
    super::BufferMappedRangeSlice::slice(&self.0)
  }
}
