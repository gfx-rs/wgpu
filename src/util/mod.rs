//! Utility structures and functions.

mod belt;

use std::{
    borrow::Cow,
    mem::{align_of, size_of},
    ptr::copy_nonoverlapping,
};

pub use belt::StagingBelt;
use std::sync::Arc;

/// Treat the given byte slice as a SPIR-V module.
///
/// # Panic
///
/// This function panics if:
///
/// - Input length isn't multiple of 4
/// - Input is longer than [`usize::max_value`]
/// - SPIR-V magic number is missing from beginning of stream
pub fn make_spirv<'a>(data: &'a [u8]) -> super::ShaderModuleSource<'a> {
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
    super::ShaderModuleSource::SpirV(words)
}

/// Utility methods not meant to be in the main API.
pub trait DeviceExt {
    /// Creates a [`Buffer`] with data to initialize it.
    fn create_buffer_init(&self, desc: &BufferInitDescriptor) -> crate::Buffer;
}

impl DeviceExt for crate::Device {
    fn create_buffer_init(&self, descriptor: &BufferInitDescriptor<'_>) -> crate::Buffer {
        let unpadded_size = descriptor.contents.len() as crate::BufferAddress;
        let padding = crate::COPY_BUFFER_ALIGNMENT - unpadded_size % crate::COPY_BUFFER_ALIGNMENT;
        let padded_size = padding + unpadded_size;

        let wgt_descriptor = crate::BufferDescriptor {
            label: descriptor.label,
            size: padded_size,
            usage: descriptor.usage,
            mapped_at_creation: true,
        };

        let mut map_context = crate::MapContext::new(padded_size);

        map_context.initial_range = 0..padded_size;

        #[cfg(target_arch = "wasm32")]
        let buffer = crate::Buffer {
            context: Arc::clone(&self.context),
            id: crate::backend::Context::create_buffer_init_polyfill(&self.id, &wgt_descriptor, descriptor.contents),
            map_context: parking_lot::Mutex::new(map_context),
            usage: descriptor.usage,
        };
        #[cfg(not(target_arch = "wasm32"))]
        let buffer = {
            let buffer = crate::Buffer {
                context: Arc::clone(&self.context),
                id: crate::Context::device_create_buffer(&*self.context, &self.id, &wgt_descriptor),
                map_context: parking_lot::Mutex::new(map_context),
                usage: descriptor.usage,
            };
    
            let range =
                crate::Context::buffer_get_mapped_range_mut(&*self.context, &buffer.id, 0..padded_size);
            range[0..unpadded_size as usize].copy_from_slice(descriptor.contents);
            for i in unpadded_size..padded_size {
                range[i as usize] = 0;
            }
    
            buffer.unmap();
            buffer
        };
        buffer
    }
}

/// Describes a [`Buffer`] when allocating.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferInitDescriptor<'a> {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: Option<&'a str>,
    /// Contents of a buffer on creation.
    pub contents: &'a [u8],
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    pub usage: crate::BufferUsage,
}
