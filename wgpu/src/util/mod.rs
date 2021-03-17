//! Utility structures and functions.

mod belt;
mod device;
mod encoder;

use std::{
    borrow::Cow,
    future::Future,
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
pub fn make_spirv(data: &[u8]) -> super::ShaderSource {
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

/// CPU accessible buffer used to download data back from the GPU.
pub struct DownloadBuffer(super::Buffer, super::BufferMappedRange);

impl DownloadBuffer {
    /// Asynchronously read the contents of a buffer.
    pub fn read_buffer(
        device: &super::Device,
        queue: &super::Queue,
        buffer: &super::BufferSlice,
    ) -> impl Future<Output = Result<Self, super::BufferAsyncError>> + Send {
        let size = match buffer.size {
            Some(size) => size.into(),
            None => buffer.buffer.map_context.lock().total_size - buffer.offset,
        };

        let download = device.create_buffer(&super::BufferDescriptor {
            size,
            usage: super::BufferUsage::COPY_DST | super::BufferUsage::MAP_READ,
            mapped_at_creation: false,
            label: None,
        });

        let mut encoder =
            device.create_command_encoder(&super::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer.buffer, buffer.offset, &download, 0, size);
        let command_buffer: super::CommandBuffer = encoder.finish();
        queue.submit(Some(command_buffer));

        let fut = download.slice(..).map_async(super::MapMode::Read);
        async move {
            fut.await?;
            let mapped_range =
                super::Context::buffer_get_mapped_range(&*download.context, &download.id, 0..size);
            Ok(Self(download, mapped_range))
        }
    }
}

impl std::ops::Deref for DownloadBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        super::BufferMappedRangeSlice::slice(&self.1)
    }
}
