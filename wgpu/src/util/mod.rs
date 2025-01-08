//! Utility structures and functions that are built on top of the main `wgpu` API.
//!
//! Nothing in this module is a part of the WebGPU API specification;
//! they are unique to the `wgpu` library.

mod belt;
mod device;
mod encoder;
mod init;

use std::sync::Arc;
use std::{
    borrow::Cow,
    mem::{align_of, size_of},
    ptr::copy_nonoverlapping,
};

pub use belt::StagingBelt;
pub use device::{BufferInitDescriptor, DeviceExt};
pub use encoder::RenderEncoder;
pub use init::*;
pub use wgt::{
    math::*, DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs, TextureDataOrder,
};

use crate::dispatch;

/// Treat the given byte slice as a SPIR-V module.
///
/// # Panic
///
/// This function panics if:
///
/// - Input length isn't multiple of 4
/// - Input is longer than [`usize::MAX`]
/// - Input is empty
/// - SPIR-V magic number is missing from beginning of stream
#[cfg(feature = "spirv")]
pub fn make_spirv(data: &[u8]) -> super::ShaderSource<'_> {
    super::ShaderSource::SpirV(make_spirv_raw(data))
}

/// Version of make_spirv intended for use with [`Device::create_shader_module_spirv`].
/// Returns raw slice instead of ShaderSource.
///
/// [`Device::create_shader_module_spirv`]: crate::Device::create_shader_module_spirv
pub fn make_spirv_raw(data: &[u8]) -> Cow<'_, [u32]> {
    const MAGIC_NUMBER: u32 = 0x0723_0203;
    assert_eq!(
        data.len() % size_of::<u32>(),
        0,
        "data size is not a multiple of 4"
    );
    assert_ne!(data.len(), 0, "data size must be larger than zero");

    // If the data happens to be aligned, directly use the byte array,
    // otherwise copy the byte array in an owned vector and use that instead.
    let mut words = if data.as_ptr().align_offset(align_of::<u32>()) == 0 {
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

    // Before checking if the data starts with the magic, check if it starts
    // with the magic in non-native endianness, own & swap the data if so.
    if words[0] == MAGIC_NUMBER.swap_bytes() {
        for word in Cow::to_mut(&mut words) {
            *word = word.swap_bytes();
        }
    }

    assert_eq!(
        words[0], MAGIC_NUMBER,
        "wrong magic word {:x}. Make sure you are using a binary SPIRV file.",
        words[0]
    );

    words
}

/// CPU accessible buffer used to download data back from the GPU.
pub struct DownloadBuffer {
    _gpu_buffer: Arc<super::Buffer>,
    mapped_range: dispatch::DispatchBufferMappedRange,
}

impl DownloadBuffer {
    /// Asynchronously read the contents of a buffer.
    pub fn read_buffer(
        device: &super::Device,
        queue: &super::Queue,
        buffer: &super::BufferSlice<'_>,
        callback: impl FnOnce(Result<Self, super::BufferAsyncError>) + Send + 'static,
    ) {
        let size = match buffer.size {
            Some(size) => size.into(),
            None => buffer.buffer.map_context.lock().total_size - buffer.offset,
        };

        let download = Arc::new(device.create_buffer(&super::BufferDescriptor {
            size,
            usage: super::BufferUsages::COPY_DST | super::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: None,
        }));

        let mut encoder =
            device.create_command_encoder(&super::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer.buffer, buffer.offset, &download, 0, size);
        let command_buffer: super::CommandBuffer = encoder.finish();
        queue.submit(Some(command_buffer));

        download
            .clone()
            .slice(..)
            .map_async(super::MapMode::Read, move |result| {
                if let Err(e) = result {
                    callback(Err(e));
                    return;
                }

                let mapped_range = download.inner.get_mapped_range(0..size);
                callback(Ok(Self {
                    _gpu_buffer: download,
                    mapped_range,
                }));
            });
    }
}

impl std::ops::Deref for DownloadBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.mapped_range.slice()
    }
}

/// A recommended key for storing [`PipelineCache`]s for the adapter
/// associated with the given [`AdapterInfo`](wgt::AdapterInfo)
/// This key will define a class of adapters for which the same cache
/// might be valid.
///
/// If this returns `None`, the adapter doesn't support [`PipelineCache`].
/// This may be because the API doesn't support application managed caches
/// (such as browser WebGPU), or that `wgpu` hasn't implemented it for
/// that API yet.
///
/// This key could be used as a filename, as seen in the example below.
///
/// # Examples
///
/// ``` no_run
/// # use std::path::PathBuf;
/// # let adapter_info = todo!();
/// let cache_dir: PathBuf = PathBuf::new();
/// let filename = wgpu::util::pipeline_cache_key(&adapter_info);
/// if let Some(filename) = filename {
///     let cache_file = cache_dir.join(&filename);
///     let cache_data = std::fs::read(&cache_file);
///     let pipeline_cache: wgpu::PipelineCache = todo!("Use data (if present) to create a pipeline cache");
///
///     let data = pipeline_cache.get_data();
///     if let Some(data) = data {
///         let temp_file = cache_file.with_extension("temp");
///         std::fs::write(&temp_file, &data)?;
///         std::fs::rename(&temp_file, &cache_file)?;
///     }
/// }
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// [`PipelineCache`]: super::PipelineCache
pub fn pipeline_cache_key(adapter_info: &wgt::AdapterInfo) -> Option<String> {
    match adapter_info.backend {
        wgt::Backend::Vulkan => Some(format!(
            // The vendor/device should uniquely define a driver
            // We/the driver will also later validate that the vendor/device and driver
            // version match, which may lead to clearing an outdated
            // cache for the same device.
            "wgpu_pipeline_cache_vulkan_{}_{}",
            adapter_info.vendor, adapter_info.device
        )),
        _ => None,
    }
}
