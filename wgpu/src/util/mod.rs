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

use crate::{
    dispatch, PipelineLayoutDescriptor, RenderPipelineDescriptor, ShaderSource,
};

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
            None => buffer.buffer.shared.map_context.lock().total_size - buffer.offset,
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

                let mapped_range = download.shared.inner.get_mapped_range(0..size);
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


pub struct TextureBlitter {
    pipeline: crate::RenderPipeline,
    bind_group_layout: crate::BindGroupLayout,
    sampler: crate::Sampler,
}

impl TextureBlitter {
    pub fn new(
        device: crate::Device,
        format: crate::TextureFormat,
        sample_type: crate::FilterMode,
    ) -> Self {
        let sampler = device.create_sampler(&crate::SamplerDescriptor {
            label: Some("wgpu::util::TextureBlitter::sampler"),
            address_mode_u: crate::AddressMode::ClampToEdge,
            address_mode_v: crate::AddressMode::ClampToEdge,
            address_mode_w: crate::AddressMode::ClampToEdge,
            mag_filter: sample_type,
            ..Default::default()
        });

        let bind_group_layout =
            device.create_bind_group_layout(&crate::BindGroupLayoutDescriptor {
                label: Some("wgpu::util::TextureBlitter::bind_group_layout"),
                entries: &[
                    crate::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: crate::ShaderStages::FRAGMENT,
                        ty: crate::BindingType::Texture {
                            sample_type: crate::TextureSampleType::Float { filterable: false },
                            view_dimension: crate::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    crate::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: crate::ShaderStages::FRAGMENT,
                        ty: crate::BindingType::Sampler(crate::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("wgpu::util::TextureBlitter::pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(crate::ShaderModuleDescriptor {
            label: Some("wgpu::util::TextureBlitter::shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(
                r#"
                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) tex_coords: vec2<f32>,
                }

                @vertex
                fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
                    var out: VertexOutput;

                    out.tex_coords = vec2<f32>(
                        f32(vi  << 1u),
                        f32(vi & u2),
                    );

                    out.position = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);

                   // Invert y so the texture is not upside down
                   out.tex_coords.y = 1.0 - out.tex_coords.y;
                   return out;
                  }

                 @group(0) @binding(0)
                 var texture: texture_2d<f32>;
                 @group(0) @binding(0)
                 var texture_sampler: Sampler;

                 @fragment
                 fn fs_main(vs: VertexOutput) -> @location(0) vec4<f32> {
                    return textureSample(texture, texture_sampler, vs.uv);
                 }
                "#,
            )),
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("wgpu::uti::TextureBlitter::pipeline"),
            layout: Some(&pipeline_layout),
            vertex: crate::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: crate::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: crate::PrimitiveState {
                topology: crate::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: crate::FrontFace::Ccw,
                cull_mode: Some(crate::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgt::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: crate::MultisampleState::default(),
            fragment: Some(crate::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: crate::PipelineCompilationOptions::default(),
                targets: &[Some(crate::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: crate::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    pub fn copy(
        &self,
        device: &crate::Device,
        encoder: &mut crate::CommandEncoder,
        target: &crate::TextureView,
        source: &crate::TextureView,
    ) {
        let bind_group = device.create_bind_group(&crate::BindGroupDescriptor {
            label: Some("wgpu::util::TextureBlitter::bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                crate::BindGroupEntry {
                    binding: 0,
                    resource: crate::BindingResource::TextureView(source),
                },
                crate::BindGroupEntry {
                    binding: 1,
                    resource: crate::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&crate::RenderPassDescriptor {
            label: Some("wgpu::util::TextureBlitter::pass"),
            color_attachments: &[Some(crate::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgt::Operations {
                    load: crate::LoadOp::Load,
                    store: crate::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
