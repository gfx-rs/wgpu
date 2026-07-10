//! Shared helpers for the resource-table GPU test suite.
//!
//! These build on the smoke-test conventions in [`super::compute`]: tiny 1x1
//! textures with distinct known red values, uploaded on the queue timeline
//! (`write_texture`, i.e. outside any later sampling command buffer, since
//! same-CB write-then-table-sample is a documented M0 limitation), plus a
//! standard "sample a table slot and read back the decoded red byte" compute
//! flow.

use wgpu::*;
use wgpu_test::{FailureCase, TestParameters, TestingContext};

/// Red channel byte stored in (and expected back from) the `k`-th texture.
///
/// Distinct, non-zero, and small so exact assertions are simple: 10, 20, 30, …
pub const fn texture_red(k: usize) -> u8 {
    ((k + 1) * 10) as u8
}

/// Standard parameters for a resource-table GPU test:
///
/// * both experimental resource-table features (so unsupported adapters
///   auto-skip rather than fail),
/// * downlevel-default limits (the WebGL2 defaults forbid storage buffers in the
///   compute stage, which the readback path needs), and
/// * Vulkan-only (M0 has no other backend implementation yet).
pub fn table_params() -> TestParameters {
    TestParameters::default()
        .features(
            Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
                | Features::EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED,
        )
        .limits(Limits::downlevel_defaults())
        .skip(FailureCase::backend(!Backends::VULKAN))
}

/// Create a 1x1 `Rgba8Unorm` sampled texture whose red channel is `red`,
/// uploaded via `queue.write_texture` (queue timeline).
pub fn make_red_texture(ctx: &TestingContext, red: u8) -> (Texture, TextureView) {
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("resource-table member texture"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let texel: [u8; 4] = [red, 0, 0, 255];
    ctx.queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &texel,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

/// Map a `MAP_READ | COPY_DST` buffer and return its contents interpreted as
/// `u32`s. Polls the device to completion so the mapping resolves regardless of
/// whether the caller already polled.
pub async fn read_u32s(ctx: &TestingContext, buffer: &Buffer) -> Vec<u32> {
    buffer.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice::<u8, u32>(&data).to_vec()
}

/// The standard sampling compute shader used by [`Sampler`].
///
/// For each invocation `i` it reads `getResource<texture_2d<f32>>(indices[i])`,
/// `textureLoad`s texel (0, 0), and writes `round(r * 255)` back to
/// `output[i]`. The per-invocation, buffer-derived slot index is non-uniform,
/// exercising the dynamic-indexing path.
pub const SAMPLING_SHADER: &str = r#"
enable resource_table;

@group(0) @binding(0)
var<storage, read> indices: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let slot = indices[i];
    let tex = getResource<texture_2d<f32>>(slot);
    let texel = textureLoad(tex, vec2<i32>(0, 0), 0);
    output[i] = u32(round(texel.r * 255.0));
}
"#;

/// A reusable single-pass sampler: one compute pass that reads a resource table
/// at the given per-invocation slot indices and writes the decoded red bytes to
/// a readback buffer.
///
/// The pipeline layout declares one bind group (group 0 = indices + output
/// storage buffers), so the table binds at set index 1.
pub struct Sampler {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    output_buffer: Buffer,
    readback_buffer: Buffer,
    size: BufferAddress,
}

impl Sampler {
    /// Build the standard sampler for `indices`.
    pub fn new(ctx: &TestingContext, indices: &[u32]) -> Self {
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("resource-table sampling shader"),
            source: ShaderSource::Wgsl(SAMPLING_SHADER.into()),
        });

        let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
        let size = index_bytes.len() as BufferAddress;

        let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("resource-table indices"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&index_buffer, 0, &index_bytes);

        let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("resource-table output"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("resource-table readback"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("resource-table sampling bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("resource-table sampling bind group"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("resource-table sampling pipeline layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
                uses_resource_table: true,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("resource-table sampling pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group,
            output_buffer,
            readback_buffer,
            size,
        }
    }

    /// Record the sampling pass (binding `table`) plus the output→readback copy
    /// into a fresh encoder and submit it. Does not poll; call [`read`](Self::read)
    /// afterwards to retrieve the results.
    pub fn submit(&self, ctx: &TestingContext, table: &ResourceTable) -> SubmissionIndex {
        let count = (self.size / 4) as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("resource-table sampling pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.set_resource_table(Some(table));
            cpass.dispatch_workgroups(count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.readback_buffer, 0, self.size);
        ctx.queue.submit(Some(encoder.finish()))
    }

    /// Map the readback buffer and return the decoded red bytes. Polls to
    /// completion internally.
    pub async fn read(&self, ctx: &TestingContext) -> Vec<u32> {
        read_u32s(ctx, &self.readback_buffer).await
    }
}

/// Build a sampler, run it against `table` for `indices`, poll, and return the
/// read-back decoded red values. For the common "just verify the result" case.
pub async fn run_sampling(
    ctx: &TestingContext,
    table: &ResourceTable,
    indices: &[u32],
) -> Vec<u32> {
    let sampler = Sampler::new(ctx, indices);
    sampler.submit(ctx, table);
    sampler.read(ctx).await
}
