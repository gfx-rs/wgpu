//! Usage-conflict coverage (work item 0.9).
//!
//! Under M0's strict (v0) semantics (D3 in `plans/resource-table.md`), a texture
//! bound in a resource table may not be *written* in a scope of any submission
//! that binds the table: the spec would hide the conflicting slot per usage
//! scope, but M0 has no visibility mechanism yet, so it rejects at submit rather
//! than silently diverge. These tests confirm on a real GPU that
//!
//! * writing a table member via a storage binding in the pass that binds the
//!   table is rejected (compute-pass write-accumulation path),
//! * writing a table member as a render-pass color target is rejected
//!   (render-pass write-accumulation path), and
//! * the legitimate write-then-sample flow *across submissions* is not rejected
//!   and produces the correct result (the write submission binds no table, so
//!   there is no conflict; the sampling submission's pass-start barrier makes the
//!   prior write visible).

use wgpu::*;
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer};

use super::common::{
    make_red_texture, read_u32s, run_sampling, table_params, texture_red, Sampler,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_MEMBER_STORAGE_WRITTEN_REJECTED,
        RESOURCE_TABLE_MEMBER_COLOR_TARGET_REJECTED,
        RESOURCE_TABLE_MEMBER_WRITTEN_THEN_BINDFUL_SAMPLED_REJECTED,
        RESOURCE_TABLE_MEMBER_COPY_SRC_SAME_CB_REJECTED,
        RESOURCE_TABLE_MEMBER_COPY_DST_SAME_CB_REJECTED,
        RESOURCE_TABLE_MEMBER_BINDFUL_SAMPLED_SAME_CB_OK,
        RESOURCE_TABLE_MEMBER_WRITE_TEXTURE_POPULATE_OK,
        RESOURCE_TABLE_WRITE_THEN_SAMPLE_ACROSS_SUBMISSIONS,
    ]);
}

/// A 1x1 `Rgba8Unorm` texture with the given `usage`, its red channel set to
/// `red` via `queue.write_texture` (queue timeline, i.e. populated outside any
/// later sampling command buffer). `usage` must include `COPY_DST`.
fn red_texture_with_usage(
    ctx: &wgpu_test::TestingContext,
    usage: TextureUsages,
    red: u8,
) -> Texture {
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: Some("conflict member texture"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage,
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
    texture
}

/// A 1x1 `Rgba8Unorm` texture usable as both a sampled resource and a storage
/// image (the two views needed to make it both a table member and a
/// storage-write target).
fn storage_sampled_texture(ctx: &wgpu_test::TestingContext) -> Texture {
    ctx.device.create_texture(&TextureDescriptor {
        label: Some("conflict storage+sampled texture"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

/// Binding a table containing a texture and writing that same texture through a
/// storage binding in the compute pass that binds the table is rejected at
/// submit (v0 semantics, work item 0.9).
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_STORAGE_WRITTEN_REJECTED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;

            let texture = storage_sampled_texture(ctx);
            let sampled_view = texture.create_view(&TextureViewDescriptor::default());
            let storage_view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &sampled_view).expect("bind member");

            // Reads slot 0 (the table member) and writes it back through the
            // storage binding — the exact member-write conflict.
            let shader = r#"
enable resource_table;
@group(0) @binding(0)
var img: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1)
fn main() {
    let t = getResource<texture_2d<f32>>(0u);
    let c = textureLoad(t, vec2<i32>(0, 0), 0);
    textureStore(img, vec2<i32>(0, 0), c);
}
"#;
            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(shader.into()),
            });
            let bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });
            let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&storage_view),
                }],
            });
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            wgpu_test::fail(
                &ctx.device,
                || {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None });
                    {
                        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.set_resource_table(Some(&table));
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    ctx.queue.submit(Some(encoder.finish()));
                },
                Some("resource table"),
            );
        });

/// Binding a table containing a texture and writing that same texture as a
/// render-pass color target (in the pass that binds the table) is rejected at
/// submit (v0 semantics, work item 0.9). An empty render pass suffices: the
/// attachment is a writable usage regardless of draws.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_COLOR_TARGET_REJECTED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;

            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: Some("conflict render+sampled texture"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind member");

            wgpu_test::fail(
                &ctx.device,
                || {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None });
                    {
                        let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: Operations {
                                    load: LoadOp::Clear(Color::BLACK),
                                    store: StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                            // The color target is also a member of this table.
                            resource_table: Some(&table),
                        });
                    }
                    ctx.queue.submit(Some(encoder.finish()));
                },
                Some("resource table"),
            );
        });

/// The reviewer-confirmed layout-safety repro (finding fix). In one compute pass
/// that binds the table, a table member `T` is first written through a bindful
/// storage binding, then *read* through an ordinary sampled bind group in a later
/// dispatch. The command-buffer tracker's end state for `T` is a read
/// (`RESOURCE`), so the original end-state-only conflict check missed the earlier
/// write and accepted the submission — leaving `T` in `GENERAL` while its table
/// descriptor declares `SHADER_READ_ONLY_OPTIMAL` (driver UB). The
/// union-over-start-and-end check now rejects it at submit.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_WRITTEN_THEN_BINDFUL_SAMPLED_REJECTED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;

            let texture = storage_sampled_texture(ctx);
            let sampled_view = texture.create_view(&TextureViewDescriptor::default());
            let storage_view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &sampled_view).expect("bind member");

            // d1: read slot 0 through the table and write `T` back through a
            // bindful storage binding (a bindful write of the member).
            let write_shader = r#"
enable resource_table;
@group(0) @binding(0)
var img: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1)
fn main() {
    let t = getResource<texture_2d<f32>>(0u);
    let c = textureLoad(t, vec2<i32>(0, 0), 0);
    textureStore(img, vec2<i32>(0, 0), c);
}
"#;
            // d2: sample `T` through an ordinary sampled bind group (`RESOURCE`),
            // flipping its tracker end state to a read.
            let sample_shader = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = u32(round(textureLoad(tex, vec2<i32>(0, 0), 0).r * 255.0));
}
"#;
            let write_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(write_shader.into()),
            });
            let sample_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(sample_shader.into()),
            });

            let write_bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });
            let write_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &write_bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&storage_view),
                }],
            });
            let write_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&write_bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let write_pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&write_layout),
                    module: &write_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            let out_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let sample_bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
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
            let sample_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &sample_bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&sampled_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: out_buffer.as_entire_binding(),
                    },
                ],
            });
            let sample_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&sample_bgl)],
                    immediate_size: 0,
                    uses_resource_table: false,
                });
            let sample_pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&sample_layout),
                    module: &sample_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            wgpu_test::fail(
                &ctx.device,
                || {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None });
                    {
                        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_resource_table(Some(&table));
                        // d1: bindful storage write of the member (+ table read).
                        pass.set_pipeline(&write_pipeline);
                        pass.set_bind_group(0, &write_bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                        // d2: bindful sample of the member (end state flips to read).
                        pass.set_pipeline(&sample_pipeline);
                        pass.set_bind_group(0, &sample_bg, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    ctx.queue.submit(Some(encoder.finish()));
                },
                Some("resource table"),
            );
        });

/// Binding a table containing a texture and, in the same command buffer, copying
/// *from* that texture with a top-level `copy_texture_to_buffer` (`COPY_SRC`,
/// which forces `TRANSFER_SRC_OPTIMAL`) is rejected at submit (finding fix). The
/// copy records directly on the command-buffer tracker rather than a pass, so this
/// exercises the top-level-transfer collection path.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_COPY_SRC_SAME_CB_REJECTED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;

            let texture = red_texture_with_usage(
                ctx,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
                texture_red(0),
            );
            let view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind member");

            let copy_dst = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 256,
                usage: BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Samples slot 0 (the member) through the table, so the command buffer
            // both binds the table and (via the top-level copy) uses the member as
            // a copy source.
            let sampler = Sampler::new(ctx, &[0]);

            wgpu_test::fail(
                &ctx.device,
                || {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None });
                    encoder.copy_texture_to_buffer(
                        TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        TexelCopyBufferInfo {
                            buffer: &copy_dst,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: None,
                                rows_per_image: None,
                            },
                        },
                        Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                    sampler.record(&mut encoder, &table);
                    ctx.queue.submit(Some(encoder.finish()));
                },
                Some("resource table"),
            );
        });

/// Binding a table containing a texture and, in the same command buffer, copying
/// *into* that texture with a top-level `copy_buffer_to_texture` (`COPY_DST`,
/// which forces `TRANSFER_DST_OPTIMAL`) is rejected at submit (finding fix). This
/// is the "copy into a member then sample it in the same command buffer" case.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_COPY_DST_SAME_CB_REJECTED: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;

            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: None,
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
            let view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind member");

            let src = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 256,
                usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue
                .write_buffer(&src, 0, &[texture_red(0), 0, 0, 255]);

            let sampler = Sampler::new(ctx, &[0]);

            wgpu_test::fail(
                &ctx.device,
                || {
                    let mut encoder = ctx
                        .device
                        .create_command_encoder(&CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_texture(
                        TexelCopyBufferInfo {
                            buffer: &src,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: None,
                                rows_per_image: None,
                            },
                        },
                        TexelCopyTextureInfo {
                            texture: &texture,
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                    );
                    sampler.record(&mut encoder, &table);
                    ctx.queue.submit(Some(encoder.finish()));
                },
                Some("resource table"),
            );
        });

/// Positive control (finding fix): a table member that is *also* sampled through
/// an ordinary bind group (`RESOURCE`) in the same command buffer is **not** a
/// conflict — bindful sampling leaves it in exactly the `SHADER_READ_ONLY_OPTIMAL`
/// layout its table descriptor declares. Submit succeeds and both reads see the
/// member's value.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_BINDFUL_SAMPLED_SAME_CB_OK: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let ctx = &ctx;
            const RED: u8 = texture_red(2); // 30

            let texture = red_texture_with_usage(
                ctx,
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                RED,
            );
            let view = texture.create_view(&TextureViewDescriptor::default());

            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind member");

            // A pass that binds the table (so the member is table-reachable) and
            // *bindfully* samples the same texture, writing the decoded red byte.
            let shader = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() {
    out[0] = u32(round(textureLoad(tex, vec2<i32>(0, 0), 0).r * 255.0));
}
"#;
            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(shader.into()),
            });
            let bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Texture {
                                sample_type: TextureSampleType::Float { filterable: true },
                                view_dimension: TextureViewDimension::D2,
                                multisampled: false,
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
            let output = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            // Layout declares no table (the shader samples bindfully); the table is
            // bound only to make the member table-reachable.
            let layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                    uses_resource_table: false,
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_resource_table(Some(&table));
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
            ctx.queue.submit(Some(encoder.finish()));

            assert_eq!(
                read_u32s(ctx, &readback).await,
                vec![RED as u32],
                "bindful sample of a table member must be accepted and correct"
            );
        });

/// Positive control (finding fix): the standard flow — populate a member with
/// `queue.write_texture` (which rides the queue's internal encoder, not the user
/// command buffer) and then sample it through the table in the same submission —
/// is **not** a conflict, and the sample sees the written value.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MEMBER_WRITE_TEXTURE_POPULATE_OK: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(0); // 10

            let (_texture, view) = make_red_texture(&ctx, RED);
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &view).expect("bind member");

            let got = run_sampling(&ctx, &table, &[0]).await;
            assert_eq!(
                got,
                vec![RED as u32],
                "write_texture populate + same-submission table sample must be accepted"
            );
        });

/// Positive control (GPU-observable): a texture written by a storage compute
/// pass in one submission (which binds no table) and then sampled through a
/// table in a *later* submission is not a conflict, and the sample sees the
/// written value. This is the legitimate write-then-sample flow — the sampling
/// submission's pass-start barrier makes the prior write visible.
#[apply(gpu_test!)]
static RESOURCE_TABLE_WRITE_THEN_SAMPLE_ACROSS_SUBMISSIONS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const VALUE: u32 = 90;

            let texture = storage_sampled_texture(&ctx);
            let storage_view = texture.create_view(&TextureViewDescriptor::default());
            let sampled_view = texture.create_view(&TextureViewDescriptor::default());

            // Submission 1: write the texture via a storage compute pass. No
            // table is bound, so there is no conflict.
            let write_shader = r#"
@group(0) @binding(0)
var img: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1)
fn main() {
    textureStore(img, vec2<i32>(0, 0), vec4<f32>(90.0 / 255.0, 0.0, 0.0, 1.0));
}
"#;
            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(write_shader.into()),
            });
            let bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });
            let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&storage_view),
                }],
            });
            let pipeline_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                    uses_resource_table: false,
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));

            // Submission 2: sample the texture through a table. No write here, so
            // no conflict; the pass-start transition makes the write visible.
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: None,
                size: 4,
            });
            table.update(0, &sampled_view).expect("bind member");

            let got = run_sampling(&ctx, &table, &[0]).await;
            assert_eq!(
                got,
                vec![VALUE],
                "sample should see the value written in the prior submission"
            );
        });
