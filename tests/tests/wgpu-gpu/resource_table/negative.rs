//! Negative / validation-path coverage:
//!
//! * dispatching or drawing with a pipeline whose layout declares a resource
//!   table but with no table bound -> `MissingResourceTable` (direct compute,
//!   indirect compute, and render draw),
//! * creating a resource table on a device without the sampling feature, and
//! * creating a pipeline whose shader uses `getResource` without the
//!   `EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED` feature (M0 has only the unchecked
//!   lowering, D4).

use wgpu::*;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

use super::common::table_params;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_MISSING_TABLE_DISPATCH,
        RESOURCE_TABLE_MISSING_TABLE_INDIRECT_DISPATCH,
        RESOURCE_TABLE_MISSING_TABLE_DRAW,
        RESOURCE_TABLE_CREATE_WITHOUT_FEATURE_FAILS,
        RESOURCE_TABLE_PIPELINE_WITHOUT_UNCHECKED_FAILS,
    ]);
}

/// A trivial compute shader used to build a pipeline whose *layout* declares a
/// resource table (the dispatch-time check keys off the layout flag, not the
/// shader), so no `getResource` or bindings are needed.
const TRIVIAL_COMPUTE: &str = "@compute @workgroup_size(1) fn main() {}";

/// Trivial vertex + fragment shaders for a render pipeline whose layout declares
/// a resource table.
const TRIVIAL_RENDER: &str = "\
@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
";

fn table_compute_pipeline(ctx: &wgpu_test::TestingContext) -> ComputePipeline {
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(TRIVIAL_COMPUTE.into()),
    });
    let layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
            uses_resource_table: true,
        });
    ctx.device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
}

/// Dispatching without a bound table when the layout declares one is a
/// validation error, surfaced at `finish()`.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MISSING_TABLE_DISPATCH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        let pipeline = table_compute_pipeline(&ctx);

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
                    pass.dispatch_workgroups(1, 1, 1);
                }
                encoder.finish()
            },
            Some("resource table"),
        );
    });

/// Same as above, but via an *indirect* dispatch: the internal indirect-buffer
/// validation dispatch re-sets the resource-table dirty flag, so the missing
/// table is still caught.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MISSING_TABLE_INDIRECT_DISPATCH: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            let pipeline = table_compute_pipeline(&ctx);

            let indirect = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 12,
                usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let dims: [u32; 3] = [1, 1, 1];
            ctx.queue
                .write_buffer(&indirect, 0, bytemuck::cast_slice(&dims));

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
                        pass.dispatch_workgroups_indirect(&indirect, 0);
                    }
                    encoder.finish()
                },
                Some("resource table"),
            );
        });

/// Drawing without a table bound via the render-pass descriptor when the layout
/// declares one is a validation error.
#[apply(gpu_test!)]
static RESOURCE_TABLE_MISSING_TABLE_DRAW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        // `create_command_encoder` requires the device's queue to still be
        // alive. This test never touches `ctx.queue`, and Rust 2021 disjoint
        // closure capture would otherwise drop it (and the underlying core
        // queue) before the future runs, so borrow the whole context to force
        // it to be captured.
        let ctx = &ctx;
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(TRIVIAL_RENDER.into()),
        });
        let layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                immediate_size: 0,
                uses_resource_table: true,
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                cache: None,
                multiview_mask: None,
            });
        let target = ctx.device.create_texture(&TextureDescriptor {
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
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&TextureViewDescriptor::default());

        wgpu_test::fail(
            &ctx.device,
            || {
                let mut encoder = ctx
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor { label: None });
                {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &target_view,
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
                        // No table bound, but the pipeline layout requires one.
                        resource_table: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.draw(0..3, 0..1);
                }
                encoder.finish()
            },
            Some("resource table"),
        );
    });

/// Creating a resource table on a device that did not enable the sampling
/// feature is a validation error. Runs with no features requested.
#[apply(gpu_test!)]
static RESOURCE_TABLE_CREATE_WITHOUT_FEATURE_FAILS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().skip(FailureCase::backend(!Backends::VULKAN)))
        .run_async(|ctx| async move {
            wgpu_test::fail(
                &ctx.device,
                || {
                    ctx.device.create_resource_table(&ResourceTableDescriptor {
                        label: None,
                        size: 4,
                    })
                },
                Some("features"),
            );
        });

/// Creating a pipeline whose shader uses `getResource` requires the unchecked
/// feature in M0 (only the unchecked lowering exists, D4). With the sampling
/// feature enabled but not the unchecked one, pipeline creation fails.
#[apply(gpu_test!)]
static RESOURCE_TABLE_PIPELINE_WITHOUT_UNCHECKED_FAILS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE)
                .limits(Limits::downlevel_defaults())
                .skip(FailureCase::backend(!Backends::VULKAN)),
        )
        .run_async(|ctx| async move {
            // A shader that reads the table (so reflection reports table use).
            let shader = "\
enable resource_table;
var<private> sink: vec4<f32>;
@compute @workgroup_size(1)
fn main() {
    sink = textureLoad(getResource<texture_2d<f32>>(0u), vec2<i32>(0, 0), 0);
}
";
            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(shader.into()),
            });
            let layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    immediate_size: 0,
                    uses_resource_table: true,
                });

            wgpu_test::fail(
                &ctx.device,
                || {
                    ctx.device
                        .create_compute_pipeline(&ComputePipelineDescriptor {
                            label: None,
                            layout: Some(&layout),
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: PipelineCompilationOptions::default(),
                            cache: None,
                        })
                },
                Some("features"),
            );
        });
