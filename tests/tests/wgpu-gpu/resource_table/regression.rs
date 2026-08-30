//! Regression tests for bugs found in the interim fresh-eyes review over M0
//! waves 1-5 (see `plans/m0-notes.md`):
//!
//! * **C1** — submit-time barrier splicing used to compute pass-start
//!   transitions in *reverse* execution order, giving a table-member texture
//!   sampled in two passes of one command buffer the wrong image layout.
//! * **M1** — a compute pass that rebinds the resource table mid-pass used to
//!   record a pass-start gap only for the *final* bound table, so an earlier
//!   table's member textures were never transitioned.

use wgpu::*;
use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
};

use super::common::{make_red_texture, read_u32s, table_params, texture_red};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_SHARED_TEXTURE_TWO_COMPUTE_PASSES,
        RESOURCE_TABLE_SHARED_TEXTURE_COMPUTE_THEN_RENDER,
        RESOURCE_TABLE_TWO_TABLES_ONE_COMPUTE_PASS,
    ]);
}

// ---------------------------------------------------------------------------
// C1: one member texture sampled in two passes of one command buffer.
// ---------------------------------------------------------------------------

/// The **C1** regression: the *same* table-member texture is sampled in two
/// separate compute passes recorded into one command buffer. Each pass records
/// its own pass-start gap referencing the shared texture; the splice must
/// transition the texture to `RESOURCE` in execution order so both passes read
/// it correctly. (Before the fix the reversed order produced a wrong layout,
/// which VVL flags and/or which corrupts the read.)
#[apply(gpu_test!)]
static RESOURCE_TABLE_SHARED_TEXTURE_TWO_COMPUTE_PASSES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(0);

            // A single texture, uploaded on the queue timeline, shared by two
            // tables (one per pass).
            let (_texture, view) = make_red_texture(&ctx, RED);

            let table1 = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("table 1"),
                size: 4,
            });
            let table2 = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("table 2"),
                size: 4,
            });
            table1.update(0, &view).expect("bind into table 1");
            table2.update(0, &view).expect("bind into table 2");

            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(super::common::SAMPLING_SHADER.into()),
            });

            // Two output buffers, one per pass.
            let index_bytes = 0u32.to_ne_bytes();
            let make_buf = |usage: BufferUsages, init: Option<&[u8]>| {
                let buf = ctx.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: 4,
                    usage,
                    mapped_at_creation: false,
                });
                if let Some(data) = init {
                    ctx.queue.write_buffer(&buf, 0, data);
                }
                buf
            };
            let index_buffer = make_buf(
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
                Some(&index_bytes),
            );
            let out1 = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let out2 = make_buf(BufferUsages::STORAGE | BufferUsages::COPY_SRC, None);
            let readback1 = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);
            let readback2 = make_buf(BufferUsages::MAP_READ | BufferUsages::COPY_DST, None);

            let bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
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
            let bg = |out: &Buffer| {
                ctx.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: index_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: out.as_entire_binding(),
                        },
                    ],
                })
            };
            let bg1 = bg(&out1);
            let bg2 = bg(&out2);

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

            // One command buffer, two passes, each sampling the shared texture.
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("pass 1"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg1, &[]);
                pass.set_resource_table(Some(&table1));
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("pass 2"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg2, &[]);
                pass.set_resource_table(Some(&table2));
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out1, 0, &readback1, 0, 4);
            encoder.copy_buffer_to_buffer(&out2, 0, &readback2, 0, 4);
            ctx.queue.submit(Some(encoder.finish()));

            let got1 = read_u32s(&ctx, &readback1).await;
            let got2 = read_u32s(&ctx, &readback2).await;
            assert_eq!(got1, vec![RED as u32], "pass 1 sampled the shared texture");
            assert_eq!(got2, vec![RED as u32], "pass 2 sampled the shared texture");
        });

/// The **C1** regression across pass *kinds*: the same member texture is
/// sampled by a compute pass and then by a render pass in one command buffer.
/// The compute pass writes the decoded red byte to a storage buffer; the render
/// pass samples the same texture in its fragment shader and writes the color to
/// a 1x1 target. Both must read the texture correctly.
#[apply(gpu_test!)]
static RESOURCE_TABLE_SHARED_TEXTURE_COMPUTE_THEN_RENDER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED: u8 = texture_red(2); // 30

            let (_texture, view) = make_red_texture(&ctx, RED);

            // Compute samples via a table bound as pass state; render samples via
            // the table bound through the render-pass descriptor. Share the
            // texture through a single table used by both.
            let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("shared table"),
                size: 4,
            });
            table.update(0, &view).expect("bind shared texture");

            // --- Compute half ------------------------------------------------
            let compute_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(super::common::SAMPLING_SHADER.into()),
            });
            let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue
                .write_buffer(&index_buffer, 0, &0u32.to_ne_bytes());
            let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let compute_bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
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
            let compute_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &compute_bgl,
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
            let compute_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&compute_bgl)],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let compute_pipeline = ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&compute_layout),
                    module: &compute_module,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

            // --- Render half -------------------------------------------------
            let render_shader = r#"
enable resource_table;
@vertex
fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(p[id], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    let tex = getResource<texture_2d<f32>>(0u);
    return textureLoad(tex, vec2<i32>(0, 0), 0);
}
"#;
            let render_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(render_shader.into()),
            });
            let render_layout = ctx
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[],
                    immediate_size: 0,
                    uses_resource_table: true,
                });
            let render_pipeline = ctx
                .device
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_layout),
                    vertex: VertexState {
                        module: &render_module,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: PipelineCompilationOptions::default(),
                    },
                    fragment: Some(FragmentState {
                        module: &render_module,
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
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let target_view = target.create_view(&TextureViewDescriptor::default());

            // One command buffer: compute pass then render pass, both sampling
            // the shared texture through the table.
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("compute half"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&compute_pipeline);
                pass.set_bind_group(0, &compute_bg, &[]);
                pass.set_resource_table(Some(&table));
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4);
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("render half"),
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
                    resource_table: Some(&table),
                });
                pass.set_pipeline(&render_pipeline);
                pass.draw(0..3, 0..1);
            }
            let readback_buffers = ReadbackBuffers::new(&ctx.device, &target);
            readback_buffers.copy_from(&ctx.device, &mut encoder, &target);
            ctx.queue.submit(Some(encoder.finish()));

            let compute_got = read_u32s(&ctx, &readback_buffer).await;
            assert_eq!(
                compute_got,
                vec![RED as u32],
                "compute pass sampled the shared texture"
            );
            readback_buffers
                .assert_buffer_contents(&ctx, &[RED, 0, 0, 255])
                .await;
        });

// ---------------------------------------------------------------------------
// M1: multiple tables in one compute pass, plus a set_resource_table(None) gap.
// ---------------------------------------------------------------------------

/// The **M1** regression: bind table A, dispatch; rebind table B, dispatch; then
/// `set_resource_table(None)`, rebind table A, dispatch — all in one compute
/// pass. Each dispatch reads through the table bound at the time of the
/// dispatch. Before the fix only the final bound table recorded a pass-start
/// gap, so earlier tables' member textures were never transitioned.
#[apply(gpu_test!)]
static RESOURCE_TABLE_TWO_TABLES_ONE_COMPUTE_PASS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(table_params())
        .run_async(|ctx| async move {
            const RED_A: u8 = texture_red(0); // 10
            const RED_B: u8 = texture_red(1); // 20

            let (_tex_a, view_a) = make_red_texture(&ctx, RED_A);
            let (_tex_b, view_b) = make_red_texture(&ctx, RED_B);

            let table_a = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("table A"),
                size: 4,
            });
            let table_b = ctx.device.create_resource_table(&ResourceTableDescriptor {
                label: Some("table B"),
                size: 4,
            });
            table_a.update(0, &view_a).expect("bind A");
            table_b.update(0, &view_b).expect("bind B");

            let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(super::common::SAMPLING_SHADER.into()),
            });

            // Three separate output buffers, one per dispatch, so each dispatch's
            // read is verified independently. Group 0 = {indices, output}; the
            // bind group is rebound before each dispatch to swap the output.
            let index_buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue
                .write_buffer(&index_buffer, 0, &0u32.to_ne_bytes());

            let outputs: Vec<Buffer> = (0..3)
                .map(|_| {
                    ctx.device.create_buffer(&BufferDescriptor {
                        label: None,
                        size: 4,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let readbacks: Vec<Buffer> = (0..3)
                .map(|_| {
                    ctx.device.create_buffer(&BufferDescriptor {
                        label: None,
                        size: 4,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();

            let bgl = ctx
                .device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
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
            let bind_groups: Vec<BindGroup> = outputs
                .iter()
                .map(|out| {
                    ctx.device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &bgl,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: index_buffer.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: out.as_entire_binding(),
                            },
                        ],
                    })
                })
                .collect();

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

            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("multi-table pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);

                // Dispatch 0: table A -> RED_A.
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_resource_table(Some(&table_a));
                pass.dispatch_workgroups(1, 1, 1);

                // Dispatch 1: rebind table B -> RED_B.
                pass.set_bind_group(0, &bind_groups[1], &[]);
                pass.set_resource_table(Some(&table_b));
                pass.dispatch_workgroups(1, 1, 1);

                // Clear the binding, then rebind table A -> RED_A. No dispatch
                // happens while the table is None (that would be an error).
                pass.set_resource_table(None);
                pass.set_bind_group(0, &bind_groups[2], &[]);
                pass.set_resource_table(Some(&table_a));
                pass.dispatch_workgroups(1, 1, 1);
            }
            for i in 0..3 {
                encoder.copy_buffer_to_buffer(&outputs[i], 0, &readbacks[i], 0, 4);
            }
            ctx.queue.submit(Some(encoder.finish()));

            let got0 = read_u32s(&ctx, &readbacks[0]).await;
            let got1 = read_u32s(&ctx, &readbacks[1]).await;
            let got2 = read_u32s(&ctx, &readbacks[2]).await;
            assert_eq!(got0, vec![RED_A as u32], "dispatch 0 read table A");
            assert_eq!(got1, vec![RED_B as u32], "dispatch 1 read table B");
            assert_eq!(
                got2,
                vec![RED_A as u32],
                "dispatch 2 read table A after a None rebind"
            );
        });
