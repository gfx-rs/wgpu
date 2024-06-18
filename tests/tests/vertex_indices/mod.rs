//! Tests that vertex buffers, vertex indices, and instance indices are properly handled.
//!
//! We need tests for these as the backends use various schemes to work around the lack
//! of support for things like `gl_BaseInstance` in shaders.

use std::{num::NonZeroU64, ops::Range};

use itertools::Itertools;
use strum::IntoEnumIterator;
use wgpu::util::{BufferInitDescriptor, DeviceExt, RenderEncoder};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};
use wgt::RenderBundleDescriptor;

/// Generic struct representing a draw call
struct Draw {
    vertex: Range<u32>,
    instance: Range<u32>,
    /// If present, is an indexed call
    base_vertex: Option<i32>,
}

impl Draw {
    /// Directly execute the draw call
    fn execute(&self, rpass: &mut dyn RenderEncoder<'_>) {
        if let Some(base_vertex) = self.base_vertex {
            rpass.draw_indexed(self.vertex.clone(), base_vertex, self.instance.clone());
        } else {
            rpass.draw(self.vertex.clone(), self.instance.clone());
        }
    }

    /// Add the draw call to the given indirect buffer
    fn add_to_buffer(&self, bytes: &mut Vec<u8>, features: wgpu::Features) {
        // The behavior of non-zero first_instance in indirect draw calls in currently undefined if INDIRECT_FIRST_INSTANCE is not supported.
        let supports_first_instance = features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        let first_instance = if supports_first_instance {
            self.instance.start
        } else {
            0
        };

        if let Some(base_vertex) = self.base_vertex {
            bytes.extend_from_slice(
                wgpu::util::DrawIndexedIndirectArgs {
                    index_count: self.vertex.end - self.vertex.start,
                    instance_count: self.instance.end - self.instance.start,
                    base_vertex,
                    first_index: self.vertex.start,
                    first_instance,
                }
                .as_bytes(),
            )
        } else {
            bytes.extend_from_slice(
                wgpu::util::DrawIndirectArgs {
                    vertex_count: self.vertex.end - self.vertex.start,
                    instance_count: self.instance.end - self.instance.start,
                    first_vertex: self.vertex.start,
                    first_instance,
                }
                .as_bytes(),
            )
        }
    }

    /// Execute the draw call from the given indirect buffer
    fn execute_indirect<'rpass>(
        &self,
        rpass: &mut dyn RenderEncoder<'rpass>,
        indirect: &'rpass wgpu::Buffer,
        offset: &mut u64,
    ) {
        if self.base_vertex.is_some() {
            rpass.draw_indexed_indirect(indirect, *offset);
            *offset += 20;
        } else {
            rpass.draw_indirect(indirect, *offset);
            *offset += 16;
        }
    }
}

#[derive(Debug, Copy, Clone, strum::EnumIter)]
enum TestCase {
    /// A single draw call with 6 vertices
    Draw,
    /// Two draw calls of 0..3 and 3..6 verts
    DrawNonZeroFirstVertex,
    /// A single draw call with 6 vertices and a vertex offset of 3
    DrawBaseVertex,
    /// A single draw call with 3 vertices and 2 instances
    DrawInstanced,
    /// Two draw calls with 3 vertices and 0..1 and 1..2 instances.
    DrawNonZeroFirstInstance,
}

impl TestCase {
    // Get the draw calls for this test case
    fn draws(&self) -> &'static [Draw] {
        match self {
            TestCase::Draw => &[Draw {
                vertex: 0..6,
                instance: 0..1,
                base_vertex: None,
            }],
            TestCase::DrawNonZeroFirstVertex => &[
                Draw {
                    vertex: 0..3,
                    instance: 0..1,
                    base_vertex: None,
                },
                Draw {
                    vertex: 3..6,
                    instance: 0..1,
                    base_vertex: None,
                },
            ],
            TestCase::DrawBaseVertex => &[Draw {
                vertex: 0..6,
                instance: 0..1,
                base_vertex: Some(3),
            }],
            TestCase::DrawInstanced => &[Draw {
                vertex: 0..3,
                instance: 0..2,
                base_vertex: None,
            }],
            TestCase::DrawNonZeroFirstInstance => &[
                Draw {
                    vertex: 0..3,
                    instance: 0..1,
                    base_vertex: None,
                },
                Draw {
                    vertex: 0..3,
                    instance: 1..2,
                    base_vertex: None,
                },
            ],
        }
    }
}

#[derive(Debug, Copy, Clone, strum::EnumIter)]
enum IdSource {
    /// Use buffers to load the vertex and instance index
    Buffers,
    /// Use builtins to load the vertex and instance index
    Builtins,
}

#[derive(Debug, Copy, Clone, strum::EnumIter)]
enum DrawCallKind {
    Direct,
    Indirect,
}

#[derive(Debug, Copy, Clone, strum::EnumIter)]
enum EncoderKind {
    RenderPass,
    RenderBundle,
}

struct Test {
    case: TestCase,
    id_source: IdSource,
    draw_call_kind: DrawCallKind,
    encoder_kind: EncoderKind,
}

impl Test {
    /// Get the expected result from this test, taking into account
    /// the various features and capabilities that may be missing.
    fn expectation(&self, ctx: &TestingContext) -> &'static [u32] {
        let is_indirect = matches!(self.draw_call_kind, DrawCallKind::Indirect);

        // Both of these failure modes require indirect rendering

        // If this is false, the first instance will be ignored.
        let non_zero_first_instance_supported = ctx
            .adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            || !is_indirect;

        // If this is false, it won't be ignored, but it won't show up in the shader
        //
        // If the IdSource is buffers, this doesn't apply
        let first_vert_instance_supported = ctx.adapter_downlevel_capabilities.flags.contains(
            wgpu::DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW,
        ) || matches!(self.id_source, IdSource::Buffers)
            || !is_indirect;

        match self.case {
            TestCase::DrawBaseVertex => {
                if !first_vert_instance_supported {
                    return &[0, 1, 2, 3, 4, 5];
                }

                &[0, 0, 0, 3, 4, 5, 6, 7, 8]
            }
            TestCase::Draw | TestCase::DrawInstanced => &[0, 1, 2, 3, 4, 5],
            TestCase::DrawNonZeroFirstVertex => {
                if !first_vert_instance_supported {
                    return &[0, 1, 2, 0, 0, 0];
                }

                &[0, 1, 2, 3, 4, 5]
            }
            TestCase::DrawNonZeroFirstInstance => {
                if !first_vert_instance_supported || !non_zero_first_instance_supported {
                    return &[0, 1, 2, 0, 0, 0];
                }

                &[0, 1, 2, 3, 4, 5]
            }
        }
    }
}

async fn vertex_index_common(ctx: TestingContext) {
    let identity_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("identity buffer"),
        contents: bytemuck::cast_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8]),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX,
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("draw.vert.wgsl"));

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                visibility: wgpu::ShaderStages::VERTEX,
                count: None,
            }],
        });

    let ppl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let mut pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&ppl),
        vertex: wgpu::VertexState {
            buffers: &[],
            module: &shader,
            entry_point: "vs_main_builtin",
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    };
    let builtin_pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    pipeline_desc.vertex.entry_point = "vs_main_buffers";
    pipeline_desc.vertex.buffers = &[
        wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Uint32],
        },
        wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![1 => Uint32],
        },
    ];
    let buffer_pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let dummy = ctx
        .device
        .create_texture_with_data(
            &ctx.queue,
            &wgpu::TextureDescriptor {
                label: Some("dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0, 0, 0, 1],
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

    let tests = TestCase::iter()
        .cartesian_product(IdSource::iter())
        .cartesian_product(DrawCallKind::iter())
        .cartesian_product(EncoderKind::iter())
        .map(|(((case, id_source), draw_call_kind), encoder_kind)| Test {
            case,
            id_source,
            draw_call_kind,
            encoder_kind,
        })
        .collect::<Vec<_>>();

    let features = ctx.adapter.features();

    let mut failed = false;
    for test in tests {
        let pipeline = match test.id_source {
            IdSource::Buffers => &buffer_pipeline,
            IdSource::Builtins => &builtin_pipeline,
        };

        let expected = test.expectation(&ctx);

        let buffer_size = (std::mem::size_of_val(&expected[0]) * expected.len()) as u64;
        let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer.as_entire_binding(),
            }],
        });

        let mut encoder1 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let render_bundle;
        let indirect_buffer;
        let mut rpass = encoder1.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &dummy,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        {
            // Need to scope render_bundle_encoder since it's not Send and would otherwise
            // infect the function if not going out of scope before an await call.
            // (it is dropped via `take` + `finish` earlier, but compiler does not take this into account)
            let mut render_bundle_encoder = match test.encoder_kind {
                EncoderKind::RenderPass => None,
                EncoderKind::RenderBundle => Some(ctx.device.create_render_bundle_encoder(
                    &wgpu::RenderBundleEncoderDescriptor {
                        label: Some("test renderbundle encoder"),
                        color_formats: &[Some(wgpu::TextureFormat::Rgba8Unorm)],
                        depth_stencil: None,
                        sample_count: 1,
                        multiview: None,
                    },
                )),
            };

            let render_encoder: &mut dyn RenderEncoder = render_bundle_encoder
                .as_mut()
                .map(|r| r as &mut dyn RenderEncoder)
                .unwrap_or(&mut rpass);

            render_encoder.set_vertex_buffer(0, identity_buffer.slice(..));
            render_encoder.set_vertex_buffer(1, identity_buffer.slice(..));
            render_encoder.set_index_buffer(identity_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_encoder.set_pipeline(pipeline);
            render_encoder.set_bind_group(0, &bg, &[]);

            let draws = test.case.draws();

            match test.draw_call_kind {
                DrawCallKind::Direct => {
                    for draw in draws {
                        draw.execute(render_encoder);
                    }
                }
                DrawCallKind::Indirect => {
                    let mut indirect_bytes = Vec::new();
                    for draw in draws {
                        draw.add_to_buffer(&mut indirect_bytes, features);
                    }
                    indirect_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
                        label: Some("indirect"),
                        contents: &indirect_bytes,
                        usage: wgpu::BufferUsages::INDIRECT,
                    });
                    let mut offset = 0;
                    for draw in draws {
                        draw.execute_indirect(render_encoder, &indirect_buffer, &mut offset);
                    }
                }
            }

            if let Some(render_bundle_encoder) = render_bundle_encoder.take() {
                render_bundle = render_bundle_encoder.finish(&RenderBundleDescriptor {
                    label: Some("test renderbundle"),
                });
                rpass.execute_bundles([&render_bundle]);
            }
        }

        drop(rpass);

        let mut encoder2 = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder2.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);

        // See https://github.com/gfx-rs/wgpu/issues/4732 for why this is split between two submissions
        // with a hard wait in between.
        ctx.queue.submit([encoder1.finish()]);
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        ctx.queue.submit([encoder2.finish()]);
        let slice = cpu_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        let case_name = format!(
            "Case {:?} getting indices from {:?} using {:?} draw calls, encoded with a {:?}",
            test.case, test.id_source, test.draw_call_kind, test.encoder_kind
        );
        if data != expected {
            eprintln!(
                "Failed: Got: {:?} Expected: {:?} - {case_name}",
                data, expected,
            );
            failed = true;
        } else {
            eprintln!("Passed: {case_name}");
        }
    }

    assert!(!failed);
}

#[gpu_test]
static VERTEX_INDICES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE)
            .features(wgpu::Features::INDIRECT_FIRST_INSTANCE),
    )
    .run_async(vertex_index_common);
