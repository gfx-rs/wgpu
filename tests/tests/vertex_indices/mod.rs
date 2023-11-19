use std::{num::NonZeroU64, ops::Range};

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

struct Draw {
    vertex: Range<u32>,
    instance: Range<u32>,
    vertex_offset: Option<i32>,
}

impl Draw {
    fn execute(&self, rpass: &mut wgpu::RenderPass<'_>) {
        if let Some(vertex_offset) = self.vertex_offset {
            rpass.draw_indexed(self.vertex.clone(), vertex_offset, self.instance.clone());
        } else {
            rpass.draw(self.vertex.clone(), self.instance.clone());
        }
    }

    fn add_to_buffer(&self, bytes: &mut Vec<u8>) {
        if let Some(vertex_offset) = self.vertex_offset {
            bytes.extend_from_slice(
                wgpu::util::DrawIndexedIndirectArgs {
                    vertex_count: self.vertex.end - self.vertex.start,
                    instance_count: self.instance.end - self.instance.start,
                    vertex_offset,
                    first_index: self.vertex.start,
                    first_instance: self.instance.start,
                }
                .as_bytes(),
            )
        } else {
            bytes.extend_from_slice(
                wgpu::util::DrawIndirectArgs {
                    vertex_count: self.vertex.end - self.vertex.start,
                    instance_count: self.instance.end - self.instance.start,
                    first_vertex: self.vertex.start,
                    first_instance: self.instance.start,
                }
                .as_bytes(),
            )
        }
    }

    fn execute_indirect<'rpass>(
        &self,
        rpass: &mut wgpu::RenderPass<'rpass>,
        indirect: &'rpass wgpu::Buffer,
        offset: &mut u64,
    ) {
        if self.vertex_offset.is_some() {
            rpass.draw_indexed_indirect(indirect, *offset);
            *offset += 20;
        } else {
            rpass.draw_indirect(indirect, *offset);
            *offset += 16;
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum TestCase {
    Draw,
    DrawVertexOffset,
    DrawBaseVertex,
    DrawInstanced,
    DrawInstancedOffset,
}

impl TestCase {
    const ARRAY: [Self; 5] = [
        Self::Draw,
        Self::DrawVertexOffset,
        Self::DrawBaseVertex,
        Self::DrawInstanced,
        Self::DrawInstancedOffset,
    ];

    fn draws(&self) -> &'static [Draw] {
        match self {
            TestCase::Draw => &[Draw {
                vertex: 0..6,
                instance: 0..1,
                vertex_offset: None,
            }],
            TestCase::DrawVertexOffset => &[
                Draw {
                    vertex: 0..3,
                    instance: 0..1,
                    vertex_offset: None,
                },
                Draw {
                    vertex: 3..6,
                    instance: 0..1,
                    vertex_offset: None,
                },
            ],
            TestCase::DrawBaseVertex => &[Draw {
                vertex: 0..6,
                instance: 0..1,
                vertex_offset: Some(3),
            }],
            TestCase::DrawInstanced => &[Draw {
                vertex: 0..3,
                instance: 0..2,
                vertex_offset: None,
            }],
            TestCase::DrawInstancedOffset => &[
                Draw {
                    vertex: 0..3,
                    instance: 0..1,
                    vertex_offset: None,
                },
                Draw {
                    vertex: 0..3,
                    instance: 1..2,
                    vertex_offset: None,
                },
            ],
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum IdSource {
    Buffers,
    Builtins,
}

impl IdSource {
    const ARRAY: [Self; 2] = [Self::Buffers, Self::Builtins];
}

#[derive(Debug, Copy, Clone)]
enum DrawCallKind {
    Direct,
    Indirect,
}

impl DrawCallKind {
    const ARRAY: [Self; 2] = [Self::Direct, Self::Indirect];
}

struct Test {
    case: TestCase,
    id_source: IdSource,
    draw_call_kind: DrawCallKind,
}

impl Test {
    fn expectation(&self, ctx: &TestingContext) -> &'static [u32] {
        // If this is false, the first instance will be ignored.
        let non_zero_first_instance_supported = ctx
            .adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        // If this is false, it won't be ignored, but it won't show up in the shader
        let first_vert_instance_supported = ctx.adapter_downlevel_capabilities.flags.contains(
            wgpu::DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_INDIRECT_FIRST,
        );

        let is_indirect = matches!(self.draw_call_kind, DrawCallKind::Indirect);

        match self.case {
            TestCase::DrawBaseVertex => {
                if !first_vert_instance_supported && is_indirect {
                    return &[0, 1, 2, 3, 4, 5];
                }

                &[0, 0, 0, 3, 4, 5, 6, 7, 8]
            }
            TestCase::Draw => &[0, 1, 2, 3, 4, 5],
            TestCase::DrawVertexOffset | TestCase::DrawInstanced => {
                if !first_vert_instance_supported && is_indirect {
                    return &[0, 1, 2, 0, 0, 0];
                }

                &[0, 1, 2, 3, 4, 5]
            }
            TestCase::DrawInstancedOffset => {
                if (!first_vert_instance_supported || !non_zero_first_instance_supported)
                    && is_indirect
                {
                    return &[0, 1, 2, 0, 0, 0];
                }

                &[0, 1, 2, 3, 4, 5]
            }
        }
    }
}

fn vertex_index_common(ctx: TestingContext) {
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
            entry_point: "vs_main_builtin",
            module: &shader,
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            entry_point: "fs_main",
            module: &shader,
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
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
            &[0, 0, 0, 1],
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut tests = Vec::with_capacity(5 * 2 * 2);
    for case in TestCase::ARRAY {
        for id_source in IdSource::ARRAY {
            for draw_call_kind in DrawCallKind::ARRAY {
                tests.push(Test {
                    case,
                    id_source,
                    draw_call_kind,
                })
            }
        }
    }

    let mut failed = false;
    for test in tests {
        let pipeline = match test.id_source {
            IdSource::Buffers => &buffer_pipeline,
            IdSource::Builtins => &builtin_pipeline,
        };

        let expected = test.expectation(&ctx);

        let buffer_size = 4 * expected.len() as u64;
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

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let indirect_buffer;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

        rpass.set_vertex_buffer(0, identity_buffer.slice(..));
        rpass.set_vertex_buffer(1, identity_buffer.slice(..));
        rpass.set_index_buffer(identity_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &bg, &[]);

        let draws = test.case.draws();

        match test.draw_call_kind {
            DrawCallKind::Direct => {
                for draw in draws {
                    draw.execute(&mut rpass);
                }
            }
            DrawCallKind::Indirect => {
                let mut indirect_bytes = Vec::new();
                for draw in draws {
                    draw.add_to_buffer(&mut indirect_bytes);
                }
                indirect_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("indirect"),
                    contents: &indirect_bytes,
                    usage: wgpu::BufferUsages::INDIRECT,
                });
                let mut offset = 0;
                for draw in draws {
                    draw.execute_indirect(&mut rpass, &indirect_buffer, &mut offset);
                }
            }
        }

        drop(rpass);

        encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, buffer_size);

        ctx.queue.submit(Some(encoder.finish()));
        let slice = cpu_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.device.poll(wgpu::Maintain::Wait);
        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        if data != expected {
            eprintln!(
                "Failed: Got: {:?} Expected: {:?} - Case {:?} getting indices from {:?} using {:?} draw calls",
                data,
                expected,
                test.case,
                test.id_source,
                test.draw_call_kind
            );
            failed = true;
        } else {
            eprintln!(
                "Passed: Case {:?} getting indices from {:?} using {:?} draw calls",
                test.case, test.id_source, test.draw_call_kind
            );
        }
    }

    assert!(!failed);
}

#[gpu_test]
static VERTEX_INDICES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_sync(vertex_index_common);
