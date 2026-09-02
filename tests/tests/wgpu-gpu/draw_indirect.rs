use std::{mem::size_of, num::NonZeroU64};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend(&[
        DRAW,
        DRAW_OOB_START,
        DRAW_OOB_COUNT,
        INSTANCED_DRAW,
        INSTANCED_DRAW_OOB_START,
        INSTANCED_DRAW_OOB_COUNT,
        INSTANCED_DRAW_OOB_INSTANCE_START,
        INSTANCED_DRAW_OOB_INSTANCE_COUNT,
        INSTANCED_DRAW_WITH_NON_ZERO_FIRST_INSTANCE,
        INSTANCED_DRAW_WITH_NON_ZERO_FIRST_INSTANCE_MISSING_FEATURE,
        INDEXED_DRAW,
        INDEXED_DRAW_OOB_START,
        INDEXED_DRAW_OOB_COUNT,
        INSTANCED_INDEXED_DRAW,
        INSTANCED_INDEXED_DRAW_OOB_START,
        INSTANCED_INDEXED_DRAW_OOB_COUNT,
        INSTANCED_INDEXED_DRAW_OOB_INSTANCE_START,
        INSTANCED_INDEXED_DRAW_OOB_INSTANCE_COUNT,
        INDIRECT_BUFFER_OFFSETS,
        MULTI_DRAW_INDEXED_INDIRECT,
        MULTI_DRAW_INDIRECT,
        MULTI_DRAW_INDIRECT_GPU_GENERATED_ARGS,
        MULTI_DRAW_INDEXED_INDIRECT_GPU_GENERATED_ARGS,
        MULTI_DRAW_INDIRECT_COUNT_READBACK,
        MULTI_DRAW_INDEXED_INDIRECT_COUNT_READBACK,
        MULTI_DRAW_INDIRECT_OVER_ICB_WORKGROUP,
        MULTI_DRAW_INDIRECT_FIRST_VERTEX_AND_INSTANCE,
        MULTI_DRAW_INDIRECT_MIXED_SEQUENCE,
        MULTI_DRAW_INDIRECT_WITH_BIND_GROUPS,
        MULTI_DRAW_INDEXED_INDIRECT_U16,
        MULTI_DRAW_INDEXED_INDIRECT_POSITIVE_BASE_VERTEX,
        MULTI_DRAW_INDEXED_INDIRECT_NEGATIVE_BASE_VERTEX,
    ]);
}

struct TestData {
    kind: Kind,
    instanced: Option<Instanced>,
}

struct Instanced {
    instance_buffer_content: &'static [f32],

    first_instance: u32,
    instance_count: u32,
}

enum Kind {
    NonIndexed {
        vertex_buffer_content: &'static [f32],

        first_vertex: u32,
        vertex_count: u32,
    },
    Indexed {
        vertex_buffer_content: &'static [f32],

        index_buffer_content: &'static [u32],

        first_index: u32,
        index_count: u32,
    },
}

impl TestData {
    fn vertex_buffer_content(&self) -> &'static [f32] {
        match self.kind {
            Kind::NonIndexed {
                vertex_buffer_content,
                ..
            } => vertex_buffer_content,
            Kind::Indexed {
                vertex_buffer_content,
                ..
            } => vertex_buffer_content,
        }
    }

    fn write_indirect_args(&self, buf: &mut Vec<u8>) {
        let (first_instance, instance_count) = match self.instanced {
            Some(ref instanced) => (instanced.first_instance, instanced.instance_count),
            None => (0, 1),
        };
        match self.kind {
            Kind::NonIndexed {
                first_vertex,
                vertex_count,
                ..
            } => {
                buf.extend_from_slice(
                    wgpu::util::DrawIndirectArgs {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    }
                    .as_bytes(),
                );
            }
            Kind::Indexed {
                first_index,
                index_count,
                ..
            } => {
                buf.extend_from_slice(
                    wgpu::util::DrawIndexedIndirectArgs {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex: 0,
                        first_instance,
                    }
                    .as_bytes(),
                );
            }
        }
    }
}

async fn run_test(ctx: TestingContext, test_data: TestData, expect_noop: bool) {
    run_test_inner(ctx, test_data, expect_noop, false).await;
}

async fn run_test_inner(
    ctx: TestingContext,
    test_data: TestData,
    expect_noop: bool,
    use_multi_draw: bool,
) {
    let mut vertex_buffer_layouts = Vec::new();
    vertex_buffer_layouts.push(Some(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![0 => Float32x2],
    }));
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(test_data.vertex_buffer_content()),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = match test_data.kind {
        Kind::NonIndexed { .. } => None,
        Kind::Indexed {
            index_buffer_content,
            ..
        } => Some(ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(index_buffer_content),
            usage: wgpu::BufferUsages::INDEX,
        })),
    };

    let instance_buffer = test_data.instanced.as_ref().map(|instanced| {
        vertex_buffer_layouts.push(Some(wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &vertex_attr_array![1 => Float32x2],
        }));
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(instanced.instance_buffer_content),
            usage: wgpu::BufferUsages::VERTEX,
        })
    });

    let shader_src = if instance_buffer.is_none() {
        "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(1.0);
            }
        "
    } else {
        "
            @vertex
            fn vs_main(@location(0) position: vec2f, @location(1) position_offset: vec2f) -> @builtin(position) vec4f {
                return vec4f(position + position_offset, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(1.0);
            }
        "
    };

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            buffers: &vertex_buffer_layouts,
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Use 2 passes to trigger internal validation buffer reuse
    let passes = 2;
    // Issue 2 draws per indirect buffer to trigger internal validation batching
    let draws = 2; // try 66000 to test multiple temporary validation buffers

    let mut indirect_bytes = Vec::new();
    for _ in 0..passes * draws {
        test_data.write_indirect_args(&mut indirect_bytes);
    }
    let indirect_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &indirect_bytes,
        usage: wgpu::BufferUsages::INDIRECT,
    });
    // Use a secondary indirect buffer to test multiple validation batches.
    let indirect_buffer2 = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &indirect_bytes,
        usage: wgpu::BufferUsages::INDIRECT,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for pass_index in 0..passes {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        if let Some(ref instance_buffer) = instance_buffer {
            rpass.set_vertex_buffer(1, instance_buffer.slice(..));
        }
        if let Some(ref index_buffer) = index_buffer {
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        }
        if use_multi_draw {
            if index_buffer.is_some() {
                rpass.multi_draw_indexed_indirect(&indirect_buffer, 0, draws);
            } else {
                rpass.multi_draw_indirect(&indirect_buffer, 0, draws);
            }
        } else {
            for draw_index in 0..draws {
                if index_buffer.is_some() {
                    let offset = (pass_index * draw_index * 20) as u64;
                    rpass.draw_indexed_indirect(&indirect_buffer, offset);
                    rpass.draw_indexed_indirect(&indirect_buffer2, offset);
                } else {
                    let offset = (pass_index * draw_index * 20) as u64;
                    rpass.draw_indirect(&indirect_buffer, offset);
                    rpass.draw_indirect(&indirect_buffer2, offset);
                }
            }
        }
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let succeeded = if expect_noop {
        data.iter().all(|b| *b == 0)
    } else {
        data.iter().all(|b| *b == u8::MAX)
    };
    assert!(succeeded);
}

fn create_gpu_generated_args_pipeline(
    ctx: &TestingContext,
    shader_src: &str,
    args_buffer: &wgpu::Buffer,
    args_size: u64,
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(args_size),
                },
                count: None,
            }],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: args_buffer.as_entire_binding(),
        }],
    });

    (pipeline, bind_group)
}

fn create_indirect_render_pipeline(
    ctx: &TestingContext,
    indexed: bool,
) -> (wgpu::RenderPipeline, wgpu::Buffer, Option<wgpu::Buffer>) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "
                    @vertex
                    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                        return vec4f(position, 0.0, 1.0);
                    }

                    @fragment
                    fn fs_main() -> @location(0) vec4f {
                        return vec4f(1.0);
                    }
                "
                .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                })],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    let vertex_buffer_content = if indexed {
        &[
            -1.0f32, -1.0, // Bottom left
            1.0, 1.0, // Top right
            -1.0, 1.0, // Top left
            1.0, -1.0, // Bottom right
        ][..]
    } else {
        &[
            -1.0f32, -1.0, // Bottom left
            1.0, 1.0, // Top right
            -1.0, 1.0, // Top left
            -1.0, -1.0, // Bottom left
            1.0, -1.0, // Bottom right
            1.0, 1.0, // Top right
        ][..]
    };
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = indexed.then(|| {
        let index_buffer_content = [
            0u32, 1, 2, // Triangle 1
            0, 3, 1, // Triangle 2
        ];
        ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&index_buffer_content),
            usage: wgpu::BufferUsages::INDEX,
        })
    });

    (pipeline, vertex_buffer, index_buffer)
}

fn create_rgba8_render_target(
    ctx: &TestingContext,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Submits `encoder` and checks that every texel of `texture` is opaque white.
async fn assert_all_white(
    ctx: &TestingContext,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) {
    let readback = ReadbackBuffers::new(&ctx.device, texture);
    readback.copy_from(&ctx.device, &mut encoder, texture);
    ctx.queue.submit([encoder.finish()]);
    let byte_count = (texture.width() * texture.height() * 4) as usize;
    readback
        .assert_buffer_contents(ctx, &vec![u8::MAX; byte_count])
        .await;
}

fn draw_indirect_bytes(args: &[wgpu::util::DrawIndirectArgs]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(args));
    for arg in args {
        bytes.extend_from_slice(arg.as_bytes());
    }
    bytes
}

fn draw_indexed_indirect_bytes(args: &[wgpu::util::DrawIndexedIndirectArgs]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_of_val(args));
    for arg in args {
        bytes.extend_from_slice(arg.as_bytes());
    }
    bytes
}

fn create_draw_indirect_buffer(
    ctx: &TestingContext,
    args: &[wgpu::util::DrawIndirectArgs],
) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &draw_indirect_bytes(args),
        usage: wgpu::BufferUsages::INDIRECT,
    })
}

fn create_draw_indexed_indirect_buffer(
    ctx: &TestingContext,
    args: &[wgpu::util::DrawIndexedIndirectArgs],
) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &draw_indexed_indirect_bytes(args),
        usage: wgpu::BufferUsages::INDIRECT,
    })
}

/// Kept in sync with `ICB_MIN_DRAW_COUNT` in `wgpu-hal/src/metal/command.rs`
/// so these tests exercise Metal's indirect-command-buffer lowering rather
/// than the small-count per-draw loop.
const ICB_MULTI_DRAW_TEST_COUNT: usize = 512;

async fn run_multi_draw_indirect_over_icb_workgroup(ctx: TestingContext) {
    let (pipeline, vertex_buffer, _) = create_indirect_render_pipeline(&ctx, false);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut args = vec![
        wgpu::util::DrawIndirectArgs {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        };
        64
    ];
    args.push(wgpu::util::DrawIndirectArgs {
        vertex_count: 6,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    });
    let indirect_buffer = create_draw_indirect_buffer(&ctx, &args);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.multi_draw_indirect(&indirect_buffer, 0, args.len() as u32);
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

async fn run_multi_draw_indirect_first_vertex_and_instance(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "
                    @vertex
                    fn vs_main(
                        @location(0) position: vec2f,
                        @location(1) position_offset: vec2f,
                    ) -> @builtin(position) vec4f {
                        return vec4f(position + position_offset, 0.0, 1.0);
                    }

                    @fragment
                    fn fs_main() -> @location(0) vec4f {
                        return vec4f(1.0);
                    }
                "
                .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attr_array![0 => Float32x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &vertex_attr_array![1 => Float32x2],
                    }),
                ],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    let vertex_buffer_content = [
        10.0f32, 10.0, // first_vertex must skip this sentinel
        -0.5, -0.5, // Triangle 1
        0.5, 0.5, -0.5, 0.5, -0.5, -0.5, // Triangle 2
        0.5, -0.5, 0.5, 0.5,
    ];
    let instance_buffer_content = [
        10.0f32, 10.0, // first_instance must skip this sentinel
        -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5,
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let instance_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&instance_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let args = vec![
        wgpu::util::DrawIndirectArgs {
            vertex_count: 6,
            instance_count: 4,
            first_vertex: 1,
            first_instance: 1,
        };
        ICB_MULTI_DRAW_TEST_COUNT
    ];
    let indirect_buffer = create_draw_indirect_buffer(&ctx, &args);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_vertex_buffer(1, instance_buffer.slice(..));
        rpass.multi_draw_indirect(&indirect_buffer, 0, ICB_MULTI_DRAW_TEST_COUNT as u32);
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

async fn run_multi_draw_indirect_mixed_sequence(ctx: TestingContext) {
    let (pipeline, vertex_buffer, _) = create_indirect_render_pipeline(&ctx, false);
    let mut args = vec![
        wgpu::util::DrawIndirectArgs {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        };
        ICB_MULTI_DRAW_TEST_COUNT + 1
    ];
    args[0] = wgpu::util::DrawIndirectArgs {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    args[ICB_MULTI_DRAW_TEST_COUNT] = wgpu::util::DrawIndirectArgs {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 3,
        first_instance: 0,
    };
    let indirect_buffer = create_draw_indirect_buffer(&ctx, &args);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.multi_draw_indirect(&indirect_buffer, 0, ICB_MULTI_DRAW_TEST_COUNT as u32);
        rpass.draw_indirect(
            &indirect_buffer,
            (ICB_MULTI_DRAW_TEST_COUNT * size_of::<wgpu::util::DrawIndirectArgs>()) as u64,
        );
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

/// Multi-draw with an active bind group; on Metal's ICB path the bind-group
/// bindings must be inherited correctly by the generated commands.
async fn run_multi_draw_indirect_with_bind_groups(ctx: TestingContext) {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "
                    struct Offset {
                        xy: vec2f,
                        _pad: vec2f,
                    };

                    @group(0) @binding(0) var<uniform> offset: Offset;

                    @vertex
                    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                        return vec4f(position + offset.xy, 0.0, 1.0);
                    }

                    @fragment
                    fn fs_main() -> @location(0) vec4f {
                        return vec4f(1.0);
                    }
                "
                .into(),
            ),
        });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                })],
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
    let shifted_vertex_buffer_content = [
        -11.0f32, -1.0, -9.0, 1.0, -11.0, 1.0, -11.0, -1.0, -9.0, -1.0, -9.0, 1.0,
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&shifted_vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let uniform_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[10.0f32, 0.0, 0.0, 0.0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let mut args = vec![
        wgpu::util::DrawIndirectArgs {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        };
        ICB_MULTI_DRAW_TEST_COUNT
    ];
    args[0] = wgpu::util::DrawIndirectArgs {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    args[1] = wgpu::util::DrawIndirectArgs {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 3,
        first_instance: 0,
    };
    let indirect_buffer = create_draw_indirect_buffer(&ctx, &args);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.multi_draw_indirect(&indirect_buffer, 0, ICB_MULTI_DRAW_TEST_COUNT as u32);
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

fn create_custom_vertex_buffer(ctx: &TestingContext, vertices: &[f32]) -> wgpu::Buffer {
    ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

async fn run_multi_draw_indexed_indirect_u16(ctx: TestingContext) {
    let (pipeline, vertex_buffer, _) = create_indirect_render_pipeline(&ctx, true);
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[0u16, 1, 2, 0, 3, 1]),
        usage: wgpu::BufferUsages::INDEX,
    });
    let mut args = vec![
        wgpu::util::DrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        ICB_MULTI_DRAW_TEST_COUNT
    ];
    args[0] = wgpu::util::DrawIndexedIndirectArgs {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    args[1] = wgpu::util::DrawIndexedIndirectArgs {
        index_count: 3,
        instance_count: 1,
        first_index: 3,
        base_vertex: 0,
        first_instance: 0,
    };
    let indirect_buffer = create_draw_indexed_indirect_buffer(&ctx, &args);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        rpass.multi_draw_indexed_indirect(&indirect_buffer, 0, ICB_MULTI_DRAW_TEST_COUNT as u32);
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

async fn run_multi_draw_indexed_indirect_base_vertex(ctx: TestingContext, base_vertex: i32) {
    let (pipeline, _, _) = create_indirect_render_pipeline(&ctx, true);
    let (vertices, indices): (&[f32], &[u32]) = if base_vertex >= 0 {
        (
            &[
                10.0, 10.0, // Sentinels that should be skipped by base_vertex.
                11.0, 11.0, 10.0, 11.0, 11.0, 10.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
            ],
            &[0, 1, 2, 0, 3, 1],
        )
    } else {
        (
            &[-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
            &[4, 5, 6, 4, 7, 5],
        )
    };
    let vertex_buffer = create_custom_vertex_buffer(&ctx, vertices);
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let mut args = vec![
        wgpu::util::DrawIndexedIndirectArgs {
            index_count: 0,
            instance_count: 1,
            first_index: 0,
            base_vertex,
            first_instance: 0,
        };
        ICB_MULTI_DRAW_TEST_COUNT
    ];
    args[0] = wgpu::util::DrawIndexedIndirectArgs {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        base_vertex,
        first_instance: 0,
    };
    args[1] = wgpu::util::DrawIndexedIndirectArgs {
        index_count: 3,
        instance_count: 1,
        first_index: 3,
        base_vertex,
        first_instance: 0,
    };
    let indirect_buffer = create_draw_indexed_indirect_buffer(&ctx, &args);
    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.multi_draw_indexed_indirect(&indirect_buffer, 0, ICB_MULTI_DRAW_TEST_COUNT as u32);
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

async fn run_multi_draw_indirect_count_readback(ctx: TestingContext, indexed: bool) {
    let (pipeline, vertex_buffer, index_buffer) = create_indirect_render_pipeline(&ctx, indexed);
    let max_draw_count = ICB_MULTI_DRAW_TEST_COUNT as u32;
    let count_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[2u32]),
        usage: wgpu::BufferUsages::INDIRECT,
    });

    let (out_texture, out_texture_view) = create_rgba8_render_target(&ctx, 256, 256);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        if let Some(index_buffer) = index_buffer.as_ref() {
            let mut args = vec![
                wgpu::util::DrawIndexedIndirectArgs {
                    index_count: 0,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                };
                ICB_MULTI_DRAW_TEST_COUNT
            ];
            args[0] = wgpu::util::DrawIndexedIndirectArgs {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
            };
            args[1] = wgpu::util::DrawIndexedIndirectArgs {
                index_count: 3,
                instance_count: 1,
                first_index: 3,
                base_vertex: 0,
                first_instance: 0,
            };
            let indirect_buffer = create_draw_indexed_indirect_buffer(&ctx, &args);
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.multi_draw_indexed_indirect_count(
                &indirect_buffer,
                0,
                &count_buffer,
                0,
                max_draw_count,
            );
        } else {
            let mut args = vec![
                wgpu::util::DrawIndirectArgs {
                    vertex_count: 0,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                };
                ICB_MULTI_DRAW_TEST_COUNT
            ];
            args[0] = wgpu::util::DrawIndirectArgs {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            };
            args[1] = wgpu::util::DrawIndirectArgs {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 3,
                first_instance: 0,
            };
            let indirect_buffer = create_draw_indirect_buffer(&ctx, &args);
            rpass.multi_draw_indirect_count(&indirect_buffer, 0, &count_buffer, 0, max_draw_count);
        }
    }

    assert_all_white(&ctx, encoder, &out_texture).await;
}

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_OVER_ICB_WORKGROUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(run_multi_draw_indirect_over_icb_workgroup);

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_FIRST_VERTEX_AND_INSTANCE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                .features(wgpu::Features::INDIRECT_FIRST_INSTANCE)
                .limits(wgpu::Limits::downlevel_defaults()),
        )
        .run_async(run_multi_draw_indirect_first_vertex_and_instance);

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_MIXED_SEQUENCE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(run_multi_draw_indirect_mixed_sequence);

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_WITH_BIND_GROUPS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(run_multi_draw_indirect_with_bind_groups);

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT_U16: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(run_multi_draw_indexed_indirect_u16);

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT_POSITIVE_BASE_VERTEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                .limits(wgpu::Limits::downlevel_defaults()),
        )
        .run_async(|ctx| run_multi_draw_indexed_indirect_base_vertex(ctx, 4));

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT_NEGATIVE_BASE_VERTEX: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                .limits(wgpu::Limits::downlevel_defaults()),
        )
        .run_async(|ctx| run_multi_draw_indexed_indirect_base_vertex(ctx, -4));

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_COUNT_READBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .features(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| run_multi_draw_indirect_count_readback(ctx, false));

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT_COUNT_READBACK: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                .features(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
                .limits(wgpu::Limits::downlevel_defaults()),
        )
        .run_async(|ctx| run_multi_draw_indirect_count_readback(ctx, true));

async fn run_gpu_generated_multi_draw_test(ctx: TestingContext, indexed: bool) {
    let draw_count = ICB_MULTI_DRAW_TEST_COUNT as u32;
    let indirect_stride = if indexed {
        size_of::<wgpu::util::DrawIndexedIndirectArgs>() as u64
    } else {
        size_of::<wgpu::util::DrawIndirectArgs>() as u64
    };
    let indirect_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(draw_count) * indirect_stride,
        usage: wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let args_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &vec![0; (u64::from(draw_count) * indirect_stride) as usize],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let draw_shader_src = "
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 32>;

        @compute @workgroup_size(1)
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
            if (id.x != 0u) {
                return;
            }
            args[0] = 3u;
            args[1] = 1u;
            args[2] = 0u;
            args[3] = 0u;
            args[4] = 3u;
            args[5] = 1u;
            args[6] = 3u;
            args[7] = 0u;
        }
    ";
    let indexed_draw_shader_src = "
        @group(0) @binding(0) var<storage, read_write> args: array<u32, 40>;

        @compute @workgroup_size(1)
        fn cs_main(@builtin(global_invocation_id) id: vec3u) {
            if (id.x != 0u) {
                return;
            }
            args[0] = 3u;
            args[1] = 1u;
            args[2] = 0u;
            args[3] = 0u;
            args[4] = 0u;
            args[5] = 3u;
            args[6] = 1u;
            args[7] = 3u;
            args[8] = 0u;
            args[9] = 0u;
        }
    ";
    let (compute_pipeline, bind_group) = create_gpu_generated_args_pipeline(
        &ctx,
        if indexed {
            indexed_draw_shader_src
        } else {
            draw_shader_src
        },
        &args_buffer,
        u64::from(draw_count) * indirect_stride,
    );
    let (render_pipeline, vertex_buffer, index_buffer) =
        create_indirect_render_pipeline(&ctx, indexed);

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let indirect_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(draw_count) * indirect_stride,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut expected_draw_args = vec![0; (u64::from(draw_count) * indirect_stride / 4) as usize];
    if indexed {
        expected_draw_args[..10].copy_from_slice(&[3, 1, 0, 0, 0, 3, 1, 3, 0, 0]);
    } else {
        expected_draw_args[..8].copy_from_slice(&[3, 1, 0, 0, 3, 1, 3, 0]);
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &args_buffer,
        0,
        &indirect_buffer,
        0,
        u64::from(draw_count) * indirect_stride,
    );
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&render_pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        if let Some(index_buffer) = index_buffer.as_ref() {
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.multi_draw_indexed_indirect(&indirect_buffer, 0, draw_count);
        } else {
            rpass.multi_draw_indirect(&indirect_buffer, 0, draw_count);
        }
    }

    encoder.copy_buffer_to_buffer(
        &indirect_buffer,
        0,
        &indirect_readback_buffer,
        0,
        u64::from(draw_count) * indirect_stride,
    );

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    let indirect_slice = indirect_readback_buffer.slice(..);
    indirect_slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let indirect_data = indirect_slice.get_mapped_range().unwrap();
    assert_eq!(
        bytemuck::cast_slice::<u8, u32>(&indirect_data),
        expected_draw_args.as_slice()
    );

    let data = slice.get_mapped_range().unwrap();
    let first_bad_pixel = data.chunks_exact(4).position(|rgba| rgba != [u8::MAX; 4]);
    if let Some(first_bad_pixel) = first_bad_pixel {
        let non_zero_pixels = data.chunks_exact(4).filter(|rgba| *rgba != [0; 4]).count();
        eprintln!(
            "first_bad_pixel={first_bad_pixel} rgba={:?} non_zero_pixels={non_zero_pixels}",
            &data[first_bad_pixel * 4..first_bad_pixel * 4 + 4]
        );
    }
    assert_eq!(first_bad_pixel, None);
}

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT_GPU_GENERATED_ARGS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
    )
    .run_async(|ctx| run_gpu_generated_multi_draw_test(ctx, false));

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT_GPU_GENERATED_ARGS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .test_features_limits()
                .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
        )
        .run_async(|ctx| run_gpu_generated_multi_draw_test(ctx, true));

macro_rules! make_test {
    ($name:ident, $test_data:expr) => {
        make_test!($name, $test_data, false, wgpu::Features::empty());
    };
    ($name:ident, $test_data:expr, $features:expr) => {
        make_test!($name, $test_data, false, $features);
    };
    ($name:ident, $test_data:expr, $expect_noop:expr, $features:expr) => {
        #[apply(gpu_test!)]
        static $name: GpuTestConfiguration = GpuTestConfiguration::new()
            .parameters({
                let params = TestParameters::default()
                    .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                    .features($features)
                    .limits(wgpu::Limits::downlevel_defaults());

                if $expect_noop {
                    params.enable_noop()
                } else {
                    params
                }
            })
            .run_async(|ctx| run_test(ctx, $test_data, $expect_noop));
    };
}
macro_rules! make_failing_test {
    ($name:ident, $test_data:expr) => {
        make_test!($name, $test_data, true, wgpu::Features::empty());
    };
    ($name:ident, $test_data:expr, $features:expr) => {
        make_test!($name, $test_data, true, $features);
    };
}

fn get_draw_test_data(first_vertex: u32, vertex_count: u32) -> TestData {
    let vertex_buffer_content = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    TestData {
        kind: Kind::NonIndexed {
            vertex_buffer_content,
            first_vertex,
            vertex_count,
        },
        instanced: None,
    }
}

make_test!(DRAW, get_draw_test_data(0, 6));
make_failing_test!(DRAW_OOB_START, get_draw_test_data(1, 6));
make_failing_test!(DRAW_OOB_COUNT, get_draw_test_data(0, 7));

fn get_instanced_draw_test_data(
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> TestData {
    let vertex_buffer_content = &[
        // Triangle 1
        -0.5, -0.5, // Bottom left
        0.5, 0.5, // Top right
        -0.5, 0.5, // Top left
        // Triangle 2
        -0.5, -0.5, // Bottom left
        0.5, -0.5, // Bottom right
        0.5, 0.5, // Top right
    ];
    let instance_buffer_content = &[
        -0.5, -0.5, // Move quad to bottom left
        0.5, 0.5, // Move quad to top right
        -0.5, 0.5, // Move quad to top left
        0.5, -0.5, // Move quad to bottom right
    ];
    TestData {
        kind: Kind::NonIndexed {
            vertex_buffer_content,
            first_vertex,
            vertex_count,
        },
        instanced: Some(Instanced {
            instance_buffer_content,
            first_instance,
            instance_count,
        }),
    }
}

make_test!(INSTANCED_DRAW, get_instanced_draw_test_data(0, 6, 0, 4));
make_failing_test!(
    INSTANCED_DRAW_OOB_START,
    get_instanced_draw_test_data(1, 6, 0, 4)
);
make_failing_test!(
    INSTANCED_DRAW_OOB_COUNT,
    get_instanced_draw_test_data(0, 7, 0, 4)
);
make_failing_test!(
    INSTANCED_DRAW_OOB_INSTANCE_START,
    get_instanced_draw_test_data(0, 6, 1, 4),
    wgpu::Features::INDIRECT_FIRST_INSTANCE
);
make_failing_test!(
    INSTANCED_DRAW_OOB_INSTANCE_COUNT,
    get_instanced_draw_test_data(0, 6, 0, 5)
);

fn get_instanced_draw_with_non_zero_first_instance_test_data() -> TestData {
    let vertex_buffer_content = &[
        // Triangle 1
        -0.5, -0.5, // Bottom left
        0.5, 0.5, // Top right
        -0.5, 0.5, // Top left
        // Triangle 2
        -0.5, -0.5, // Bottom left
        0.5, -0.5, // Bottom right
        0.5, 0.5, // Top right
    ];
    let instance_buffer_content = &[
        10.0, 10.0, // unused
        -0.5, -0.5, // Move quad to bottom left
        0.5, 0.5, // Move quad to top right
        -0.5, 0.5, // Move quad to top left
        0.5, -0.5, // Move quad to bottom right
    ];
    TestData {
        kind: Kind::NonIndexed {
            vertex_buffer_content,
            first_vertex: 0,
            vertex_count: 6,
        },
        instanced: Some(Instanced {
            instance_buffer_content,
            first_instance: 1,
            instance_count: 4,
        }),
    }
}

make_test!(
    INSTANCED_DRAW_WITH_NON_ZERO_FIRST_INSTANCE,
    get_instanced_draw_with_non_zero_first_instance_test_data(),
    wgpu::Features::INDIRECT_FIRST_INSTANCE
);
make_failing_test!(
    INSTANCED_DRAW_WITH_NON_ZERO_FIRST_INSTANCE_MISSING_FEATURE,
    get_instanced_draw_with_non_zero_first_instance_test_data()
);

fn get_indexed_draw_test_data(first_index: u32, index_count: u32) -> TestData {
    let vertex_buffer_content = &[
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        1.0, -1.0, // Bottom right
    ];
    let index_buffer_content = &[
        0, 1, 2, // Triangle 1
        0, 3, 1, // Triangle 2
    ];
    TestData {
        kind: Kind::Indexed {
            vertex_buffer_content,
            index_buffer_content,
            first_index,
            index_count,
        },
        instanced: None,
    }
}

make_test!(INDEXED_DRAW, get_indexed_draw_test_data(0, 6));
make_failing_test!(INDEXED_DRAW_OOB_START, get_indexed_draw_test_data(1, 6));
make_failing_test!(INDEXED_DRAW_OOB_COUNT, get_indexed_draw_test_data(0, 7));

fn get_instanced_indexed_draw_test_data(
    first_index: u32,
    index_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> TestData {
    let vertex_buffer_content = &[
        -0.5, -0.5, // Bottom left
        0.5, 0.5, // Top right
        -0.5, 0.5, // Top left
        0.5, -0.5, // Bottom right
    ];
    let index_buffer_content = &[
        0, 1, 2, // Triangle 1
        0, 3, 1, // Triangle 2
    ];
    let instance_buffer_content = &[
        -0.5, -0.5, // Move quad to bottom left
        0.5, 0.5, // Move quad to top right
        -0.5, 0.5, // Move quad to top left
        0.5, -0.5, // Move quad to bottom right
    ];
    TestData {
        kind: Kind::Indexed {
            vertex_buffer_content,
            index_buffer_content,
            first_index,
            index_count,
        },
        instanced: Some(Instanced {
            instance_buffer_content,
            first_instance,
            instance_count,
        }),
    }
}

make_test!(
    INSTANCED_INDEXED_DRAW,
    get_instanced_indexed_draw_test_data(0, 6, 0, 4)
);
make_failing_test!(
    INSTANCED_INDEXED_DRAW_OOB_START,
    get_instanced_indexed_draw_test_data(1, 6, 0, 4)
);
make_failing_test!(
    INSTANCED_INDEXED_DRAW_OOB_COUNT,
    get_instanced_indexed_draw_test_data(0, 7, 0, 4)
);
make_failing_test!(
    INSTANCED_INDEXED_DRAW_OOB_INSTANCE_START,
    get_instanced_indexed_draw_test_data(0, 6, 1, 4),
    wgpu::Features::INDIRECT_FIRST_INSTANCE
);
make_failing_test!(
    INSTANCED_INDEXED_DRAW_OOB_INSTANCE_COUNT,
    get_instanced_indexed_draw_test_data(0, 6, 0, 5)
);

#[apply(gpu_test!)]
static INDIRECT_BUFFER_OFFSETS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .features(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(indirect_buffer_offsets);

/// Tests that indirect draw calls work properly with offsets that straddle 16 byte boundaries (size of DrawIndirectArgs).
async fn indirect_buffer_offsets(ctx: TestingContext) {
    // The first 2 draws are successful, the third one is not.
    let indirect_args_offsets = [0, 4, 8];

    let indirect_args = [
        //     1st draw       | 2nd draw       | 3rd draw
        9,  // vertex_count   |                |
        9,  // instance_count | vertex_count   |
        1,  // first_vertex   | instance_count | vertex_count
        0,  // first_instance | first_vertex   | instance_count
        9,  //                | first_instance | first_vertex
        10, //                |                | first_instance
    ];

    // 1st draw (first_vertex: 1): ◤ ◢ ◢
    // 2nd draw (first_vertex: 0): ◤ ◣ ◢
    let vertex_buffer_content = [
        -0.5, 0.5, // Top left
        // Triangle 1
        -0.5, -0.5, // Bottom left
        0.5, 0.5, // Top right
        -0.5, 0.5, // Top left
        // Triangle 2
        -0.5, -0.5, // Bottom left
        0.5, -0.5, // Bottom right
        0.5, 0.5, // Top right
        // Triangle 3 (same as Triangle 2)
        -0.5, -0.5, // Bottom left
        0.5, -0.5, // Bottom right
        0.5, 0.5, // Top right
    ];
    #[rustfmt::skip]
    let instance_buffer_content = [
        // Move quad to top left (for 1st draw):
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        -0.5, 0.5,
        // Move quad to top right (for 2nd draw):
        0.5, 0.5,
    ];

    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(&vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let instance_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(&instance_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let shader_src = "
        @vertex
        fn vs_main(@location(0) position: vec2f, @location(1) position_offset: vec2f) -> @builtin(position) vec4f {
            return vec4f(position + position_offset, 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4f {
            return vec4f(1.0);
        }
    ";

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &vertex_attr_array![1 => Float32x2],
                }),
            ],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let indirect_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<u32, u8>(&indirect_args),
        usage: wgpu::BufferUsages::INDIRECT,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        rpass.set_vertex_buffer(1, instance_buffer.slice(..));
        for offset in indirect_args_offsets {
            rpass.draw_indirect(&indirect_buffer, offset);
        }
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let half = data.len() / 2;
    let succeeded =
        data[..half].iter().all(|b| *b == u8::MAX) && data[half..].iter().all(|b| *b == 0);
    assert!(succeeded);
}

#[apply(gpu_test!)]
static MULTI_DRAW_INDEXED_INDIRECT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| run_test_inner(ctx, get_indexed_draw_test_data(0, 6), false, true));

#[apply(gpu_test!)]
static MULTI_DRAW_INDIRECT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| run_test_inner(ctx, get_draw_test_data(0, 6), false, true));
