use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

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
    let mut vertex_buffer_layouts = Vec::new();
    vertex_buffer_layouts.push(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &vertex_attr_array![0 => Float32x2],
    });
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
        vertex_buffer_layouts.push(wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &vertex_attr_array![1 => Float32x2],
        });
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
        multiview: None,
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
        });

        rpass.set_pipeline(&pipeline);
        rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
        if let Some(ref instance_buffer) = instance_buffer {
            rpass.set_vertex_buffer(1, instance_buffer.slice(..));
        }
        if let Some(ref index_buffer) = index_buffer {
            rpass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        }
        for draw_index in 0..draws {
            if index_buffer.is_some() {
                let offset = pass_index * draw_index * 20;
                rpass.draw_indexed_indirect(&indirect_buffer, offset);
                rpass.draw_indexed_indirect(&indirect_buffer2, offset);
            } else {
                let offset = pass_index * draw_index * 20;
                rpass.draw_indirect(&indirect_buffer, offset);
                rpass.draw_indirect(&indirect_buffer2, offset);
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

    ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

    let data = slice.get_mapped_range();
    let succeeded = if expect_noop {
        data.iter().all(|b| *b == 0)
    } else {
        data.iter().all(|b| *b == u8::MAX)
    };
    assert!(succeeded);
}

macro_rules! make_test {
    ($name:ident, $test_data:expr) => {
        make_test!($name, $test_data, false, wgpu::Features::empty());
    };
    ($name:ident, $test_data:expr, $features:expr) => {
        make_test!($name, $test_data, false, $features);
    };
    ($name:ident, $test_data:expr, $expect_noop:expr, $features:expr) => {
        #[gpu_test]
        static $name: GpuTestConfiguration = GpuTestConfiguration::new()
            .parameters(
                TestParameters::default()
                    .downlevel_flags(wgpu::DownlevelFlags::INDIRECT_EXECUTION)
                    .features($features)
                    .limits(wgpu::Limits::downlevel_defaults()),
            )
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

#[gpu_test]
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
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x2],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &vertex_attr_array![1 => Float32x2],
                },
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
        multiview: None,
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

    ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

    let data = slice.get_mapped_range();
    let half = data.len() / 2;
    let succeeded =
        data[..half].iter().all(|b| *b == u8::MAX) && data[half..].iter().all(|b| *b == 0);
    assert!(succeeded);
}
