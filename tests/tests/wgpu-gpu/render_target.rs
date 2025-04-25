use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static DRAW_TO_2D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, wgpu::TextureViewDimension::D2, false));

#[gpu_test]
static DRAW_TO_2D_ARRAY_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, wgpu::TextureViewDimension::D2Array, false));

#[gpu_test]
static RESOLVE_TO_2D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, wgpu::TextureViewDimension::D2, true));

#[gpu_test]
static RESOLVE_TO_2D_ARRAY_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| run_test(ctx, wgpu::TextureViewDimension::D2Array, true));

async fn run_test(
    ctx: TestingContext,
    view_dimension: wgpu::TextureViewDimension,
    multisample: bool,
) {
    let vertex_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let shader_src = "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
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
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2],
            }],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: if multisample { 4 } else { 1 },
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
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

    const SIZE: u32 = 512;
    const LAYERS: u32 = 2;
    const MIPS: u32 = 2;
    const fn size_for_mips(mips: u32) -> u64 {
        let mut out: u64 = 0;
        let mut mip = 0;
        while mip < mips {
            let size = SIZE as u64 >> mip;
            out += size * size;

            mip += 1;
        }
        out * LAYERS as u64
    }

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: LAYERS,
        },
        mip_level_count: MIPS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_for_mips(MIPS),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for mip in 0..MIPS {
        let ms_texture_view = if multisample {
            let ms_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE >> mip,
                    height: SIZE >> mip,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 4,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let ms_texture_view = ms_texture.create_view(&wgpu::TextureViewDescriptor::default());
            Some(ms_texture_view)
        } else {
            None
        };
        for layer in 0..LAYERS {
            let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: Some(wgpu::TextureFormat::R8Unorm),
                dimension: Some(view_dimension),
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
            });
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ms_texture_view.as_ref().unwrap_or(&out_texture_view),
                    depth_slice: None,
                    resolve_target: multisample.then_some(&out_texture_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: if multisample {
                            wgpu::StoreOp::Discard
                        } else {
                            wgpu::StoreOp::Store
                        },
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }

    for mip in 0..MIPS {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: size_for_mips(mip),
                    bytes_per_row: Some(SIZE >> mip),
                    rows_per_image: Some(SIZE >> mip),
                },
            },
            wgpu::Extent3d {
                width: SIZE >> mip,
                height: SIZE >> mip,
                depth_or_array_layers: LAYERS,
            },
        );
    }

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

    let data = slice.get_mapped_range();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}

#[gpu_test]
static DRAW_TO_3D_VIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().limits(wgpu::Limits {
        max_texture_dimension_3d: 512,
        ..wgpu::Limits::downlevel_webgl2_defaults()
    }))
    .run_async(run_test_3d);

async fn run_test_3d(ctx: TestingContext) {
    let vertex_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(vertex_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let shader_src = "
            @vertex
            fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
                return vec4f(position, 0.0, 1.0);
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
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attr_array![0 => Float32x2],
            }],
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

    const SIZE: u32 = 512;
    const DEPTH: u32 = 2;
    const MIPS: u32 = 2;
    const fn size_for_mips(mips: u32) -> u64 {
        let mut out: u64 = 0;
        let mut mip = 0;
        while mip < mips {
            let size = SIZE as u64 >> mip;
            let z = DEPTH as u64 >> mip;
            out += size * size * z;

            mip += 1;
        }
        out
    }

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: DEPTH,
        },
        mip_level_count: MIPS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size_for_mips(MIPS),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for mip in 0..MIPS {
        let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        });
        for layer in 0..DEPTH >> mip {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_texture_view,
                    depth_slice: Some(layer),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..6, 0..1);
        }
    }

    for mip in 0..MIPS {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out_texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: size_for_mips(mip),
                    bytes_per_row: Some(SIZE >> mip),
                    rows_per_image: Some(SIZE >> mip),
                },
            },
            wgpu::Extent3d {
                width: SIZE >> mip,
                height: SIZE >> mip,
                depth_or_array_layers: DEPTH >> mip,
            },
        );
    }

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait()).await.unwrap();

    let data = slice.get_mapped_range();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}
