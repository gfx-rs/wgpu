use std::{fmt::Write, num::NonZeroU64};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

pub fn all_tests(vec: &mut Vec<wgpu_test::GpuTestInitializer>) {
    vec.push(PIPELINE_CACHE);
    vec.push(PIPELINE_CACHE_RENDER);
}

/// We want to test that using a pipeline cache doesn't cause failure
///
/// It would be nice if we could also assert that reusing a pipeline cache would make compilation
/// be faster however, some drivers use a fallback pipeline cache, which makes this inconsistent
/// (both intra- and inter-run).
#[gpu_test]
static PIPELINE_CACHE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::PIPELINE_CACHE),
    )
    .run_async(pipeline_cache_test);

/// Set to a higher value if adding a timing based assertion. This is otherwise fast to compile
const ARRAY_SIZE: u64 = 256;

/// Create a shader which should be slow-ish to compile
fn shader() -> String {
    let mut body = String::new();
    for idx in 0..ARRAY_SIZE {
        // "Safety": There will only be a single workgroup, and a single thread in that workgroup
        writeln!(body, "    output[{idx}] = {idx}u;")
            .expect("`u64::fmt` and `String::write_fmt` are infallible");
    }

    format!(
        r#"
        @group(0) @binding(0)
        var<storage, read_write> output: array<u32>;

        @compute @workgroup_size(1)
        fn main() {{
        {body}
        }}
        "#,
    )
}

async fn pipeline_cache_test(ctx: TestingContext) {
    let shader = shader();
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(ARRAY_SIZE * 4),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: ARRAY_SIZE * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let first_cache_data;
    {
        let first_cache = unsafe {
            ctx.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("pipeline_cache"),
                    data: None,
                    fallback: false,
                })
        };
        let first_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: Some(&pipeline_layout),
                module: &sm,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: Some(&first_cache),
            });
        validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;
        first_cache_data = first_cache.get_data();
    }
    assert!(first_cache_data.is_some());

    let second_cache = unsafe {
        ctx.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("pipeline_cache"),
                data: first_cache_data.as_deref(),
                fallback: false,
            })
    };
    let first_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: Some(&second_cache),
        });
    validate_pipeline(&ctx, first_pipeline, &bind_group, &gpu_buffer, &cpu_buffer).await;

    // Ideally, we could assert here that the second compilation was faster than the first
    // However, that doesn't actually work, because drivers have their own internal caches.
    // This does work on my machine if I set `MESA_DISABLE_PIPELINE_CACHE=1`
    // before running the test; but of course that is not a realistic scenario
}

async fn validate_pipeline(
    ctx: &TestingContext,
    pipeline: wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    gpu_buffer: &wgpu::Buffer,
    cpu_buffer: &wgpu::Buffer,
) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, Some(bind_group), &[]);

        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(gpu_buffer, 0, cpu_buffer, 0, ARRAY_SIZE * 4);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = cpu_buffer.slice(..).get_mapped_range().unwrap();

    let arrays: &[u32] = bytemuck::cast_slice(&data);

    assert_eq!(arrays.len(), ARRAY_SIZE as usize);
    for (idx, value) in arrays.iter().copied().enumerate() {
        assert_eq!(value as usize, idx);
    }
    drop(data);
    cpu_buffer.unmap();
}

/// Render-pipeline analogue of [`PIPELINE_CACHE`]. The test above only covers
/// compute pipelines; backends thread the pipeline cache through render-pipeline
/// creation on a separate path (e.g. the Metal backend sets `binaryArchives` on
/// the render descriptor and grows the archive with `addRenderPipelineFunctions`),
/// so exercise that here too: create a render pipeline with a cache, draw with it,
/// `get_data`, then reseed a second cache from that data and draw again.
#[gpu_test]
static PIPELINE_CACHE_RENDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::PIPELINE_CACHE),
    )
    .run_async(pipeline_cache_render_test);

/// 64 wide so the texture→buffer copy's bytes-per-row (64 * 4 = 256) is already
/// `COPY_BYTES_PER_ROW_ALIGNMENT`-aligned (no row padding to account for).
const RENDER_WIDTH: u32 = 64;
const RENDER_HEIGHT: u32 = 64;
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// The constant colour the fragment shader writes, as stored in `Rgba8Unorm`
/// (linear — no sRGB encoding): `round([0.25, 0.5, 0.75, 1.0] * 255)`.
const RENDER_EXPECTED: [u8; 4] = [64, 128, 191, 255];

fn render_shader() -> &'static str {
    r#"
        @vertex
        fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
            // Oversized triangle that covers the whole framebuffer.
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -3.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 3.0,  1.0),
            );
            return vec4<f32>(positions[idx], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(0.25, 0.5, 0.75, 1.0);
        }
    "#
}

async fn pipeline_cache_render_test(ctx: TestingContext) {
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render shader"),
            source: wgpu::ShaderSource::Wgsl(render_shader().into()),
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline_layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

    let make_pipeline = |cache: &wgpu::PipelineCache| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    buffers: &[],
                    module: &sm,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &sm,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: RENDER_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: Some(cache),
            })
    };

    let first_cache_data;
    {
        let first_cache = unsafe {
            ctx.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("render pipeline_cache"),
                    data: None,
                    fallback: false,
                })
        };
        let pipeline = make_pipeline(&first_cache);
        validate_render_pipeline(&ctx, &pipeline).await;
        first_cache_data = first_cache.get_data();
    }
    assert!(first_cache_data.is_some());

    let second_cache = unsafe {
        ctx.device
            .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("render pipeline_cache"),
                data: first_cache_data.as_deref(),
                fallback: false,
            })
    };
    let pipeline = make_pipeline(&second_cache);
    validate_render_pipeline(&ctx, &pipeline).await;
}

async fn validate_render_pipeline(ctx: &TestingContext, pipeline: &wgpu::RenderPipeline) {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render target"),
        size: wgpu::Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RENDER_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("render readback"),
        size: u64::from(RENDER_WIDTH * RENDER_HEIGHT * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                resolve_target: None,
                view: &view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(pipeline);
        rpass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &cpu_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(RENDER_WIDTH * 4),
                rows_per_image: Some(RENDER_HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = cpu_buffer.slice(..).get_mapped_range().unwrap();
    // Every pixel should be the fragment shader's constant colour (±1 for rounding).
    for px in data.chunks_exact(4) {
        for (channel, (&got, &want)) in px.iter().zip(RENDER_EXPECTED.iter()).enumerate() {
            assert!(
                i16::from(got).abs_diff(i16::from(want)) <= 1,
                "channel {channel}: got {got}, expected ~{want}"
            );
        }
    }
    drop(data);
    cpu_buffer.unmap();
}
