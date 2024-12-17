//! Tests that vertex formats pass through to vertex shaders accurately.

use std::{mem::size_of_val, num::NonZeroU64};

use wgpu::util::{BufferInitDescriptor, DeviceExt};

use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

#[derive(Debug, Copy, Clone)]
enum TestCase {
    UnormsAndSnorms,
    UintsAndSintsSmall,
    UintsBig,
    SintsBig,
    Floats,
    Unorm1010102,
}

struct Test<'a> {
    case: TestCase,
    entry_point: &'a str,
    attributes: &'a [wgt::VertexAttribute],
    input: &'a [u8],
    checksums: &'a [f32],
}

async fn vertex_formats_all(ctx: TestingContext) {
    let attributes_block_0 = &wgpu::vertex_attr_array![
        0 => Unorm8x4,
        1 => Unorm16x2,
        2 => Unorm16x4,
        3 => Snorm8x4,
        4 => Snorm16x2,
        5 => Snorm16x4,
        6 => Unorm8x2,
        7 => Snorm8x2,
    ];

    let attributes_block_1 = &wgpu::vertex_attr_array![
        0 => Uint8x4,
        1 => Uint16x2,
        2 => Uint16x4,
        3 => Sint8x4,
        4 => Sint16x2,
        5 => Sint16x4,
        6 => Uint8x2,
        7 => Sint8x2,
    ];

    let attributes_block_2 = &wgpu::vertex_attr_array![
        0 => Uint32,
        1 => Uint32x2,
        2 => Uint32x3,
        3 => Uint32x4,
    ];

    let attributes_block_3 = &wgpu::vertex_attr_array![
        0 => Sint32,
        1 => Sint32x2,
        2 => Sint32x3,
        3 => Sint32x4,
    ];

    let attributes_block_4 = &wgpu::vertex_attr_array![
        0 => Float32,
        1 => Float32x2,
        2 => Float32x3,
        3 => Float32x4,
        4 => Float16x2,
        5 => Float16x4,
    ];

    let tests = vec![
        Test {
            case: TestCase::UnormsAndSnorms,
            entry_point: "vertex_block_0",
            attributes: attributes_block_0,
            input: &[
                128u8, 128u8, 128u8, 128u8, // Unorm8x4 (0.5, 0.5, 0.5, 0.5)
                0u8, 128u8, 0u8, 128u8, // Unorm16x2 (0.5, 0.5)
                0u8, 64u8, 0u8, 64u8, 0u8, 64u8, 0u8,
                64u8, // Unorm16x4 (0.25, 0.25, 0.25, 0.25)
                127u8, 127u8, 127u8, 127u8, // Snorm8x4 (1, 1, 1, 1)
                0u8, 128u8, 0u8, 128u8, // Snorm16x2 (-1, -1)
                255u8, 127u8, 255u8, 127u8, 255u8, 127u8, 255u8,
                127u8, // Snorm16x4 (1, 1, 1, 1)
                255u8, 255u8, // Unorm8x2 (1, 1)
                128u8, 128u8, // Snorm8x2 (-1, -1)
            ],
            checksums: &[0.0, 0.0, 6.0, 4.0, 0.0, 0.0],
        },
        Test {
            case: TestCase::UintsAndSintsSmall,
            entry_point: "vertex_block_1",
            attributes: attributes_block_1,
            input: &[
                4u8, 8u8, 16u8, 32u8, // Uint8x4 (4, 8, 16, 32)
                64u8, 0u8, 128u8, 0u8, // Uint16x2 (64, 128)
                0u8, 1u8, 0u8, 2u8, 0u8, 4u8, 0u8, 8u8, // Uint16x4 (256, 512, 1024, 2048)
                127u8, 127u8, 2u8, 0u8, // Sint8x4 (127, 127, 2, 0)
                255u8, 255u8, 1u8, 0u8, // Sint16x2 (-1, 1)
                128u8, 255u8, 128u8, 255u8, 0u8, 1u8, 240u8,
                255u8, // Sint16x4 (-128, -128, 256, -16)
                1u8, 2u8, // Uint8x2 (1, 2)
                128u8, 128u8, // Sint8x2 (-128, -128)
            ],
            checksums: &[4095.0, -16.0, 0.0, 0.0, 0.0, 0.0],
        },
        Test {
            case: TestCase::UintsBig,
            entry_point: "vertex_block_2",
            attributes: attributes_block_2,
            input: &[
                1u8, 0u8, 0u8, 0u8, // Uint32x2 (1)
                2u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, // Uint32x2 (2, 4)
                8u8, 0u8, 0u8, 0u8, 16u8, 0u8, 0u8, 0u8, 32u8, 0u8, 0u8,
                0u8, // Uint32x3 (8, 16, 32)
                64u8, 0u8, 0u8, 0u8, 128u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 2u8, 0u8,
                0u8, // Uint32x4 (64, 128, 256, 512)
            ],
            checksums: &[1023.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        Test {
            case: TestCase::SintsBig,
            entry_point: "vertex_block_3",
            attributes: attributes_block_3,
            input: &[
                128u8, 255u8, 255u8, 255u8, // Sint32 (-128)
                120u8, 0u8, 0u8, 0u8, 8u8, 0u8, 0u8, 0u8, // Sint32x2 (120, 8)
                252u8, 255u8, 255u8, 255u8, 2u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8,
                0u8, // Sint32x3 (-4, 2, 2)
                24u8, 252u8, 255u8, 255u8, 88u8, 2u8, 0u8, 0u8, 44u8, 1u8, 0u8, 0u8, 99u8, 0u8,
                0u8, 0u8, // Sint32x4 (-1000, 600, 300, 99)
            ],
            checksums: &[0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        },
        Test {
            case: TestCase::Floats,
            entry_point: "vertex_block_4",
            attributes: attributes_block_4,
            input: &[
                0u8, 0u8, 0u8, 63u8, // Float32 (0.5)
                0u8, 0u8, 0u8, 191u8, 0u8, 0u8, 128u8, 64u8, // Float32x2 (-0.5, 4.0)
                0u8, 0u8, 0u8, 192u8, 0u8, 0u8, 204u8, 194u8, 0u8, 0u8, 200u8,
                66u8, // Float32x3 (-2.0, -102.0, 100.0)
                0u8, 0u8, 92u8, 66u8, 0u8, 0u8, 72u8, 194u8, 0u8, 0u8, 32u8, 65u8, 0u8, 0u8, 128u8,
                63u8, // Float32x4 (55.0, -50.0, 10.0, 1.0)
                0u8, 60u8, 72u8, 53u8, // Float16x2 (1.0, 0.33)
                72u8, 57u8, 0u8, 192u8, 0u8, 188u8, 0u8,
                184u8, // Float16x4 (0.66, -2.0, -1.0, -0.5)
            ],
            checksums: &[0.0, 0.0, 0.0, 0.0, -1.5, 16.0],
        },
    ];

    vertex_formats_common(ctx, &tests).await;
}

async fn vertex_formats_10_10_10_2(ctx: TestingContext) {
    let attributes_block_5 = &wgpu::vertex_attr_array![
        0 => Unorm10_10_10_2,
    ];

    let tests = vec![Test {
        case: TestCase::Unorm1010102,
        entry_point: "vertex_block_5",
        attributes: attributes_block_5,
        input: &[
            // We are aiming for rgba of (0.5, 0.5, 0.5, 0.66)
            // Packing   AA BB BBBB BBBB GGGG GGGG GG RR RRRR RRRR
            // Binary    10 10 0000 0000 1000 0000 00 10 0000 0000
            // Hex               A0        08         02        00
            // Decimal          160         8          2         0
            // unorm   0.66          0.5          0.5          0.5 = 2.16
            0u8, 2u8, 8u8, 160u8, // Unorm10_10_10_2
        ],
        checksums: &[0.0, 0.0, 2.16, 0.0, 0.0, 0.0],
    }];

    vertex_formats_common(ctx, &tests).await;
}

async fn vertex_formats_common(ctx: TestingContext, tests: &[Test<'_>]) {
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

    let mut failed = false;
    for test in tests {
        let buffer_input = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(test.input),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let pipeline_desc = wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&ppl),
            vertex: wgpu::VertexState {
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 0, // Calculate, please!
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: test.attributes,
                }],
                module: &shader,
                entry_point: Some(test.entry_point),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
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

        let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

        let expected = test.checksums;
        let buffer_size = (size_of_val(&expected[0]) * expected.len()) as u64;
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

        rpass.set_vertex_buffer(0, buffer_input.slice(..));
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bg, &[]);

        // Draw three vertices and no instance, which is enough to generate the
        // checksums.
        rpass.draw(0..3, 0..1);

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
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

        let case_name = format!("Case {:?}", test.case);

        // Calculate the difference between data and expected. Since the data is
        // a bunch of float checksums, we allow a fairly large epsilon, which helps
        // with the accumulation of float rounding errors.
        const EPSILON: f32 = 0.01;

        let mut deltas = data.iter().zip(expected.iter()).map(|(d, e)| (d - e).abs());
        if deltas.any(|x| x > EPSILON) {
            eprintln!("Failed: Got: {data:?} Expected: {expected:?} - {case_name}",);
            failed = true;
            continue;
        }

        eprintln!("Passed: {case_name}");
    }

    assert!(!failed);
}

#[gpu_test]
static VERTEX_FORMATS_ALL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_async(vertex_formats_all);

// Some backends can handle Unorm-10-10-2, but GL backends seem to throw this error:
// Validation Error: GL_INVALID_ENUM in glVertexAttribFormat(type = GL_UNSIGNED_INT_10_10_10_2)
#[gpu_test]
static VERTEX_FORMATS_10_10_10_2: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .expect_fail(FailureCase::backend(wgpu::Backends::GL))
            .test_features_limits()
            .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
    )
    .run_async(vertex_formats_10_10_10_2);
