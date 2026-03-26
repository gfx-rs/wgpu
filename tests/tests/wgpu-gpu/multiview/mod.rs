use std::num::NonZero;

use wgpu::{Features, Limits};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DRAW_MULTIVIEW_SINGLE);
    vec.push(DRAW_MULTIVIEW);
    vec.push(DRAW_MULTIVIEW_NONCONTIGUOUS);
    vec.push(DRAW_MULTIVIEW_MULTISAMPLE);
}

#[gpu_test]
static DRAW_MULTIVIEW_SINGLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 1,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b1, 1));

#[gpu_test]
static DRAW_MULTIVIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b11, 1));

#[gpu_test]
static DRAW_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters {
        required_features: Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW,
        required_limits: Limits {
            max_multiview_view_count: 4,
            ..Limits::defaults()
        },
        // https://github.com/gfx-rs/wgpu/issues/9184 and https://github.com/gfx-rs/wgpu/issues/9187
        failures: wgpu_test::FailureCase::mac_vulkan(|case| {
            case.panic(
                "assertion `left == right` failed: Expected 0\n  left: Some(255)\n right: None",
            )
        }),
        ..Default::default()
    })
    .run_async(|ctx| run_test(ctx, 0b1001, 1));

#[gpu_test]
static DRAW_MULTIVIEW_MULTISAMPLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::MULTISAMPLE_ARRAY)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b11, 4));

async fn run_test(ctx: TestingContext, layer_mask: u32, sample_count: u32) {
    let num_layers = 32 - layer_mask.leading_zeros();

    let shader_src = include_str!("shader.wgsl");

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        vertex: wgpu::VertexState {
            buffers: &[],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
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
        multiview_mask: NonZero::new(layer_mask),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        layout: None,
        depth_stencil: None,
        cache: None,
    };

    const TEXTURE_SIZE: u32 = 256;
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);

    let texture_desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: 32 - layer_mask.leading_zeros(),
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    let texture = ctx.device.create_texture(&texture_desc);
    let texture_view_desc = wgpu::TextureViewDescriptor {
        label: None,
        format: Some(wgpu::TextureFormat::R8Unorm),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: Some(num_layers),
    };
    let entire_texture_view = texture.create_view(&texture_view_desc);

    let (resolve_texture, resolve_texture_view) = if sample_count != 1 {
        let mut texture_desc = texture_desc.clone();
        texture_desc.sample_count = 1;

        let resolve_texture = ctx.device.create_texture(&texture_desc);
        let resolve_texture_view = resolve_texture.create_view(&texture_view_desc);

        (Some(resolve_texture), Some(resolve_texture_view))
    } else {
        (None, None)
    };

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: TEXTURE_SIZE as u64 * TEXTURE_SIZE as u64 * num_layers as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let clear_color = 0.0;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &entire_texture_view,
                depth_slice: None,
                resolve_target: resolve_texture_view.as_ref(),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: NonZero::new(layer_mask),
        });
        rpass.set_pipeline(&pipeline);
        rpass.draw(0..6, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: resolve_texture.as_ref().unwrap_or(&texture),
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEXTURE_SIZE),
                rows_per_image: Some(TEXTURE_SIZE),
            },
        },
        wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: num_layers,
        },
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range();
    let each_texture_size = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
    assert_eq!(data.len(), each_texture_size * num_layers as usize);
    for view_idx in 0..num_layers as usize {
        let target_value = if (layer_mask & (1 << view_idx)) != 0 {
            (32 + 64 * view_idx) as u8
        } else {
            (clear_color * 255.0) as u8
        };
        // Some metal devices automatically initialize stuff to 255, so I decided to use 128 instead of that
        let failed_value = data[each_texture_size * view_idx..each_texture_size * (view_idx + 1)]
            .iter()
            .copied()
            .find(|b| b.abs_diff(target_value) > 1);
        assert_eq!(failed_value, None, "Expected {target_value}");
    }
}
