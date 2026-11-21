use std::num::NonZero;

use wgpu::{Features, Limits};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DRAW_MULTIVIEW_SINGLE);
    vec.push(DRAW_MULTIVIEW);
    vec.push(DRAW_MULTIVIEW_NONCONTIGUOUS);
    vec.push(DRAW_MULTIVIEW_MULTISAMPLE);
    vec.push(DISCARD_MULTIVIEW);
    vec.push(DISCARD_MULTIVIEW_NONCONTIGUOUS);
    vec.push(DISCARD_MULTI_LAYER_NO_MASK);
    vec.push(CLEAR_STORE_MULTIVIEW_NONCONTIGUOUS);
    vec.push(CLEAR_STORE_MULTIVIEW_BASE_LAYER);
    vec.push(CLEAR_DEPTH_MULTIVIEW_NONCONTIGUOUS);
    vec.push(CLEAR_DEPTH_STENCIL_MULTIVIEW_NONCONTIGUOUS);
}

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
static DRAW_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters {
        required_features: Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW,
        required_limits: Limits {
            max_multiview_view_count: 4,
            ..Limits::defaults()
        },
        ..Default::default()
    })
    .run_async(|ctx| run_test(ctx, 0b1001, 1));

#[apply(gpu_test!)]
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

#[apply(gpu_test!)]
static DISCARD_MULTIVIEW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            })
            // Lavapipe with `LVP_POISON_MEMORY` crashes when a multiview pass discards a color
            // attachment: it uses the view index as the attachment index.
            .skip(FailureCase::backend_adapter(
                wgpu::Backends::VULKAN,
                "llvmpipe",
            )),
    )
    .run_async(|ctx| run_clear_test(ctx, NonZero::new(0b11), 0, 2, wgpu::StoreOp::Discard));

#[apply(gpu_test!)]
static DISCARD_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 4,
                ..Limits::defaults()
            })
            // Lavapipe with `LVP_POISON_MEMORY` crashes when a multiview pass discards a color
            // attachment: it uses the view index as the attachment index.
            .skip(FailureCase::backend_adapter(
                wgpu::Backends::VULKAN,
                "llvmpipe",
            )),
    )
    .run_async(|ctx| run_clear_test(ctx, NonZero::new(0b1001), 0, 4, wgpu::StoreOp::Discard));

#[apply(gpu_test!)]
static DISCARD_MULTI_LAYER_NO_MASK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_clear_test(ctx, None, 0, 2, wgpu::StoreOp::Discard));

#[apply(gpu_test!)]
static CLEAR_STORE_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 4,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_clear_test(ctx, NonZero::new(0b1001), 0, 4, wgpu::StoreOp::Store));

#[apply(gpu_test!)]
static CLEAR_STORE_MULTIVIEW_BASE_LAYER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 4,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_clear_test(ctx, NonZero::new(0b1001), 2, 4, wgpu::StoreOp::Store));

#[apply(gpu_test!)]
static CLEAR_DEPTH_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 4,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_clear_depth_test(ctx, wgpu::TextureFormat::Depth32Float));

#[apply(gpu_test!)]
static CLEAR_DEPTH_STENCIL_MULTIVIEW_NONCONTIGUOUS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(
                    Features::MULTIVIEW
                        | Features::SELECTIVE_MULTIVIEW
                        | Features::DEPTH32FLOAT_STENCIL8,
                )
                .limits(Limits {
                    max_multiview_view_count: 4,
                    ..Limits::defaults()
                }),
        )
        .run_async(|ctx| run_clear_depth_test(ctx, wgpu::TextureFormat::Depth32FloatStencil8));

/// Fills a texture of `base_layer + num_layers` layers with 128, then runs a pass
/// with `LoadOp::Clear(WHITE)` and `store_op` on a view of the layers from
/// `base_layer`. Checks that the rendered layers read back as 255 after `Store` or
/// 0 after `Discard`, and the other layers keep 128. With no mask, only the base
/// layer of the view is rendered.
async fn run_clear_test(
    ctx: TestingContext,
    layer_mask: Option<NonZero<u32>>,
    base_layer: u32,
    num_layers: u32,
    store_op: wgpu::StoreOp,
) {
    const TEXTURE_SIZE: u32 = 256;
    const FILL_VALUE: u8 = 128;

    let texture_layers = base_layer + num_layers;
    let size = wgpu::Extent3d {
        width: TEXTURE_SIZE,
        height: TEXTURE_SIZE,
        depth_or_array_layers: texture_layers,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(TEXTURE_SIZE),
        rows_per_image: Some(TEXTURE_SIZE),
    };
    let layer_size = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &vec![FILL_VALUE; layer_size * texture_layers as usize],
        layout,
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: base_layer,
        ..Default::default()
    });
    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (layer_size * texture_layers as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: store_op,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: layer_mask,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout,
        },
        size,
    );
    ctx.queue.submit([encoder.finish()]);

    let mask = layer_mask.map_or(1, NonZero::get) << base_layer;
    check_layers(&ctx, &readback_buffer, texture_layers, |layer| {
        if mask & (1 << layer) == 0 {
            FILL_VALUE
        } else if store_op == wgpu::StoreOp::Store {
            255
        } else {
            0
        }
    })
    .await;
}

/// Fills each of 4 depth layers with 0.5, then runs a depth-only pass with mask
/// 0b1001, `LoadOp::Clear(1.0)` and `StoreOp::Store`. Checks that layers 0 and 3
/// read back as 1.0 and layers 1 and 2 keep 0.5. The stencil aspect, if any, is
/// read-only.
async fn run_clear_depth_test(ctx: TestingContext, format: wgpu::TextureFormat) {
    const TEXTURE_SIZE: u32 = 64;
    const NUM_LAYERS: u32 = 4;
    const LAYER_MASK: u32 = 0b1001;

    let size = wgpu::Extent3d {
        width: TEXTURE_SIZE,
        height: TEXTURE_SIZE,
        depth_or_array_layers: NUM_LAYERS,
    };
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut clear_depth = |view: &wgpu::TextureView, value, multiview_mask| {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(value),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask,
        });
    };
    for layer in 0..NUM_LAYERS {
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        clear_depth(&view, 0.5, None);
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    clear_depth(&view, 1.0, NonZero::new(LAYER_MASK));

    let layer_size = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (layer_size * NUM_LAYERS as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEXTURE_SIZE * 4),
                rows_per_image: Some(TEXTURE_SIZE),
            },
        },
        size,
    );
    ctx.queue.submit([encoder.finish()]);

    check_layers(&ctx, &readback_buffer, NUM_LAYERS, |layer| {
        if LAYER_MASK & (1 << layer) != 0 {
            1.0f32
        } else {
            0.5
        }
    })
    .await;
}

/// Maps `buffer`, which holds `num_layers` layers of equal size, and checks that
/// every value in each layer equals `expected(layer)`.
async fn check_layers<T: bytemuck::Pod + PartialEq + core::fmt::Debug>(
    ctx: &TestingContext,
    buffer: &wgpu::Buffer,
    num_layers: u32,
    expected: impl Fn(u32) -> T,
) {
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    let values: &[T] = bytemuck::cast_slice(&data);
    let layers = values.chunks_exact(values.len() / num_layers as usize);
    for (layer, layer_data) in (0..).zip(layers) {
        let expected = expected(layer);
        let failed_value = layer_data.iter().find(|&&v| v != expected);
        assert_eq!(failed_value, None, "Expected {expected:?} in layer {layer}");
    }
}

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
        texture_binding_view_dimension: None,
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
        swizzle: wgpu::TextureComponentSwizzle::default(),
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

    let data = slice.get_mapped_range().unwrap();
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
