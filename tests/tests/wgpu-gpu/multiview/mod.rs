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
    vec.push(DRAW_MULTIVIEW_SKIP_FIRST_LAYER);
    vec.push(DRAW_MULTIVIEW_MULTISAMPLE);
    vec.push(DRAW_MULTIVIEW_MESH);
    vec.push(DRAW_MULTIVIEW_MESH_NONCONTIGUOUS);
    vec.push(DRAW_MULTIVIEW_MESH_SKIP_FIRST_LAYER);
    vec.push(DRAW_MULTIVIEW_VIEWPORT);
    vec.push(MULTIVIEW_OCCLUSION_QUERY);
    vec.push(MULTIVIEW_DISCARD_RESETS_ALL_LAYERS);
    vec.push(SELECTIVE_MULTIVIEW_CLEAR_INITS_ONLY_MASKED_LAYERS);
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
    .run_async(|ctx| run_test(ctx, 0b1, 1, false));

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
    .run_async(|ctx| run_test(ctx, 0b11, 1, false));

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
    .run_async(|ctx| run_test(ctx, 0b1001, 1, false));

/// Renders only to layer 1, so `view_index` differs from the index of the view within the mask.
#[apply(gpu_test!)]
static DRAW_MULTIVIEW_SKIP_FIRST_LAYER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b10, 1, false));

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
    .run_async(|ctx| run_test(ctx, 0b11, 4, false));

#[apply(gpu_test!)]
static DRAW_MULTIVIEW_MESH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::MULTIVIEW
                    | Features::EXPERIMENTAL_MESH_SHADER
                    | Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW,
            )
            .limits(Limits {
                max_multiview_view_count: 2,
                max_mesh_multiview_view_count: 2,
                ..Limits::defaults().using_recommended_minimum_mesh_shader_values()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b11, 1, true));

#[apply(gpu_test!)]
static DRAW_MULTIVIEW_MESH_NONCONTIGUOUS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::MULTIVIEW
                    | Features::SELECTIVE_MULTIVIEW
                    | Features::EXPERIMENTAL_MESH_SHADER
                    | Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW,
            )
            .limits(Limits {
                max_multiview_view_count: 4,
                max_mesh_multiview_view_count: 4,
                ..Limits::defaults().using_recommended_minimum_mesh_shader_values()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b1001, 1, true));

#[apply(gpu_test!)]
static DRAW_MULTIVIEW_MESH_SKIP_FIRST_LAYER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                Features::MULTIVIEW
                    | Features::SELECTIVE_MULTIVIEW
                    | Features::EXPERIMENTAL_MESH_SHADER
                    | Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW,
            )
            .limits(Limits {
                max_multiview_view_count: 2,
                max_mesh_multiview_view_count: 2,
                ..Limits::defaults().using_recommended_minimum_mesh_shader_values()
            }),
    )
    .run_async(|ctx| run_test(ctx, 0b10, 1, true));

/// The viewport applies to every view.
#[apply(gpu_test!)]
static DRAW_MULTIVIEW_VIEWPORT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| async move {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                }),
                multiview_mask: NonZero::new(0b11),
                cache: None,
            });

        let texture = zero_init_texture(&ctx);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: NonZero::new(0b11),
            });
            pass.set_pipeline(&pipeline);
            let half = ZERO_INIT_SIZE as f32 / 2.0;
            pass.set_viewport(0.0, 0.0, half, ZERO_INIT_SIZE as f32, 0.0, 1.0);
            pass.draw(0..3, 0..1);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let row_bytes = (ZERO_INIT_SIZE * 4) as usize;
        for (layer, data) in read_layers(&ctx, &texture).await.iter().enumerate() {
            for (y, row) in data.chunks(row_bytes).enumerate() {
                let (left, right) = row.split_at(row_bytes / 2);
                assert!(
                    left.iter().all(|&b| b != 0),
                    "layer {layer} row {y}: the left half was not drawn"
                );
                assert!(
                    right.iter().all(|&b| b == 0),
                    "layer {layer} row {y}: the right half, outside the viewport, was drawn"
                );
            }
        }
    });

async fn run_test(ctx: TestingContext, layer_mask: u32, sample_count: u32, mesh: bool) {
    let num_layers = 32 - layer_mask.leading_zeros();

    let shader_src = if mesh {
        include_str!("mesh.wgsl")
    } else {
        include_str!("shader.wgsl")
    };

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let fragment = wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
    };
    let multisample = wgpu::MultisampleState {
        count: sample_count,
        ..Default::default()
    };
    let pipeline = if mesh {
        ctx.device
            .create_mesh_pipeline(&wgpu::MeshPipelineDescriptor {
                label: None,
                layout: None,
                task: None,
                mesh: wgpu::MeshState {
                    module: &shader,
                    entry_point: Some("ms_main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample,
                fragment: Some(fragment),
                multiview_mask: NonZero::new(layer_mask),
                cache: None,
            })
    } else {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                vertex: wgpu::VertexState {
                    buffers: &[],
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                fragment: Some(fragment),
                multiview_mask: NonZero::new(layer_mask),
                multisample,
                layout: None,
                depth_stencil: None,
                cache: None,
            })
    };

    const TEXTURE_SIZE: u32 = 256;

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
        if mesh {
            rpass.draw_mesh_tasks(1, 1, 1);
        } else {
            rpass.draw(0..6, 0..1);
        }
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

/// Occlusion queries in a multiview pass use one slot per view, and the slots of each query sum
/// to its result.
#[apply(gpu_test!)]
static MULTIVIEW_OCCLUSION_QUERY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: 2,
                ..Limits::defaults()
            })
            .expect_fail(FailureCase::webgl2()),
    )
    .run_async(|ctx| async move {
        const QUERY_COUNT: u32 = 6;

        let depth_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
            texture_binding_view_dimension: None,
        });
        let multiview_depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let single_depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            array_layer_count: Some(1),
            ..Default::default()
        });

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../occlusion_query/shader.wgsl"));
        let create_pipeline = |multiview_mask| {
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: None,
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: None,
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: NonZero::new(multiview_mask),
                    cache: None,
                })
        };
        let single_pipeline = create_pipeline(0);
        let multiview_pipeline = create_pipeline(0b11);

        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Occlusion,
            count: QUERY_COUNT,
        });

        let depth_attachment = |view| {
            Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            // Leave a nonzero result in slot 5, which the last multiview query also uses.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: depth_attachment(&single_depth_view),
                timestamp_writes: None,
                occlusion_query_set: Some(&query_set),
                multiview_mask: None,
            });
            pass.set_pipeline(&single_pipeline);
            pass.begin_occlusion_query(5);
            pass.draw(0..3, 0..1);
            pass.end_occlusion_query();
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: depth_attachment(&multiview_depth_view),
                timestamp_writes: None,
                occlusion_query_set: Some(&query_set),
                multiview_mask: NonZero::new(0b11),
            });
            pass.set_pipeline(&multiview_pipeline);

            // Not occluded (z = 0.5, nothing drawn yet)
            pass.begin_occlusion_query(0);
            pass.draw(4..7, 0..1);
            pass.end_occlusion_query();

            // Not occluded (z = 0.0)
            pass.begin_occlusion_query(2);
            pass.draw(0..3, 0..1);
            pass.end_occlusion_query();

            // Occluded (z = 0.5)
            pass.begin_occlusion_query(4);
            pass.draw(4..7, 0..1);
            pass.end_occlusion_query();
        }

        let query_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<u64>() as u64 * u64::from(QUERY_COUNT),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(&query_set, 0..QUERY_COUNT, &query_buffer, 0);

        let mapping_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: query_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&query_buffer, 0, &mapping_buffer, 0, query_buffer.size());

        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let view = mapping_buffer.slice(..).get_mapped_range().unwrap();
        let results: &[u64] = bytemuck::cast_slice(&view);

        // Occlusion results are only defined as zero or nonzero.
        assert_ne!(results[0] + results[1], 0, "results: {results:?}");
        assert_ne!(results[2] + results[3], 0, "results: {results:?}");
        assert_eq!(results[4] + results[5], 0, "results: {results:?}");
    });

// Wide enough that a row is `COPY_BYTES_PER_ROW_ALIGNMENT` bytes.
const ZERO_INIT_SIZE: u32 = 64;
const ZERO_INIT_LAYERS: u32 = 2;
const ZERO_INIT_BYTES_PER_LAYER: u32 = ZERO_INIT_SIZE * ZERO_INIT_SIZE * 4;

fn zero_init_texture(ctx: &TestingContext) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: ZERO_INIT_SIZE,
            height: ZERO_INIT_SIZE,
            depth_or_array_layers: ZERO_INIT_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        texture_binding_view_dimension: None,
    })
}

/// Writes `0xff` to every byte of every layer of `texture`.
fn fill_layers(ctx: &TestingContext, texture: &wgpu::Texture) {
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &vec![0xff; (ZERO_INIT_BYTES_PER_LAYER * ZERO_INIT_LAYERS) as usize],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(ZERO_INIT_SIZE * 4),
            rows_per_image: Some(ZERO_INIT_SIZE),
        },
        texture.size(),
    );
}

fn encode_color_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    ops: wgpu::Operations<wgpu::Color>,
    multiview_mask: u32,
) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: NonZero::new(multiview_mask),
    });
}

/// Copies every layer of `texture` to a buffer and returns its bytes, one `Vec` per layer.
async fn read_layers(ctx: &TestingContext, texture: &wgpu::Texture) -> Vec<Vec<u8>> {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(ZERO_INIT_BYTES_PER_LAYER * ZERO_INIT_LAYERS),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ZERO_INIT_SIZE * 4),
                rows_per_image: Some(ZERO_INIT_SIZE),
            },
        },
        texture.size(),
    );
    ctx.queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    data.chunks(ZERO_INIT_BYTES_PER_LAYER as usize)
        .map(<[u8]>::to_vec)
        .collect()
}

/// A discard in a multiview pass resets every layer, so later reads see zeros.
#[apply(gpu_test!)]
static MULTIVIEW_DISCARD_RESETS_ALL_LAYERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::MULTIVIEW)
            .limits(Limits {
                max_multiview_view_count: ZERO_INIT_LAYERS,
                ..Limits::defaults()
            }),
    )
    .run_async(|ctx| async move {
        let texture = zero_init_texture(&ctx);
        fill_layers(&ctx, &texture);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode_color_pass(
            &mut encoder,
            &view,
            wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Discard,
            },
            0b11,
        );
        ctx.queue.submit(Some(encoder.finish()));

        for (layer, data) in read_layers(&ctx, &texture).await.iter().enumerate() {
            assert_eq!(
                data.iter().position(|&b| b != 0),
                None,
                "layer {layer} has a nonzero byte"
            );
        }
    });

/// A clear in a selective multiview pass only marks the layers in the mask as initialized, so
/// discarded layers outside the mask still read as zeros.
#[apply(gpu_test!)]
static SELECTIVE_MULTIVIEW_CLEAR_INITS_ONLY_MASKED_LAYERS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW)
                .limits(Limits {
                    max_multiview_view_count: ZERO_INIT_LAYERS,
                    ..Limits::defaults()
                }),
        )
        .run_async(|ctx| async move {
            let texture = zero_init_texture(&ctx);
            fill_layers(&ctx, &texture);

            // Discard layer 1 so that it is uninitialized but may still hold the old data.
            let layer_1 = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: 1,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let all_layers = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encode_color_pass(
                &mut encoder,
                &layer_1,
                wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                },
                0,
            );
            encode_color_pass(
                &mut encoder,
                &all_layers,
                wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                0b01,
            );
            ctx.queue.submit(Some(encoder.finish()));

            // Backends may or may not clear layers outside the mask, but either way they are zero.
            for (layer, data) in read_layers(&ctx, &texture).await.iter().enumerate() {
                assert_eq!(
                    data.iter().position(|&b| b != 0),
                    None,
                    "layer {layer} has a nonzero byte"
                );
            }
        });
