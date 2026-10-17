use wgpu::*;
use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        METAL_MANY_PASSES,
        DISCARDED_TEXTURE_BETWEEN_COALESCED_PASSES,
        DISCARDED_TEXTURE_RENDER_BINDING,
    ]);
}

#[apply(gpu_test!)]
static METAL_MANY_PASSES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .skip(FailureCase::backend(!Backends::METAL))
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        // More passes than Metal's outstanding command-buffer limit, but only
        // one application command buffer. No per-pass submission is necessary.
        run_passes(&ctx, 2100, false).await;
    });

#[apply(gpu_test!)]
static DISCARDED_TEXTURE_BETWEEN_COALESCED_PASSES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
                .limits(Limits::downlevel_defaults()),
        )
        .run_async(|ctx| async move {
            run_passes(&ctx, 8, true).await;
        });

async fn run_passes(ctx: &TestingContext, count: u32, discard: bool) {
    let descriptor = TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let texture = ctx.device.create_texture(&descriptor);
    if discard {
        ctx.queue.write_texture(
            texture.as_image_copy(),
            &[255; 4],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            descriptor.size,
        );
    }
    let view = texture.create_view(&Default::default());
    let stencil = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        format: TextureFormat::Stencil8,
        usage: TextureUsages::RENDER_ATTACHMENT,
        ..descriptor
    });
    let stencil_view = stencil.create_view(&Default::default());
    let module = ctx
        .device
        .create_shader_module(include_wgsl!("pass_coalescing.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
    let result = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 8,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 8,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: result.as_entire_binding(),
            },
        ],
    });

    // Also exercise dropping an encoded, unsubmitted group of passes.
    for submit in [false, true] {
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.push_debug_group("mixed passes");
        for i in 0..count {
            drop(encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: if discard {
                            LoadOp::Load
                        } else {
                            LoadOp::Clear(if i % 2 == 0 {
                                Color::WHITE
                            } else {
                                Color::BLACK
                            })
                        },
                        store: if discard {
                            StoreOp::Discard
                        } else {
                            StoreOp::Store
                        },
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &stencil_view,
                    depth_ops: None,
                    stencil_ops: Some(Operations {
                        load: LoadOp::Clear(1),
                        store: StoreOp::Discard,
                    }),
                }),
                ..Default::default()
            }));
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.pop_debug_group();
        let commands = encoder.finish();
        if submit {
            let mut copy = ctx.device.create_command_encoder(&Default::default());
            copy.copy_buffer_to_buffer(&result, 0, &readback, 0, 8);
            ctx.queue.submit([commands, copy.finish()]);
        }
    }

    readback
        .slice(..)
        .map_async(MapMode::Read, |result| result.unwrap());
    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let values: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(
        values,
        &[count, if discard { 0 } else { count.div_ceil(2) }]
    );
    drop(mapped);
    readback.unmap();
}

#[apply(gpu_test!)]
static DISCARDED_TEXTURE_RENDER_BINDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let descriptor = TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let source = ctx.device.create_texture(&descriptor);
        let target = ctx.device.create_texture(&descriptor);
        let source_view = source.create_view(&Default::default());
        let target_view = target.create_view(&Default::default());
        ctx.queue.write_texture(
            source.as_image_copy(),
            &[255; 4],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
            descriptor.size,
        );
        let module = ctx
            .device
            .create_shader_module(include_wgsl!("pass_coalescing.wgsl"));
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(TextureFormat::Rgba8Unorm.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&source_view),
            }],
        });
        let readback = ReadbackBuffers::new(&ctx.device, &target);
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &source_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Discard,
                },
            })],
            ..Default::default()
        }));
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::WHITE),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        readback.copy_from(&ctx.device, &mut encoder, &target);
        ctx.queue.submit([encoder.finish()]);
        readback.assert_buffer_contents(&ctx, &[0; 4]).await;
    });
