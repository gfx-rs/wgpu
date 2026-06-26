use wgpu::util::DeviceExt;
use wgpu::{ComputePassTimestampWrites, QueryType, RenderPassTimestampWrites};
use wgpu_test::{
    fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        BUFFER_DESTROY,
        TEXTURE_DESTROY,
        BUFFER_DESTROY_BEFORE_SUBMIT,
        BUFFER_DESTROY_AFTER_SUBMIT,
        TEXTURE_DESTROY_BEFORE_SUBMIT,
        TEXTURE_DESTROY_AFTER_SUBMIT,
        EXTERNAL_TEXTURE_DESTROY_BEFORE_SUBMIT,
        QUERY_SET_DESTROY_BEFORE_SUBMIT,
        QUERY_SET_DESTROY_AFTER_SUBMIT,
        REPLACED_BIND_GROUP,
    ]);
}

#[gpu_test]
static BUFFER_DESTROY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer"),
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        buffer.destroy();

        buffer.destroy();

        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        fail(
            &ctx.device,
            || {
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Write, move |_| {});
            },
            Some("buffer with 'buffer' label has been destroyed"),
        );

        buffer.destroy();

        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        buffer.destroy();

        buffer.destroy();

        let descriptor = wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        };

        // Scopes to mix up the drop/poll ordering.
        {
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
        }
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        {
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            buffer.destroy();
            let buffer = ctx.device.create_buffer(&descriptor);
            ctx.async_poll(wgpu::PollType::wait_indefinitely())
                .await
                .unwrap();
            buffer.destroy();
        }
        let buffer = ctx.device.create_buffer(&descriptor);
        buffer.destroy();
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
    });

#[gpu_test]
static TEXTURE_DESTROY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_async(|ctx| async move {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // multisampling is not supported for clear
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        texture.destroy();

        texture.destroy();

        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        texture.destroy();

        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        texture.destroy();

        texture.destroy();
    });

#[derive(Copy, Clone)]
enum UsageKind {
    Direct,
    RenderPass,
    ComputePass,
    RenderBundle,
}

const BUFFER_RENDER_SHADER: &str = "\
@group(0) @binding(0) var<uniform> buf: vec4<f32>;
@vertex fn vs() -> @builtin(position) vec4<f32> { return buf; }
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0); }";

const BUFFER_COMPUTE_SHADER: &str = "\
@group(0) @binding(0) var<uniform> buf: vec4<f32>;
@compute @workgroup_size(1) fn main() { _ = buf; }";

const TEXTURE_RENDER_SHADER: &str = "\
@group(0) @binding(0) var tex: texture_2d<f32>;
@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }
@fragment fn fs() -> @location(0) vec4<f32> { return textureLoad(tex, vec2(0), 0); }";

const TEXTURE_COMPUTE_SHADER: &str = "\
@group(0) @binding(0) var tex: texture_2d<f32>;
@compute @workgroup_size(1) fn main() { _ = textureLoad(tex, vec2(0), 0); }";

const EXTERNAL_TEXTURE_RENDER_SHADER: &str = "\
@group(0) @binding(0) var tex: texture_external;
@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }
@fragment fn fs() -> @location(0) vec4<f32> { return textureLoad(tex, vec2(0)); }";

const EXTERNAL_TEXTURE_COMPUTE_SHADER: &str = "\
@group(0) @binding(0) var tex: texture_external;
@compute @workgroup_size(1) fn main() { _ = textureLoad(tex, vec2(0)); }";

const EMPTY_COMPUTE_SHADER: &str = "\
@compute @workgroup_size(1) fn main() {}";

const EMPTY_RENDER_SHADER: &str = "\
@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0); }";

fn create_render_target(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_render_pipeline(device: &wgpu::Device, shader_src: &str) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_compute_pipeline(device: &wgpu::Device, shader_src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Records a bind group usage into an encoder and returns the encoder.
fn record_encoder_with_resource(
    ctx: &TestingContext,
    usage: UsageKind,
    resource: wgpu::BindingResource<'_>,
    render_shader: &str,
    compute_shader: &str,
) -> wgpu::CommandEncoder {
    let (_render_target, rt_view) = create_render_target(&ctx.device);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    match usage {
        UsageKind::Direct => unreachable!(),
        UsageKind::RenderPass | UsageKind::RenderBundle => {
            let pipeline = create_render_pipeline(&ctx.device, render_shader);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource,
                }],
            });

            let color_attachment = [Some(wgpu::RenderPassColorAttachment {
                view: &rt_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];

            if matches!(usage, UsageKind::RenderPass) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &color_attachment,
                    ..Default::default()
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..0, 0..0);
            } else {
                let mut rbe =
                    ctx.device
                        .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                            label: None,
                            color_formats: &[Some(wgpu::TextureFormat::Rgba8Unorm)],
                            depth_stencil: None,
                            sample_count: 1,
                            multiview: None,
                        });
                rbe.set_pipeline(&pipeline);
                rbe.set_bind_group(0, &bind_group, &[]);
                rbe.draw(0..0, 0..0);
                let bundle = rbe.finish(&wgpu::RenderBundleDescriptor::default());
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &color_attachment,
                    ..Default::default()
                });
                pass.execute_bundles([&bundle]);
            }
        }
        UsageKind::ComputePass => {
            let pipeline = create_compute_pipeline(&ctx.device, compute_shader);
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource,
                }],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(0, 0, 0);
        }
    }

    encoder
}

fn test_buffer_destroy_before_submit(ctx: &TestingContext, usage: UsageKind) {
    if matches!(usage, UsageKind::Direct) {
        let buffer_source = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let buffer_dest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);

        buffer_source.destroy();
        buffer_dest.destroy();

        fail(
            &ctx.device,
            || ctx.queue.submit([encoder.finish()]),
            Some("Buffer with '' label has been destroyed"),
        );
        return;
    }

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let encoder = record_encoder_with_resource(
        ctx,
        usage,
        buffer.as_entire_binding(),
        BUFFER_RENDER_SHADER,
        BUFFER_COMPUTE_SHADER,
    );

    buffer.destroy();

    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("Buffer with '' label has been destroyed"),
    );
}

// Test that destroying a buffer between command encoding and submission fails gracefully.
#[gpu_test]
static BUFFER_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop(),
    )
    .run_sync(|ctx| {
        test_buffer_destroy_before_submit(&ctx, UsageKind::Direct);
        test_buffer_destroy_before_submit(&ctx, UsageKind::RenderPass);
        test_buffer_destroy_before_submit(&ctx, UsageKind::ComputePass);
        test_buffer_destroy_before_submit(&ctx, UsageKind::RenderBundle);
    });

fn test_buffer_destroy_after_submit(ctx: &TestingContext, usage: UsageKind) {
    if matches!(usage, UsageKind::Direct) {
        let buffer_source = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let buffer_dest = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);
        ctx.queue.submit([encoder.finish()]);

        buffer_source.destroy();
        buffer_dest.destroy();

        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        return;
    }

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let encoder = record_encoder_with_resource(
        ctx,
        usage,
        buffer.as_entire_binding(),
        BUFFER_RENDER_SHADER,
        BUFFER_COMPUTE_SHADER,
    );

    ctx.queue.submit([encoder.finish()]);

    buffer.destroy();

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

// Test that destroying a buffer between submission and GPU completion is handled correctly.
#[gpu_test]
static BUFFER_DESTROY_AFTER_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop(),
    )
    .run_sync(|ctx| {
        test_buffer_destroy_after_submit(&ctx, UsageKind::Direct);
        test_buffer_destroy_after_submit(&ctx, UsageKind::RenderPass);
        test_buffer_destroy_after_submit(&ctx, UsageKind::ComputePass);
        test_buffer_destroy_after_submit(&ctx, UsageKind::RenderBundle);
    });

fn test_texture_destroy_before_submit(ctx: &TestingContext, usage: UsageKind) {
    if matches!(usage, UsageKind::Direct) {
        let descriptor = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let texture_1 = ctx.device.create_texture(&descriptor);
        let texture_2 = ctx.device.create_texture(&descriptor);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_1,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture_2,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );

        texture_1.destroy();
        texture_2.destroy();

        fail(
            &ctx.device,
            || ctx.queue.submit([encoder.finish()]),
            Some("Texture with '' label has been destroyed"),
        );
        return;
    }

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let encoder = record_encoder_with_resource(
        ctx,
        usage,
        wgpu::BindingResource::TextureView(&view),
        TEXTURE_RENDER_SHADER,
        TEXTURE_COMPUTE_SHADER,
    );

    texture.destroy();

    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("Texture with '' label has been destroyed"),
    );
}

// Test that destroying a texture between command encoding and submission fails gracefully.
#[gpu_test]
static TEXTURE_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop()
            .features(wgpu::Features::CLEAR_TEXTURE),
    )
    .run_sync(|ctx| {
        test_texture_destroy_before_submit(&ctx, UsageKind::Direct);
        test_texture_destroy_before_submit(&ctx, UsageKind::RenderPass);
        test_texture_destroy_before_submit(&ctx, UsageKind::ComputePass);
        test_texture_destroy_before_submit(&ctx, UsageKind::RenderBundle);
    });

fn test_texture_destroy_after_submit(ctx: &TestingContext, usage: UsageKind) {
    if matches!(usage, UsageKind::Direct) {
        let descriptor = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Snorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture_1 = ctx.device.create_texture(&descriptor);
        let texture_2 = ctx.device.create_texture(&descriptor);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture_1,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture_2,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit([encoder.finish()]);

        texture_1.destroy();
        texture_2.destroy();

        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        return;
    }

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let encoder = record_encoder_with_resource(
        ctx,
        usage,
        wgpu::BindingResource::TextureView(&view),
        TEXTURE_RENDER_SHADER,
        TEXTURE_COMPUTE_SHADER,
    );

    ctx.queue.submit([encoder.finish()]);

    texture.destroy();

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

// Test that destroying a texture between submission and GPU completion is handled correctly.
#[gpu_test]
static TEXTURE_DESTROY_AFTER_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop()
            .features(wgpu::Features::CLEAR_TEXTURE),
    )
    .run_sync(|ctx| {
        test_texture_destroy_after_submit(&ctx, UsageKind::Direct);
        test_texture_destroy_after_submit(&ctx, UsageKind::RenderPass);
        test_texture_destroy_after_submit(&ctx, UsageKind::ComputePass);
        test_texture_destroy_after_submit(&ctx, UsageKind::RenderBundle);
    });

fn test_external_texture_destroy_before_submit(ctx: &TestingContext, usage: UsageKind) {
    let plane_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let external_texture = ctx.device.create_external_texture(
        &wgpu::ExternalTextureDescriptor {
            label: None,
            width: 1,
            height: 1,
            format: wgpu::ExternalTextureFormat::Rgba,
            yuv_conversion_matrix: [0.0; 16],
            gamut_conversion_matrix: [0.0; 9],
            src_transfer_function: Default::default(),
            dst_transfer_function: Default::default(),
            sample_transform: [0.0; 6],
            load_transform: [0.0; 6],
        },
        &[&plane_texture.create_view(&wgpu::TextureViewDescriptor::default())],
    );

    let encoder = record_encoder_with_resource(
        ctx,
        usage,
        wgpu::BindingResource::ExternalTexture(&external_texture),
        EXTERNAL_TEXTURE_RENDER_SHADER,
        EXTERNAL_TEXTURE_COMPUTE_SHADER,
    );

    plane_texture.destroy();
    external_texture.destroy();

    // External textures use a buffer and several textures internally. We consider which one
    // triggers the error to be an implementation detail and match either.
    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("with '' label has been destroyed"),
    );
}

// Test that destroying an external texture between command encoding and submission fails
// gracefully.
#[gpu_test]
static EXTERNAL_TEXTURE_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop()
            .features(wgpu::Features::EXTERNAL_TEXTURE | wgpu::Features::CLEAR_TEXTURE),
    )
    .run_sync(|ctx| {
        // UsageKind::Direct does not apply because external textures only support TEXTURE_BINDING.
        test_external_texture_destroy_before_submit(&ctx, UsageKind::RenderPass);
        test_external_texture_destroy_before_submit(&ctx, UsageKind::ComputePass);
        test_external_texture_destroy_before_submit(&ctx, UsageKind::RenderBundle);
    });

fn test_query_set_destroy_before_submit(ctx: &TestingContext, ty: QueryType, usage: UsageKind) {
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        count: 2,
        ty,
    });

    let (_render_target, rt_view) = create_render_target(&ctx.device);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    match usage {
        UsageKind::RenderPass => {
            let pipeline = create_render_pipeline(&ctx.device, EMPTY_RENDER_SHADER);
            let color_attachment = [Some(wgpu::RenderPassColorAttachment {
                view: &rt_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let (occlusion_query_set, timestamp_writes) = match ty {
                QueryType::Occlusion => (Some(&query_set), None),
                QueryType::Timestamp => (
                    None,
                    Some(RenderPassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                ),
                QueryType::PipelineStatistics(_) => unreachable!(),
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &color_attachment,
                occlusion_query_set,
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..0, 0..0);
        }
        UsageKind::ComputePass => {
            let pipeline = create_compute_pipeline(&ctx.device, EMPTY_COMPUTE_SHADER);
            let timestamp_writes = match ty {
                QueryType::Timestamp => Some(ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                _ => unreachable!(),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(0, 0, 0);
        }
        UsageKind::RenderBundle => unreachable!(),
        UsageKind::Direct => unreachable!(),
    }

    query_set.destroy();

    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("QuerySet with '' label has been destroyed"),
    );
}

// Test that destroying a query set between command encoding and submission fails
// gracefully.
#[gpu_test]
static QUERY_SET_DESTROY_BEFORE_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop()
            .features(wgpu::Features::TIMESTAMP_QUERY),
    )
    .run_sync(|ctx| {
        test_query_set_destroy_before_submit(&ctx, QueryType::Occlusion, UsageKind::RenderPass);
        test_query_set_destroy_before_submit(&ctx, QueryType::Timestamp, UsageKind::RenderPass);
        test_query_set_destroy_before_submit(&ctx, QueryType::Timestamp, UsageKind::ComputePass);
    });

fn test_query_set_destroy_after_submit(ctx: &TestingContext, ty: QueryType, usage: UsageKind) {
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        count: 2,
        ty,
    });

    let (_render_target, rt_view) = create_render_target(&ctx.device);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    match usage {
        UsageKind::RenderPass => {
            let pipeline = create_render_pipeline(&ctx.device, EMPTY_RENDER_SHADER);
            let color_attachment = [Some(wgpu::RenderPassColorAttachment {
                view: &rt_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let (occlusion_query_set, timestamp_writes) = match ty {
                QueryType::Occlusion => (Some(&query_set), None),
                QueryType::Timestamp => (
                    None,
                    Some(RenderPassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                ),
                QueryType::PipelineStatistics(_) => unreachable!(),
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &color_attachment,
                occlusion_query_set,
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..0, 0..0);
        }
        UsageKind::ComputePass => {
            let pipeline = create_compute_pipeline(&ctx.device, EMPTY_COMPUTE_SHADER);
            let timestamp_writes = match ty {
                QueryType::Timestamp => Some(ComputePassTimestampWrites {
                    query_set: &query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                _ => unreachable!(),
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                timestamp_writes,
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.dispatch_workgroups(0, 0, 0);
        }
        UsageKind::RenderBundle => unreachable!(),
        UsageKind::Direct => unreachable!(),
    }

    ctx.queue.submit([encoder.finish()]);

    query_set.destroy();

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
}

// Test that destroying a query set between submission and GPU completion is handled correctly.
#[gpu_test]
static QUERY_SET_DESTROY_AFTER_SUBMIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop()
            .features(wgpu::Features::TIMESTAMP_QUERY),
    )
    .run_sync(|ctx| {
        test_query_set_destroy_after_submit(&ctx, QueryType::Occlusion, UsageKind::RenderPass);
        test_query_set_destroy_after_submit(&ctx, QueryType::Timestamp, UsageKind::RenderPass);
        test_query_set_destroy_after_submit(&ctx, QueryType::Timestamp, UsageKind::ComputePass);
    });

fn test_replaced_bind_group(ctx: &TestingContext, usage: UsageKind) {
    let buffer_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let buffer_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let (_render_target, rt_view) = create_render_target(&ctx.device);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    match usage {
        UsageKind::RenderPass | UsageKind::RenderBundle => {
            let pipeline = create_render_pipeline(&ctx.device, BUFFER_RENDER_SHADER);
            let layout = pipeline.get_bind_group_layout(0);
            let bind_group_a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                }],
            });
            let bind_group_b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_b.as_entire_binding(),
                }],
            });

            let color_attachment = [Some(wgpu::RenderPassColorAttachment {
                view: &rt_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];

            if matches!(usage, UsageKind::RenderPass) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &color_attachment,
                    ..Default::default()
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group_a, &[]);
                pass.set_bind_group(0, &bind_group_b, &[]);
                pass.draw(0..0, 0..0);
            } else {
                let mut rbe =
                    ctx.device
                        .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                            label: None,
                            color_formats: &[Some(wgpu::TextureFormat::Rgba8Unorm)],
                            depth_stencil: None,
                            sample_count: 1,
                            multiview: None,
                        });
                rbe.set_pipeline(&pipeline);
                rbe.set_bind_group(0, &bind_group_a, &[]);
                rbe.set_bind_group(0, &bind_group_b, &[]);
                rbe.draw(0..0, 0..0);
                let bundle = rbe.finish(&wgpu::RenderBundleDescriptor::default());
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &color_attachment,
                    ..Default::default()
                });
                pass.execute_bundles([&bundle]);
            }
        }
        UsageKind::ComputePass => {
            let pipeline = create_compute_pipeline(&ctx.device, BUFFER_COMPUTE_SHADER);
            let layout = pipeline.get_bind_group_layout(0);
            let bind_group_a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                }],
            });
            let bind_group_b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_b.as_entire_binding(),
                }],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group_a, &[]);
            pass.set_bind_group(0, &bind_group_b, &[]);
            pass.dispatch_workgroups(0, 0, 0);
        }
        UsageKind::Direct => unreachable!(),
    }

    buffer_a.destroy();

    fail(
        &ctx.device,
        || ctx.queue.submit([encoder.finish()]),
        Some("Buffer with '' label has been destroyed"),
    );
}

/// Test that bind groups that are replaced before use in a draw/dispatch are still
/// considered in submit-time liveness checks.
#[gpu_test]
static REPLACED_BIND_GROUP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .enable_noop(),
    )
    .run_sync(|ctx| {
        test_replaced_bind_group(&ctx, UsageKind::RenderPass);
        test_replaced_bind_group(&ctx, UsageKind::ComputePass);
        test_replaced_bind_group(&ctx, UsageKind::RenderBundle);
    });
