use wgpu::util::DeviceExt;
use wgpu::CommandEncoder;
use wgpu_test::{
    fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext,
};

#[gpu_test]
static DROP_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    drop(encoder);
});

#[gpu_test]
static DROP_QUEUE_BEFORE_CREATING_COMMAND_ENCODER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().expect_fail(FailureCase::always()))
        .run_sync(|ctx| {
            // Use the device after the queue is dropped. Currently this panics
            // but it probably shouldn't
            let device = ctx.device.clone();
            drop(ctx);
            let _encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        });

#[gpu_test]
static DROP_ENCODER_AFTER_ERROR: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let target_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("renderpass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &target_view,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set a bad viewport on renderpass, triggering an error.
        fail(
            &ctx.device,
            || {
                renderpass.set_viewport(0.0, 0.0, -1.0, -1.0, 0.0, 1.0);
                drop(renderpass);
            },
            None,
        );

        // This is the actual interesting error condition. We've created
        // a CommandEncoder which errored out when processing a command.
        // The encoder is still open!
        drop(encoder);
    });

// TODO: This should also apply to render passes once the lifetime bound is lifted.
#[gpu_test]
static ENCODER_OPERATIONS_FAIL_WHILE_COMPUTE_PASS_ALIVE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(
            wgpu::Features::CLEAR_TEXTURE
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ))
        .run_sync(encoder_operations_fail_while_compute_pass_alive);

fn encoder_operations_fail_while_compute_pass_alive(ctx: TestingContext) {
    let buffer_source = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let buffer_dest = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::COPY_DST,
        });

    let texture_desc = wgpu::TextureDescriptor {
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
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let texture_dst = ctx.device.create_texture(&texture_desc);
    let texture_src = ctx.device.create_texture(&wgpu::TextureDescriptor {
        usage: wgpu::TextureUsages::COPY_SRC,
        ..texture_desc
    });
    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        count: 1,
        ty: wgpu::QueryType::Timestamp,
        label: None,
    });

    #[allow(clippy::type_complexity)]
    let recording_ops: Vec<(_, Box<dyn Fn(&mut CommandEncoder)>)> = vec![
        (
            "begin_compute_pass",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            }),
        ),
        (
            "begin_render_pass",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor::default());
            }),
        ),
        (
            "copy_buffer_to_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_buffer_to_buffer(&buffer_source, 0, &buffer_dest, 0, 4);
            }),
        ),
        (
            "copy_buffer_to_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer_source,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.as_image_copy(),
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &buffer_dest,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(4),
                            rows_per_image: None,
                        },
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "copy_texture_to_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture_src,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &texture_dst,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    texture_dst.size(),
                );
            }),
        ),
        (
            "clear_texture",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.clear_texture(&texture_dst, &wgpu::ImageSubresourceRange::default());
            }),
        ),
        (
            "clear_buffer",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.clear_buffer(&buffer_dest, 0, None);
            }),
        ),
        (
            "insert_debug_marker",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.insert_debug_marker("marker");
            }),
        ),
        (
            "push_debug_group",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.push_debug_group("marker");
            }),
        ),
        (
            "pop_debug_group",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.pop_debug_group();
            }),
        ),
        (
            "resolve_query_set",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.resolve_query_set(&query_set, 0..1, &buffer_dest, 0);
            }),
        ),
        (
            "write_timestamp",
            Box::new(|encoder: &mut wgpu::CommandEncoder| {
                encoder.write_timestamp(&query_set, 0);
            }),
        ),
    ];

    for (op_name, op) in recording_ops.iter() {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
            .forget_lifetime();

        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);

        log::info!("Testing operation {} on a locked command encoder", op_name);
        fail(
            &ctx.device,
            || op(&mut encoder),
            Some("Command encoder is locked"),
        );

        // Drop the pass - this also fails now since the encoder is invalid:
        fail(
            &ctx.device,
            || drop(pass),
            Some("Command encoder is invalid"),
        );
        // Also, it's not possible to create a new pass on the encoder:
        fail(
            &ctx.device,
            || encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default()),
            Some("Command encoder is invalid"),
        );
    }

    // Test encoder finishing separately since it consumes the encoder and doesn't fit above pattern.
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
            .forget_lifetime();
        fail(
            &ctx.device,
            || encoder.finish(),
            Some("Command encoder is locked"),
        );
        fail(
            &ctx.device,
            || drop(pass),
            Some("Command encoder is invalid"),
        );
    }
}
