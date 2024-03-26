use wgpu_test::{fail, gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

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
