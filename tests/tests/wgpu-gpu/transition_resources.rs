use wgpu_test::{gpu_test, GpuTestConfiguration};

#[gpu_test]
static TRANSITION_RESOURCES: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.transition_resources(
        std::iter::empty(),
        [wgpu::TextureTransition {
            texture: &texture,
            selector: None,
            state: wgpu::TextureUses::COLOR_TARGET,
        }]
        .into_iter(),
    );

    ctx.queue.submit([encoder.finish()]);
});
