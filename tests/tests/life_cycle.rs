use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static BUFFER_DESTROY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().expect_fail(FailureCase::always()))
    .run_sync(|ctx| {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        buffer.destroy();

        buffer.destroy();

        ctx.device.poll(wgpu::MaintainBase::Wait);

        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |_| {});

        buffer.destroy();

        ctx.device.poll(wgpu::MaintainBase::Wait);

        buffer.destroy();

        buffer.destroy();
    });

#[gpu_test]
static TEXTURE_DESTROY: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
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

    ctx.device.poll(wgpu::MaintainBase::Wait);

    texture.destroy();

    ctx.device.poll(wgpu::MaintainBase::Wait);

    texture.destroy();

    texture.destroy();
});
