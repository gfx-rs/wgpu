use wgpu_test::{gpu_test, GpuTestConfiguration};

#[gpu_test]
static TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_sync(|ctx| {
        let source = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let blitter = wgpu::util::TextureBlitterBuilder::new(
            &ctx.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .sample_type(wgpu::FilterMode::Linear)
        .build();
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        blitter.copy(
            &ctx.device,
            &mut encoder,
            &source.create_view(&wgpu::TextureViewDescriptor::default()),
            &target.create_view(&wgpu::TextureViewDescriptor::default()),
        );
    });

#[gpu_test]
static TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_sync(|ctx| {
        let source = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 100,
                height: 100,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let blitter = wgpu::util::TextureBlitterBuilder::new(
            &ctx.device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        )
        .sample_type(wgpu::FilterMode::Linear)
        .build();

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        blitter.copy(
            &ctx.device,
            &mut encoder,
            &source.create_view(&wgpu::TextureViewDescriptor::default()),
            &target.create_view(&wgpu::TextureViewDescriptor::default()),
        );
    });
