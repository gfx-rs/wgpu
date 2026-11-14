use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST,
        TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST,
        TEXTURE_BLIT_DOWNSAMPLE_LINEAR_TEST,
        TEXTURE_BLIT_DOWNSAMPLE_NEAREST_TEST,
    ]);
}

#[apply(gpu_test!)]
static TEXTURE_BLIT_WITH_LINEAR_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
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
            texture_binding_view_dimension: None,
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
            texture_binding_view_dimension: None,
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

#[apply(gpu_test!)]
static TEXTURE_BLIT_WITH_NEAREST_FILTER_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
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
            texture_binding_view_dimension: None,
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
            texture_binding_view_dimension: None,
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

/// Blits a 2x2 texture with a black left column and a white right column into a
/// 1x1 texture, and returns the red channel of the result.
async fn downsample_red_channel(ctx: &TestingContext, filter: wgpu::FilterMode) -> u8 {
    let source = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });

    let column = [[0, 0, 0, 255], [255, 255, 255, 255]];
    let texels = [column, column].concat().concat();
    ctx.queue.write_texture(
        source.as_image_copy(),
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: None,
        },
        source.size(),
    );

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        texture_binding_view_dimension: None,
    });

    let blitter =
        wgpu::util::TextureBlitterBuilder::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm)
            .sample_type(filter)
            .build();

    let readback = ReadbackBuffers::new(&ctx.device, &target);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    blitter.copy(
        &ctx.device,
        &mut encoder,
        &source.create_view(&wgpu::TextureViewDescriptor::default()),
        &target.create_view(&wgpu::TextureViewDescriptor::default()),
    );
    readback.copy_from(&ctx.device, &mut encoder, &target);
    ctx.queue.submit(Some(encoder.finish()));

    let data = readback.retrieve(ctx, wgpu::TextureAspect::All).await;
    data.all()[0]
}

#[apply(gpu_test!)]
static TEXTURE_BLIT_DOWNSAMPLE_LINEAR_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let red = downsample_red_channel(&ctx, wgpu::FilterMode::Linear).await;
        assert!(
            (120..=135).contains(&red),
            "linear downsampling should blend the texels, got {red}"
        );
    });

#[apply(gpu_test!)]
static TEXTURE_BLIT_DOWNSAMPLE_NEAREST_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let red = downsample_red_channel(&ctx, wgpu::FilterMode::Nearest).await;
        assert!(
            red == 0 || red == 255,
            "nearest downsampling should pick a single texel, got {red}"
        );
    });
