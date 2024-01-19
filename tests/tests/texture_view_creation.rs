use wgpu::*;
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static STENCIL_ONLY_VIEW_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .skip(FailureCase::webgl2()) // WebGL doesn't have stencil only views
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        for format in [TextureFormat::Stencil8, TextureFormat::Depth24PlusStencil8] {
            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC
                    | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let _view = texture.create_view(&TextureViewDescriptor {
                aspect: TextureAspect::StencilOnly,
                ..Default::default()
            });
        }
    });

#[gpu_test]
static DEPTH_ONLY_VIEW_CREATION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        for format in [
            TextureFormat::Depth16Unorm,
            TextureFormat::Depth24Plus,
            TextureFormat::Depth24PlusStencil8,
        ] {
            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::COPY_DST
                    | TextureUsages::COPY_SRC
                    | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let _view = texture.create_view(&TextureViewDescriptor {
                aspect: TextureAspect::DepthOnly,
                ..Default::default()
            });
        }
    });
