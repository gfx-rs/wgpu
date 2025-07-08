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
        for (format, expected_view_format) in [
            (TextureFormat::Stencil8, TextureFormat::Stencil8),
            (TextureFormat::Depth24PlusStencil8, TextureFormat::Stencil8),
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
            let view = texture.create_view(&TextureViewDescriptor {
                aspect: TextureAspect::StencilOnly,
                ..Default::default()
            });

            assert!(view.render_extent().is_none());

            let descriptor = view.descriptor();
            assert_eq!(descriptor.format, Some(expected_view_format));
            assert_eq!(descriptor.dimension, Some(TextureViewDimension::D2));
            assert_eq!(
                descriptor.usage,
                Some(
                    TextureUsages::COPY_DST
                        | TextureUsages::COPY_SRC
                        | TextureUsages::TEXTURE_BINDING,
                )
            );
            assert_eq!(descriptor.mip_level_count, Some(1));
            assert_eq!(descriptor.array_layer_count, Some(1));
        }
    });

#[gpu_test]
static DEPTH_ONLY_VIEW_CREATION: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        for (format, expected_view_format) in [
            (TextureFormat::Depth16Unorm, TextureFormat::Depth16Unorm),
            (TextureFormat::Depth24Plus, TextureFormat::Depth24Plus),
            (
                TextureFormat::Depth24PlusStencil8,
                TextureFormat::Depth24Plus,
            ),
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
            let view = texture.create_view(&TextureViewDescriptor {
                aspect: TextureAspect::DepthOnly,
                ..Default::default()
            });

            assert!(view.render_extent().is_none());

            let descriptor = view.descriptor();
            assert_eq!(descriptor.format, Some(expected_view_format));
            assert_eq!(descriptor.dimension, Some(TextureViewDimension::D2));
            assert_eq!(
                descriptor.usage,
                Some(
                    TextureUsages::COPY_DST
                        | TextureUsages::COPY_SRC
                        | TextureUsages::TEXTURE_BINDING,
                )
            );
            assert_eq!(descriptor.mip_level_count, Some(1));
            assert_eq!(descriptor.array_layer_count, Some(1));
        }
    });

#[gpu_test]
static SHARED_USAGE_VIEW_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().downlevel_flags(DownlevelFlags::VIEW_FORMATS))
    .run_async(|ctx| async move {
        {
            let (texture_format, view_format) =
                (TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb);
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
                format: texture_format,
                usage: TextureUsages::COPY_DST
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[TextureFormat::Rgba8UnormSrgb],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                aspect: TextureAspect::All,
                format: Some(view_format),
                usage: Some(
                    TextureUsages::COPY_DST
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT,
                ),
                ..Default::default()
            });

            assert_eq!(
                view.render_extent(),
                Some(Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                })
            );

            let descriptor = view.descriptor();
            assert_eq!(descriptor.format, Some(view_format));
            assert_eq!(descriptor.dimension, Some(TextureViewDimension::D2));
            assert_eq!(
                descriptor.usage,
                Some(
                    TextureUsages::COPY_DST
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::RENDER_ATTACHMENT,
                )
            );
            assert_eq!(descriptor.mip_level_count, Some(1));
            assert_eq!(descriptor.array_layer_count, Some(1));
        }
    });
