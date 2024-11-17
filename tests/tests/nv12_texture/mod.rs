//! Tests for nv12 texture creation and sampling.

use wgpu_test::{fail, gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static NV12_TEXTURE_CREATION_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        let target_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("nv12_texture.wgsl"));
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("nv12 pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(target_format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: Some(wgpu::IndexFormat::Uint32),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let y_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R8Unorm),
            aspect: wgpu::TextureAspect::Plane0,
            ..Default::default()
        });
        let uv_view = tex.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rg8Unorm),
            aspect: wgpu::TextureAspect::Plane1,
            ..Default::default()
        });
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
            ],
        });

        let target_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &target_view,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
        drop(rpass);
        ctx.queue.submit(Some(encoder.finish()));
    });

#[gpu_test]
static NV12_TEXTURE_VIEW_PLANE_ON_NON_PLANAR_FORMAT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
        .run_sync(|ctx| {
            let size = wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            };
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                dimension: wgpu::TextureDimension::D2,
                size,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });
            fail(
                &ctx.device,
                || {
                    let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::Plane0,
                        ..Default::default()
                    });
                },
                Some("aspect plane0 is not in the source texture format r8unorm"),
            );
        });

#[gpu_test]
static NV12_TEXTURE_VIEW_PLANE_OUT_OF_BOUNDS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        fail(
            &ctx.device,
            || {
                let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu::TextureFormat::R8Unorm),
                    aspect: wgpu::TextureAspect::Plane2,
                    ..Default::default()
                });
            },
            Some("aspect plane2 is not in the source texture format nv12"),
        );
    });

#[gpu_test]
static NV12_TEXTURE_BAD_FORMAT_VIEW_PLANE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: wgpu::TextureFormat::NV12,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        fail(
            &ctx.device,
            || {
                let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu::TextureFormat::Rg8Unorm),
                    aspect: wgpu::TextureAspect::Plane0,
                    ..Default::default()
                });
            },
            Some("unable to view texture nv12 as rg8unorm"),
        );
    });

#[gpu_test]
static NV12_TEXTURE_BAD_SIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::TEXTURE_FORMAT_NV12))
    .run_sync(|ctx| {
        let size = wgpu::Extent3d {
            width: 255,
            height: 255,
            depth_or_array_layers: 1,
        };

        fail(
            &ctx.device,
            || {
                let _ = ctx.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    size,
                    format: wgpu::TextureFormat::NV12,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some("width 255 is not a multiple of nv12's width multiple requirement"),
        );
    });
