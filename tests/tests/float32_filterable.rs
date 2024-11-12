//! Tests for FLOAT32_FILTERABLE feature.

use wgpu_test::{fail, gpu_test, GpuTestConfiguration, TestParameters};

fn create_texture_binding(device: &wgpu::Device, format: wgpu::TextureFormat, filterable: bool) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                multisampled: false,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });

    let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
}

#[gpu_test]
static FLOAT32_FILTERABLE_WITHOUT_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_sync(|ctx| {
        let device = &ctx.device;
        // Unorm textures are always filterable
        create_texture_binding(device, wgpu::TextureFormat::R8Unorm, true);
        create_texture_binding(device, wgpu::TextureFormat::R8Unorm, false);
        // As are float16 textures
        create_texture_binding(device, wgpu::TextureFormat::R16Float, true);
        create_texture_binding(device, wgpu::TextureFormat::R16Float, false);
        // Float 32 textures can be used as non-filterable only
        create_texture_binding(device, wgpu::TextureFormat::R32Float, false);
        // This is supposed to fail, since we have not activated the feature
        fail(
            &ctx.device,
            || {
                create_texture_binding(device, wgpu::TextureFormat::R32Float, true);
            },
            Some(concat!(
                "texture binding 0 expects sample type float { filterable: true }, ",
                "but was given a view with format r32float ",
                "(sample type float { filterable: false })"
            )),
        );
    });

#[gpu_test]
static FLOAT32_FILTERABLE_WITH_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::FLOAT32_FILTERABLE))
    .run_sync(|ctx| {
        let device = &ctx.device;
        // With the feature enabled, it does work!
        create_texture_binding(device, wgpu::TextureFormat::R32Float, true);
        create_texture_binding(device, wgpu::TextureFormat::Rg32Float, true);
        create_texture_binding(device, wgpu::TextureFormat::Rgba32Float, true);
    });
