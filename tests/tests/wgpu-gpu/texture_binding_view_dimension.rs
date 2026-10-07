//! Tests for TextureDescriptor::texture_binding_view_dimension.
//!
//! Adapters without `ARBITRARY_BINDING_VIEW_DIMENSIONS` (GLES) apply WebGPU compatibility-mode
//! validation, others ignore the field.

use wgpu::TextureViewDimension as ViewDim;
use wgpu_test::{
    apply, fail_if, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(CREATE_TEXTURE_VALIDATION);
    tests.push(SAMPLED_BIND_GROUP_VALIDATION);
    tests.push(STORAGE_BIND_GROUP_VALIDATION);
}

/// `D2` requires 1 layer, `Cube` requires 6 layers, and `CubeArray` is not allowed.
#[apply(gpu_test!)]
static CREATE_TEXTURE_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let compat = !ctx
            .adapter_downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::ARBITRARY_BINDING_VIEW_DIMENSIONS);

        for (layers, dim) in [
            (2, ViewDim::D2),
            (5, ViewDim::Cube),
            (6, ViewDim::CubeArray),
        ] {
            fail_if(
                &ctx.device,
                compat,
                || {
                    ctx.device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d {
                            width: 4,
                            height: 4,
                            depth_or_array_layers: layers,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                        texture_binding_view_dimension: Some(dim),
                    })
                },
                None,
            );
        }
    });

/// A sampled texture view must have the texture's binding view dimension.
#[apply(gpu_test!)]
static SAMPLED_BIND_GROUP_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
        let compat = !ctx
            .adapter_downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::ARBITRARY_BINDING_VIEW_DIMENSIONS);

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
            texture_binding_view_dimension: Some(ViewDim::D2Array),
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(ViewDim::D2),
            ..Default::default()
        });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: ViewDim::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        fail_if(
            &ctx.device,
            compat,
            || {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                })
            },
            None,
        );
    });

/// A storage texture view must have the texture's binding view dimension.
#[apply(gpu_test!)]
static STORAGE_BIND_GROUP_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        let compat = !ctx
            .adapter_downlevel_capabilities
            .flags
            .contains(wgpu::DownlevelFlags::ARBITRARY_BINDING_VIEW_DIMENSIONS);

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
            texture_binding_view_dimension: Some(ViewDim::D2Array),
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(ViewDim::D2),
            ..Default::default()
        });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: ViewDim::D2,
                    },
                    count: None,
                }],
            });

        fail_if(
            &ctx.device,
            compat,
            || {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                })
            },
            None,
        );
    });
