use wgpu::*;
use wgpu_test::{fail, valid};

/// Ensures a [`TextureView`] can be bound to a [`BindingType::ExternalTexture`]
/// resource binding.
#[test]
fn external_texture_binding_texture_view() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::EXTERNAL_TEXTURE,
        ..Default::default()
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::ExternalTexture,
            count: None,
        }],
    });

    let texture_descriptor = TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let texture = device.create_texture(&texture_descriptor);
    let view = texture.create_view(&TextureViewDescriptor::default());
    valid(&device, || {
        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        })
    });

    // Invalid usages (must include TEXTURE_BINDING)
    let texture = device.create_texture(&TextureDescriptor {
        usage: TextureUsages::STORAGE_BINDING,
        ..texture_descriptor
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    fail(
        &device,
        || {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            })
        },
        Some("Usage flags TextureUsages(STORAGE_BINDING) of TextureView with '' label do not contain required usage flags TextureUsages(TEXTURE_BINDING"),
    );

    // Invalid dimension (must be D2)
    let texture = device.create_texture(&TextureDescriptor {
        dimension: TextureDimension::D3,
        ..texture_descriptor
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    fail(
        &device,
        || {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            })
        },
        Some("Texture binding 0 expects dimension = D2, but given a view with dimension = D3"),
    );

    // Invalid mip_level_count (must be 1)
    let texture = device.create_texture(&TextureDescriptor {
        mip_level_count: 2,
        ..texture_descriptor
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    fail(
        &device,
        || {

            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                ],
            })
        },
        Some("External texture bindings must have a single mip level, but given a view with mip_level_count = 2 at binding 0")
    );

    // Invalid format (must be Rgba8Unorm, Bgra8Unorm, or Rgba16float)
    let texture = device.create_texture(&TextureDescriptor {
        format: TextureFormat::Rgba8Uint,
        ..texture_descriptor
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    fail(
        &device,
        || {

            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                ],
            })
        },
        Some("External texture bindings must have a format of `rgba8unorm`, `bgra8unorm`, or `rgba16float, but given a view with format = Rgba8Uint at binding 0")
    );

    // Invalid sample count (must be 1)
    let texture = device.create_texture(&TextureDescriptor {
        sample_count: 4,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        ..texture_descriptor
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    fail(
        &device,
        || {
            device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            })
        },
        Some("Texture binding 0 expects multisampled = false, but given a view with samples = 4"),
    );
}
