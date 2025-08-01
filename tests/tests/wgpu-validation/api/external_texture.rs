use wgpu::*;
use wgpu_test::{fail, valid};

/// Ensures an [`ExternalTexture`] can be created from a valid descriptor and planes,
/// but appropriate errors are returned for invalid descriptors and planes.
#[test]
fn create_external_texture() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::EXTERNAL_TEXTURE,
        ..Default::default()
    });

    let texture_descriptor = TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };

    let r_texture = device.create_texture(&TextureDescriptor {
        format: TextureFormat::R8Unorm,
        ..texture_descriptor
    });
    let r_view = r_texture.create_view(&TextureViewDescriptor::default());
    let rg_texture = device.create_texture(&TextureDescriptor {
        format: TextureFormat::Rg8Unorm,
        ..texture_descriptor
    });
    let rg_view = rg_texture.create_view(&TextureViewDescriptor::default());
    let rgba_texture = device.create_texture(&TextureDescriptor {
        format: TextureFormat::Rgba8Unorm,
        ..texture_descriptor
    });
    let rgba_view = rgba_texture.create_view(&TextureViewDescriptor::default());

    let _ = valid(&device, || {
        device.create_external_texture(
            &ExternalTextureDescriptor {
                format: ExternalTextureFormat::Rgba,
                label: None,
                width: r_texture.width(),
                height: r_texture.height(),
                yuv_conversion_matrix: [0.0; 16],
                sample_transform: [0.0; 6],
                load_transform: [0.0; 6],
            },
            &[&rgba_view],
        )
    });
    let _ = valid(&device, || {
        device.create_external_texture(
            &ExternalTextureDescriptor {
                format: ExternalTextureFormat::Nv12,
                label: None,
                width: r_texture.width(),
                height: r_texture.height(),
                yuv_conversion_matrix: [0.0; 16],
                sample_transform: [0.0; 6],
                load_transform: [0.0; 6],
            },
            &[&r_view, &rg_view],
        )
    });
    let _ = valid(&device, || {
        device.create_external_texture(
            &ExternalTextureDescriptor {
                format: ExternalTextureFormat::Yu12,
                label: None,
                width: r_texture.width(),
                height: r_texture.height(),
                yuv_conversion_matrix: [0.0; 16],
                sample_transform: [0.0; 6],
                load_transform: [0.0; 6],
            },
            &[&r_view, &r_view, &r_view],
        )
    });

    // Wrong number of planes for format
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view, &r_view],
            )
        },
        Some("External texture format Rgba expects 1 planes, but given 2"),
    );
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Nv12,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view],
            )
        },
        Some("External texture format Nv12 expects 2 planes, but given 1"),
    );
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Yu12,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view, &r_view],
            )
        },
        Some("External texture format Yu12 expects 3 planes, but given 2"),
    );

    // Wrong plane formats
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view],
            )
        },
        Some("External texture format Rgba plane 0 expects format with 4 components but given view with format R8Unorm (1 components)"),
    );
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Nv12,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view, &rgba_view],
            )
        },
        Some("External texture format Nv12 plane 1 expects format with 2 components but given view with format Rgba8Unorm (4 components)"),
    );
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Yu12,
                    label: None,
                    width: r_texture.width(),
                    height: r_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&r_view, &rg_view, &r_view],
            )
        },
        Some("External texture format Yu12 plane 1 expects format with 1 components but given view with format Rg8Unorm (2 components)"),
    );

    // Wrong sample type
    let uint_texture = device.create_texture(&TextureDescriptor {
        format: TextureFormat::Rgba8Uint,
        ..texture_descriptor
    });
    let uint_view = uint_texture.create_view(&TextureViewDescriptor::default());
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: uint_texture.width(),
                    height: uint_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&uint_view],
            )
        },
        Some("External texture planes expect a filterable float sample type, but given view with format Rgba8Uint (sample type Uint)"),
    );

    // Wrong texture dimension
    let d3_texture = device.create_texture(&TextureDescriptor {
        dimension: TextureDimension::D3,
        ..texture_descriptor
    });
    let d3_view = d3_texture.create_view(&TextureViewDescriptor::default());
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: d3_texture.width(),
                    height: d3_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&d3_view],
            )
        },
        Some("External texture planes expect 2D dimension, but given view with dimension = D3"),
    );

    // Multisampled
    let multisampled_texture = device.create_texture(&TextureDescriptor {
        sample_count: 4,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        ..texture_descriptor
    });
    let multisampled_view = multisampled_texture.create_view(&TextureViewDescriptor::default());
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: multisampled_texture.width(),
                    height: multisampled_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&multisampled_view],
            )
        },
        Some("External texture planes cannot be multisampled, but given view with samples = 4"),
    );

    // Missing TEXTURE_BINDING
    let non_binding_texture = device.create_texture(&TextureDescriptor {
        usage: TextureUsages::STORAGE_BINDING,
        ..texture_descriptor
    });
    let non_binding_view = non_binding_texture.create_view(&TextureViewDescriptor::default());
    let _ = fail(
        &device,
        || {
            device.create_external_texture(
                &ExternalTextureDescriptor {
                    format: ExternalTextureFormat::Rgba,
                    label: None,
                    width: non_binding_texture.width(),
                    height: non_binding_texture.height(),
                    yuv_conversion_matrix: [0.0; 16],
                    sample_transform: [0.0; 6],
                    load_transform: [0.0; 6],
                },
                &[&non_binding_view],
            )
        },
        Some("Usage flags TextureUsages(STORAGE_BINDING) of TextureView with '' label do not contain required usage flags TextureUsages(TEXTURE_BINDING)"),
    );
}

/// Ensures an [`ExternalTexture`] can be bound to a [`BindingType::ExternalTexture`]
/// resource binding.
#[test]
fn external_texture_binding() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::EXTERNAL_TEXTURE,
        ..Default::default()
    });

    let bgl = valid(&device, || {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::ExternalTexture,
                count: None,
            }],
        })
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
    let external_texture_descriptor = ExternalTextureDescriptor {
        label: None,
        width: texture_descriptor.size.width,
        height: texture_descriptor.size.height,
        format: ExternalTextureFormat::Rgba,
        yuv_conversion_matrix: [0.0; 16],
        sample_transform: [0.0; 6],
        load_transform: [0.0; 6],
    };

    valid(&device, || {
        let texture = device.create_texture(&texture_descriptor);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let external_texture =
            device.create_external_texture(&external_texture_descriptor, &[&view]);

        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::ExternalTexture(&external_texture),
            }],
        })
    });
}

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

/// Ensures that submitting a command buffer referencing an external texture, any of
/// whose plane textures have already been destroyed, results in an error.
#[test]
fn destroyed_external_texture_plane() {
    let (device, queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::EXTERNAL_TEXTURE,
        ..Default::default()
    });

    let target_texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&TextureViewDescriptor::default());

    let plane_texture = device.create_texture(&TextureDescriptor {
        label: Some("External texture plane"),
        size: Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let plane_view = plane_texture.create_view(&TextureViewDescriptor::default());

    let external_texture = device.create_external_texture(
        &ExternalTextureDescriptor {
            format: ExternalTextureFormat::Rgba,
            label: None,
            width: plane_texture.width(),
            height: plane_texture.height(),
            yuv_conversion_matrix: [0.0; 16],
            sample_transform: [0.0; 6],
            load_transform: [0.0; 6],
        },
        &[&plane_view],
    );

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            "
@group(0) @binding(0)
var tex: texture_external;
@vertex fn vert_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0); }
@fragment fn frag_main() -> @location(0) vec4<f32> { return textureLoad(tex, vec2(0)); }",
        )),
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: &module,
            entry_point: None,
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &module,
            entry_point: None,
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: target_texture.format(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::ExternalTexture(&external_texture),
        }],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &target_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..0, 0..0);
    drop(pass);

    plane_texture.destroy();

    fail(
        &device,
        || queue.submit([encoder.finish()]),
        Some("Texture with 'External texture plane' label has been destroyed"),
    );
}
