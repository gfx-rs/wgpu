//! Tests of [`wgpu::Texture`] and related.

use wgpu_test::{fail, valid};

/// Ensures that submitting a command buffer referencing an already destroyed texture
/// results in an error.
#[test]
#[should_panic = "Texture with 'dst' label has been destroyed"]
fn destroyed_texture() {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let texture_src = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("src"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_dst = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dst"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture_src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture_dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );

    texture_dst.destroy();

    queue.submit([encoder.finish()]);
}

/// Ensures that creating a texture view from a specific plane of a planar
/// texture works as expected.
#[test]
fn planar_texture_view_plane() {
    let required_features = wgpu::Features::TEXTURE_FORMAT_NV12
        | wgpu::Features::TEXTURE_FORMAT_P010
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            wgpu::TextureFormat::NV12,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureAspect::Plane0,
        ),
        (
            wgpu::TextureFormat::NV12,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureAspect::Plane1,
        ),
        (
            wgpu::TextureFormat::P010,
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureAspect::Plane0,
        ),
        (
            wgpu::TextureFormat::P010,
            wgpu::TextureFormat::Rg16Unorm,
            wgpu::TextureAspect::Plane1,
        ),
    ] {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: tex_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        valid(&device, || {
            let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                aspect: view_aspect,
                ..Default::default()
            });
        });
    }
}

/// Ensures that attempting to create a texture view from a specific plane of a
/// non-planar texture fails validation.
#[test]
fn non_planar_texture_view_plane() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
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
        &device,
        || {
            let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::Plane0,
                ..Default::default()
            });
        },
        Some("Aspect Plane0 is not a valid aspect of the source texture format R8Unorm"),
    );
}

/// Ensures that attempting to create a texture view from an invalid plane of a
/// planar texture fails validation.
#[test]
fn planar_texture_view_plane_out_of_bounds() {
    let required_features = wgpu::Features::TEXTURE_FORMAT_NV12
        | wgpu::Features::TEXTURE_FORMAT_P010
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            wgpu::TextureFormat::NV12,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureAspect::Plane2,
        ),
        (
            wgpu::TextureFormat::P010,
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureAspect::Plane2,
        ),
    ] {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: tex_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        fail(
            &device,
            || {
                let _ = tex.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(view_format),
                    aspect: view_aspect,
                    ..Default::default()
                });
            },
            Some(&format!(
                "Aspect {view_aspect:?} is not a valid aspect of the source texture format {tex_format:?}"
            )),
        );
    }
}

/// Ensures that attempting to create a texture view from a specific plane of a
/// planar texture with an invalid format fails validation.
#[test]
fn planar_texture_bad_view_format() {
    let required_features = wgpu::Features::TEXTURE_FORMAT_NV12
        | wgpu::Features::TEXTURE_FORMAT_P010
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };
    for (tex_format, view_format) in [
        (wgpu::TextureFormat::NV12, wgpu::TextureFormat::Rg8Unorm),
        (wgpu::TextureFormat::P010, wgpu::TextureFormat::Rg16Unorm),
    ] {
        fail(
            &device,
            || {
                let _ = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    size,
                    format: tex_format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[view_format],
                });
            },
            Some(&format!(
                "The view format {view_format:?} is not compatible with texture \
                 format {tex_format:?}, only changing srgb-ness is allowed."
            )),
        );
    }
}

/// Ensures that attempting to create a planar texture with an invalid size
/// fails validation.
#[test]
fn planar_texture_bad_size() {
    let required_features =
        wgpu::Features::TEXTURE_FORMAT_NV12 | wgpu::Features::TEXTURE_FORMAT_P010;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 255,
        height: 255,
        depth_or_array_layers: 1,
    };
    for format in [wgpu::TextureFormat::NV12, wgpu::TextureFormat::P010] {
        fail(
            &device,
            || {
                let _ = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    dimension: wgpu::TextureDimension::D2,
                    size,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: &[],
                });
            },
            Some(&format!(
                "width {} is not a multiple of {format:?}'s width multiple requirement",
                size.width
            )),
        );
    }
}

/// Ensures that creating a planar textures that support `RENDER_ATTACHMENT` usage
/// is possible.
#[test]
fn planar_texture_render_attachment() {
    let required_features = wgpu::Features::TEXTURE_FORMAT_NV12;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    for (tex_format, view_format, view_aspect) in [
        (
            wgpu::TextureFormat::NV12,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureAspect::Plane0,
        ),
        (
            wgpu::TextureFormat::NV12,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureAspect::Plane1,
        ),
    ] {
        valid(&device, || {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                dimension: wgpu::TextureDimension::D2,
                size,
                format: tex_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });

            let _ = texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(view_format),
                aspect: view_aspect,
                ..Default::default()
            });
        });
    }
}

/// Ensures that creating a planar textures with `RENDER_ATTACHMENT`
/// for non renderable planar formats fails validation.
#[test]
fn planar_texture_render_attachment_unsupported() {
    let required_features =
        wgpu::Features::TEXTURE_FORMAT_P010 | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    fail(
        &device,
        || {
            let _ = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                dimension: wgpu::TextureDimension::D2,
                size,
                format: wgpu::TextureFormat::P010,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[],
            });
        },
        Some("Texture usages TextureUsages(RENDER_ATTACHMENT) are not allowed on a texture of type P010"),
    );
}

/// Creates a texture and a buffer, and encodes a copy from the texture to the
/// buffer.
fn encode_copy_texture_to_buffer(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    aspect: wgpu::TextureAspect,
    bytes_per_texel: u32,
) -> wgpu::CommandEncoder {
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (size.width * size.height * bytes_per_texel) as u64,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_texel),
                rows_per_image: None,
            },
        },
        size,
    );

    encoder
}

/// Ensures that attempting to copy a texture with a forbidden source format to
/// a buffer fails validation.
#[test]
fn copy_texture_to_buffer_forbidden_format() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let format = wgpu::TextureFormat::Depth24Plus;

    let encoder = encode_copy_texture_to_buffer(&device, format, wgpu::TextureAspect::All, 4);

    fail(
        &device,
        || {
            encoder.finish();
        },
        Some(&format!(
            "Copying from textures with format {format:?} is forbidden"
        )),
    );
}

/// Ensures that attempting ta copy a texture with a forbidden source
/// format/aspect combination to a buffer fails validation.
#[test]
fn copy_texture_to_buffer_forbidden_format_aspect() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let format = wgpu::TextureFormat::Depth24PlusStencil8;
    let aspect = wgpu::TextureAspect::DepthOnly;

    let encoder = encode_copy_texture_to_buffer(&device, format, aspect, 4);

    fail(
        &device,
        || {
            encoder.finish();
        },
        Some(&format!(
            "Copying from textures with format {format:?} and aspect {aspect:?} is forbidden"
        )),
    );
}

/// Creates a texture and a buffer, and encodes a copy from the buffer to the
/// texture.
fn encode_copy_buffer_to_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    aspect: wgpu::TextureAspect,
    bytes_per_texel: u32,
) -> wgpu::CommandEncoder {
    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (size.width * size.height * bytes_per_texel) as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * bytes_per_texel),
                rows_per_image: None,
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        size,
    );

    encoder
}

/// Ensures that attempting to copy a buffer to a texture with a forbidden
/// destination format fails validation.
#[test]
fn copy_buffer_to_texture_forbidden_format() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    for format in [
        wgpu::TextureFormat::Depth24Plus,
        wgpu::TextureFormat::Depth32Float,
    ] {
        let encoder = encode_copy_buffer_to_texture(&device, format, wgpu::TextureAspect::All, 4);

        fail(
            &device,
            || {
                encoder.finish();
            },
            Some(&format!(
                "Copying to textures with format {format:?} is forbidden"
            )),
        );
    }
}

/// Ensures that attempting to copy a buffer to a texture with a forbidden
/// destination format/aspect combination fails validation.
#[test]
fn copy_buffer_to_texture_forbidden_format_aspect() {
    let required_features = wgpu::Features::DEPTH32FLOAT_STENCIL8;
    let device_desc = wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    };
    let (device, _queue) = wgpu::Device::noop(&device_desc);

    let aspect = wgpu::TextureAspect::DepthOnly;

    for (format, bytes_per_texel) in [
        (wgpu::TextureFormat::Depth24PlusStencil8, 4),
        (wgpu::TextureFormat::Depth32FloatStencil8, 8),
    ] {
        let encoder = encode_copy_buffer_to_texture(&device, format, aspect, bytes_per_texel);

        fail(
            &device,
            || {
                encoder.finish();
            },
            Some(&format!(
                "Copying to textures with format {format:?} and aspect {aspect:?} is forbidden"
            )),
        );
    }
}

/// Ensures that attempting to create a texture with [`wgpu::TextureUsages::TRANSIENT`]
/// and its unsupported usages fails validation.
#[test]
fn transient_invalid_usage() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let invalid_usages = wgpu::TextureUsages::all()
        - wgpu::TextureUsages::RENDER_ATTACHMENT
        - wgpu::TextureUsages::TRANSIENT;

    for usage in invalid_usages {
        let invalid_texture_descriptor = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT | usage,
            view_formats: &[],
        };
        fail(
            &device,
            || device.create_texture(&invalid_texture_descriptor),
            Some(&format!("Texture usage TextureUsages(TRANSIENT) is not compatible with texture usage {usage:?}")),
        );
    }

    let invalid_texture_descriptor = wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TRANSIENT,
        view_formats: &[],
    };
    fail(
        &device,
        || device.create_texture(&invalid_texture_descriptor),
        Some("Invalid usage flags TextureUsages(TRANSIENT)"),
    );
}

/// Ensures that attempting to use a texture of [`wgpu::TextureUsages::TRANSIENT`]
/// with [`wgpu::StoreOp::Store`] fails validation.
#[test]
fn transient_invalid_storeop() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let transient_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT,
        view_formats: &[],
    });

    fail(
        &device,
        || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let invalid_render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &transient_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            drop(invalid_render_pass);

            encoder.finish()
        },
      Some("Color attachment's usage contains TextureUsages(TRANSIENT). This can only be used with StoreOp::Discard, but StoreOp::Store was provided")
    );
}
