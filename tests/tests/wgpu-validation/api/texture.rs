//! Tests of [`wgpu::Texture`] and related.

use std::sync::Arc;

use wgpu_test::{fail, fail_if, valid};

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

/// Ensures that `bytes_per_row` of buffer-texture copies must be a multiple of 256, or only
/// of the texel block copy size with [`wgpu::Features::UNRESTRICTED_BYTES_PER_ROW`].
#[test]
fn copy_bytes_per_row_alignment() {
    for (features, format, should_fail) in [
        (wgpu::Features::empty(), wgpu::TextureFormat::R8Unorm, true),
        (
            wgpu::Features::UNRESTRICTED_BYTES_PER_ROW,
            wgpu::TextureFormat::R8Unorm,
            false,
        ),
        (
            wgpu::Features::UNRESTRICTED_BYTES_PER_ROW,
            wgpu::TextureFormat::Rg8Unorm,
            true,
        ),
    ] {
        let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
            required_features: features,
            ..Default::default()
        });
        let size = wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffer_copy = wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(101),
                rows_per_image: None,
            },
        };

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_texture(buffer_copy, texture.as_image_copy(), size);
        fail_if(
            &device,
            should_fail,
            || encoder.finish(),
            Some("is not a multiple of"),
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(texture.as_image_copy(), buffer_copy, size);
        fail_if(
            &device,
            should_fail,
            || encoder.finish(),
            Some("is not a multiple of"),
        );
    }
}

/// Ensures that attempting to create a texture with [`wgpu::TextureUsages::TRANSIENT_ATTACHMENT`]
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
        - wgpu::TextureUsages::TRANSIENT_ATTACHMENT;

    for usage in invalid_usages {
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TRANSIENT_ATTACHMENT
            | usage;
        let invalid_texture_descriptor = wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        };
        fail(
            &device,
            || device.create_texture(&invalid_texture_descriptor),
            Some(&format!("Transient texture usage must be equal to `TRANSIENT_ATTACHMENT | RENDER_ATTACHMENT`, but got `{usage:?}`")),
        );
    }

    let invalid_texture_descriptor = wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
        view_formats: &[],
    };
    fail(
        &device,
        || device.create_texture(&invalid_texture_descriptor),
        Some("Transient texture usage must be equal to `TRANSIENT_ATTACHMENT | RENDER_ATTACHMENT`, but got `TextureUsages(TRANSIENT_ATTACHMENT)`"),
    );
}

/// Ensures that attempting to use a texture of [`wgpu::TextureUsages::TRANSIENT_ATTACHMENT`]
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT_ATTACHMENT,
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
                multiview_mask: None,
            });

            drop(invalid_render_pass);

            encoder.finish()
        },
      Some("Color attachment with `TRANSIENT_ATTACHMENT` usage can only be used with `LoadOp::Clear` or `LoadOp::DontCare` (if it is available) and  `StoreOp::Discard`. Operations `(Clear(Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }), Store)` were provided")
    );
}

/// <https://bugzilla.mozilla.org/show_bug.cgi?id=2053860>
#[test]
fn no_overflow_in_texture_selector() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
    // ignore validation errors to not panic before reaching `TextureView::invalid`
    device.on_uncaptured_error(Arc::new(|_| {}));
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 59,
            height: 401,
            depth_or_array_layers: 1948,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let _view = texture.create_view(&wgpu::TextureViewDescriptor {
        base_array_layer: 3385121660,
        array_layer_count: Some(3815184380),
        ..Default::default()
    });
}

fn planar_copy_device() -> (wgpu::Device, wgpu::Queue) {
    wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TEXTURE_FORMAT_NV12
            | wgpu::Features::TEXTURE_FORMAT_P010
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        ..Default::default()
    })
}

fn planar_copy_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    mip_level_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

fn copy_view(
    texture: &wgpu::Texture,
    mip_level: u32,
    aspect: wgpu::TextureAspect,
    (x, y): (u32, u32),
) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level,
        origin: wgpu::Origin3d { x, y, z: 0 },
        aspect,
    }
}

fn encode_texture_to_texture(
    device: &wgpu::Device,
    source: wgpu::TexelCopyTextureInfo<'_>,
    destination: wgpu::TexelCopyTextureInfo<'_>,
    copy_size: wgpu::Extent3d,
) -> wgpu::CommandBuffer {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_texture(source, destination, copy_size);
    encoder.finish()
}

/// Ensures that copies between a plane of a multi-planar texture and a texture of the
/// plane's format, between matching planes, and between whole multi-planar textures of
/// the same format pass validation.
#[test]
fn planar_copy_texture_to_texture_compatible() {
    use wgpu::{TextureAspect as Ta, TextureFormat as Tf};

    let (device, _queue) = planar_copy_device();
    let luma = extent(256, 128, 1);
    let chroma = extent(128, 64, 1);

    for (src_format, src_size, src_aspect, dst_format, dst_size, dst_aspect, copy_size) in [
        (Tf::NV12, luma, Ta::All, Tf::NV12, luma, Ta::All, luma),
        (Tf::P010, luma, Ta::All, Tf::P010, luma, Ta::All, luma),
        (Tf::NV12, luma, Ta::Plane0, Tf::R8Unorm, luma, Ta::All, luma),
        (
            Tf::NV12,
            luma,
            Ta::Plane1,
            Tf::Rg8Unorm,
            chroma,
            Ta::All,
            chroma,
        ),
        (
            Tf::P010,
            luma,
            Ta::Plane0,
            Tf::R16Unorm,
            luma,
            Ta::All,
            luma,
        ),
        (
            Tf::P010,
            luma,
            Ta::Plane1,
            Tf::Rg16Unorm,
            chroma,
            Ta::All,
            chroma,
        ),
        (Tf::R8Unorm, luma, Ta::All, Tf::NV12, luma, Ta::Plane0, luma),
        (
            Tf::Rg8Unorm,
            chroma,
            Ta::All,
            Tf::NV12,
            luma,
            Ta::Plane1,
            chroma,
        ),
        (
            Tf::Rg16Unorm,
            chroma,
            Ta::All,
            Tf::P010,
            luma,
            Ta::Plane1,
            chroma,
        ),
        (Tf::NV12, luma, Ta::Plane0, Tf::NV12, luma, Ta::Plane0, luma),
        (
            Tf::NV12,
            luma,
            Ta::Plane1,
            Tf::NV12,
            luma,
            Ta::Plane1,
            chroma,
        ),
    ] {
        let src = planar_copy_texture(&device, src_format, src_size, 1);
        let dst = planar_copy_texture(&device, dst_format, dst_size, 1);
        valid(&device, || {
            encode_texture_to_texture(
                &device,
                copy_view(&src, 0, src_aspect, (0, 0)),
                copy_view(&dst, 0, dst_aspect, (0, 0)),
                copy_size,
            )
        });
    }
}

/// Ensures that copies involving multi-planar textures fail validation when the
/// source and destination do not resolve to the same format.
#[test]
fn planar_copy_texture_to_texture_incompatible() {
    use wgpu::{TextureAspect as Ta, TextureFormat as Tf};

    let (device, _queue) = planar_copy_device();
    let size = extent(256, 128, 1);
    let copy_size = extent(2, 2, 1);

    for (src_format, src_aspect, dst_format, dst_aspect, expected) in [
        (
            Tf::NV12,
            Ta::All,
            Tf::P010,
            Ta::All,
            "(NV12) and destination format (P010)",
        ),
        (
            Tf::NV12,
            Ta::All,
            Tf::R8Unorm,
            Ta::All,
            "(NV12) and destination format (R8Unorm)",
        ),
        (
            Tf::R8Unorm,
            Ta::All,
            Tf::NV12,
            Ta::All,
            "(R8Unorm) and destination format (NV12)",
        ),
        (
            Tf::NV12,
            Ta::All,
            Tf::NV12,
            Ta::Plane0,
            "(NV12) and destination format (R8Unorm, in plane 0 of NV12)",
        ),
        (
            Tf::NV12,
            Ta::Plane0,
            Tf::NV12,
            Ta::All,
            "(R8Unorm, in plane 0 of NV12) and destination format (NV12)",
        ),
        (
            Tf::NV12,
            Ta::Plane0,
            Tf::NV12,
            Ta::Plane1,
            "(R8Unorm, in plane 0 of NV12) and destination format (Rg8Unorm, in plane 1 of NV12)",
        ),
        (
            Tf::NV12,
            Ta::Plane0,
            Tf::P010,
            Ta::Plane0,
            "(R8Unorm, in plane 0 of NV12) and destination format (R16Unorm, in plane 0 of P010)",
        ),
        (
            Tf::NV12,
            Ta::Plane1,
            Tf::R8Unorm,
            Ta::All,
            "(Rg8Unorm, in plane 1 of NV12) and destination format (R8Unorm)",
        ),
        (
            Tf::R8Unorm,
            Ta::All,
            Tf::NV12,
            Ta::Plane1,
            "(R8Unorm) and destination format (Rg8Unorm, in plane 1 of NV12)",
        ),
    ] {
        let src = planar_copy_texture(&device, src_format, size, 1);
        let dst = planar_copy_texture(&device, dst_format, size, 1);
        fail(
            &device,
            || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&src, 0, src_aspect, (0, 0)),
                    copy_view(&dst, 0, dst_aspect, (0, 0)),
                    copy_size,
                )
            },
            Some(&format!("Source format {expected} are not copy-compatible")),
        );
    }
}

/// Ensures that texture-to-texture copies fail validation when they select an aspect
/// that a multi-planar format does not have.
#[test]
fn planar_copy_texture_to_texture_invalid_aspect() {
    use wgpu::{TextureAspect as Ta, TextureFormat as Tf};

    let (device, _queue) = planar_copy_device();
    let size = extent(256, 128, 1);
    let copy_size = extent(2, 2, 1);
    let nv12 = planar_copy_texture(&device, Tf::NV12, size, 1);
    let r8 = planar_copy_texture(&device, Tf::R8Unorm, size, 1);

    for aspect in [Ta::Plane2, Ta::DepthOnly, Ta::StencilOnly] {
        let expected = format!("Unable to select texture aspect {aspect:?} from format NV12");
        fail(
            &device,
            || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&nv12, 0, aspect, (0, 0)),
                    copy_view(&r8, 0, Ta::All, (0, 0)),
                    copy_size,
                )
            },
            Some(&expected),
        );
        fail(
            &device,
            || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&r8, 0, Ta::All, (0, 0)),
                    copy_view(&nv12, 0, aspect, (0, 0)),
                    copy_size,
                )
            },
            Some(&expected),
        );
    }
}

/// Ensures that texture-to-texture copies of a single plane of a multi-planar texture
/// are bounds-checked against the size of that plane, and are not subject to the
/// chroma-subsampling alignment required for copies of all planes.
#[test]
fn planar_copy_texture_to_texture_plane_bounds() {
    use wgpu::{TextureAspect as Ta, TextureFormat as Tf};

    let (device, _queue) = planar_copy_device();
    let nv12 = planar_copy_texture(&device, Tf::NV12, extent(256, 128, 2), 2);
    let r8 = planar_copy_texture(&device, Tf::R8Unorm, extent(256, 128, 2), 1);
    let rg8 = planar_copy_texture(&device, Tf::Rg8Unorm, extent(256, 128, 3), 1);

    // (mip level, aspect, origin, copy size)
    for (mip_level, aspect, origin, copy_size) in [
        (0, Ta::Plane0, (0, 0), extent(256, 128, 2)),
        (0, Ta::Plane1, (0, 0), extent(128, 64, 2)),
        (0, Ta::Plane1, (2, 4), extent(126, 60, 1)),
        (1, Ta::Plane0, (0, 0), extent(128, 64, 1)),
        (1, Ta::Plane1, (0, 0), extent(64, 32, 1)),
        (0, Ta::Plane0, (1, 3), extent(5, 7, 1)),
        (0, Ta::Plane1, (1, 3), extent(3, 5, 1)),
    ] {
        let plane_dst = if aspect == Ta::Plane0 { &r8 } else { &rg8 };
        valid(&device, || {
            encode_texture_to_texture(
                &device,
                copy_view(&nv12, mip_level, aspect, origin),
                copy_view(plane_dst, 0, Ta::All, (0, 0)),
                copy_size,
            )
        });
        valid(&device, || {
            encode_texture_to_texture(
                &device,
                copy_view(plane_dst, 0, Ta::All, (0, 0)),
                copy_view(&nv12, mip_level, aspect, origin),
                copy_size,
            )
        });
    }

    // (mip level, origin, copy size, expected overrun, dimension, plane size)
    for (mip_level, origin, copy_size, expected, dimension, plane_size) in [
        (0, (0, 0), extent(256, 64, 1), "X 0..256", "X", 128),
        (0, (0, 0), extent(128, 128, 1), "Y 0..128", "Y", 64),
        (0, (2, 0), extent(128, 64, 1), "X 2..130", "X", 128),
        (0, (0, 2), extent(128, 64, 1), "Y 2..66", "Y", 64),
        (0, (0, 0), extent(128, 64, 3), "Z 0..3", "Z", 2),
        (1, (0, 0), extent(128, 32, 1), "X 0..128", "X", 64),
        (1, (0, 0), extent(64, 64, 1), "Y 0..64", "Y", 32),
    ] {
        fail(
            &device,
            || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&nv12, mip_level, Ta::Plane1, origin),
                    copy_view(&rg8, 0, Ta::All, (0, 0)),
                    copy_size,
                )
            },
            Some(&format!(
                "Copy of {expected} would end up overrunning the bounds of plane 1 of the \
                 Source texture of {dimension} size {plane_size}"
            )),
        );
        fail(
            &device,
            || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&rg8, 0, Ta::All, (0, 0)),
                    copy_view(&nv12, mip_level, Ta::Plane1, origin),
                    copy_size,
                )
            },
            Some(&format!(
                "Copy of {expected} would end up overrunning the bounds of plane 1 of the \
                 Destination texture of {dimension} size {plane_size}"
            )),
        );
    }
}

/// Ensures that texture-to-texture copies of all planes of a multi-planar texture are
/// bounds-checked against the size of plane 0, and must be aligned to the chroma
/// subsampling.
#[test]
fn planar_copy_texture_to_texture_all_planes_bounds_and_alignment() {
    use wgpu::{TextureAspect as Ta, TextureFormat as Tf};

    let (device, _queue) = planar_copy_device();

    for format in [Tf::NV12, Tf::P010] {
        let src = planar_copy_texture(&device, format, extent(256, 128, 1), 1);
        let dst = planar_copy_texture(&device, format, extent(256, 128, 1), 1);

        // (source origin, destination origin, copy size)
        for (src_origin, dst_origin, copy_size) in [
            ((0, 0), (0, 0), extent(256, 128, 1)),
            ((2, 4), (6, 2), extent(4, 8, 1)),
            ((0, 0), (0, 0), extent(0, 0, 1)),
        ] {
            valid(&device, || {
                encode_texture_to_texture(
                    &device,
                    copy_view(&src, 0, Ta::All, src_origin),
                    copy_view(&dst, 0, Ta::All, dst_origin),
                    copy_size,
                )
            });
        }

        // (source origin, destination origin, copy size, expected error)
        for (src_origin, dst_origin, copy_size, expected) in [
            (
                (1, 0),
                (0, 0),
                extent(2, 2, 1),
                "Copy origin's x component is not a multiple of the required alignment",
            ),
            (
                (0, 1),
                (0, 0),
                extent(2, 2, 1),
                "Copy origin's y component is not a multiple of the required alignment",
            ),
            (
                (0, 0),
                (1, 0),
                extent(2, 2, 1),
                "Copy origin's x component is not a multiple of the required alignment",
            ),
            (
                (0, 0),
                (0, 1),
                extent(2, 2, 1),
                "Copy origin's y component is not a multiple of the required alignment",
            ),
            (
                (0, 0),
                (0, 0),
                extent(3, 2, 1),
                "Copy width is not a multiple of the required alignment",
            ),
            (
                (0, 0),
                (0, 0),
                extent(2, 3, 1),
                "Copy height is not a multiple of the required alignment",
            ),
            (
                (0, 0),
                (0, 0),
                extent(258, 2, 1),
                "Copy of X 0..258 would end up overrunning the bounds of the Source texture \
                 of X size 256",
            ),
            (
                (2, 0),
                (0, 0),
                extent(256, 2, 1),
                "Copy of X 2..258 would end up overrunning the bounds of the Source texture \
                 of X size 256",
            ),
            (
                (0, 0),
                (0, 2),
                extent(2, 128, 1),
                "Copy of Y 2..130 would end up overrunning the bounds of the Destination \
                 texture of Y size 128",
            ),
        ] {
            fail(
                &device,
                || {
                    encode_texture_to_texture(
                        &device,
                        copy_view(&src, 0, Ta::All, src_origin),
                        copy_view(&dst, 0, Ta::All, dst_origin),
                        copy_size,
                    )
                },
                Some(expected),
            );
        }
    }
}

/// Ensures that texture-buffer copies of a single plane of a multi-planar texture are
/// bounds-checked against the size of that plane, and that copies of all planes or of a
/// nonexistent plane fail validation.
#[test]
fn planar_copy_texture_buffer() {
    use wgpu::TextureAspect as Ta;

    let (device, queue) = planar_copy_device();
    let nv12 = planar_copy_texture(&device, wgpu::TextureFormat::NV12, extent(256, 128, 1), 1);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 128 * 2,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let data = vec![0u8; 256 * 128 * 2];
    let layout = |bytes_per_row| wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(bytes_per_row),
        rows_per_image: None,
    };

    let to_buffer = |aspect, copy_size, bytes_per_row| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            copy_view(&nv12, 0, aspect, (0, 0)),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: layout(bytes_per_row),
            },
            copy_size,
        );
        encoder.finish()
    };
    let from_buffer = |aspect, copy_size, bytes_per_row| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: layout(bytes_per_row),
            },
            copy_view(&nv12, 0, aspect, (0, 0)),
            copy_size,
        );
        encoder.finish()
    };
    let write = |aspect, copy_size, bytes_per_row| {
        queue.write_texture(
            copy_view(&nv12, 0, aspect, (0, 0)),
            &data,
            layout(bytes_per_row),
            copy_size,
        );
    };

    for (aspect, copy_size, bytes_per_row) in [
        (Ta::Plane0, extent(256, 128, 1), 256),
        (Ta::Plane1, extent(128, 64, 1), 256),
    ] {
        valid(&device, || to_buffer(aspect, copy_size, bytes_per_row));
        valid(&device, || from_buffer(aspect, copy_size, bytes_per_row));
        valid(&device, || write(aspect, copy_size, bytes_per_row));
    }

    // (copy size, expected overrun)
    for (copy_size, expected) in [
        (
            extent(256, 64, 1),
            "X 0..256 would end up overrunning the bounds of plane 1",
        ),
        (
            extent(128, 128, 1),
            "Y 0..128 would end up overrunning the bounds of plane 1",
        ),
    ] {
        fail(
            &device,
            || to_buffer(Ta::Plane1, copy_size, 512),
            Some(&format!("{expected} of the Source texture")),
        );
        fail(
            &device,
            || from_buffer(Ta::Plane1, copy_size, 512),
            Some(&format!("{expected} of the Destination texture")),
        );
        fail(
            &device,
            || write(Ta::Plane1, copy_size, 512),
            Some(&format!("{expected} of the Destination texture")),
        );
    }

    let aspect_not_one = "Copy aspect must refer to a single aspect of texture format";
    let size = extent(256, 128, 1);
    fail(
        &device,
        || to_buffer(Ta::All, size, 256),
        Some(aspect_not_one),
    );
    fail(
        &device,
        || from_buffer(Ta::All, size, 256),
        Some(aspect_not_one),
    );
    fail(&device, || write(Ta::All, size, 256), Some(aspect_not_one));

    let size = extent(2, 2, 1);
    let invalid_aspect = "Unable to select texture aspect Plane2 from format NV12";
    fail(
        &device,
        || to_buffer(Ta::Plane2, size, 256),
        Some(invalid_aspect),
    );
    fail(
        &device,
        || from_buffer(Ta::Plane2, size, 256),
        Some(invalid_aspect),
    );
    fail(
        &device,
        || write(Ta::Plane2, size, 256),
        Some(invalid_aspect),
    );
}
