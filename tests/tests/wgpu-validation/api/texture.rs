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
        Some("Aspect Plane0 is not in the source texture format R8Unorm"),
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
                "Aspect {view_aspect:?} is not in the source texture format {tex_format:?}"
            )),
        );
    }
}

/// Ensures that attempting to create a texture view from a specific plane of a
/// planar texture with an invalid format fails validation.
#[test]
fn planar_texture_view_plane_bad_format() {
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
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureAspect::Plane0,
        ),
        (
            wgpu::TextureFormat::P010,
            wgpu::TextureFormat::Rg16Unorm,
            wgpu::TextureAspect::Plane0,
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
                "unable to view texture {tex_format:?} as {view_format:?}"
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
