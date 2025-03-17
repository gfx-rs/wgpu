//! Tests of [`wgpu::Texture`] and related.

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
