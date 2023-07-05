use wgpu_test::{fail, initialize_test, TestParameters};

#[test]
fn copy_overflow_z() {
    // A simple crash test exercising validation that used to happen a bit too
    // late, letting an integer overflow slip through.
    initialize_test(TestParameters::default(), |ctx| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let t1 = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let t2 = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::Rgba8Uint,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        fail(&ctx.device, || {
            // Validation should catch the silly selected z layer range without panicking.
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &t1,
                    mip_level: 1,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &t2,
                    mip_level: 1,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: 3824276442,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: 100,
                    height: 3,
                    depth_or_array_layers: 613286111,
                },
            );
            ctx.queue.submit(Some(encoder.finish()));
        });
    })
}
