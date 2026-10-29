//! Tests for the TEXTURE_COMPRESSION_UNALIGNED feature.
//!
//! Validation-only tests for this feature live in
//! `wgpu-validation/api/texture.rs`.

use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration, GpuTestInitializer,
    TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(UNALIGNED_WRITE_READBACK);
}

#[apply(gpu_test!)]
static UNALIGNED_WRITE_READBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_COMPRESSION_BC
                    | wgpu::Features::TEXTURE_COMPRESSION_UNALIGNED,
            )
            // compressed texture copy to buffer not yet implemented in the GL backend
            .expect_fail(FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_async(|ctx| async move {
        // A 5x5 BC1 texture has a physical size of 8x8: 2x2 blocks of 8 bytes each.
        // Copies address whole blocks, so writing and reading back the full physical
        // size must round-trip the block data, including the partial edge blocks.
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 5,
                height: 5,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bc1RgbaUnorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let data: Vec<u8> = (0..32).collect();
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );

        let buffers = ReadbackBuffers::new(&ctx.device, &texture);
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        buffers.copy_from(&ctx.device, &mut encoder, &texture);
        ctx.queue.submit([encoder.finish()]);
        buffers.assert_buffer_contents(&ctx, &data).await;
    });
