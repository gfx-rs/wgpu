//! Tests for the TEXTURE_COMPRESSION_UNALIGNED feature.

use wgpu_test::{
    apply, fail, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        UNALIGNED_SIZE_REQUIRES_FEATURE,
        UNALIGNED_SIZE_WITH_FEATURE,
        UNALIGNED_WRITE_READBACK,
    ]);
}

fn descriptor(width: u32, height: u32, mip_level_count: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }
}

const UNALIGNED_SIZES: [(u32, u32); 4] = [(5, 4), (4, 5), (5, 5), (1, 1)];

#[apply(gpu_test!)]
static UNALIGNED_SIZE_REQUIRES_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::TEXTURE_COMPRESSION_BC)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        let _aligned = ctx.device.create_texture(&descriptor(4, 4, 1));
        for (width, height) in UNALIGNED_SIZES {
            fail(
                &ctx.device,
                || drop(ctx.device.create_texture(&descriptor(width, height, 1))),
                Some("is not a multiple of"),
            );
        }
    });

#[apply(gpu_test!)]
static UNALIGNED_SIZE_WITH_FEATURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_COMPRESSION_BC
                    | wgpu::Features::TEXTURE_COMPRESSION_UNALIGNED,
            )
            .enable_noop(),
    )
    .run_sync(|ctx| {
        for (width, height) in UNALIGNED_SIZES {
            let _ = ctx.device.create_texture(&descriptor(width, height, 1));
        }
        // Full mip chains: the smaller mips of an unaligned texture also have
        // partial edge blocks.
        let _ = ctx.device.create_texture(&descriptor(5, 5, 3));
        let _ = ctx.device.create_texture(&descriptor(7, 3, 3));
    });

#[apply(gpu_test!)]
static UNALIGNED_WRITE_READBACK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(
        wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_UNALIGNED,
    ))
    .run_async(|ctx| async move {
        // A 5x5 BC1 texture has a physical size of 8x8: 2x2 blocks of 8 bytes each.
        // Copies address whole blocks, so writing and reading back the full physical
        // size must round-trip the block data, including the partial edge blocks.
        let texture = ctx.device.create_texture(&descriptor(5, 5, 1));
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
