use wgpu::util::DeviceExt;
use wgpu_test::{
    apply, fail, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
    TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COPY_OVERFLOW_Z);
    vec.push(COPY_TEXTURE_TO_TEXTURE_3D);
}

#[apply(gpu_test!)]
static COPY_TEXTURE_TO_TEXTURE_3D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(copy_texture_to_texture_3d);

async fn copy_texture_to_texture_3d(ctx: wgpu_test::TestingContext) {
    struct Case {
        name: &'static str,
        src_origin: wgpu::Origin3d,
        dst_origin: wgpu::Origin3d,
        size: wgpu::Extent3d,
    }

    const TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 7,
        height: 5,
        depth_or_array_layers: 6,
    };

    let cases = [
        Case {
            name: "full texture",
            src_origin: wgpu::Origin3d::ZERO,
            dst_origin: wgpu::Origin3d::ZERO,
            size: TEXTURE_SIZE,
        },
        Case {
            name: "depth subrange",
            src_origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            dst_origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
            size: wgpu::Extent3d {
                width: 7,
                height: 5,
                depth_or_array_layers: 4,
            },
        },
        Case {
            name: "offset subvolume",
            src_origin: wgpu::Origin3d { x: 1, y: 1, z: 2 },
            dst_origin: wgpu::Origin3d { x: 3, y: 0, z: 1 },
            size: wgpu::Extent3d {
                width: 3,
                height: 3,
                depth_or_array_layers: 2,
            },
        },
    ];

    let descriptor = wgpu::TextureDescriptor {
        label: None,
        dimension: wgpu::TextureDimension::D3,
        size: TEXTURE_SIZE,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        mip_level_count: 1,
        sample_count: 1,
        view_formats: &[],
    };
    let texel_count =
        (TEXTURE_SIZE.width * TEXTURE_SIZE.height * TEXTURE_SIZE.depth_or_array_layers) as usize;
    let src_data: Vec<u8> = (1..=texel_count).map(|value| value as u8).collect();
    let src = ctx.device.create_texture_with_data(
        &ctx.queue,
        &descriptor,
        wgpu::util::TextureDataOrder::LayerMajor,
        &src_data,
    );
    let index = |origin: wgpu::Origin3d, z, y| {
        (((origin.z + z) * TEXTURE_SIZE.height + origin.y + y) * TEXTURE_SIZE.width + origin.x)
            as usize
    };

    for case in cases {
        let mut expected = vec![0; texel_count];
        for z in 0..case.size.depth_or_array_layers {
            for y in 0..case.size.height {
                let src_start = index(case.src_origin, z, y);
                let dst_start = index(case.dst_origin, z, y);
                let width = case.size.width as usize;
                expected[dst_start..dst_start + width]
                    .copy_from_slice(&src_data[src_start..src_start + width]);
            }
        }

        let dst = ctx.device.create_texture(&descriptor);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                origin: case.src_origin,
                ..src.as_image_copy()
            },
            wgpu::TexelCopyTextureInfo {
                origin: case.dst_origin,
                ..dst.as_image_copy()
            },
            case.size,
        );

        let readback = ReadbackBuffers::new(&ctx.device, &dst);
        readback.copy_from(&ctx.device, &mut encoder, &dst);
        ctx.queue.submit(Some(encoder.finish()));

        let actual = readback.retrieve(&ctx, wgpu::TextureAspect::All).await;
        assert_eq!(actual.all(), expected, "{}", case.name);
    }
}

#[apply(gpu_test!)]
static COPY_OVERFLOW_Z: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|ctx| {
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

        fail(
            &ctx.device,
            || {
                // Validation should catch the silly selected z layer range without panicking.
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &t1,
                        mip_level: 1,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
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
            },
            Some("unable to select texture mip level"),
        );
    });
