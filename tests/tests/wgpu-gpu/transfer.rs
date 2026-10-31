use wgpu::util::DeviceExt;
use wgpu_test::{
    apply, fail, gpu_test, image::ReadbackBuffers, FailureCase, GpuTestConfiguration,
    GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COPY_OVERFLOW_Z);
    vec.push(COPY_TEXTURE_TO_TEXTURE_2D);
    vec.push(COPY_TEXTURE_TO_TEXTURE_3D);
    vec.push(COPY_UNALIGNED_ROWS);
    vec.push(COPY_UNALIGNED_COMPRESSED_ROWS);
}

#[apply(gpu_test!)]
static COPY_TEXTURE_TO_TEXTURE_2D: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| copy_texture_to_texture(ctx, wgpu::TextureDimension::D2));

#[apply(gpu_test!)]
static COPY_TEXTURE_TO_TEXTURE_3D: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| copy_texture_to_texture(ctx, wgpu::TextureDimension::D3));

async fn copy_texture_to_texture(
    ctx: wgpu_test::TestingContext,
    dimension: wgpu::TextureDimension,
) {
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
            name: "single slice",
            src_origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            dst_origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
            size: wgpu::Extent3d {
                width: 7,
                height: 5,
                depth_or_array_layers: 1,
            },
        },
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
        dimension,
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

#[apply(gpu_test!)]
static COPY_UNALIGNED_ROWS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::UNRESTRICTED_BYTES_PER_ROW))
    .run_async(|ctx| {
        copy_unaligned_rows(
            ctx,
            &[
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureFormat::Depth16Unorm,
                wgpu::TextureFormat::Stencil8,
            ],
        )
    });

#[apply(gpu_test!)]
static COPY_UNALIGNED_COMPRESSED_ROWS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::UNRESTRICTED_BYTES_PER_ROW | wgpu::Features::TEXTURE_COMPRESSION_BC,
            )
            // GL doesn't implement copies from compressed textures to buffers.
            .skip(FailureCase::backend(wgpu::Backends::GL)),
    )
    .run_async(|ctx| {
        copy_unaligned_rows(
            ctx,
            &[
                wgpu::TextureFormat::Bc1RgbaUnorm,
                wgpu::TextureFormat::Bc7RgbaUnorm,
            ],
        )
    });

/// Copies an array texture from a buffer and back, with rows and images padded by one block
/// and a non-zero offset, so copies must honour a `bytes_per_row` that is neither tight nor a
/// multiple of 256.
async fn copy_unaligned_rows(ctx: wgpu_test::TestingContext, formats: &[wgpu::TextureFormat]) {
    const BLOCKS: wgpu::Extent3d = wgpu::Extent3d {
        width: 3,
        height: 2,
        depth_or_array_layers: 2,
    };
    const OFFSET: u32 = 16;
    for &format in formats {
        if format.is_depth_stencil_format()
            && !ctx
                .adapter_downlevel_capabilities
                .flags
                .contains(wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
        {
            continue;
        }
        let (block_width, block_height) = format.block_dimensions();
        let block_size = format.block_copy_size(None).unwrap();
        let size = wgpu::Extent3d {
            width: BLOCKS.width * block_width,
            height: BLOCKS.height * block_height,
            depth_or_array_layers: BLOCKS.depth_or_array_layers,
        };
        let bytes_per_row = (BLOCKS.width + 1) * block_size;
        let rows_per_image = BLOCKS.height + 1;
        let layout = wgpu::TexelCopyBufferLayout {
            offset: OFFSET.into(),
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(rows_per_image),
        };
        let image_bytes = bytes_per_row * rows_per_image;
        let data: Vec<u8> = (0..OFFSET + image_bytes * BLOCKS.depth_or_array_layers)
            .map(|index| {
                let Some(byte) = index.checked_sub(OFFSET) else {
                    return 0;
                };
                let row = byte % image_bytes / bytes_per_row;
                let row_byte = byte % bytes_per_row;
                if row < BLOCKS.height && row_byte < BLOCKS.width * block_size {
                    byte as u8 | 1
                } else {
                    0
                }
            })
            .collect();
        let upload = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &data,
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &upload,
                layout,
            },
            texture.as_image_copy(),
            size,
        );
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout,
            },
            size,
        );
        ctx.queue.submit(Some(encoder.finish()));

        readback.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        assert_eq!(
            *readback.slice(..).get_mapped_range().unwrap(),
            data[..],
            "{format:?}"
        );
    }
}
