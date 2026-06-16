use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(HOST_IMAGE_UPLOAD);
    vec.push(HOST_IMAGE_UPLOAD_MAPPED_AT_CREATION);
    vec.push(HOST_IMAGE_DOWNLOAD);
    vec.push(HOST_IMAGE_READ_UNINITIALIZED);
    vec.push(HOST_IMAGE_PARTIAL_WRITE);
    vec.push(HOST_IMAGE_DEPTH_ROUNDTRIP);
    vec.push(HOST_IMAGE_PLANAR_ROUNDTRIP);
}

/// Upload pixel data into a HOST_VISIBLE texture via the encoder-mapped path
/// (`map_texture_on_completion`), then read it back through a COPY_DST buffer
/// and verify the round-trip.
async fn host_image_upload(ctx: TestingContext, start_mapped: bool) {
    let width = 4u32;
    let height = 4u32;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let bytes_per_pixel = 4u32;

    // Dense row stride for host (CPU) copies — no alignment requirement.
    let host_bytes_per_row = bytes_per_pixel * width; // 16
    let host_data_size = (host_bytes_per_row * height) as usize; // 64

    // GPU copies require rows aligned to COPY_BYTES_PER_ROW_ALIGNMENT.
    let gpu_bytes_per_row = host_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT); // 256
    let gpu_data_size = (gpu_bytes_per_row * height) as usize; // 1024

    // Dense pixel data — each byte is its own index.
    let upload_data: Vec<u8> = (0..host_data_size).map(|i| i as u8).collect();

    // Texture needs HOST_VISIBLE (for CPU access) and COPY_SRC (for readback).
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        mapped_at_creation: start_mapped,
    });

    // Ask the GPU to transition the texture to HOST_COPY layout on completion.
    if !start_mapped {
        let mut map_encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("map encoder"),
            });
        map_encoder.map_texture_on_completion(&texture, Box::new(|_| {}));
        ctx.queue.submit(Some(map_encoder.finish()));

        // Wait until the submission is done — texture is now CPU-accessible.
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
    }

    // Write and verify via CPU.
    {
        let mapped = texture.get_mapped().expect("texture should be mapped");

        // Upload dense pixel data.
        mapped.copy_from_memory(
            wgpu::TexelCopyTextureInfoBase {
                texture: (),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(host_bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Verify the CPU write is visible immediately.
        let mut cpu_readback = vec![0u8; host_data_size];
        mapped.copy_to_memory(
            wgpu::TexelCopyTextureInfoBase {
                texture: (),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mut cpu_readback,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(host_bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        assert_eq!(cpu_readback, upload_data, "CPU round-trip failed");
    } // MappedTexture dropped — safe to unmap.

    // Return the texture to GPU ownership.
    texture.unmap();

    // Copy texture → buffer for GPU readback.
    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: gpu_data_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut copy_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy encoder"),
        });
    copy_encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(gpu_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(Some(copy_encoder.finish()));

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    // GPU readback has aligned rows (host_bytes_per_row pixel bytes + padding per row).
    // Verify only the actual pixel bytes within each row.
    let data = slice.get_mapped_range().unwrap();
    for row in 0..height as usize {
        let gpu_row_start = row * gpu_bytes_per_row as usize;
        let host_row_start = row * host_bytes_per_row as usize;
        assert_eq!(
            &data[gpu_row_start..gpu_row_start + host_bytes_per_row as usize],
            &upload_data[host_row_start..host_row_start + host_bytes_per_row as usize],
            "row {row} mismatch",
        );
    }
}

#[gpu_test]
static HOST_IMAGE_UPLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_upload(ctx, false).await;
    });

#[gpu_test]
static HOST_IMAGE_UPLOAD_MAPPED_AT_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_upload(ctx, true).await;
    });

/// Upload pixel data into a HOST_VISIBLE texture via a GPU queue transfer
/// (`queue.write_texture`), then read it back via `copy_to_memory` and verify.
async fn host_image_download(ctx: TestingContext) {
    let width = 4u32;
    let height = 4u32;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let bytes_per_pixel = 4u32;

    let host_bytes_per_row = bytes_per_pixel * width; // 16
    let host_data_size = (host_bytes_per_row * height) as usize; // 64

    // queue.write_texture requires rows aligned to COPY_BYTES_PER_ROW_ALIGNMENT.
    let gpu_bytes_per_row = host_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT); // 256
    let gpu_data_size = (gpu_bytes_per_row * height) as usize; // 1024

    // Dense pixel data, then scattered into an aligned upload buffer.
    let pixel_data: Vec<u8> = (0..host_data_size).map(|i| i as u8).collect();
    let mut upload_data = vec![0u8; gpu_data_size];
    for row in 0..height as usize {
        let src = row * host_bytes_per_row as usize;
        let dst = row * gpu_bytes_per_row as usize;
        upload_data[dst..dst + host_bytes_per_row as usize]
            .copy_from_slice(&pixel_data[src..src + host_bytes_per_row as usize]);
    }

    // HOST_VISIBLE for CPU read, COPY_DST for GPU write.
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible download texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
        mapped_at_creation: false,
    });

    // Upload via GPU queue transfer.
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &upload_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(gpu_bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // Transition texture to HOST_COPY layout after the queue write completes.
    let mut map_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("map encoder"),
        });
    map_encoder.map_texture_on_completion(&texture, Box::new(|_| {}));
    ctx.queue.submit(Some(map_encoder.finish()));

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    // Download via host image copy, using dense (unaligned) row stride.
    let mut cpu_readback = vec![0u8; host_data_size];
    {
        let mapped = texture.get_mapped().expect("texture should be mapped");
        mapped.copy_to_memory(
            wgpu::TexelCopyTextureInfoBase {
                texture: (),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &mut cpu_readback,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(host_bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.unmap();

    assert_eq!(cpu_readback, pixel_data, "host image download mismatch");
}

#[gpu_test]
static HOST_IMAGE_DOWNLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_download(ctx).await;
    });

/// A HOST_VISIBLE texture that has never been written must read back as zeros
/// on the host (lazy zero-init), not as uninitialized driver memory.
async fn host_image_read_uninitialized(ctx: TestingContext) {
    let width = 4u32;
    let height = 4u32;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let bytes_per_pixel = 4u32;
    let host_bytes_per_row = bytes_per_pixel * width;
    let host_data_size = (host_bytes_per_row * height) as usize;

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("uninitialized host texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::HOST_VISIBLE,
        view_formats: &[],
        mapped_at_creation: true,
    });

    let mapped = texture.get_mapped().expect("texture should be mapped");

    // Pre-fill with a sentinel so a no-op read would be visibly wrong.
    let mut readback = vec![0xABu8; host_data_size];
    mapped.copy_to_memory(
        wgpu::TexelCopyTextureInfoBase {
            texture: (),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &mut readback,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(host_bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    drop(mapped);
    texture.unmap();

    assert!(
        readback.iter().all(|&b| b == 0),
        "never-written HOST_VISIBLE texture must read back as zeros, got {readback:?}",
    );
}

#[gpu_test]
static HOST_IMAGE_READ_UNINITIALIZED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_read_uninitialized(ctx).await;
    });

/// A partial host write must not leave the uncovered part of the layer as
/// uninitialized memory: the untouched region must read back as zero while the
/// written sub-rect keeps its data.
async fn host_image_partial_write(ctx: TestingContext) {
    let width = 4u32;
    let height = 4u32;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let bytes_per_pixel = 4u32;
    let host_bytes_per_row = bytes_per_pixel * width;
    let host_data_size = (host_bytes_per_row * height) as usize;

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("partial-write host texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::HOST_VISIBLE,
        view_formats: &[],
        mapped_at_creation: true,
    });

    let mapped = texture.get_mapped().expect("texture should be mapped");

    // Write only the top-left 2x2 sub-rect, with a non-zero pattern.
    let sub_w = 2u32;
    let sub_h = 2u32;
    let sub_bytes_per_row = bytes_per_pixel * sub_w;
    let sub_data: Vec<u8> = (0..(sub_bytes_per_row * sub_h))
        .map(|i| i as u8 | 0x80)
        .collect();
    mapped.copy_from_memory(
        wgpu::TexelCopyTextureInfoBase {
            texture: (),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &sub_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(sub_bytes_per_row),
            rows_per_image: Some(sub_h),
        },
        wgpu::Extent3d {
            width: sub_w,
            height: sub_h,
            depth_or_array_layers: 1,
        },
    );

    // Read back the whole texture.
    let mut full = vec![0u8; host_data_size];
    mapped.copy_to_memory(
        wgpu::TexelCopyTextureInfoBase {
            texture: (),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &mut full,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(host_bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    drop(mapped);
    texture.unmap();

    for y in 0..height {
        for x in 0..width {
            let off = (y * host_bytes_per_row + x * bytes_per_pixel) as usize;
            let pixel = &full[off..off + bytes_per_pixel as usize];
            if x < sub_w && y < sub_h {
                let soff = (y * sub_bytes_per_row + x * bytes_per_pixel) as usize;
                assert_eq!(
                    pixel,
                    &sub_data[soff..soff + bytes_per_pixel as usize],
                    "covered pixel ({x},{y}) lost its data",
                );
            } else {
                assert_eq!(
                    pixel,
                    &[0, 0, 0, 0],
                    "uncovered pixel ({x},{y}) must be zero-initialized",
                );
            }
        }
    }
}

#[gpu_test]
static HOST_IMAGE_PARTIAL_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_partial_write(ctx).await;
    });

/// Round-trip a depth texture (`Depth32Float`) through host copies: write depth
/// values from the CPU, read them straight back, and verify. Exercises the
/// non-color, single depth-aspect host-copy + zero-init path.
async fn host_image_depth_roundtrip(ctx: TestingContext) {
    let format = wgpu::TextureFormat::Depth32Float;
    // Host-copy support is per-format (e.g. Vulkan's host-transfer format
    // feature); many drivers don't support it for depth formats. Skip if so.
    if !ctx
        .adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
    {
        log::info!("skipping host_image_depth_roundtrip: {format:?} is not host-copyable");
        return;
    }

    let width = 4u32;
    let height = 4u32;
    let bytes_per_row = 4 * width; // Depth32Float: 4 bytes/texel
    let data_size = (bytes_per_row * height) as usize;

    // Arbitrary depth values in [0, 1], copied verbatim (no format conversion).
    let upload: Vec<u8> = (0..width * height)
        .flat_map(|i| (i as f32 / (width * height) as f32).to_ne_bytes())
        .collect();

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        mapped_at_creation: true,
    });

    let mapped = texture.get_mapped().expect("texture should be mapped");
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(bytes_per_row),
        rows_per_image: Some(height),
    };
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    mapped.copy_from_memory(
        wgpu::TexelCopyTextureInfoBase {
            texture: (),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &upload,
        layout,
        size,
    );
    let mut readback = vec![0u8; data_size];
    mapped.copy_to_memory(
        wgpu::TexelCopyTextureInfoBase {
            texture: (),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &mut readback,
        layout,
        size,
    );
    assert_eq!(readback, upload, "depth host round-trip failed");

    drop(mapped);
    texture.unmap();
}

#[gpu_test]
static HOST_IMAGE_DEPTH_ROUNDTRIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_IMAGE_COPY))
    .run_async(|ctx| async move {
        host_image_depth_roundtrip(ctx).await;
    });

/// Round-trip both planes of an `NV12` texture through host copies, verifying
/// per-plane (`Plane0` full-res / `Plane1` half-res) host addressing.
///
/// NOTE: gated on `TEXTURE_FORMAT_NV12`, so it is skipped on backends that don't
/// expose NV12 (including Metal). Validated only where NV12 + `HOST_IMAGE_COPY`
/// are both available (Vulkan / DX12).
async fn host_image_planar_roundtrip(ctx: TestingContext) {
    // Host-copy support is per-format; skip if this adapter can't host-copy NV12.
    if !ctx
        .adapter
        .get_texture_format_features(wgpu::TextureFormat::NV12)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
    {
        log::info!("skipping host_image_planar_roundtrip: NV12 is not host-copyable");
        return;
    }

    let width = 4u32;
    let height = 4u32;

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible NV12 texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::NV12,
        usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        mapped_at_creation: true,
    });
    let mapped = texture.get_mapped().expect("texture should be mapped");

    // Plane 0 (Y): full resolution, R8 (1 byte/texel).
    // Plane 1 (UV): half resolution, Rg8 (2 bytes/texel). Distinct patterns so a
    // plane swap or plane-0 fallback fails the assertion.
    let y: Vec<u8> = (0..width * height).map(|i| i as u8).collect();
    let uv: Vec<u8> = (0..(width / 2) * (height / 2) * 2)
        .map(|i| (i ^ 0xA5) as u8)
        .collect();

    for (aspect, data, w, h, bytes_per_texel) in [
        (wgpu::TextureAspect::Plane0, &y, width, height, 1u32),
        (wgpu::TextureAspect::Plane1, &uv, width / 2, height / 2, 2u32),
    ] {
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * bytes_per_texel),
            rows_per_image: Some(h),
        };
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        mapped.copy_from_memory(
            wgpu::TexelCopyTextureInfoBase {
                texture: (),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            data,
            layout,
            size,
        );
        let mut readback = vec![0u8; (w * bytes_per_texel * h) as usize];
        mapped.copy_to_memory(
            wgpu::TexelCopyTextureInfoBase {
                texture: (),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            &mut readback,
            layout,
            size,
        );
        assert_eq!(&readback, data, "NV12 plane {aspect:?} host round-trip failed");
    }

    drop(mapped);
    texture.unmap();
}

#[gpu_test]
static HOST_IMAGE_PLANAR_ROUNDTRIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_IMAGE_COPY | wgpu::Features::TEXTURE_FORMAT_NV12),
    )
    .run_async(|ctx| async move {
        host_image_planar_roundtrip(ctx).await;
    });
