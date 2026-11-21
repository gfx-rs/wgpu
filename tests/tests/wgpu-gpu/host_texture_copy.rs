use wgpu::util::DeviceExt as _;
use wgpu_test::{
    apply, fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(HOST_TEXTURE_MAP_INVALID_CALL_CALLBACK_ERR);
    vec.push(HOST_TEXTURE_MAP_INVALIDATED_ENCODER_CALLBACK_ERR);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_WHILE_PENDING_MAP);
    vec.push(HOST_TEXTURE_MAP_AFTER_USE_IN_SAME_SUBMIT);
    vec.push(HOST_TEXTURE_USE_AFTER_MAP_IN_SAME_SUBMIT);
    vec.push(HOST_TEXTURE_ZERO_EXTENT_COPIES);
    vec.push(HOST_TEXTURE_WRITE_LAYOUTS);
    vec.push(HOST_TEXTURE_READ_LAYOUTS);
    vec.push(HOST_TEXTURE_BC_LAYOUTS);
    vec.push(HOST_TEXTURE_BC_3D_UNALIGNED_OFFSET);
    vec.push(HOST_TEXTURE_BC_SMALL_MIPS);
    vec.push(HOST_TEXTURE_UNALIGNED_BYTES_PER_ROW);
    #[cfg(not(target_arch = "wasm32"))]
    vec.push(HOST_TEXTURE_MULTITHREADED);
    vec.push(HOST_TEXTURE_TRANSITION_TO_HOST_COPY);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_ORDERING);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_HOST_VISIBLE);
    vec.push(HOST_TEXTURE_UPLOAD);
    vec.push(HOST_TEXTURE_UPLOAD_MAPPED_AT_CREATION);
    vec.push(HOST_TEXTURE_DOWNLOAD);
    vec.push(HOST_TEXTURE_READ_UNINITIALIZED);
    vec.push(HOST_TEXTURE_PARTIAL_WRITE);
    vec.push(HOST_TEXTURE_DEPTH_ROUNDTRIP);
    vec.push(HOST_TEXTURE_PLANAR_ROUNDTRIP);
    vec.push(HOST_TEXTURE_GPU_ZERO_INIT_UNWRITTEN);
    vec.push(HOST_TEXTURE_DEPTH_GPU_ZERO_INIT_UNWRITTEN);
    vec.push(HOST_TEXTURE_MAP_CALLBACK_OK);
    vec.push(HOST_TEXTURE_MAP_CANCEL_ON_ENCODER_DROP);
    vec.push(HOST_TEXTURE_MAP_CANCEL_ON_COMMAND_BUFFER_DROP);
    vec.push(HOST_TEXTURE_MAP_CANCEL_ON_SUBMIT_FAILURE);
    vec.push(HOST_TEXTURE_SUBMIT_WHILE_MAPPED);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_WHILE_MAPPED);
}

/// A 4x4 `Rgba8Unorm` texture with the given usage, for the lifecycle/validation
/// tests below (which don't care about contents).
fn host_test_texture(
    device: &wgpu::Device,
    usage: wgpu::TextureUsages,
    mapped_at_creation: bool,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible texture"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
        texture_binding_view_dimension: None,
        mapped_at_creation,
    })
}

type MapResultReceiver = std::sync::mpsc::Receiver<Result<(), wgpu::TextureAsyncError>>;

/// Queue a `map_texture_on_completion` whose callback forwards its result over a
/// channel, so a test can assert on the exact `Ok`/`Err` the caller would see.
fn queue_map_with_channel(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> MapResultReceiver {
    let (tx, rx) = std::sync::mpsc::channel();
    encoder.map_texture_on_completion(
        texture,
        Box::new(move |result| {
            let _ = tx.send(result);
        }),
    );
    rx
}

/// Upload pixel data into a HOST_VISIBLE texture via the encoder-mapped path
/// (`map_texture_on_completion`), then read it back through a COPY_DST buffer
/// and verify the round-trip.
async fn host_texture_upload(ctx: TestingContext, start_mapped: bool) {
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
        texture_binding_view_dimension: None,
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

#[apply(gpu_test!)]
static HOST_TEXTURE_UPLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_upload(ctx, false).await;
    });

#[apply(gpu_test!)]
static HOST_TEXTURE_UPLOAD_MAPPED_AT_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_upload(ctx, true).await;
    });

/// Upload pixel data into a HOST_VISIBLE texture via a GPU queue transfer
/// (`queue.write_texture`), then read it back via `copy_to_memory` and verify.
async fn host_texture_download(ctx: TestingContext) {
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
        texture_binding_view_dimension: None,
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

    // Download via host texture copy, using dense (unaligned) row stride.
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

    assert_eq!(cpu_readback, pixel_data, "host texture download mismatch");
}

#[apply(gpu_test!)]
static HOST_TEXTURE_DOWNLOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_download(ctx).await;
    });

/// A HOST_VISIBLE texture that has never been written must read back as zeros
/// on the host (lazy zero-init), not as uninitialized driver memory.
async fn host_texture_read_uninitialized(ctx: TestingContext) {
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
        texture_binding_view_dimension: None,
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

#[apply(gpu_test!)]
static HOST_TEXTURE_READ_UNINITIALIZED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_read_uninitialized(ctx).await;
    });

/// A partial host write must not leave the uncovered part of the layer as
/// uninitialized memory: the untouched region must read back as zero while the
/// written sub-rect keeps its data.
async fn host_texture_partial_write(ctx: TestingContext) {
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
        texture_binding_view_dimension: None,
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

#[apply(gpu_test!)]
static HOST_TEXTURE_PARTIAL_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_partial_write(ctx).await;
    });

/// Round-trip a depth texture (`Depth32Float`) through host copies: write depth
/// values from the CPU, read them straight back, and verify. Exercises the
/// non-color, single depth-aspect host-copy + zero-init path.
async fn host_texture_depth_roundtrip(ctx: TestingContext) {
    let format = wgpu::TextureFormat::Depth32Float;
    // Host-copy support is per-format (e.g. Vulkan's host-transfer format
    // feature); many drivers don't support it for depth formats. Skip if so.
    if !ctx
        .adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
    {
        log::info!("skipping host_texture_depth_roundtrip: {format:?} is not host-copyable");
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
        texture_binding_view_dimension: None,
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

#[apply(gpu_test!)]
static HOST_TEXTURE_DEPTH_ROUNDTRIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_depth_roundtrip(ctx).await;
    });

/// Round-trip both planes of an `NV12` texture through host copies, verifying
/// per-plane (`Plane0` full-res / `Plane1` half-res) host addressing.
///
/// NOTE: gated on `TEXTURE_FORMAT_NV12`, so it is skipped on backends that don't
/// expose NV12 (including Metal). Validated only where NV12 + `HOST_TEXTURE_COPY`
/// are both available (Vulkan / DX12).
async fn host_texture_planar_roundtrip(ctx: TestingContext) {
    // Host-copy support is per-format; skip if this adapter can't host-copy NV12.
    if !ctx
        .adapter
        .get_texture_format_features(wgpu::TextureFormat::NV12)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
    {
        log::info!("skipping host_texture_planar_roundtrip: NV12 is not host-copyable");
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
        texture_binding_view_dimension: None,
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
        (
            wgpu::TextureAspect::Plane1,
            &uv,
            width / 2,
            height / 2,
            2u32,
        ),
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
        assert_eq!(
            &readback, data,
            "NV12 plane {aspect:?} host round-trip failed"
        );
    }

    drop(mapped);
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_PLANAR_ROUNDTRIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_TEXTURE_COPY | wgpu::Features::TEXTURE_FORMAT_NV12),
    )
    .run_async(|ctx| async move {
        host_texture_planar_roundtrip(ctx).await;
    });

/// Regression: a HOST_VISIBLE texture that is mapped then unmapped *without any
/// host write* must still be zero-initialized when the GPU reads it.
///
/// This exercises the GPU-side init path (`clear_texture` via buffer copies) for
/// host-visible textures, which requires them to carry `COPY_DST` hal usage.
/// Without it, the clear would fail (Vulkan: no `TRANSFER_DST`) or expose
/// uninitialized memory.
async fn host_texture_gpu_zero_init_unwritten(ctx: TestingContext) {
    let width = 4u32;
    let height = 4u32;
    let bytes_per_pixel = 4u32;
    let host_bytes_per_row = bytes_per_pixel * width; // 16
    let gpu_bytes_per_row = host_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT); // 256
    let gpu_data_size = (gpu_bytes_per_row * height) as usize;

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible unwritten texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        texture_binding_view_dimension: None,
        mapped_at_creation: true,
    });

    // Map then unmap without writing anything: the texture stays uninitialized.
    texture.unmap();

    // GPU readback must observe zero-initialized memory, not garbage.
    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: gpu_data_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy encoder"),
        });
    encoder.copy_texture_to_buffer(
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
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    for row in 0..height as usize {
        let start = row * gpu_bytes_per_row as usize;
        let pixels = &data[start..start + host_bytes_per_row as usize];
        assert!(
            pixels.iter().all(|&b| b == 0),
            "row {row} of an unwritten host-visible texture must be zero-initialized, got {pixels:?}",
        );
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_GPU_ZERO_INIT_UNWRITTEN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_gpu_zero_init_unwritten(ctx).await;
    });

/// Like `host_texture_gpu_zero_init_unwritten`, but for a depth (`Depth32Float`)
/// HOST_VISIBLE texture. This exercises a different GPU zero-init path: for
/// depth formats `map_texture_usage_for_texture` selects `DEPTH_STENCIL_WRITE`
/// (a render-pass clear), not the `COPY_DST` buffer clear the color case uses,
/// so a never-written host-visible depth texture must still read back as zeros
/// once the GPU initializes it.
async fn host_texture_depth_gpu_zero_init_unwritten(ctx: TestingContext) {
    let format = wgpu::TextureFormat::Depth32Float;
    // Host-copy support is per-format; HOST_VISIBLE depth requires it. Skip if
    // this adapter can't host-copy the format (same guard as the round-trip test).
    if !ctx
        .adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
    {
        log::info!(
            "skipping host_texture_depth_gpu_zero_init_unwritten: {format:?} is not host-copyable"
        );
        return;
    }

    let width = 4u32;
    let height = 4u32;
    let host_bytes_per_row = 4 * width; // Depth32Float: 4 bytes/texel
    let gpu_bytes_per_row = host_bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let gpu_data_size = (gpu_bytes_per_row * height) as usize;

    // HOST_VISIBLE | COPY_SRC, never written from the host: it stays
    // uninitialized until the GPU copy below forces a zero-init.
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("host-visible unwritten depth texture"),
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
        texture_binding_view_dimension: None,
        mapped_at_creation: false,
    });

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: gpu_data_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy encoder"),
        });
    encoder.copy_texture_to_buffer(
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
    ctx.queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range().unwrap();
    for row in 0..height as usize {
        let start = row * gpu_bytes_per_row as usize;
        let pixels = &data[start..start + host_bytes_per_row as usize];
        assert!(
            pixels.iter().all(|&b| b == 0),
            "row {row} of an unwritten host-visible depth texture must be zero-initialized, got {pixels:?}",
        );
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_DEPTH_GPU_ZERO_INIT_UNWRITTEN: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_depth_gpu_zero_init_unwritten(ctx).await;
        });

/// The map callback must actually be delivered with `Ok` once the submission
/// that queues the map completes — the happy-path round-trip tests pass a
/// callback that discards its result, so nothing else asserts this.
async fn host_texture_map_callback_ok(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, &texture);
    ctx.queue.submit(Some(encoder.finish()));
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let result = rx
        .try_recv()
        .expect("map callback should have fired after the submission completed");
    assert!(
        result.is_ok(),
        "map callback should report Ok, got {result:?}"
    );
    assert!(
        texture.get_mapped().is_some(),
        "texture should be mapped after a successful map"
    );
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_CALLBACK_OK: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_map_callback_ok(ctx).await;
    });

/// Dropping the encoder a map was queued into, without finishing it, must fire
/// the callback with `Err` (not silently drop it) and leave the texture unmapped.
async fn host_texture_map_cancel_on_encoder_drop(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, &texture);
    drop(encoder);

    let result = rx
        .try_recv()
        .expect("map callback should fire when the encoder is dropped");
    assert!(
        result.is_err(),
        "cancelled map callback must report Err, got {result:?}"
    );
    assert!(
        texture.get_mapped().is_none(),
        "texture must not be mapped after the map was cancelled"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_CANCEL_ON_ENCODER_DROP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_map_cancel_on_encoder_drop(ctx).await;
    });

/// Finishing the encoder but dropping the command buffer without submitting it
/// must also fire the callback with `Err`.
async fn host_texture_map_cancel_on_command_buffer_drop(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, &texture);
    let command_buffer = encoder.finish();
    drop(command_buffer);

    let result = rx
        .try_recv()
        .expect("map callback should fire when the command buffer is dropped");
    assert!(
        result.is_err(),
        "cancelled map callback must report Err, got {result:?}"
    );
    assert!(
        texture.get_mapped().is_none(),
        "texture must not be mapped after the map was cancelled"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_CANCEL_ON_COMMAND_BUFFER_DROP: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_map_cancel_on_command_buffer_drop(ctx).await;
        });

/// A submission that fails (here: a second command buffer using a destroyed
/// buffer) must not strand a map queued by another command buffer in the same
/// submit. The callback must fire with `Err` and the texture revert to unmapped,
/// otherwise the texture is pinned in `MappingQueued` forever and awaiting
/// callers hang.
async fn host_texture_map_cancel_on_submit_failure(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);

    let mut map_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut map_encoder, &texture);
    let map_command_buffer = map_encoder.finish();

    // A second command buffer whose submit fails validation: copy from a buffer
    // that gets destroyed before submit.
    let src = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("doomed src"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dst"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut bad_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    bad_encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 4);
    let bad_command_buffer = bad_encoder.finish();
    src.destroy();

    fail(
        &ctx.device,
        || {
            ctx.queue.submit([map_command_buffer, bad_command_buffer]);
        },
        Some("destroyed"),
    );

    let result = rx
        .try_recv()
        .expect("map callback should fire when the submission fails");
    assert!(
        result.is_err(),
        "map callback must report Err on submit failure, got {result:?}"
    );
    assert!(
        texture.get_mapped().is_none(),
        "texture must revert to unmapped after a failed submit"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_CANCEL_ON_SUBMIT_FAILURE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_map_cancel_on_submit_failure(ctx).await;
        });

// NOTE: the device-loss path (`drain_pending_texture_maps`, which fires queued
// maps with `Err` in `Queue::drop` when the device is invalid) is not covered
// here: it only triggers for a submission still in flight at device loss, and
// there's no deterministic way to hold a submission in flight across adapters
// (fast software adapters complete it before the drain runs, so the callback
// resolves `Ok`). Left for a future test with a lower-level fence hook.

/// Submitting a command buffer that uses a still-mapped texture must fail
/// validation (`TextureStillMapped`).
async fn host_texture_submit_while_mapped(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        true,
    );

    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    let command_buffer = encoder.finish();

    // The texture was never unmapped, so the submit must reject it.
    fail(
        &ctx.device,
        || {
            ctx.queue.submit(Some(command_buffer));
        },
        Some("mapped or queued for mapping"),
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_SUBMIT_WHILE_MAPPED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_submit_while_mapped(ctx).await;
    });

/// `queue.write_texture` to a still-mapped texture must fail validation
/// (`TransferError::TextureNotAvailable`).
async fn host_texture_write_texture_while_mapped(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_DST,
        true,
    );

    let data = [0u8; 4 * 4 * 4];
    fail(
        &ctx.device,
        || {
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
                    rows_per_image: Some(4),
                },
                wgpu::Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
            );
        },
        Some("unmapped"),
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_WHILE_MAPPED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_write_texture_while_mapped(ctx).await;
    });

const PADDING_BYTE: u8 = 0xEE;

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

fn host_range(mip_level: u32, origin: wgpu::Origin3d) -> wgpu::TexelCopyTextureInfoBase<()> {
    wgpu::TexelCopyTextureInfoBase {
        texture: (),
        mip_level,
        origin,
        aspect: wgpu::TextureAspect::All,
    }
}

fn tight_layout(format: wgpu::TextureFormat, size: wgpu::Extent3d) -> wgpu::TexelCopyBufferLayout {
    let (row_bytes, rows) = copy_geometry(format, size);
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(row_bytes),
        rows_per_image: Some(rows),
    }
}

/// Non-zero bytes, so they can't be mistaken for zero-initialized memory.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| ((i + seed as usize) % 251) as u8 + 1)
        .collect()
}

/// Bytes per block row and number of block rows of a copy of `size`.
fn copy_geometry(format: wgpu::TextureFormat, size: wgpu::Extent3d) -> (u32, u32) {
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap();
    (
        size.width.div_ceil(block_width) * block_size,
        size.height.div_ceil(block_height),
    )
}

/// Lays out tightly packed `dense` data according to `layout`, filling the
/// gaps with `PADDING_BYTE`.
fn pack(
    dense: &[u8],
    row_bytes: u32,
    rows: u32,
    depth: u32,
    layout: wgpu::TexelCopyBufferLayout,
) -> Vec<u8> {
    let row_bytes = row_bytes as usize;
    let bytes_per_row = layout.bytes_per_row.unwrap() as usize;
    let bytes_per_image = bytes_per_row * layout.rows_per_image.unwrap() as usize;
    let offset = layout.offset as usize;
    let len = offset
        + bytes_per_image * (depth as usize - 1)
        + bytes_per_row * (rows as usize - 1)
        + row_bytes;
    let mut out = vec![PADDING_BYTE; len];
    for z in 0..depth as usize {
        for y in 0..rows as usize {
            let src = (z * rows as usize + y) * row_bytes;
            let dst = offset + z * bytes_per_image + y * bytes_per_row;
            out[dst..dst + row_bytes].copy_from_slice(&dense[src..src + row_bytes]);
        }
    }
    out
}

/// Places tightly packed `dense` data for a copy at `origin` into an otherwise
/// zeroed, tightly packed image of `full_size`.
fn embed(
    format: wgpu::TextureFormat,
    full_size: wgpu::Extent3d,
    origin: wgpu::Origin3d,
    copy_size: wgpu::Extent3d,
    dense: &[u8],
) -> Vec<u8> {
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap();
    let (full_row_bytes, full_rows) = copy_geometry(format, full_size);
    let (row_bytes, rows) = copy_geometry(format, copy_size);
    let (full_row_bytes, full_rows, row_bytes, rows) = (
        full_row_bytes as usize,
        full_rows as usize,
        row_bytes as usize,
        rows as usize,
    );
    let x = (origin.x / block_width * block_size) as usize;
    let y0 = (origin.y / block_height) as usize;
    let mut out = vec![0; full_row_bytes * full_rows * full_size.depth_or_array_layers as usize];
    for z in 0..copy_size.depth_or_array_layers as usize {
        for y in 0..rows {
            let src = (z * rows + y) * row_bytes;
            let dst = ((origin.z as usize + z) * full_rows + y0 + y) * full_row_bytes + x;
            out[dst..dst + row_bytes].copy_from_slice(&dense[src..src + row_bytes]);
        }
    }
    out
}

/// Reads a region of `texture` through a GPU copy, tightly packed.
async fn gpu_read(
    ctx: &TestingContext,
    texture: &wgpu::Texture,
    mip_level: u32,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
) -> Vec<u8> {
    let (row_bytes, rows) = copy_geometry(texture.format(), size);
    let bytes_per_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback buffer"),
        size: u64::from(bytes_per_row * rows * size.depth_or_array_layers),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows),
            },
        },
        size,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let data = slice.get_mapped_range().unwrap();
    let mut dense = Vec::new();
    for image in data.chunks(bytes_per_row as usize * rows as usize) {
        for row in image.chunks(bytes_per_row as usize) {
            dense.extend_from_slice(&row[..row_bytes as usize]);
        }
    }
    dense
}

/// Records a GPU copy of tightly packed `dense` data into a region of `texture`.
fn gpu_write(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level: u32,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
    dense: &[u8],
) {
    let (row_bytes, rows) = copy_geometry(texture.format(), size);
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
        rows_per_image: Some(rows),
    };
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("upload buffer"),
        contents: &pack(dense, row_bytes, rows, size.depth_or_array_layers, layout),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout,
        },
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
}

async fn map_and_wait(ctx: &TestingContext, texture: &wgpu::Texture) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, texture);
    ctx.queue.submit(Some(encoder.finish()));
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    rx.try_recv()
        .expect("map callback should have fired")
        .expect("map should succeed");
}

fn host_read_all(texture: &wgpu::Texture) -> Vec<u8> {
    let size = texture.size();
    let layout = tight_layout(texture.format(), size);
    let mut out = vec![
        PADDING_BYTE;
        (layout.bytes_per_row.unwrap() * layout.rows_per_image.unwrap()) as usize
            * size.depth_or_array_layers as usize
    ];
    let mapped = texture.get_mapped().expect("texture should be mapped");
    mapped.copy_to_memory(host_range(0, wgpu::Origin3d::ZERO), &mut out, layout, size);
    out
}

fn host_write_all(texture: &wgpu::Texture, data: &[u8]) {
    let size = texture.size();
    let mapped = texture.get_mapped().expect("texture should be mapped");
    mapped.copy_from_memory(
        host_range(0, wgpu::Origin3d::ZERO),
        data,
        tight_layout(texture.format(), size),
        size,
    );
}

fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: wgpu::Origin3d,
    size: wgpu::Extent3d,
    data: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        tight_layout(texture.format(), size),
        size,
    );
}

fn clear_pass(encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, color: wgpu::Color) {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
}

async fn host_texture_map_invalid_call_callback_err(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::COPY_DST, false);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = fail(
        &ctx.device,
        || {
            let rx = queue_map_with_channel(&mut encoder, &texture);
            encoder.finish();
            rx
        },
        Some("HOST_VISIBLE"),
    );
    let result = rx
        .try_recv()
        .expect("map callback should fire for an invalid call");
    assert!(result.is_err(), "expected Err, got {result:?}");
    assert!(rx.try_recv().is_err(), "callback must fire only once");
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_INVALID_CALL_CALLBACK_ERR: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_map_invalid_call_callback_err(ctx).await;
        });

async fn host_texture_map_invalidated_encoder_callback_err(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, &texture);
    let pass = encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor::default())
        .forget_lifetime();
    // Recording while a pass is open invalidates the encoder.
    encoder.clear_buffer(&buffer, 0, None);
    drop(pass);
    let command_buffer = fail(&ctx.device, || encoder.finish(), Some("locked"));

    let result = rx
        .try_recv()
        .expect("map callback should fire when the encoder is invalidated");
    assert!(result.is_err(), "expected Err, got {result:?}");

    fail(&ctx.device, || ctx.queue.submit(Some(command_buffer)), None);
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    assert!(rx.try_recv().is_err(), "callback must fire only once");
    assert!(texture.get_mapped().is_none());
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_INVALIDATED_ENCODER_CALLBACK_ERR: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_map_invalidated_encoder_callback_err(ctx).await;
        });

async fn host_texture_write_texture_while_pending_map(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        false,
    );
    let size = texture.size();
    let before = pattern(64, 1);
    let after = pattern(64, 2);
    write_texture(&ctx.queue, &texture, wgpu::Origin3d::ZERO, size, &before);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut encoder, &texture);
    ctx.queue.submit(Some(encoder.finish()));

    fail(
        &ctx.device,
        || write_texture(&ctx.queue, &texture, wgpu::Origin3d::ZERO, size, &after),
        Some("unmapped"),
    );

    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut use_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    use_encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        size,
    );
    let use_command_buffer = use_encoder.finish();
    fail(
        &ctx.device,
        || ctx.queue.submit(Some(use_command_buffer)),
        Some("queued for mapping"),
    );

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let result = rx.try_recv().expect("map callback should have fired");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    assert_eq!(host_read_all(&texture), before);
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_WHILE_PENDING_MAP: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_write_texture_while_pending_map(ctx).await;
        });

async fn host_texture_map_after_use_in_same_submit(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_DST,
        false,
    );
    let data = pattern(64, 3);

    let mut write_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    gpu_write(
        &ctx.device,
        &mut write_encoder,
        &texture,
        0,
        wgpu::Origin3d::ZERO,
        texture.size(),
        &data,
    );
    let mut map_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut map_encoder, &texture);
    let command_buffers = [write_encoder.finish(), map_encoder.finish()];
    valid(&ctx.device, || ctx.queue.submit(command_buffers));

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    let result = rx.try_recv().expect("map callback should have fired");
    assert!(result.is_ok(), "expected Ok, got {result:?}");
    assert_eq!(host_read_all(&texture), data);
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_AFTER_USE_IN_SAME_SUBMIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_map_after_use_in_same_submit(ctx).await;
        });

async fn host_texture_use_after_map_in_same_submit(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_DST,
        false,
    );

    let mut map_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let rx = queue_map_with_channel(&mut map_encoder, &texture);
    let mut write_encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    gpu_write(
        &ctx.device,
        &mut write_encoder,
        &texture,
        0,
        wgpu::Origin3d::ZERO,
        texture.size(),
        &pattern(64, 4),
    );
    let command_buffers = [map_encoder.finish(), write_encoder.finish()];
    fail(
        &ctx.device,
        || ctx.queue.submit(command_buffers),
        Some("queued for mapping"),
    );

    let result = rx
        .try_recv()
        .expect("map callback should fire when the submission fails");
    assert!(result.is_err(), "expected Err, got {result:?}");
    assert!(texture.get_mapped().is_none());

    // The failed submit must not leave the texture stuck as pending mapping.
    map_and_wait(&ctx, &texture).await;
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_USE_AFTER_MAP_IN_SAME_SUBMIT: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
        .run_async(|ctx| async move {
            host_texture_use_after_map_in_same_submit(ctx).await;
        });

async fn host_texture_zero_extent_copies(ctx: TestingContext) {
    let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, true);
    let layout = tight_layout(texture.format(), texture.size());
    let data = pattern(64, 5);
    let mut out = vec![PADDING_BYTE; 64];
    let mapped = texture.get_mapped().unwrap();

    valid(&ctx.device, || {
        for (origin, size) in [
            (wgpu::Origin3d::ZERO, extent(0, 4, 1)),
            (wgpu::Origin3d::ZERO, extent(4, 0, 1)),
            (wgpu::Origin3d::ZERO, extent(4, 4, 0)),
            (wgpu::Origin3d { x: 4, y: 4, z: 0 }, extent(0, 0, 1)),
            (wgpu::Origin3d { x: 0, y: 0, z: 1 }, extent(4, 4, 0)),
        ] {
            mapped.copy_from_memory(host_range(0, origin), &data, layout, size);
            mapped.copy_to_memory(host_range(0, origin), &mut out, layout, size);
        }
    });
    assert!(out.iter().all(|&b| b == PADDING_BYTE));
    drop(mapped);

    assert!(host_read_all(&texture).iter().all(|&b| b == 0));
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_ZERO_EXTENT_COPIES: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_zero_extent_copies(ctx).await;
    });

#[derive(Clone, Copy, Debug)]
struct LayoutCase {
    name: &'static str,
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureDimension,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    mip_level: u32,
    origin: wgpu::Origin3d,
    copy_size: wgpu::Extent3d,
    offset: u64,
    extra_bytes_per_row: u32,
    extra_rows_per_image: u32,
}

impl LayoutCase {
    const BASE: Self = Self {
        name: "",
        format: wgpu::TextureFormat::Rgba8Unorm,
        dimension: wgpu::TextureDimension::D2,
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        copy_size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        offset: 0,
        extra_bytes_per_row: 0,
        extra_rows_per_image: 0,
    };

    fn create_texture(
        &self,
        device: &wgpu::Device,
        usage: wgpu::TextureUsages,
        mapped_at_creation: bool,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(self.name),
            size: self.size,
            mip_level_count: self.mip_level_count,
            sample_count: 1,
            dimension: self.dimension,
            format: self.format,
            usage,
            view_formats: &[],
            texture_binding_view_dimension: None,
            mapped_at_creation,
        })
    }

    /// Returns the host layout, its tightly packed data, and the data laid out
    /// in it.
    fn host_data(&self, seed: u8) -> (wgpu::TexelCopyBufferLayout, Vec<u8>, Vec<u8>) {
        let (row_bytes, rows) = copy_geometry(self.format, self.copy_size);
        let depth = self.copy_size.depth_or_array_layers;
        let layout = wgpu::TexelCopyBufferLayout {
            offset: self.offset,
            bytes_per_row: Some(row_bytes + self.extra_bytes_per_row),
            rows_per_image: Some(rows + self.extra_rows_per_image),
        };
        let dense = pattern((row_bytes * rows * depth) as usize, seed);
        let packed = pack(&dense, row_bytes, rows, depth, layout);
        (layout, dense, packed)
    }

    fn mip_size(&self) -> wgpu::Extent3d {
        self.size.mip_level_size(self.mip_level, self.dimension)
    }
}

fn rgba8_layout_cases() -> Vec<LayoutCase> {
    let base = LayoutCase::BASE;
    vec![
        LayoutCase {
            name: "mip level 1",
            size: extent(16, 16, 1),
            mip_level_count: 3,
            mip_level: 1,
            ..base
        },
        LayoutCase {
            name: "array layer 1",
            size: extent(8, 8, 3),
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            ..base
        },
        LayoutCase {
            name: "3d slices",
            dimension: wgpu::TextureDimension::D3,
            size: extent(8, 8, 4),
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            copy_size: extent(8, 8, 2),
            extra_rows_per_image: 3,
            ..base
        },
        LayoutCase {
            name: "non-zero origin",
            origin: wgpu::Origin3d { x: 2, y: 1, z: 0 },
            copy_size: extent(3, 4, 1),
            ..base
        },
        LayoutCase {
            name: "padded layout",
            size: extent(8, 8, 2),
            copy_size: extent(5, 6, 2),
            offset: 8,
            extra_bytes_per_row: 12,
            extra_rows_per_image: 2,
            ..base
        },
        LayoutCase {
            name: "unaligned offset",
            size: extent(8, 8, 2),
            origin: wgpu::Origin3d { x: 1, y: 1, z: 0 },
            copy_size: extent(5, 6, 2),
            offset: 7,
            extra_bytes_per_row: 12,
            extra_rows_per_image: 1,
            ..base
        },
    ]
}

fn bc1_layout_cases() -> Vec<LayoutCase> {
    let base = LayoutCase {
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        size: extent(16, 16, 1),
        copy_size: extent(16, 16, 1),
        ..LayoutCase::BASE
    };
    vec![
        LayoutCase {
            name: "bc1 mip level 1",
            mip_level_count: 2,
            mip_level: 1,
            copy_size: extent(8, 8, 1),
            offset: 8,
            extra_bytes_per_row: 8,
            ..base
        },
        LayoutCase {
            name: "bc1 non-zero origin",
            origin: wgpu::Origin3d { x: 4, y: 8, z: 0 },
            copy_size: extent(8, 4, 1),
            offset: 3,
            extra_bytes_per_row: 8,
            ..base
        },
        LayoutCase {
            name: "bc1 array layer 1",
            size: extent(16, 16, 3),
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            copy_size: extent(16, 16, 2),
            extra_rows_per_image: 1,
            ..base
        },
    ]
}

/// Host write with `case`'s layout, then verify through a GPU copy.
async fn host_write_layout_case(ctx: &TestingContext, case: &LayoutCase) {
    let texture = case.create_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
        true,
    );
    let (layout, dense, packed) = case.host_data(0);
    let range = host_range(case.mip_level, case.origin);
    {
        let mapped = texture.get_mapped().unwrap();
        mapped.copy_from_memory(range, &packed, layout, case.copy_size);
        let mut readback = vec![PADDING_BYTE; packed.len()];
        mapped.copy_to_memory(range, &mut readback, layout, case.copy_size);
        assert_eq!(readback, packed, "{}: host read-back mismatch", case.name);
    }
    texture.unmap();

    let mip_size = case.mip_size();
    let gpu = gpu_read(
        ctx,
        &texture,
        case.mip_level,
        wgpu::Origin3d::ZERO,
        mip_size,
    )
    .await;
    assert_eq!(
        gpu,
        embed(case.format, mip_size, case.origin, case.copy_size, &dense),
        "{}: GPU read-back mismatch",
        case.name
    );
}

/// GPU write, map, then host read with `case`'s layout.
async fn host_read_layout_case(ctx: &TestingContext, case: &LayoutCase) {
    let texture = case.create_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        false,
    );
    let (layout, dense, packed) = case.host_data(1);
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    gpu_write(
        &ctx.device,
        &mut encoder,
        &texture,
        case.mip_level,
        case.origin,
        case.copy_size,
        &dense,
    );
    let rx = queue_map_with_channel(&mut encoder, &texture);
    ctx.queue.submit(Some(encoder.finish()));
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
    rx.try_recv().unwrap().unwrap();

    let mapped = texture.get_mapped().unwrap();
    let mut readback = vec![PADDING_BYTE; packed.len()];
    mapped.copy_to_memory(
        host_range(case.mip_level, case.origin),
        &mut readback,
        layout,
        case.copy_size,
    );
    assert_eq!(readback, packed, "{}: host read mismatch", case.name);

    let mip_size = case.mip_size();
    let full_layout = tight_layout(case.format, mip_size);
    let expected = embed(case.format, mip_size, case.origin, case.copy_size, &dense);
    let mut full = vec![PADDING_BYTE; expected.len()];
    mapped.copy_to_memory(
        host_range(case.mip_level, wgpu::Origin3d::ZERO),
        &mut full,
        full_layout,
        mip_size,
    );
    assert_eq!(full, expected, "{}: full mip host read mismatch", case.name);
    drop(mapped);
    texture.unmap();
}

fn host_copyable(ctx: &TestingContext, format: wgpu::TextureFormat) -> bool {
    ctx.adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(wgpu::TextureUsages::HOST_VISIBLE)
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        for case in rgba8_layout_cases() {
            host_write_layout_case(&ctx, &case).await;
        }
    });

#[apply(gpu_test!)]
static HOST_TEXTURE_READ_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        for case in rgba8_layout_cases() {
            host_read_layout_case(&ctx, &case).await;
        }
    });

#[apply(gpu_test!)]
static HOST_TEXTURE_BC_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_TEXTURE_COPY | wgpu::Features::TEXTURE_COMPRESSION_BC),
    )
    .run_async(|ctx| async move {
        if !host_copyable(&ctx, wgpu::TextureFormat::Bc1RgbaUnorm) {
            log::info!("skipping: Bc1RgbaUnorm is not host-copyable");
            return;
        }
        for case in bc1_layout_cases() {
            host_write_layout_case(&ctx, &case).await;
            host_read_layout_case(&ctx, &case).await;
        }
    });

fn bc1_3d_layout_cases() -> Vec<LayoutCase> {
    let base = LayoutCase {
        format: wgpu::TextureFormat::Bc1RgbaUnorm,
        dimension: wgpu::TextureDimension::D3,
        offset: 1,
        ..LayoutCase::BASE
    };
    vec![
        LayoutCase {
            name: "bc1 3d whole texture",
            size: extent(4, 4, 2),
            copy_size: extent(4, 4, 2),
            ..base
        },
        LayoutCase {
            name: "bc1 3d sub-region",
            size: extent(8, 8, 3),
            origin: wgpu::Origin3d { x: 4, y: 4, z: 1 },
            copy_size: extent(4, 4, 2),
            extra_bytes_per_row: 8,
            ..base
        },
    ]
}

async fn host_texture_bc_3d_unaligned_offset(ctx: TestingContext) {
    for case in bc1_3d_layout_cases() {
        host_write_layout_case(&ctx, &case).await;
        host_read_layout_case(&ctx, &case).await;
    }

    let format = wgpu::TextureFormat::Bc1RgbaUnorm;
    let size = extent(8, 8, 2);
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
        texture_binding_view_dimension: None,
        mapped_at_creation: false,
    });
    let origin = wgpu::Origin3d { x: 0, y: 4, z: 0 };
    let copy_size = extent(8, 4, 2);
    let dense = pattern(32, 9);
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 1,
        bytes_per_row: Some(16),
        rows_per_image: Some(1),
    };
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        &pack(&dense, 16, 1, 2, layout),
        layout,
        copy_size,
    );
    assert_eq!(
        gpu_read(&ctx, &texture, 0, wgpu::Origin3d::ZERO, size).await,
        embed(format, size, origin, copy_size, &dense),
        "write_texture to bc1 3d texture"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_BC_3D_UNALIGNED_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(
        wgpu::Features::HOST_TEXTURE_COPY
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
    ))
    .run_async(|ctx| async move {
        if !host_copyable(&ctx, wgpu::TextureFormat::Bc1RgbaUnorm) {
            log::info!("skipping: Bc1RgbaUnorm is not host-copyable");
            return;
        }
        host_texture_bc_3d_unaligned_offset(ctx).await;
    });

/// Mips 2, 3 and 4 of a 16x16 BC texture are 4x4, 2x2 and 1x1, and their copies
/// cover one physical block.
async fn host_texture_bc_small_mips(ctx: TestingContext) {
    let format = wgpu::TextureFormat::Bc1RgbaUnorm;
    let size = extent(16, 16, 1);
    let create = |mapped_at_creation| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 5,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
            texture_binding_view_dimension: None,
            mapped_at_creation,
        })
    };
    let block = extent(4, 4, 1);
    let block_layout = tight_layout(format, block);

    let texture = create(true);
    let writes = [(3, pattern(8, 3)), (4, pattern(8, 4))];
    {
        let mapped = texture.get_mapped().unwrap();
        for (mip, data) in &writes {
            let range = host_range(*mip, wgpu::Origin3d::ZERO);
            mapped.copy_from_memory(range, data, block_layout, block);
            let mut readback = vec![PADDING_BYTE; 8];
            mapped.copy_to_memory(range, &mut readback, block_layout, block);
            assert_eq!(&readback, data, "mip {mip}: host read-back mismatch");
        }
    }
    texture.unmap();
    for (mip, data) in &writes {
        let gpu = gpu_read(&ctx, &texture, *mip, wgpu::Origin3d::ZERO, block).await;
        assert_eq!(&gpu, data, "mip {mip}: GPU read-back mismatch");
    }
    let gpu = gpu_read(&ctx, &texture, 2, wgpu::Origin3d::ZERO, block).await;
    assert_eq!(gpu, vec![0; 8], "mip 2 should be zero");

    let texture = create(false);
    map_and_wait(&ctx, &texture).await;
    {
        let mapped = texture.get_mapped().unwrap();
        for mip in 2..5 {
            let mut readback = vec![PADDING_BYTE; 8];
            mapped.copy_to_memory(
                host_range(mip, wgpu::Origin3d::ZERO),
                &mut readback,
                block_layout,
                block,
            );
            assert_eq!(readback, vec![0; 8], "mip {mip}: host read should be zero");
        }
    }
    texture.unmap();
    for mip in 2..5 {
        let gpu = gpu_read(&ctx, &texture, mip, wgpu::Origin3d::ZERO, block).await;
        assert_eq!(gpu, vec![0; 8], "mip {mip}: GPU read should be zero");
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_BC_SMALL_MIPS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_TEXTURE_COPY | wgpu::Features::TEXTURE_COMPRESSION_BC),
    )
    .run_async(|ctx| async move {
        if !host_copyable(&ctx, wgpu::TextureFormat::Bc1RgbaUnorm) {
            log::info!("skipping: Bc1RgbaUnorm is not host-copyable");
            return;
        }
        host_texture_bc_small_mips(ctx).await;
    });

/// A `bytes_per_row` that is not a multiple of the texel size is valid for host
/// copies and for `write_texture`, which uses host copies for idle
/// HOST_VISIBLE textures.
async fn host_texture_unaligned_bytes_per_row(ctx: TestingContext) {
    let case = LayoutCase {
        name: "unaligned bytes_per_row",
        size: extent(8, 8, 2),
        origin: wgpu::Origin3d { x: 1, y: 1, z: 0 },
        copy_size: extent(5, 6, 2),
        offset: 7,
        extra_bytes_per_row: 13,
        extra_rows_per_image: 1,
        ..LayoutCase::BASE
    };
    host_write_layout_case(&ctx, &case).await;
    host_read_layout_case(&ctx, &case).await;

    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        false,
    );
    let data = pattern(64, 13);
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 3,
        bytes_per_row: Some(17),
        rows_per_image: Some(4),
    };
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &pack(&data, 16, 4, 1, layout),
        layout,
        texture.size(),
    );
    assert_eq!(
        gpu_read(&ctx, &texture, 0, wgpu::Origin3d::ZERO, texture.size()).await,
        data
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_UNALIGNED_BYTES_PER_ROW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_unaligned_bytes_per_row(ctx).await;
    });

#[cfg(not(target_arch = "wasm32"))]
async fn host_texture_multithreaded(ctx: TestingContext) {
    const THREADS: u32 = 4;
    const SIZE: u32 = 64;
    const ROW_BYTES: usize = SIZE as usize * 4;

    let create = || {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent(SIZE, SIZE, 1),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::HOST_VISIBLE | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
            texture_binding_view_dimension: None,
            mapped_at_creation: true,
        })
    };
    let write_rows = |mapped: &wgpu::MappedTexture, data: &[u8], rows: core::ops::Range<u32>| {
        for y in rows {
            let start = y as usize * ROW_BYTES;
            mapped.copy_from_memory(
                host_range(0, wgpu::Origin3d { x: 0, y, z: 0 }),
                &data[start..start + ROW_BYTES],
                wgpu::TexelCopyBufferLayout::default(),
                extent(SIZE, 1, 1),
            );
        }
    };

    let own_textures: Vec<_> = (0..THREADS).map(|_| create()).collect();
    let own_data: Vec<_> = (0..THREADS)
        .map(|i| pattern(ROW_BYTES * SIZE as usize, i as u8 * 31))
        .collect();
    let shared = create();
    let shared_data = pattern(ROW_BYTES * SIZE as usize, 200);
    let unwritten = create();

    let shared_mapped = shared.get_mapped().unwrap();
    std::thread::scope(|s| {
        for (i, (texture, data)) in own_textures.iter().zip(&own_data).enumerate() {
            let shared_mapped = &shared_mapped;
            let shared_data = &shared_data;
            s.spawn(move || {
                let mapped = texture.get_mapped().unwrap();
                write_rows(&mapped, data, 0..SIZE);
                let band = SIZE / THREADS;
                let i = i as u32;
                write_rows(shared_mapped, shared_data, i * band..(i + 1) * band);
            });
        }
    });
    drop(shared_mapped);

    std::thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| {
                for _ in 0..8 {
                    assert_eq!(host_read_all(&shared), shared_data);
                    assert!(host_read_all(&unwritten).iter().all(|&b| b == 0));
                }
            });
        }
    });

    for (texture, data) in own_textures
        .iter()
        .zip(&own_data)
        .chain([(&shared, &shared_data)])
    {
        texture.unmap();
        let gpu = gpu_read(&ctx, texture, 0, wgpu::Origin3d::ZERO, texture.size()).await;
        assert_eq!(&gpu, data);
    }
    unwritten.unmap();
}

#[cfg(not(target_arch = "wasm32"))]
#[apply(gpu_test!)]
static HOST_TEXTURE_MULTITHREADED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_multithreaded(ctx).await;
    });

#[apply(gpu_test!)]
static HOST_TEXTURE_TRANSITION_TO_HOST_COPY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_sync(|ctx| {
        let texture = host_test_texture(&ctx.device, wgpu::TextureUsages::HOST_VISIBLE, false);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        fail(
            &ctx.device,
            || {
                encoder.transition_resources(
                    core::iter::empty(),
                    core::iter::once(wgpu::TextureTransition {
                        texture: &texture,
                        selector: None,
                        state: wgpu::TextureUses::HOST_COPY,
                    }),
                );
                encoder.finish()
            },
            Some("HOST_COPY"),
        );
    });

/// `Queue::write_texture` may use host copies internally, but must keep its
/// ordering relative to submissions.
async fn write_texture_ordering(ctx: TestingContext) {
    let ctx = &ctx;
    let size = extent(4, 4, 1);
    let create = || {
        host_test_texture(
            &ctx.device,
            wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            false,
        )
    };
    let red = wgpu::Color::RED;
    let red_bytes = [255u8, 0, 0, 255].repeat(16);
    let b = pattern(64, 7);
    let c = pattern(64, 8);
    let clear = |texture: &wgpu::Texture| {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        clear_pass(&mut encoder, texture, red);
        encoder.finish()
    };
    let read = |texture: wgpu::Texture| async move {
        gpu_read(ctx, &texture, 0, wgpu::Origin3d::ZERO, size).await
    };

    let t = create();
    write_texture(&ctx.queue, &t, wgpu::Origin3d::ZERO, size, &b);
    assert_eq!(read(t).await, b, "write only");

    let t = create();
    ctx.queue.submit(Some(clear(&t)));
    write_texture(&ctx.queue, &t, wgpu::Origin3d::ZERO, size, &b);
    assert_eq!(read(t).await, b, "clear then write");

    let t = create();
    write_texture(&ctx.queue, &t, wgpu::Origin3d::ZERO, size, &b);
    ctx.queue.submit(Some(clear(&t)));
    assert_eq!(read(t).await, red_bytes, "write then clear");

    let t = create();
    let clear_command_buffer = clear(&t);
    write_texture(&ctx.queue, &t, wgpu::Origin3d::ZERO, size, &b);
    ctx.queue.submit(Some(clear_command_buffer));
    assert_eq!(
        read(t).await,
        red_bytes,
        "clear recorded, then write, then clear submitted"
    );

    let t = create();
    write_texture(&ctx.queue, &t, wgpu::Origin3d::ZERO, size, &b);
    write_texture(
        &ctx.queue,
        &t,
        wgpu::Origin3d { x: 1, y: 1, z: 0 },
        extent(2, 2, 1),
        &c[..16],
    );
    let mut expected = b.clone();
    for y in 1..3 {
        let row = (y - 1) * 8;
        expected[y * 16 + 4..y * 16 + 12].copy_from_slice(&c[row..row + 8]);
    }
    assert_eq!(read(t).await, expected, "write then partial write");

    let t = create();
    write_texture(
        &ctx.queue,
        &t,
        wgpu::Origin3d { x: 1, y: 2, z: 0 },
        extent(2, 2, 1),
        &c[..16],
    );
    assert_eq!(
        read(t).await,
        embed(
            wgpu::TextureFormat::Rgba8Unorm,
            size,
            wgpu::Origin3d { x: 1, y: 2, z: 0 },
            extent(2, 2, 1),
            &c[..16]
        ),
        "partial write"
    );

    let t = create();
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 3,
        bytes_per_row: Some(17),
        rows_per_image: Some(4),
    };
    ctx.queue
        .write_texture(t.as_image_copy(), &pack(&b, 16, 4, 1, layout), layout, size);
    assert_eq!(read(t).await, b, "unaligned layout");
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_ORDERING: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(write_texture_ordering);

async fn host_texture_write_texture_host_visible(ctx: TestingContext) {
    let texture = host_test_texture(
        &ctx.device,
        wgpu::TextureUsages::HOST_VISIBLE
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        false,
    );
    let size = texture.size();
    let b = pattern(64, 10);
    let c = pattern(64, 11);
    let d = pattern(64, 12);

    write_texture(&ctx.queue, &texture, wgpu::Origin3d::ZERO, size, &b);
    map_and_wait(&ctx, &texture).await;
    assert_eq!(host_read_all(&texture), b);
    texture.unmap();

    write_texture(&ctx.queue, &texture, wgpu::Origin3d::ZERO, size, &c);
    assert_eq!(
        gpu_read(&ctx, &texture, 0, wgpu::Origin3d::ZERO, size).await,
        c
    );

    write_texture(&ctx.queue, &texture, wgpu::Origin3d::ZERO, size, &d);
    map_and_wait(&ctx, &texture).await;
    assert_eq!(host_read_all(&texture), d);
    host_write_all(&texture, &b);
    texture.unmap();
    assert_eq!(
        gpu_read(&ctx, &texture, 0, wgpu::Origin3d::ZERO, size).await,
        b
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_HOST_VISIBLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        host_texture_write_texture_host_visible(ctx).await;
    });
