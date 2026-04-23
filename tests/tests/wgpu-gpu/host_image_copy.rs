use wgpu_test::{
    gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(HOST_IMAGE_UPLOAD);
    vec.push(HOST_IMAGE_UPLOAD_MAPPED_AT_CREATION);
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
        let mapped = texture.get_mapped();

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
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_IMAGE_COPY)
            // llvmpipe reports HOST_IMAGE_COPY support but cannot load the
            // vkCopyMemoryToImageEXT function pointers (driver bug).
            .skip(FailureCase::adapter("llvmpipe")),
    )
    .run_async(|ctx| async move {
        host_image_upload(ctx, false).await;
    });

#[gpu_test]
static HOST_IMAGE_UPLOAD_MAPPED_AT_CREATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_IMAGE_COPY)
            // llvmpipe reports HOST_IMAGE_COPY support but cannot load the
            // vkCopyMemoryToImageEXT function pointers (driver bug).
            .skip(FailureCase::adapter("llvmpipe")),
    )
    .run_async(|ctx| async move {
        host_image_upload(ctx, true).await;
    });
