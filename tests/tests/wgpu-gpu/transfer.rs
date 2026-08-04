use wgpu_test::{apply, fail, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COPY_OVERFLOW_Z);
    vec.push(COPY_TEXTURE_TO_TEXTURE_3D);
}

/// Regression test for the GLES backend copying only z slice 0 of a 3D texture.
#[apply(gpu_test!)]
static COPY_TEXTURE_TO_TEXTURE_3D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        // Each slice is filled with a distinct value, so a dropped or misplaced slice is
        // caught rather than only a wrong copy length. `width` is a multiple of
        // `COPY_BYTES_PER_ROW_ALIGNMENT` so `bytes_per_row` stays aligned.
        let width = 256;
        let height = 2;
        let depth = 4;
        let slice_len = (width * height) as usize;

        let descriptor = wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D3,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        };
        let src = ctx.device.create_texture(&descriptor);
        let dst = ctx.device.create_texture(&descriptor);

        let data: Vec<u8> = (0..depth)
            .flat_map(|z| vec![(z + 1) as u8; slice_len])
            .collect();
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        };
        ctx.queue
            .write_texture(src.as_image_copy(), &data, layout, descriptor.size);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), descriptor.size);
        encoder.copy_texture_to_buffer(
            dst.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout,
            },
            descriptor.size,
        );
        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();
        let read: Vec<u8> = slice.get_mapped_range().unwrap().to_vec();

        for z in 0..depth as usize {
            let expected = (z + 1) as u8;
            let got = &read[z * slice_len..(z + 1) * slice_len];
            assert!(
                got.iter().all(|&texel| texel == expected),
                "slice z={z} not copied: expected all {expected}, got {:?}",
                &got[..8],
            );
        }
    });

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
