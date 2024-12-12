//! Tests for texture copy

use wgpu_test::{gpu_test, GpuTestConfiguration};

#[gpu_test]
static WRITE_TEXTURE_SUBSET_2D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let size = 256;

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; size as usize * 2];
        // Write the first two rows
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: 2,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(None);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * size) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        let data: Vec<u8> = slice.get_mapped_range().to_vec();

        for byte in &data[..(size as usize * 2)] {
            assert_eq!(*byte, 1);
        }
        for byte in &data[(size as usize * 2)..] {
            assert_eq!(*byte, 0);
        }
    });

#[gpu_test]
static WRITE_TEXTURE_SUBSET_3D: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let size = 256;
        let depth = 4;
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D3,
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: depth,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; (size * size) as usize * 2];
        // Write the first two slices
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 2,
            },
        );

        ctx.queue.submit(None);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * size * depth) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: depth,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.async_poll(wgpu::Maintain::wait())
            .await
            .panic_on_timeout();
        let data: Vec<u8> = slice.get_mapped_range().to_vec();

        for byte in &data[..((size * size) as usize * 2)] {
            assert_eq!(*byte, 1);
        }
        for byte in &data[((size * size) as usize * 2)..] {
            assert_eq!(*byte, 0);
        }
    });

#[gpu_test]
static WRITE_TEXTURE_NO_OOB: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let size = 256;

        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });
        let data = vec![1u8; size as usize * 2 + 100]; // check that we don't attempt to copy OOB internally by adding 100 bytes here
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: 2,
                depth_or_array_layers: 1,
            },
        );
    });
