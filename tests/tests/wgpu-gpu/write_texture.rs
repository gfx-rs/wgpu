//! Tests for texture copy

use wgpu::*;
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
        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();
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
        ctx.async_poll(wgpu::PollType::wait()).await.unwrap();
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

// Test a writeTexture operation that will use the staging buffer.
// If run with the address sanitizer, this serves as a regression
// test for https://github.com/gfx-rs/wgpu/pull/7893.
#[gpu_test]
static WRITE_TEXTURE_VIA_STAGING_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        let width = 89;
        let height = 17;

        let tex = ctx.device.create_texture(&TextureDescriptor {
            label: None,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::R8Uint,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
        });

        let write_width: u32 = 31;
        let write_height: u32 = 5;
        let write_bytes_per_row: u32 = 113;
        let write_data = (0..(write_height - 1) * write_bytes_per_row + write_width)
            .map(|b| (b % 256) as u8)
            .collect::<Vec<_>>();

        ctx.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &write_data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(write_bytes_per_row),
                rows_per_image: Some(19),
            },
            Extent3d {
                width: write_width,
                height: write_height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(None);

        let read_bytes_per_row = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let read_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: (height * read_bytes_per_row) as u64,
            usage: BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &read_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(read_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait()).await.unwrap();
        let read_data: Vec<u8> = slice.get_mapped_range().to_vec();

        for x in 0..write_width {
            for y in 0..write_height {
                assert_eq!(
                    read_data[(y * read_bytes_per_row + x) as usize],
                    write_data[(y * write_bytes_per_row + x) as usize]
                );
            }
        }
    });
