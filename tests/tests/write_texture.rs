//! Tests for texture copy

use wgpu_test::{initialize_test, FailureCase, TestParameters};

use wasm_bindgen_test::*;

#[test]
#[wasm_bindgen_test]
fn write_texture_subset_2d() {
    let size = 256;
    let parameters =
        TestParameters::default().expect_fail(FailureCase::backend(wgpu::Backends::DX12));
    initialize_test(parameters, |ctx| {
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
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
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

        let read_buffer_0 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 2) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_buffer_1 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * (size - 2)) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buffer_0,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: 2,
                depth_or_array_layers: 1,
            },
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 2, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buffer_1,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size - 2,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice_0 = read_buffer_0.slice(..);
        slice_0.map_async(wgpu::MapMode::Read, |_| ());
        let slice_1 = read_buffer_1.slice(..);
        slice_1.map_async(wgpu::MapMode::Read, |_| ());

        ctx.device.poll(wgpu::Maintain::Wait);

        let data_0 = slice_0.get_mapped_range();
        let data_1 = slice_1.get_mapped_range();

        for byte in data_0.as_ref() {
            assert_eq!(*byte, 1);
        }
        for byte in data_1.as_ref() {
            assert_eq!(*byte, 0);
        }
    });
}

#[test]
#[wasm_bindgen_test]
fn write_texture_subset_3d() {
    let size = 256;
    let depth = 4;
    let parameters = TestParameters::default();
    initialize_test(parameters, |ctx| {
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
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
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
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buffer,
                layout: wgpu::ImageDataLayout {
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
        ctx.device.poll(wgpu::Maintain::Wait);
        let data: Vec<u8> = slice.get_mapped_range().to_vec();

        for byte in &data[..((size * size) as usize * 2)] {
            assert_eq!(*byte, 1);
        }
        for byte in &data[((size * size) as usize * 2)..] {
            assert_eq!(*byte, 0);
        }
    });
}
