//! Tests for texture copy

use wgpu_test::{initialize_test, TestParameters};

use wasm_bindgen_test::*;

struct LayoutDesc {
    bytes_per_row: u32,
    rows_per_image: u32,
    total_size_in_bytes: u32,
}

fn total_bytes_in_copy(
    texture_format: wgpu::TextureFormat,
    bytes_per_row: u32,
    rows_per_image: u32,
    copy_extent: wgpu::Extent3d,
) -> u32 {
    let block_size = texture_format.block_size(None).unwrap_or(1);

    let block_dim = texture_format.block_dimensions();

    let block_width = copy_extent.width / block_dim.0;
    let block_height = copy_extent.height / block_dim.1;

    let bytes_per_image = bytes_per_row * rows_per_image;
    let mut total_bytes = bytes_per_image * (copy_extent.depth_or_array_layers - 1);

    if block_height != 0 {
        let last_row_bytes = block_width * block_size;
        let last_image_bytes = bytes_per_row * (block_height - 1) + last_row_bytes;
        total_bytes += last_image_bytes;
    }

    total_bytes
}

fn test(
    format: wgpu::TextureFormat,
    layout_desc: LayoutDesc,
    mip_level: u32,
    mips_count: u32,
    tex_size: wgpu::Extent3d,
    write_size: wgpu::Extent3d,
    copy_size: wgpu::Extent3d,
) {
    let LayoutDesc {
        bytes_per_row,
        total_size_in_bytes,
        rows_per_image,
    } = layout_desc;
    let parameters = TestParameters::default();
    initialize_test(parameters, |ctx| {
        let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size: tex_size,
            format,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: mips_count,
            sample_count: 1,
            view_formats: &[],
        });

        let val = (mip_level + 1) as u8;
        let data = vec![val; total_size_in_bytes as usize];

        // Write the first two rows
        ctx.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            write_size,
        );

        ctx.queue.submit(None);

        let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (total_size_in_bytes) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &tex,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &read_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows_per_image),
                },
            },
            copy_size,
        );

        ctx.queue.submit(Some(encoder.finish()));

        let slice = read_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        ctx.device.poll(wgpu::Maintain::Wait);
        let data: Vec<u8> = slice.get_mapped_range().to_vec();

        for byte in &data[..(total_size_in_bytes as usize)] {
            assert_eq!(*byte, val);
        }
        for byte in &data[(total_size_in_bytes as usize)..] {
            assert_eq!(*byte, 0);
        }
    });
}

#[test]
#[wasm_bindgen_test]
fn write_texture_subset_2d() {
    let format = wgpu::TextureFormat::R8Uint;
    let mips_count = 1;
    let tex_size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
    };

    let bytes_per_row = tex_size.width;
    let rows_per_image = tex_size.height;

    let total_bytes_in_copy = total_bytes_in_copy(format, bytes_per_row, rows_per_image, tex_size);

    test(
        format,
        LayoutDesc {
            bytes_per_row,
            rows_per_image,
            total_size_in_bytes: total_bytes_in_copy,
        },
        0,
        mips_count,
        tex_size,
        tex_size,
        tex_size,
    );
}

#[test]
#[wasm_bindgen_test]
fn write_texture_subset_2d_mips() {
    let format = wgpu::TextureFormat::R8Uint;
    let mips_count = 3;
    let tex_size = wgpu::Extent3d {
        width: 2048,
        height: 2048,
        depth_or_array_layers: 1,
    };
    for mip_level in 0..mips_count {
        let mip_w = tex_size.width / (1 << mip_level);
        let mip_h = tex_size.height / (1 << mip_level);
        let bytes_per_row = mip_w;
        let rows_per_image = mip_h;

        let mip_extent = wgpu::Extent3d {
            width: mip_w,
            height: 2,
            depth_or_array_layers: tex_size.depth_or_array_layers,
        };

        let total_bytes_in_copy =
            total_bytes_in_copy(format, bytes_per_row, rows_per_image, mip_extent);

        test(
            format,
            LayoutDesc {
                bytes_per_row,
                rows_per_image,
                total_size_in_bytes: total_bytes_in_copy,
            },
            mip_level,
            mips_count,
            tex_size,
            mip_extent,
            mip_extent,
        );
    }
}

#[test]
#[wasm_bindgen_test]
fn write_texture_subset_3d() {
    let format = wgpu::TextureFormat::R8Uint;
    let mips_count = 1;
    let tex_size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 4,
    };
    let copy_size = wgpu::Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 2,
    };

    let bytes_per_row = tex_size.width;
    let rows_per_image = tex_size.height;

    let total_bytes_in_copy = total_bytes_in_copy(format, bytes_per_row, rows_per_image, copy_size);

    test(
        format,
        LayoutDesc {
            bytes_per_row,
            rows_per_image,
            total_size_in_bytes: total_bytes_in_copy,
        },
        0,
        mips_count,
        tex_size,
        copy_size,
        copy_size,
    );
}
