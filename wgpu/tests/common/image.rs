use std::{
    borrow::Cow,
    ffi::{OsStr, OsString},
    io,
    path::Path,
    str::FromStr,
};
use wgpu::util::DeviceExt;
use wgpu::*;

fn read_png(path: impl AsRef<Path>, width: u32, height: u32) -> Option<Vec<u8>> {
    let data = match std::fs::read(&path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!(
                "image comparison invalid: file io error when comparing {}: {}",
                path.as_ref().display(),
                e
            );
            return None;
        }
    };
    let decoder = png::Decoder::new(io::Cursor::new(data));
    let mut reader = decoder.read_info().ok()?;

    let mut buffer = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buffer).ok()?;
    if info.width != width {
        log::warn!("image comparison invalid: size mismatch");
        return None;
    }
    if info.height != height {
        log::warn!("image comparison invalid: size mismatch");
        return None;
    }
    if info.color_type != png::ColorType::Rgba {
        log::warn!("image comparison invalid: color type mismatch");
        return None;
    }
    if info.bit_depth != png::BitDepth::Eight {
        log::warn!("image comparison invalid: bit depth mismatch");
        return None;
    }

    Some(buffer)
}

#[allow(unused_variables)]
fn write_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u8],
    compression: png::Compression,
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let file = io::BufWriter::new(std::fs::File::create(path).unwrap());

        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(compression);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(data).unwrap();
    }
}

pub fn calc_difference(lhs: u8, rhs: u8) -> u8 {
    (lhs as i16 - rhs as i16).unsigned_abs() as u8
}

pub fn compare_image_output(
    path: impl AsRef<Path> + AsRef<OsStr>,
    width: u32,
    height: u32,
    data: &[u8],
    tolerance: u8,
    max_outliers: usize,
) {
    let comparison_data = read_png(&path, width, height);

    if let Some(cmp) = comparison_data {
        assert_eq!(cmp.len(), data.len());

        let difference_data: Vec<_> = cmp
            .chunks_exact(4)
            .zip(data.chunks_exact(4))
            .flat_map(|(cmp_chunk, data_chunk)| {
                [
                    calc_difference(cmp_chunk[0], data_chunk[0]),
                    calc_difference(cmp_chunk[1], data_chunk[1]),
                    calc_difference(cmp_chunk[2], data_chunk[2]),
                    255,
                ]
            })
            .collect();

        let outliers: usize = difference_data
            .chunks_exact(4)
            .map(|colors| {
                (colors[0] > tolerance) as usize
                    + (colors[1] > tolerance) as usize
                    + (colors[2] > tolerance) as usize
            })
            .sum();

        let max_difference = difference_data
            .chunks_exact(4)
            .map(|colors| colors[0].max(colors[1]).max(colors[2]))
            .max()
            .unwrap();

        if outliers > max_outliers {
            // Because the data is mismatched, lets output the difference to a file.
            let old_path = Path::new(&path);
            let actual_path = Path::new(&path).with_file_name(
                OsString::from_str(
                    &(old_path.file_stem().unwrap().to_string_lossy() + "-actual.png"),
                )
                .unwrap(),
            );
            let difference_path = Path::new(&path).with_file_name(
                OsString::from_str(
                    &(old_path.file_stem().unwrap().to_string_lossy() + "-difference.png"),
                )
                .unwrap(),
            );

            write_png(actual_path, width, height, data, png::Compression::Fast);
            write_png(
                difference_path,
                width,
                height,
                &difference_data,
                png::Compression::Fast,
            );

            panic!(
                "Image data mismatch! Outlier count {outliers} over limit {max_outliers}. Max difference {max_difference}"
            )
        } else {
            println!("{outliers} outliers over max difference {max_difference}");
        }
    } else {
        write_png(&path, width, height, data, png::Compression::Best);
    }
}

fn copy_via_compute(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    aspect: TextureAspect,
) {
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: match aspect {
                        TextureAspect::DepthOnly => TextureSampleType::Float { filterable: false },
                        TextureAspect::StencilOnly => TextureSampleType::Uint,
                        _ => unreachable!(),
                    },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let view = texture.create_view(&TextureViewDescriptor {
        aspect,
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });

    let output_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("output buffer"),
        size: buffer.size(),
        usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bg = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &output_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let pll = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let source = String::from(include_str!("copy_texture_to_buffer.wgsl"));

    let processed_source = source.replace(
        "{{type}}",
        match aspect {
            TextureAspect::DepthOnly => "f32",
            TextureAspect::StencilOnly => "u32",
            _ => unreachable!(),
        },
    );

    let sm = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("shader copy_texture_to_buffer.wgsl"),
        source: ShaderSource::Wgsl(Cow::Borrowed(&processed_source)),
    });

    let pipeline_copy = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("pipeline read"),
        layout: Some(&pll),
        module: &sm,
        entry_point: "copy_texture_to_buffer",
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_pipeline(&pipeline_copy);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, buffer, 0, buffer.size());
}

fn copy_texture_to_buffer_with_aspect(
    encoder: &mut CommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    buffer_stencil: &Option<Buffer>,
    aspect: TextureAspect,
) {
    let (block_width, block_height) = texture.format().block_dimensions();
    let block_size = texture.format().block_size(Some(aspect)).unwrap();
    let mip_level = 0;
    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            texture,
            mip_level,
            origin: Origin3d::ZERO,
            aspect,
        },
        ImageCopyBuffer {
            buffer: match aspect {
                TextureAspect::StencilOnly => buffer_stencil.as_ref().unwrap(),
                _ => buffer,
            },
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some((texture.width() / block_width) * block_size),
                rows_per_image: Some(texture.height() / block_height),
            },
        },
        texture
            .size()
            .mip_level_size(mip_level, texture.dimension()),
    );
}

fn copy_texture_to_buffer(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    buffer_stencil: &Option<Buffer>,
) {
    match texture.format() {
        TextureFormat::Depth24Plus => {
            copy_via_compute(device, encoder, texture, buffer, TextureAspect::DepthOnly);
        }
        TextureFormat::Depth24PlusStencil8 => {
            copy_via_compute(device, encoder, texture, buffer, TextureAspect::DepthOnly);
            // copy_via_compute(
            //     device,
            //     encoder,
            //     texture,
            //     buffer_stencil.as_ref().unwrap(),
            //     TextureAspect::StencilOnly,
            // );
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                buffer,
                buffer_stencil,
                TextureAspect::StencilOnly,
            );
        }
        TextureFormat::Depth32FloatStencil8 => {
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                buffer,
                buffer_stencil,
                TextureAspect::DepthOnly,
            );
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                buffer,
                buffer_stencil,
                TextureAspect::StencilOnly,
            );
        }
        _ => {
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                buffer,
                buffer_stencil,
                TextureAspect::All,
            );
        }
    }
}

pub struct ReadbackBuffers {
    /// buffer for color or depth aspects
    buffer: Buffer,
    /// buffer for stencil aspect
    buffer_stencil: Option<Buffer>,
}

impl ReadbackBuffers {
    pub fn new(device: &Device, texture: &Texture) -> Self {
        let (block_width, block_height) = texture.format().block_dimensions();
        let base_size = (texture.width() / block_width)
            * (texture.height() / block_height)
            * texture.depth_or_array_layers();
        if texture.format().is_combined_depth_stencil_format() {
            let buffer_size = base_size
                * texture
                    .format()
                    .block_size(Some(TextureAspect::DepthOnly))
                    .unwrap_or(4);
            let buffer_stencil_size = base_size
                * texture
                    .format()
                    .block_size(Some(TextureAspect::StencilOnly))
                    .unwrap();
            let buffer = device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Texture Readback"),
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                contents: &vec![255; buffer_size as usize],
            });
            let buffer_stencil = device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Texture Stencil-Aspect Readback"),
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                contents: &vec![255; buffer_stencil_size as usize],
            });
            ReadbackBuffers {
                buffer,
                buffer_stencil: Some(buffer_stencil),
            }
        } else {
            let buffer_size = base_size * texture.format().block_size(None).unwrap_or(4);
            let buffer = device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("Texture Readback"),
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                contents: &vec![255; buffer_size as usize],
            });
            ReadbackBuffers {
                buffer,
                buffer_stencil: None,
            }
        }
    }

    // TODO: also copy and check mips
    pub fn copy_from(&self, device: &Device, encoder: &mut CommandEncoder, texture: &Texture) {
        copy_texture_to_buffer(device, encoder, texture, &self.buffer, &self.buffer_stencil);
    }

    pub fn are_zero(&self, device: &Device) -> bool {
        fn is_zero(device: &Device, buffer: &Buffer) -> bool {
            let is_zero = {
                let buffer_slice = buffer.slice(..);
                buffer_slice.map_async(MapMode::Read, |_| ());
                device.poll(Maintain::Wait);
                let buffer_view = buffer_slice.get_mapped_range();
                buffer_view.iter().all(|b| *b == 0)
            };
            buffer.unmap();
            is_zero
        }

        is_zero(device, &self.buffer)
            && self
                .buffer_stencil
                .as_ref()
                .map(|buffer_stencil| is_zero(device, buffer_stencil))
                .unwrap_or(true)
    }
}
