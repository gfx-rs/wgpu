use std::{borrow::Cow, ffi::OsStr, io, path::Path};
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

fn add_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(3)
        .flat_map(|chunk| [chunk[0], chunk[1], chunk[2], 255])
        .collect()
}

fn remove_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(4)
        .flat_map(|chunk| &chunk[0..3])
        .copied()
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn print_flip(pool: &mut nv_flip::FlipPool) {
    println!("\tMean: {:.6}", pool.mean());
    println!("\tMin Value: {:.6}", pool.min_value());
    for percentile in [25, 50, 75, 95, 99] {
        println!(
            "\t      {percentile}%: {:.6}",
            pool.get_percentile(percentile as f32 / 100.0, true)
        );
    }
    println!("\tMax Value: {:.6}", pool.max_value());
}

/// The FLIP library generates a per-pixel error map where 0.0 represents "no error"
/// and 1.0 represents "maximum error" between the images. This is then put into
/// a weighted-histogram, which we query to determine if the errors between
/// the test and reference image is high enough to count as "different".
///
/// Error thresholds will be different for every test, but good initial values
/// to look at are in the [0.01, 0.1] range. The larger the area that might have
/// inherent variance, the larger this base value is. Using a high percentile comparison
/// (e.g. 95% or 99%) is good for images that are likely to have a lot of error
/// in a small area when they fail.
#[derive(Debug, Clone, Copy)]
pub enum ComparisonType {
    /// If the mean error is greater than the given value, the test will fail.
    Mean(f32),
    /// If the given percentile is greater than the given value, the test will fail.
    ///
    /// The percentile is given in the range [0, 1].
    Percentile { percentile: f32, threshold: f32 },
}

impl ComparisonType {
    #[cfg(not(target_arch = "wasm32"))]
    fn check(&self, pool: &mut nv_flip::FlipPool) -> bool {
        match *self {
            ComparisonType::Mean(v) => {
                let mean = pool.mean();
                let within = mean <= v;
                println!(
                    "\tExpected Mean ({:.6}) to be under expected maximum ({}): {}",
                    mean,
                    v,
                    if within { "PASS" } else { "FAIL" }
                );
                within
            }
            ComparisonType::Percentile {
                percentile: p,
                threshold: v,
            } => {
                let percentile = pool.get_percentile(p, true);
                let within = percentile <= v;
                println!(
                    "\tExpected {}% ({:.6}) to be under expected maximum ({}): {}",
                    p * 100.0,
                    percentile,
                    v,
                    if within { "PASS" } else { "FAIL" }
                );
                within
            }
        }
    }
}

pub fn compare_image_output(
    path: impl AsRef<Path> + AsRef<OsStr>,
    backend: wgpu::Backend,
    width: u32,
    height: u32,
    test_with_alpha: &[u8],
    checks: &[ComparisonType],
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::{ffi::OsString, str::FromStr};

        let reference_with_alpha = read_png(&path, width, height);

        let reference = match reference_with_alpha {
            Some(v) => remove_alpha(&v),
            None => {
                write_png(
                    &path,
                    width,
                    height,
                    test_with_alpha,
                    png::Compression::Best,
                );
                return;
            }
        };
        let test = remove_alpha(test_with_alpha);

        assert_eq!(reference.len(), test.len());

        let reference_flip = nv_flip::FlipImageRgb8::with_data(width, height, &reference);
        let test_flip = nv_flip::FlipImageRgb8::with_data(width, height, &test);

        let error_map_flip = nv_flip::flip(
            reference_flip,
            test_flip,
            nv_flip::DEFAULT_PIXELS_PER_DEGREE,
        );
        let mut pool = nv_flip::FlipPool::from_image(&error_map_flip);

        let reference_path = Path::new(&path);
        println!(
            "Starting image comparison test with reference image \"{}\"",
            reference_path.display()
        );

        print_flip(&mut pool);

        // If there are no checks, we want to fail the test.
        let mut all_passed = !checks.is_empty();
        // We always iterate all of these, as the call to check prints
        for check in checks {
            all_passed &= check.check(&mut pool);
        }

        let file_stem = reference_path.file_stem().unwrap().to_string_lossy();
        // Determine the paths to write out the various intermediate files
        let actual_path = Path::new(&path).with_file_name(
            OsString::from_str(&format!("{}-{}-actual.png", file_stem, backend.to_str(),)).unwrap(),
        );
        let difference_path = Path::new(&path).with_file_name(
            OsString::from_str(&format!(
                "{}-{}-difference.png",
                file_stem,
                backend.to_str(),
            ))
            .unwrap(),
        );

        // Convert the error values to a false color reprensentation
        let magma_image = error_map_flip
            .apply_color_lut(&nv_flip::magma_lut())
            .to_vec();
        let magma_image_with_alpha = add_alpha(&magma_image);

        write_png(
            actual_path,
            width,
            height,
            test_with_alpha,
            png::Compression::Fast,
        );
        write_png(
            difference_path,
            width,
            height,
            &magma_image_with_alpha,
            png::Compression::Fast,
        );

        if !all_passed {
            panic!("Image data mismatch!")
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = (path, backend, width, height, test_with_alpha, checks);
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
