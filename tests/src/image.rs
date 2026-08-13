//! Image comparison utilities

use std::{borrow::Cow, ffi::OsStr, path::Path};

use wgpu::util::{align_to, DeviceExt};
use wgpu::*;

use crate::TestingContext;

#[cfg(not(any(target_arch = "wasm32", miri)))]
async fn read_png(path: impl AsRef<Path>, width: u32, height: u32) -> Option<Vec<u8>> {
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
    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().ok()?;

    let buffer_len = reader
        .output_buffer_size()
        .expect("output buffer would not fit in memory");
    let mut buffer = vec![0; buffer_len];
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

#[cfg(not(any(target_arch = "wasm32", miri)))]
async fn write_png(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    data: &[u8],
    compression: png::Compression,
) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());

    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(compression);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(data).unwrap();
}

#[cfg_attr(any(target_arch = "wasm32", miri), allow(unused))]
fn add_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(3)
        .flat_map(|chunk| [chunk[0], chunk[1], chunk[2], 255])
        .collect()
}

#[cfg_attr(any(target_arch = "wasm32", miri), allow(unused))]
fn remove_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(4)
        .flat_map(|chunk| &chunk[0..3])
        .copied()
        .collect()
}

#[cfg(not(any(target_arch = "wasm32", miri)))]
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
    #[cfg(not(any(target_arch = "wasm32", miri)))]
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

#[cfg(not(any(target_arch = "wasm32", miri)))]
pub async fn compare_image_output(
    path: impl AsRef<Path> + AsRef<OsStr>,
    adapter_info: &wgpu::AdapterInfo,
    width: u32,
    height: u32,
    test_with_alpha: &[u8],
    checks: &[ComparisonType],
) {
    use std::{ffi::OsString, str::FromStr};

    let reference_path = Path::new(&path);
    let reference_with_alpha = read_png(&path, width, height).await;

    let reference = match reference_with_alpha {
        Some(v) => remove_alpha(&v),
        None => {
            write_png(
                &path,
                width,
                height,
                test_with_alpha,
                png::Compression::High,
            )
            .await;
            return;
        }
    };
    let test = remove_alpha(test_with_alpha);

    assert_eq!(reference.len(), test.len());

    let file_stem = reference_path.file_stem().unwrap().to_string_lossy();
    let renderer = format!(
        "{}-{}-{}",
        adapter_info.backend,
        sanitize_for_path(&adapter_info.name),
        sanitize_for_path(&adapter_info.driver)
    );
    // Determine the paths to write out the various intermediate files
    let actual_path = Path::new(&path)
        .with_file_name(OsString::from_str(&format!("{file_stem}-{renderer}-actual.png")).unwrap());
    let difference_path = Path::new(&path).with_file_name(
        OsString::from_str(&format!("{file_stem}-{renderer}-difference.png",)).unwrap(),
    );

    let mut all_passed;
    let magma_image_with_alpha;
    {
        let reference_flip = nv_flip::FlipImageRgb8::with_data(width, height, &reference);
        let test_flip = nv_flip::FlipImageRgb8::with_data(width, height, &test);

        let error_map_flip = nv_flip::flip(
            reference_flip,
            test_flip,
            nv_flip::DEFAULT_PIXELS_PER_DEGREE,
        );
        let mut pool = nv_flip::FlipPool::from_image(&error_map_flip);

        println!(
            "Starting image comparison test with reference image \"{}\"",
            reference_path.display()
        );

        print_flip(&mut pool);

        // If there are no checks, we want to fail the test.
        all_passed = !checks.is_empty();
        // We always iterate all of these, as the call to check prints
        for check in checks {
            all_passed &= check.check(&mut pool);
        }

        // Convert the error values to a false color representation
        let magma_image = error_map_flip
            .apply_color_lut(&nv_flip::magma_lut())
            .to_vec();
        magma_image_with_alpha = add_alpha(&magma_image);
    }

    write_png(
        actual_path,
        width,
        height,
        test_with_alpha,
        png::Compression::Fast,
    )
    .await;
    write_png(
        &difference_path,
        width,
        height,
        &magma_image_with_alpha,
        png::Compression::Fast,
    )
    .await;

    if !all_passed {
        panic!("Image data mismatch: {}", difference_path.display())
    }
}

#[cfg(any(target_arch = "wasm32", miri))]
pub async fn compare_image_output(
    path: impl AsRef<Path> + AsRef<OsStr>,
    adapter_info: &wgpu::AdapterInfo,
    width: u32,
    height: u32,
    test_with_alpha: &[u8],
    checks: &[ComparisonType],
) {
    #[cfg(any(target_arch = "wasm32", miri))]
    {
        let _ = (path, adapter_info, width, height, test_with_alpha, checks);
    }
}

#[cfg_attr(any(target_arch = "wasm32", miri), allow(unused))]
fn sanitize_for_path(s: &str) -> String {
    s.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

/// Value that readback buffers are filled with before use, so that a copy that fails to
/// happen is not mistaken for zeroed memory.
const POISON: u8 = 255;

/// Layout of one mip level of one aspect of a texture within a readback buffer.
///
/// The array layers (for `D2` textures) or depth slices (for `D3`) of a mip level are laid
/// out consecutively within its region, each `bytes_per_row * rows_per_image` bytes long.
#[derive(Clone, Copy, Debug)]
struct MipLayout {
    /// Byte offset of this mip level's region within the buffer. A multiple of
    /// [`COPY_BYTES_PER_ROW_ALIGNMENT`], which satisfies every alignment requirement that
    /// applies: copy offsets, storage binding offsets, and buffer mapping.
    offset: u64,
    /// Row stride within the buffer. Aligned to [`COPY_BYTES_PER_ROW_ALIGNMENT`], which
    /// `copy_texture_to_buffer` requires, except on the compute shader path, whose shader
    /// writes tightly packed rows.
    bytes_per_row: u32,
    /// Bytes of texel data in each row, without the padding included in `bytes_per_row`.
    unpadded_bytes_per_row: u32,
    /// Number of block rows in each array layer or depth slice.
    rows_per_image: u32,
    /// Physical (block aligned) extent of this mip level.
    size: Extent3d,
    /// Bytes reserved for this mip level, including the padding that keeps the offset of
    /// the next mip level aligned.
    region_size: u64,
}

impl MipLayout {
    /// Bytes of texel data in one array layer or depth slice, excluding row padding.
    fn unpadded_subresource_size(&self) -> usize {
        self.unpadded_bytes_per_row as usize * self.rows_per_image as usize
    }

    /// Bytes of texel data in this mip level, excluding all padding.
    fn unpadded_size(&self) -> usize {
        self.unpadded_subresource_size() * self.size.depth_or_array_layers as usize
    }
}

/// Whether the readable aspect of `format` has to be read with a compute shader because it
/// cannot be the source of a `copy_texture_to_buffer`.
fn needs_compute_readback(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8
    )
}

/// Compute the layout of every mip level of one aspect of `texture`.
///
/// When `dense_rows` is `true`, computes a layout with tightly packed rows for
/// the compute shader path. When false, computes a layout with padding to
/// `COPY_BYTES_PER_ROW_ALIGNMENT` for `copy_texture_to_buffer`.
fn mip_layouts(
    texture: &Texture,
    aspect: Option<TextureAspect>,
    dense_rows: bool,
) -> Vec<MipLayout> {
    let format = texture.format();
    let (block_width, block_height) = format.block_dimensions();
    // `block_copy_size` returns `None` for the aspects that are read with a compute
    // shader, which writes one `f32` or `u32` per texel.
    let block_size = format.block_copy_size(aspect).unwrap_or(4);

    let mut offset = 0;
    (0..texture.mip_level_count())
        .map(|mip_level| {
            // The physical size is what a copy of a whole mip level must use, and it is
            // what the buffer has to hold: for a compressed format, the last row or
            // column of blocks extends past the logical size.
            let size = texture
                .size()
                .mip_level_size(mip_level, texture.dimension())
                .physical_size(format);
            let unpadded_bytes_per_row = (size.width / block_width) * block_size;
            let bytes_per_row = if dense_rows {
                unpadded_bytes_per_row
            } else {
                align_to(unpadded_bytes_per_row, COPY_BYTES_PER_ROW_ALIGNMENT)
            };
            let rows_per_image = size.height / block_height;
            let layout = MipLayout {
                offset,
                bytes_per_row,
                unpadded_bytes_per_row,
                rows_per_image,
                size,
                region_size: align_to(
                    u64::from(bytes_per_row)
                        * u64::from(rows_per_image)
                        * u64::from(size.depth_or_array_layers),
                    u64::from(COPY_BYTES_PER_ROW_ALIGNMENT),
                ),
            };
            offset += layout.region_size;
            layout
        })
        .collect()
}

fn create_readback_buffer(device: &Device, label: &str, layouts: &[MipLayout]) -> Buffer {
    let size = layouts.last().map_or(0, |l| l.offset + l.region_size);
    device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some(label),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        contents: &vec![POISON; size as usize],
    })
}

fn copy_via_compute(
    device: &Device,
    encoder: &mut CommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    layouts: &[MipLayout],
    aspect: TextureAspect,
) {
    assert_eq!(
        texture.dimension(),
        TextureDimension::D2,
        "the compute shader readback path binds the texture as `D2Array`"
    );

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

    // The shader writes one texel at a time, so anything it fails to cover keeps the
    // poison value rather than reading back as zero.
    let output_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("output buffer"),
        usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        contents: &vec![POISON; buffer.size() as usize],
    });

    // A view of a single mip level rebases mip level 0 onto it, so the shader reads the
    // right level without knowing which one it is.
    let bind_groups: Vec<BindGroup> = layouts
        .iter()
        .enumerate()
        .map(|(mip_level, layout)| {
            let view = texture.create_view(&TextureViewDescriptor {
                aspect,
                dimension: Some(TextureViewDimension::D2Array),
                base_mip_level: mip_level as u32,
                mip_level_count: Some(1),
                ..Default::default()
            });
            device.create_bind_group(&BindGroupDescriptor {
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
                            offset: layout.offset,
                            size: BufferSize::new(layout.region_size),
                        }),
                    },
                ],
            })
        })
        .collect();

    let pll = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
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
        entry_point: Some("copy_texture_to_buffer"),
        compilation_options: Default::default(),
        cache: None,
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_pipeline(&pipeline_copy);
        for bind_group in &bind_groups {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, buffer, 0, output_buffer.size());
}

fn copy_texture_to_buffer_with_aspect(
    encoder: &mut CommandEncoder,
    texture: &Texture,
    buffer: &Buffer,
    layouts: &[MipLayout],
    aspect: TextureAspect,
) {
    for (mip_level, layout) in layouts.iter().enumerate() {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: mip_level as u32,
                origin: Origin3d::ZERO,
                aspect,
            },
            TexelCopyBufferInfo {
                buffer,
                layout: TexelCopyBufferLayout {
                    offset: layout.offset,
                    bytes_per_row: Some(layout.bytes_per_row),
                    rows_per_image: Some(layout.rows_per_image),
                },
            },
            layout.size,
        );
    }
}

/// Buffers holding a readback of every mip level and every array layer or depth slice of a
/// texture.
pub struct ReadbackBuffers {
    /// format of the texture this was created for
    texture_format: TextureFormat,
    /// buffer for color or depth aspects
    buffer: Buffer,
    /// layout of `buffer`, one entry per mip level
    layouts: Vec<MipLayout>,
    /// buffer for stencil aspect
    buffer_stencil: Option<Buffer>,
    /// layout of `buffer_stencil`, empty if there is none
    stencil_layouts: Vec<MipLayout>,
}

impl ReadbackBuffers {
    pub fn new(device: &Device, texture: &Texture) -> Self {
        let texture_format = texture.format();
        let combined = texture_format.is_combined_depth_stencil_format();

        // When using the compute shader path (`depth24plus` and
        // `depth24plus-stencil8`), rows are packed tightly instead of padded to
        // `COPY_BYTES_PER_ROW_ALIGNMENT`.
        let layouts = mip_layouts(
            texture,
            combined.then_some(TextureAspect::DepthOnly),
            needs_compute_readback(texture_format),
        );
        let buffer = create_readback_buffer(device, "Texture Readback", &layouts);

        let (buffer_stencil, stencil_layouts) = if combined {
            let layouts = mip_layouts(texture, Some(TextureAspect::StencilOnly), false);
            let buffer =
                create_readback_buffer(device, "Texture Stencil-Aspect Readback", &layouts);
            (Some(buffer), layouts)
        } else {
            (None, Vec::new())
        };

        ReadbackBuffers {
            texture_format,
            buffer,
            layouts,
            buffer_stencil,
            stencil_layouts,
        }
    }

    pub fn copy_from(&self, device: &Device, encoder: &mut CommandEncoder, texture: &Texture) {
        assert_eq!(
            (texture.format(), texture.mip_level_count() as usize),
            (self.texture_format, self.layouts.len()),
            "texture does not match the one this `ReadbackBuffers` was created for"
        );

        if needs_compute_readback(self.texture_format) {
            copy_via_compute(
                device,
                encoder,
                texture,
                &self.buffer,
                &self.layouts,
                TextureAspect::DepthOnly,
            );
        } else {
            let aspect = if self.buffer_stencil.is_some() {
                TextureAspect::DepthOnly
            } else {
                TextureAspect::All
            };
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                &self.buffer,
                &self.layouts,
                aspect,
            );
        }

        if let Some(buffer_stencil) = &self.buffer_stencil {
            copy_texture_to_buffer_with_aspect(
                encoder,
                texture,
                buffer_stencil,
                &self.stencil_layouts,
                TextureAspect::StencilOnly,
            );
        }
    }

    /// Map `buffer` and return its texel data, tightly packed, with the row padding that
    /// `layouts` describes removed. Mip level 0 comes first.
    ///
    /// The caller is responsible for unmapping `buffer`.
    async fn retrieve_buffer(
        &self,
        ctx: &TestingContext,
        buffer: &Buffer,
        layouts: &[MipLayout],
    ) -> Vec<u8> {
        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        let data: BufferView = buffer_slice.get_mapped_range().unwrap();

        let mut result = Vec::with_capacity(layouts.iter().map(MipLayout::unpadded_size).sum());
        for layout in layouts {
            let start = layout.offset as usize;
            let rows = layout.rows_per_image as usize * layout.size.depth_or_array_layers as usize;
            let region = &data[start..start + layout.bytes_per_row as usize * rows];
            for row in region.chunks_exact(layout.bytes_per_row as usize) {
                result.extend_from_slice(&row[..layout.unpadded_bytes_per_row as usize]);
            }
        }
        result
    }

    /// Read back one aspect of the texture.
    ///
    /// Reads the stencil buffer if `aspect` is [`TextureAspect::StencilOnly`] and the
    /// texture has a separate stencil aspect, and the color or depth buffer otherwise.
    pub async fn retrieve(&self, ctx: &TestingContext, aspect: TextureAspect) -> ReadbackData {
        let (buffer, layouts) = match (aspect, &self.buffer_stencil) {
            (TextureAspect::StencilOnly, Some(buffer)) => (buffer, &self.stencil_layouts),
            _ => (&self.buffer, &self.layouts),
        };
        let data = self.retrieve_buffer(ctx, buffer, layouts).await;
        buffer.unmap();
        ReadbackData {
            data,
            layouts: layouts.clone(),
        }
    }

    async fn is_zero(&self, ctx: &TestingContext, buffer: &Buffer, layouts: &[MipLayout]) -> bool {
        let is_zero = self
            .retrieve_buffer(ctx, buffer, layouts)
            .await
            .iter()
            .all(|b| *b == 0);
        buffer.unmap();
        is_zero
    }

    pub async fn are_zero(&self, ctx: &TestingContext) -> bool {
        let buffer_zero = self.is_zero(ctx, &self.buffer, &self.layouts).await;
        let mut stencil_buffer_zero = true;
        if let Some(buffer) = &self.buffer_stencil {
            stencil_buffer_zero = self.is_zero(ctx, buffer, &self.stencil_layouts).await;
        };
        buffer_zero && stencil_buffer_zero
    }

    /// Assert that the color or depth aspect starts with `expected_data`.
    ///
    /// Only a prefix of the readback is checked, so for a texture with more than one mip
    /// level this covers mip level 0 alone unless `expected_data` holds every mip level.
    pub async fn assert_buffer_contents(&self, ctx: &TestingContext, expected_data: &[u8]) {
        self.assert_buffer_contents_imprecise(ctx, expected_data, 0)
            .await;
    }

    pub async fn assert_buffer_contents_imprecise(
        &self,
        ctx: &TestingContext,
        expected_data: &[u8],
        max_diff: u8,
    ) {
        let result_buffer = self.retrieve_buffer(ctx, &self.buffer, &self.layouts).await;
        assert!(
            result_buffer.len() >= expected_data.len(),
            "Result buffer ({}) smaller than expected buffer ({})",
            result_buffer.len(),
            expected_data.len()
        );
        let result_buffer = &result_buffer[..expected_data.len()];
        assert!(result_buffer
            .iter()
            .zip(expected_data)
            .all(|(a, b)| a.abs_diff(*b) <= max_diff));
        self.buffer.unmap();
    }
}

/// Tightly packed contents of one aspect of a texture, as read back by
/// [`ReadbackBuffers::retrieve`].
pub struct ReadbackData {
    data: Vec<u8>,
    layouts: Vec<MipLayout>,
}

impl ReadbackData {
    /// Every mip level, tightly packed, mip level 0 first.
    pub fn all(&self) -> &[u8] {
        &self.data
    }

    /// Contents of one array layer (for a `D2` texture) or depth slice (for `D3`) of one
    /// mip level, tightly packed.
    pub fn subresource(&self, mip_level: u32, index: u32) -> &[u8] {
        let layouts = &self.layouts[..mip_level as usize + 1];
        let (layout, preceding) = layouts.split_last().unwrap();
        assert!(
            index < layout.size.depth_or_array_layers,
            "mip level {mip_level} has {} array layers or depth slices, asked for {index}",
            layout.size.depth_or_array_layers,
        );
        let size = layout.unpadded_subresource_size();
        let start = preceding
            .iter()
            .map(MipLayout::unpadded_size)
            .sum::<usize>()
            + index as usize * size;
        &self.data[start..start + size]
    }
}
