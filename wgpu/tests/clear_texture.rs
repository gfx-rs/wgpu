use crate::common::{image::ReadbackBuffers, initialize_test, TestParameters, TestingContext};
use wasm_bindgen_test::*;

static TEXTURE_FORMATS_UNCOMPRESSED_GLES_COMPAT: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R8Snorm,
    wgpu::TextureFormat::R8Uint,
    wgpu::TextureFormat::R8Sint,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::R16Sint,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::R32Uint,
    wgpu::TextureFormat::R32Sint,
    wgpu::TextureFormat::R32Float,
    wgpu::TextureFormat::Rg16Uint,
    wgpu::TextureFormat::Rg16Sint,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Snorm,
    wgpu::TextureFormat::Rgba8Uint,
    wgpu::TextureFormat::Rgba8Sint,
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rg11b10Float,
    wgpu::TextureFormat::Rg32Uint,
    wgpu::TextureFormat::Rg32Sint,
    wgpu::TextureFormat::Rg32Float,
    wgpu::TextureFormat::Rgba16Uint,
    wgpu::TextureFormat::Rgba16Sint,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Uint,
    wgpu::TextureFormat::Rgba32Sint,
    wgpu::TextureFormat::Rgba32Float,
];

static TEXTURE_FORMATS_UNCOMPRESSED: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Rg8Unorm,
    wgpu::TextureFormat::Rg8Snorm,
    wgpu::TextureFormat::Rg8Uint,
    wgpu::TextureFormat::Rg8Sint,
    wgpu::TextureFormat::Rgb9e5Ufloat,
];

static TEXTURE_FORMATS_DEPTH: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Stencil8,
    wgpu::TextureFormat::Depth16Unorm,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth24PlusStencil8,
    wgpu::TextureFormat::Depth32Float,
];

// needs TEXTURE_COMPRESSION_BC
static TEXTURE_FORMATS_BC: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Bc1RgbaUnorm,
    wgpu::TextureFormat::Bc1RgbaUnormSrgb,
    wgpu::TextureFormat::Bc2RgbaUnorm,
    wgpu::TextureFormat::Bc2RgbaUnormSrgb,
    wgpu::TextureFormat::Bc3RgbaUnorm,
    wgpu::TextureFormat::Bc3RgbaUnormSrgb,
    wgpu::TextureFormat::Bc4RUnorm,
    wgpu::TextureFormat::Bc4RSnorm,
    wgpu::TextureFormat::Bc5RgUnorm,
    wgpu::TextureFormat::Bc5RgSnorm,
    wgpu::TextureFormat::Bc6hRgbUfloat,
    wgpu::TextureFormat::Bc6hRgbFloat,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
];

// needs TEXTURE_COMPRESSION_ETC2
static TEXTURE_FORMATS_ETC2: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Etc2Rgb8Unorm,
    wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
    wgpu::TextureFormat::Etc2Rgb8A1Unorm,
    wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
    wgpu::TextureFormat::Etc2Rgba8Unorm,
    wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
    wgpu::TextureFormat::EacR11Unorm,
    wgpu::TextureFormat::EacR11Snorm,
    wgpu::TextureFormat::EacRg11Unorm,
    wgpu::TextureFormat::EacRg11Snorm,
];

// needs TEXTURE_COMPRESSION_ASTC
use wgpu::{AstcBlock, AstcChannel};
static TEXTURE_FORMATS_ASTC: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B4x4,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B5x4,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B5x5,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B6x5,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B6x6,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x5,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x6,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x8,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x5,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x6,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x8,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x10,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B12x10,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B12x12,
        channel: AstcChannel::Unorm,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B4x4,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B5x4,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B5x5,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B6x5,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B6x6,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x5,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x6,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B8x8,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x5,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x6,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x8,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B10x10,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B12x10,
        channel: AstcChannel::UnormSrgb,
    },
    wgpu::TextureFormat::Astc {
        block: AstcBlock::B12x12,
        channel: AstcChannel::UnormSrgb,
    },
];

fn single_texture_clear_test(
    ctx: &TestingContext,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    dimension: wgpu::TextureDimension,
) {
    log::info!(
        "clearing texture with {:?}, dimension {:?}, size {:?}",
        format,
        dimension,
        size
    );

    let extra_usages = match format {
        wgpu::TextureFormat::Depth24Plus | wgpu::TextureFormat::Depth24PlusStencil8 => {
            wgpu::TextureUsages::TEXTURE_BINDING
        }
        _ => wgpu::TextureUsages::empty(),
    };

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("texture {format:?}")),
        size,
        mip_level_count: if dimension == wgpu::TextureDimension::D1 {
            1
        } else {
            // arbitrary value between 2 and max
            3
        },
        sample_count: 1, // multisampling is not supported for clear
        dimension,
        format,
        usage: wgpu::TextureUsages::COPY_SRC | extra_usages,
        view_formats: &[],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.clear_texture(
        &texture,
        &wgpu::ImageSubresourceRange {
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        },
    );

    let readback_buffers = ReadbackBuffers::new(&ctx.device, &texture);

    readback_buffers.copy_from(&ctx.device, &mut encoder, &texture);

    ctx.queue.submit([encoder.finish()]);

    assert!(
        readback_buffers.are_zero(&ctx.device),
        "texture with format {format:?} was not fully cleared"
    );
}

fn clear_texture_tests(ctx: &TestingContext, formats: &[wgpu::TextureFormat]) {
    for &format in formats {
        let (block_width, block_height) = format.block_dimensions();
        let rounded_width = block_width * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let rounded_height = block_height * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let is_compressed_or_depth_stencil_format =
            format.is_compressed() || format.is_depth_stencil_format();
        let supports_1d = !is_compressed_or_depth_stencil_format;
        let supports_3d = !is_compressed_or_depth_stencil_format;

        // 1D texture
        if supports_1d {
            single_texture_clear_test(
                ctx,
                format,
                wgpu::Extent3d {
                    width: rounded_width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D1,
            );
        }
        // 2D texture
        single_texture_clear_test(
            ctx,
            format,
            wgpu::Extent3d {
                width: rounded_width,
                height: rounded_height,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
        );
        // 2D array texture
        single_texture_clear_test(
            ctx,
            format,
            wgpu::Extent3d {
                width: rounded_width,
                height: rounded_height,
                depth_or_array_layers: 4,
            },
            wgpu::TextureDimension::D2,
        );
        if supports_3d {
            // volume texture
            single_texture_clear_test(
                ctx,
                format,
                wgpu::Extent3d {
                    width: rounded_width,
                    height: rounded_height,
                    depth_or_array_layers: 16,
                },
                wgpu::TextureDimension::D3,
            );
        }
    }
}

#[test]
#[wasm_bindgen_test]
fn clear_texture_uncompressed_gles_compat() {
    initialize_test(
        TestParameters::default()
            .webgl2_failure()
            .features(wgpu::Features::CLEAR_TEXTURE),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_UNCOMPRESSED_GLES_COMPAT);
        },
    )
}

#[test]
#[wasm_bindgen_test]
fn clear_texture_uncompressed() {
    initialize_test(
        TestParameters::default()
            .webgl2_failure()
            .backend_failure(wgpu::Backends::GL)
            .features(wgpu::Features::CLEAR_TEXTURE),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_UNCOMPRESSED);
        },
    )
}

#[test]
#[wasm_bindgen_test]
fn clear_texture_depth() {
    initialize_test(
        TestParameters::default()
            .webgl2_failure()
            .downlevel_flags(
                wgpu::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES
                    | wgpu::DownlevelFlags::COMPUTE_SHADERS,
            )
            .limits(wgpu::Limits::downlevel_defaults())
            .features(wgpu::Features::CLEAR_TEXTURE),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_DEPTH);
        },
    )
}

#[test]
#[wasm_bindgen_test]
fn clear_texture_d32_s8() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::DEPTH32FLOAT_STENCIL8),
        |ctx| {
            clear_texture_tests(&ctx, &[wgpu::TextureFormat::Depth32FloatStencil8]);
        },
    )
}

#[test]
fn clear_texture_bc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_BC)
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false) // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .backend_failure(wgpu::Backends::GL), // compressed texture copy to buffer not yet implemented
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_BC);
        },
    )
}

#[test]
fn clear_texture_astc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ASTC)
            .limits(wgpu::Limits {
                max_texture_dimension_2d: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT * 12,
                ..wgpu::Limits::downlevel_defaults()
            })
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false) // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .backend_failure(wgpu::Backends::GL), // compressed texture copy to buffer not yet implemented
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ASTC);
        },
    )
}

#[test]
fn clear_texture_etc2() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ETC2)
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false) // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
            .backend_failure(wgpu::Backends::GL), // compressed texture copy to buffer not yet implemented
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ETC2);
        },
    )
}
