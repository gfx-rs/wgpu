use crate::common::{initialize_test, TestParameters, TestingContext};
use wasm_bindgen_test::*;
use wgpu::util::align_to;

static TEXTURE_FORMATS_UNCOMPRESSED: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R8Snorm,
    wgpu::TextureFormat::R8Uint,
    wgpu::TextureFormat::R8Sint,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::R16Sint,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::Rg8Unorm,
    wgpu::TextureFormat::Rg8Snorm,
    wgpu::TextureFormat::Rg8Uint,
    wgpu::TextureFormat::Rg8Sint,
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
    wgpu::TextureFormat::Rgb9e5Ufloat,
];

static TEXTURE_FORMATS_DEPTH: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Stencil8,
    wgpu::TextureFormat::Depth16Unorm,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth24PlusStencil8,
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
    wgpu::TextureFormat::Bc6hRgbSfloat,
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

// needs TEXTURE_COMPRESSION_ASTC_LDR
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
        // Forces internally the required usages to be able to clear it.
        // This is not visible on the API level.
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
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
    ctx.queue.submit([encoder.finish()]);

    // TODO: Read back and check zeroness?
}

fn clear_texture_tests(
    ctx: &TestingContext,
    formats: &[wgpu::TextureFormat],
    supports_1d: bool,
    supports_3d: bool,
) {
    for &format in formats {
        let desc = format.describe();
        let rounded_width = align_to(64, desc.block_dimensions.0 as u32);
        let rounded_height = align_to(64, desc.block_dimensions.1 as u32);

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
fn clear_texture_2d_uncompressed() {
    initialize_test(
        TestParameters::default()
            .webgl2_failure()
            .features(wgpu::Features::CLEAR_TEXTURE),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_UNCOMPRESSED, true, true);
            clear_texture_tests(&ctx, TEXTURE_FORMATS_DEPTH, false, false);
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
            clear_texture_tests(
                &ctx,
                &[wgpu::TextureFormat::Depth32FloatStencil8],
                false,
                false,
            );
        },
    )
}

#[test]
fn clear_texture_2d_bc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_BC)
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false), // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_BC, false, false);
        },
    )
}

#[test]
fn clear_texture_2d_astc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR)
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false), // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ASTC, false, false);
        },
    )
}

#[test]
fn clear_texture_2d_etc2() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ETC2)
            .specific_failure(Some(wgpu::Backends::GL), None, Some("ANGLE"), false), // https://bugs.chromium.org/p/angleproject/issues/detail?id=7056
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ETC2, false, false);
        },
    )
}
