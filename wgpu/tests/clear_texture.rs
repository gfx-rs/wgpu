use crate::common::{initialize_test, TestParameters, TestingContext};

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
    wgpu::TextureFormat::Depth32Float,
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
static TEXTURE_FORMATS_ASTC: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Astc4x4RgbaUnorm,
    wgpu::TextureFormat::Astc4x4RgbaUnormSrgb,
    wgpu::TextureFormat::Astc5x4RgbaUnorm,
    wgpu::TextureFormat::Astc5x4RgbaUnormSrgb,
    wgpu::TextureFormat::Astc5x5RgbaUnorm,
    wgpu::TextureFormat::Astc5x5RgbaUnormSrgb,
    wgpu::TextureFormat::Astc6x5RgbaUnorm,
    wgpu::TextureFormat::Astc6x5RgbaUnormSrgb,
    wgpu::TextureFormat::Astc6x6RgbaUnorm,
    wgpu::TextureFormat::Astc6x6RgbaUnormSrgb,
    wgpu::TextureFormat::Astc8x5RgbaUnorm,
    wgpu::TextureFormat::Astc8x5RgbaUnormSrgb,
    wgpu::TextureFormat::Astc8x6RgbaUnorm,
    wgpu::TextureFormat::Astc8x6RgbaUnormSrgb,
    wgpu::TextureFormat::Astc10x5RgbaUnorm,
    wgpu::TextureFormat::Astc10x5RgbaUnormSrgb,
    wgpu::TextureFormat::Astc10x6RgbaUnorm,
    wgpu::TextureFormat::Astc10x6RgbaUnormSrgb,
    wgpu::TextureFormat::Astc8x8RgbaUnorm,
    wgpu::TextureFormat::Astc8x8RgbaUnormSrgb,
    wgpu::TextureFormat::Astc10x8RgbaUnorm,
    wgpu::TextureFormat::Astc10x8RgbaUnormSrgb,
    wgpu::TextureFormat::Astc10x10RgbaUnorm,
    wgpu::TextureFormat::Astc10x10RgbaUnormSrgb,
    wgpu::TextureFormat::Astc12x10RgbaUnorm,
    wgpu::TextureFormat::Astc12x10RgbaUnormSrgb,
    wgpu::TextureFormat::Astc12x12RgbaUnorm,
    wgpu::TextureFormat::Astc12x12RgbaUnormSrgb,
];

fn single_texture_clear_test(
    ctx: &TestingContext,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    dimension: wgpu::TextureDimension,
) {
    println!(
        "clearing texture with {:?}, dimension {:?}, size {:?}",
        format, dimension, size
    );

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("texture {:?}", format)),
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

fn clear_texture_tests(ctx: &TestingContext, formats: &[wgpu::TextureFormat], supports_1d: bool) {
    for &format in formats {
        // 1D texture
        if supports_1d {
            single_texture_clear_test(
                ctx,
                format,
                wgpu::Extent3d {
                    width: 64,
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
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
        );
        // 2D array texture
        single_texture_clear_test(
            ctx,
            format,
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 4,
            },
            wgpu::TextureDimension::D2,
        );
        // volume texture
        if format.describe().sample_type != wgt::TextureSampleType::Depth {
            single_texture_clear_test(
                ctx,
                format,
                wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 16,
                },
                wgpu::TextureDimension::D3,
            );
        }
    }
}

#[test]
fn clear_texture_2d_uncompressed() {
    initialize_test(
        TestParameters::default().features(wgpu::Features::CLEAR_TEXTURE),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_UNCOMPRESSED, true);
            clear_texture_tests(&ctx, TEXTURE_FORMATS_DEPTH, false);
        },
    )
}

#[test]
fn clear_texture_2d_bc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_BC),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_BC, false);
        },
    )
}

#[test]
fn clear_texture_2d_astc() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ASTC, false);
        },
    )
}

#[test]
fn clear_texture_2d_etc2() {
    initialize_test(
        TestParameters::default()
            .features(wgpu::Features::CLEAR_TEXTURE | wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        |ctx| {
            clear_texture_tests(&ctx, TEXTURE_FORMATS_ETC2, false);
        },
    )
}
