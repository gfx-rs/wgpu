// Lets keep these on one line
#[rustfmt::skip]
pub const TEXTURE_FORMAT_LIST: [wgpu::TextureFormat; 117] = [
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R8Snorm,
    wgpu::TextureFormat::R8Uint,
    wgpu::TextureFormat::R8Sint,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::R16Sint,
    wgpu::TextureFormat::R16Unorm,
    wgpu::TextureFormat::R16Snorm,
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
    wgpu::TextureFormat::Rg16Unorm,
    wgpu::TextureFormat::Rg16Snorm,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Snorm,
    wgpu::TextureFormat::Rgba8Uint,
    wgpu::TextureFormat::Rgba8Sint,
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgb9e5Ufloat,
    wgpu::TextureFormat::Rgb10a2Uint,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rg11b10Ufloat,
    wgpu::TextureFormat::R64Uint,
    wgpu::TextureFormat::Rg32Uint,
    wgpu::TextureFormat::Rg32Sint,
    wgpu::TextureFormat::Rg32Float,
    wgpu::TextureFormat::Rgba16Uint,
    wgpu::TextureFormat::Rgba16Sint,
    wgpu::TextureFormat::Rgba16Unorm,
    wgpu::TextureFormat::Rgba16Snorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Uint,
    wgpu::TextureFormat::Rgba32Sint,
    wgpu::TextureFormat::Rgba32Float,
    wgpu::TextureFormat::Stencil8,
    wgpu::TextureFormat::Depth16Unorm,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth24PlusStencil8,
    wgpu::TextureFormat::Depth32Float,
    wgpu::TextureFormat::Depth32FloatStencil8,
    wgpu::TextureFormat::NV12,
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
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Hdr },
];

#[test]
fn test_uniqueness_in_texture_format_list() {
    use std::collections::HashSet;

    let uniq: HashSet<wgpu::TextureFormat> = TEXTURE_FORMAT_LIST.into_iter().collect();
    let mut duplicated = TEXTURE_FORMAT_LIST.to_vec();
    uniq.iter().for_each(|u| {
        let first_occurrence = duplicated.iter().position(|el| u == el).unwrap();
        duplicated.remove(first_occurrence);
    });
    assert_eq!(duplicated, vec![]);
}

pub fn max_texture_format_string_size() -> usize {
    TEXTURE_FORMAT_LIST
        .into_iter()
        .map(|f| texture_format_name(f).len())
        .max()
        .unwrap()
}

pub fn texture_format_name(format: wgpu::TextureFormat) -> String {
    match format {
        wgpu::TextureFormat::Astc { block, channel } => {
            format!("Astc{block:?}{channel:?}:")
        }
        _ => {
            format!("{format:?}:")
        }
    }
}
