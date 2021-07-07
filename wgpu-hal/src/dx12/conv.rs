use winapi::{
    shared::{dxgi1_2, dxgiformat},
    um::d3d12,
};

pub(super) fn map_texture_format(format: wgt::TextureFormat) -> dxgiformat::DXGI_FORMAT {
    use wgt::TextureFormat as Tf;
    use winapi::shared::dxgiformat::*;

    match format {
        Tf::R8Unorm => DXGI_FORMAT_R8_UNORM,
        Tf::R8Snorm => DXGI_FORMAT_R8_SNORM,
        Tf::R8Uint => DXGI_FORMAT_R8_UINT,
        Tf::R8Sint => DXGI_FORMAT_R8_SINT,
        Tf::R16Uint => DXGI_FORMAT_R16_UINT,
        Tf::R16Sint => DXGI_FORMAT_R16_SINT,
        Tf::R16Float => DXGI_FORMAT_R16_FLOAT,
        Tf::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        Tf::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        Tf::Rg8Uint => DXGI_FORMAT_R8G8_UINT,
        Tf::Rg8Sint => DXGI_FORMAT_R8G8_SINT,
        Tf::R32Uint => DXGI_FORMAT_R32_UINT,
        Tf::R32Sint => DXGI_FORMAT_R32_SINT,
        Tf::R32Float => DXGI_FORMAT_R32_FLOAT,
        Tf::Rg16Uint => DXGI_FORMAT_R16G16_UINT,
        Tf::Rg16Sint => DXGI_FORMAT_R16G16_SINT,
        Tf::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        Tf::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        Tf::Rgba8UnormSrgb => DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        Tf::Bgra8UnormSrgb => DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        Tf::Rgba8Snorm => DXGI_FORMAT_R8G8B8A8_SNORM,
        Tf::Bgra8Unorm => DXGI_FORMAT_B8G8R8A8_UNORM,
        Tf::Rgba8Uint => DXGI_FORMAT_R8G8B8A8_UINT,
        Tf::Rgba8Sint => DXGI_FORMAT_R8G8B8A8_SINT,
        Tf::Rgb10a2Unorm => DXGI_FORMAT_R10G10B10A2_UNORM,
        Tf::Rg11b10Float => DXGI_FORMAT_R11G11B10_FLOAT,
        Tf::Rg32Uint => DXGI_FORMAT_R32G32_UINT,
        Tf::Rg32Sint => DXGI_FORMAT_R32G32_SINT,
        Tf::Rg32Float => DXGI_FORMAT_R32G32_FLOAT,
        Tf::Rgba16Uint => DXGI_FORMAT_R16G16B16A16_UINT,
        Tf::Rgba16Sint => DXGI_FORMAT_R16G16B16A16_SINT,
        Tf::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Tf::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        Tf::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        Tf::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        Tf::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        Tf::Depth24Plus => DXGI_FORMAT_D24_UNORM_S8_UINT,
        Tf::Depth24PlusStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        Tf::Bc1RgbaUnorm => DXGI_FORMAT_BC1_UNORM,
        Tf::Bc1RgbaUnormSrgb => DXGI_FORMAT_BC1_UNORM_SRGB,
        Tf::Bc2RgbaUnorm => DXGI_FORMAT_BC2_UNORM,
        Tf::Bc2RgbaUnormSrgb => DXGI_FORMAT_BC2_UNORM_SRGB,
        Tf::Bc3RgbaUnorm => DXGI_FORMAT_BC3_UNORM,
        Tf::Bc3RgbaUnormSrgb => DXGI_FORMAT_BC3_UNORM_SRGB,
        Tf::Bc4RUnorm => DXGI_FORMAT_BC4_UNORM,
        Tf::Bc4RSnorm => DXGI_FORMAT_BC4_SNORM,
        Tf::Bc5RgUnorm => DXGI_FORMAT_BC5_UNORM,
        Tf::Bc5RgSnorm => DXGI_FORMAT_BC5_SNORM,
        Tf::Bc6hRgbUfloat => DXGI_FORMAT_BC6H_UF16,
        Tf::Bc6hRgbSfloat => DXGI_FORMAT_BC6H_SF16,
        Tf::Bc7RgbaUnorm => DXGI_FORMAT_BC7_UNORM,
        Tf::Bc7RgbaUnormSrgb => DXGI_FORMAT_BC7_UNORM_SRGB,
        Tf::Etc2RgbUnorm
        | Tf::Etc2RgbUnormSrgb
        | Tf::Etc2RgbA1Unorm
        | Tf::Etc2RgbA1UnormSrgb
        | Tf::EacRUnorm
        | Tf::EacRSnorm
        | Tf::EacRgUnorm
        | Tf::EacRgSnorm
        | Tf::Astc4x4RgbaUnorm
        | Tf::Astc4x4RgbaUnormSrgb
        | Tf::Astc5x4RgbaUnorm
        | Tf::Astc5x4RgbaUnormSrgb
        | Tf::Astc5x5RgbaUnorm
        | Tf::Astc5x5RgbaUnormSrgb
        | Tf::Astc6x5RgbaUnorm
        | Tf::Astc6x5RgbaUnormSrgb
        | Tf::Astc6x6RgbaUnorm
        | Tf::Astc6x6RgbaUnormSrgb
        | Tf::Astc8x5RgbaUnorm
        | Tf::Astc8x5RgbaUnormSrgb
        | Tf::Astc8x6RgbaUnorm
        | Tf::Astc8x6RgbaUnormSrgb
        | Tf::Astc10x5RgbaUnorm
        | Tf::Astc10x5RgbaUnormSrgb
        | Tf::Astc10x6RgbaUnorm
        | Tf::Astc10x6RgbaUnormSrgb
        | Tf::Astc8x8RgbaUnorm
        | Tf::Astc8x8RgbaUnormSrgb
        | Tf::Astc10x8RgbaUnorm
        | Tf::Astc10x8RgbaUnormSrgb
        | Tf::Astc10x10RgbaUnorm
        | Tf::Astc10x10RgbaUnormSrgb
        | Tf::Astc12x10RgbaUnorm
        | Tf::Astc12x10RgbaUnormSrgb
        | Tf::Astc12x12RgbaUnorm
        | Tf::Astc12x12RgbaUnormSrgb => unreachable!(),
    }
}

pub fn map_texture_format_nosrgb(format: wgt::TextureFormat) -> dxgiformat::DXGI_FORMAT {
    // NOTE: DXGI doesn't allow sRGB format on the swapchain, but
    //       creating RTV of swapchain buffers with sRGB works
    match format {
        wgt::TextureFormat::Bgra8UnormSrgb => dxgiformat::DXGI_FORMAT_B8G8R8A8_UNORM,
        wgt::TextureFormat::Rgba8UnormSrgb => dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
        _ => map_texture_format(format),
    }
}

pub fn map_acomposite_alpha_mode(mode: crate::CompositeAlphaMode) -> dxgi1_2::DXGI_ALPHA_MODE {
    use crate::CompositeAlphaMode as Cam;
    match mode {
        Cam::Opaque => dxgi1_2::DXGI_ALPHA_MODE_IGNORE,
        Cam::PreMultiplied => dxgi1_2::DXGI_ALPHA_MODE_PREMULTIPLIED,
        Cam::PostMultiplied => dxgi1_2::DXGI_ALPHA_MODE_STRAIGHT,
    }
}

pub fn map_buffer_usage_to_resource_flags(usage: crate::BufferUses) -> d3d12::D3D12_RESOURCE_FLAGS {
    let mut flags = 0;
    if usage.contains(crate::BufferUses::STORAGE_STORE) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if !usage.intersects(crate::BufferUses::UNIFORM | crate::BufferUses::STORAGE_LOAD) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }
    flags
}

pub fn map_texture_dimension(dim: wgt::TextureDimension) -> d3d12::D3D12_RESOURCE_DIMENSION {
    match dim {
        wgt::TextureDimension::D1 => d3d12::D3D12_RESOURCE_DIMENSION_TEXTURE1D,
        wgt::TextureDimension::D2 => d3d12::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        wgt::TextureDimension::D3 => d3d12::D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    }
}

pub fn map_texture_usage_to_resource_flags(
    usage: crate::TextureUses,
) -> d3d12::D3D12_RESOURCE_FLAGS {
    let mut flags = 0;

    if usage.contains(crate::TextureUses::COLOR_TARGET) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if usage.intersects(
        crate::TextureUses::DEPTH_STENCIL_READ | crate::TextureUses::DEPTH_STENCIL_WRITE,
    ) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    if usage.contains(crate::TextureUses::STORAGE_STORE) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    if !usage.intersects(crate::TextureUses::SAMPLED | crate::TextureUses::STORAGE_LOAD) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }

    flags
}

pub fn map_address_mode(mode: wgt::AddressMode) -> d3d12::D3D12_TEXTURE_ADDRESS_MODE {
    use wgt::AddressMode as Am;
    match mode {
        Am::Repeat => d3d12::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        Am::MirrorRepeat => d3d12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        Am::ClampToEdge => d3d12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        Am::ClampToBorder => d3d12::D3D12_TEXTURE_ADDRESS_MODE_BORDER,
        //Am::MirrorClamp => d3d12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
    }
}

pub fn map_filter_mode(mode: wgt::FilterMode) -> d3d12::D3D12_FILTER_TYPE {
    match mode {
        wgt::FilterMode::Nearest => d3d12::D3D12_FILTER_TYPE_POINT,
        wgt::FilterMode::Linear => d3d12::D3D12_FILTER_TYPE_LINEAR,
    }
}

pub fn map_comparison(func: wgt::CompareFunction) -> d3d12::D3D12_COMPARISON_FUNC {
    use wgt::CompareFunction as Cf;
    match func {
        Cf::Never => d3d12::D3D12_COMPARISON_FUNC_NEVER,
        Cf::Less => d3d12::D3D12_COMPARISON_FUNC_LESS,
        Cf::LessEqual => d3d12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Cf::Equal => d3d12::D3D12_COMPARISON_FUNC_EQUAL,
        Cf::GreaterEqual => d3d12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Cf::Greater => d3d12::D3D12_COMPARISON_FUNC_GREATER,
        Cf::NotEqual => d3d12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Cf::Always => d3d12::D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

pub fn map_border_color(border_color: Option<wgt::SamplerBorderColor>) -> [f32; 4] {
    use wgt::SamplerBorderColor as Sbc;
    match border_color {
        Some(Sbc::TransparentBlack) | None => [0.0; 4],
        Some(Sbc::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
        Some(Sbc::OpaqueWhite) => [1.0; 4],
    }
}

pub fn map_visibility(visibility: wgt::ShaderStages) -> native::ShaderVisibility {
    match visibility {
        wgt::ShaderStages::VERTEX => native::ShaderVisibility::VS,
        wgt::ShaderStages::FRAGMENT => native::ShaderVisibility::PS,
        _ => native::ShaderVisibility::All,
    }
}

pub fn map_binding_type(ty: &wgt::BindingType) -> native::DescriptorRangeType {
    use wgt::BindingType as Bt;
    match *ty {
        Bt::Sampler { .. } => native::DescriptorRangeType::Sampler,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Uniform,
            ..
        } => native::DescriptorRangeType::CBV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: true },
            ..
        }
        | Bt::Texture { .. }
        | Bt::StorageTexture {
            access: wgt::StorageTextureAccess::ReadOnly,
            ..
        } => native::DescriptorRangeType::SRV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: false },
            ..
        }
        | Bt::StorageTexture { .. } => native::DescriptorRangeType::UAV,
    }
}
