use std::iter;
use winapi::{
    shared::{dxgi1_2, dxgiformat},
    um::{d3d12, d3dcommon},
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
        Tf::R16Unorm => DXGI_FORMAT_R16_UNORM,
        Tf::R16Snorm => DXGI_FORMAT_R16_SNORM,
        Tf::R16Float => DXGI_FORMAT_R16_FLOAT,
        Tf::Rg8Unorm => DXGI_FORMAT_R8G8_UNORM,
        Tf::Rg8Snorm => DXGI_FORMAT_R8G8_SNORM,
        Tf::Rg8Uint => DXGI_FORMAT_R8G8_UINT,
        Tf::Rg8Sint => DXGI_FORMAT_R8G8_SINT,
        Tf::Rg16Unorm => DXGI_FORMAT_R16G16_UNORM,
        Tf::Rg16Snorm => DXGI_FORMAT_R16G16_SNORM,
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
        Tf::Rgba16Unorm => DXGI_FORMAT_R16G16B16A16_UNORM,
        Tf::Rgba16Snorm => DXGI_FORMAT_R16G16B16A16_SNORM,
        Tf::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Tf::Rgba32Uint => DXGI_FORMAT_R32G32B32A32_UINT,
        Tf::Rgba32Sint => DXGI_FORMAT_R32G32B32A32_SINT,
        Tf::Rgba32Float => DXGI_FORMAT_R32G32B32A32_FLOAT,
        Tf::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        Tf::Depth24Plus => DXGI_FORMAT_D24_UNORM_S8_UINT,
        Tf::Depth24PlusStencil8 => DXGI_FORMAT_D24_UNORM_S8_UINT,
        Tf::Rgb9e5Ufloat => DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
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
        Tf::Etc2Rgb8Unorm
        | Tf::Etc2Rgb8UnormSrgb
        | Tf::Etc2Rgb8A1Unorm
        | Tf::Etc2Rgb8A1UnormSrgb
        | Tf::Etc2Rgba8Unorm
        | Tf::Etc2Rgba8UnormSrgb
        | Tf::EacR11Unorm
        | Tf::EacR11Snorm
        | Tf::EacRg11Unorm
        | Tf::EacRg11Snorm
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

//Note: DXGI doesn't allow sRGB format on the swapchain,
// but creating RTV of swapchain buffers with sRGB works.
pub fn map_texture_format_nosrgb(format: wgt::TextureFormat) -> dxgiformat::DXGI_FORMAT {
    match format {
        wgt::TextureFormat::Bgra8UnormSrgb => dxgiformat::DXGI_FORMAT_B8G8R8A8_UNORM,
        wgt::TextureFormat::Rgba8UnormSrgb => dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
        _ => map_texture_format(format),
    }
}

//Note: SRV and UAV can't use the depth formats directly
//TODO: stencil views?
pub fn map_texture_format_nodepth(format: wgt::TextureFormat) -> dxgiformat::DXGI_FORMAT {
    match format {
        wgt::TextureFormat::Depth32Float => dxgiformat::DXGI_FORMAT_R32_FLOAT,
        wgt::TextureFormat::Depth24Plus | wgt::TextureFormat::Depth24PlusStencil8 => {
            dxgiformat::DXGI_FORMAT_R24_UNORM_X8_TYPELESS
        }
        _ => {
            assert_eq!(
                crate::FormatAspects::from(format),
                crate::FormatAspects::COLOR
            );
            map_texture_format(format)
        }
    }
}

pub fn map_texture_format_depth_typeless(format: wgt::TextureFormat) -> dxgiformat::DXGI_FORMAT {
    match format {
        wgt::TextureFormat::Depth32Float => dxgiformat::DXGI_FORMAT_R32_TYPELESS,
        wgt::TextureFormat::Depth24Plus | wgt::TextureFormat::Depth24PlusStencil8 => {
            dxgiformat::DXGI_FORMAT_R24G8_TYPELESS
        }
        _ => unreachable!(),
    }
}

pub fn map_index_format(format: wgt::IndexFormat) -> dxgiformat::DXGI_FORMAT {
    match format {
        wgt::IndexFormat::Uint16 => dxgiformat::DXGI_FORMAT_R16_UINT,
        wgt::IndexFormat::Uint32 => dxgiformat::DXGI_FORMAT_R32_UINT,
    }
}

pub fn map_vertex_format(format: wgt::VertexFormat) -> dxgiformat::DXGI_FORMAT {
    use wgt::VertexFormat as Vf;
    use winapi::shared::dxgiformat::*;

    match format {
        Vf::Unorm8x2 => DXGI_FORMAT_R8G8_UNORM,
        Vf::Snorm8x2 => DXGI_FORMAT_R8G8_SNORM,
        Vf::Uint8x2 => DXGI_FORMAT_R8G8_UINT,
        Vf::Sint8x2 => DXGI_FORMAT_R8G8_SINT,
        Vf::Unorm8x4 => DXGI_FORMAT_R8G8B8A8_UNORM,
        Vf::Snorm8x4 => DXGI_FORMAT_R8G8B8A8_SNORM,
        Vf::Uint8x4 => DXGI_FORMAT_R8G8B8A8_UINT,
        Vf::Sint8x4 => DXGI_FORMAT_R8G8B8A8_SINT,
        Vf::Unorm16x2 => DXGI_FORMAT_R16G16_UNORM,
        Vf::Snorm16x2 => DXGI_FORMAT_R16G16_SNORM,
        Vf::Uint16x2 => DXGI_FORMAT_R16G16_UINT,
        Vf::Sint16x2 => DXGI_FORMAT_R16G16_SINT,
        Vf::Float16x2 => DXGI_FORMAT_R16G16_FLOAT,
        Vf::Unorm16x4 => DXGI_FORMAT_R16G16B16A16_UNORM,
        Vf::Snorm16x4 => DXGI_FORMAT_R16G16B16A16_SNORM,
        Vf::Uint16x4 => DXGI_FORMAT_R16G16B16A16_UINT,
        Vf::Sint16x4 => DXGI_FORMAT_R16G16B16A16_SINT,
        Vf::Float16x4 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        Vf::Uint32 => DXGI_FORMAT_R32_UINT,
        Vf::Sint32 => DXGI_FORMAT_R32_SINT,
        Vf::Float32 => DXGI_FORMAT_R32_FLOAT,
        Vf::Uint32x2 => DXGI_FORMAT_R32G32_UINT,
        Vf::Sint32x2 => DXGI_FORMAT_R32G32_SINT,
        Vf::Float32x2 => DXGI_FORMAT_R32G32_FLOAT,
        Vf::Uint32x3 => DXGI_FORMAT_R32G32B32_UINT,
        Vf::Sint32x3 => DXGI_FORMAT_R32G32B32_SINT,
        Vf::Float32x3 => DXGI_FORMAT_R32G32B32_FLOAT,
        Vf::Uint32x4 => DXGI_FORMAT_R32G32B32A32_UINT,
        Vf::Sint32x4 => DXGI_FORMAT_R32G32B32A32_SINT,
        Vf::Float32x4 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        Vf::Float64 | Vf::Float64x2 | Vf::Float64x3 | Vf::Float64x4 => unimplemented!(),
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
    if usage.contains(crate::BufferUses::STORAGE_WRITE) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
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
        if !usage.contains(crate::TextureUses::RESOURCE) {
            flags |= d3d12::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        }
    }
    if usage.contains(crate::TextureUses::STORAGE_WRITE) {
        flags |= d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
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
        | Bt::Texture { .. } => native::DescriptorRangeType::SRV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: false },
            ..
        }
        | Bt::StorageTexture { .. } => native::DescriptorRangeType::UAV,
    }
}

pub fn map_label(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(iter::once(0)).collect()
}

pub fn map_buffer_usage_to_state(usage: crate::BufferUses) -> d3d12::D3D12_RESOURCE_STATES {
    use crate::BufferUses as Bu;
    let mut state = d3d12::D3D12_RESOURCE_STATE_COMMON;

    if usage.intersects(Bu::COPY_SRC) {
        state |= d3d12::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Bu::COPY_DST) {
        state |= d3d12::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Bu::INDEX) {
        state |= d3d12::D3D12_RESOURCE_STATE_INDEX_BUFFER;
    }
    if usage.intersects(Bu::VERTEX | Bu::UNIFORM) {
        state |= d3d12::D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    }
    if usage.intersects(Bu::STORAGE_WRITE) {
        state |= d3d12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else if usage.intersects(Bu::STORAGE_READ) {
        state |= d3d12::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | d3d12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Bu::INDIRECT) {
        state |= d3d12::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }
    state
}

pub fn map_texture_usage_to_state(usage: crate::TextureUses) -> d3d12::D3D12_RESOURCE_STATES {
    use crate::TextureUses as Tu;
    let mut state = d3d12::D3D12_RESOURCE_STATE_COMMON;
    //Note: `RESOLVE_SOURCE` and `RESOLVE_DEST` are not used here
    //Note: `PRESENT` is the same as `COMMON`
    if usage == crate::TextureUses::UNINITIALIZED {
        return state;
    }

    if usage.intersects(Tu::COPY_SRC) {
        state |= d3d12::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Tu::COPY_DST) {
        state |= d3d12::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Tu::RESOURCE) {
        state |= d3d12::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | d3d12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Tu::COLOR_TARGET) {
        state |= d3d12::D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_READ) {
        state |= d3d12::D3D12_RESOURCE_STATE_DEPTH_READ;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_WRITE) {
        state |= d3d12::D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    if usage.intersects(Tu::STORAGE_READ | Tu::STORAGE_WRITE) {
        state |= d3d12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    state
}

pub fn map_topology(
    topology: wgt::PrimitiveTopology,
) -> (
    d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    d3d12::D3D12_PRIMITIVE_TOPOLOGY,
) {
    match topology {
        wgt::PrimitiveTopology::PointList => (
            d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        ),
        wgt::PrimitiveTopology::LineList => (
            d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        ),
        wgt::PrimitiveTopology::LineStrip => (
            d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        ),
        wgt::PrimitiveTopology::TriangleList => (
            d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        ),
        wgt::PrimitiveTopology::TriangleStrip => (
            d3d12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
        ),
    }
}

pub fn map_polygon_mode(mode: wgt::PolygonMode) -> d3d12::D3D12_FILL_MODE {
    match mode {
        wgt::PolygonMode::Point => {
            log::error!("Point rasterization is not supported");
            d3d12::D3D12_FILL_MODE_WIREFRAME
        }
        wgt::PolygonMode::Line => d3d12::D3D12_FILL_MODE_WIREFRAME,
        wgt::PolygonMode::Fill => d3d12::D3D12_FILL_MODE_SOLID,
    }
}

fn map_blend_factor(factor: wgt::BlendFactor, is_alpha: bool) -> d3d12::D3D12_BLEND {
    use wgt::BlendFactor as Bf;
    match factor {
        Bf::Zero => d3d12::D3D12_BLEND_ZERO,
        Bf::One => d3d12::D3D12_BLEND_ONE,
        Bf::Src if is_alpha => d3d12::D3D12_BLEND_SRC_ALPHA,
        Bf::Src => d3d12::D3D12_BLEND_SRC_COLOR,
        Bf::OneMinusSrc if is_alpha => d3d12::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::OneMinusSrc => d3d12::D3D12_BLEND_INV_SRC_COLOR,
        Bf::Dst if is_alpha => d3d12::D3D12_BLEND_DEST_ALPHA,
        Bf::Dst => d3d12::D3D12_BLEND_DEST_COLOR,
        Bf::OneMinusDst if is_alpha => d3d12::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::OneMinusDst => d3d12::D3D12_BLEND_INV_DEST_COLOR,
        Bf::SrcAlpha => d3d12::D3D12_BLEND_SRC_ALPHA,
        Bf::OneMinusSrcAlpha => d3d12::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::DstAlpha => d3d12::D3D12_BLEND_DEST_ALPHA,
        Bf::OneMinusDstAlpha => d3d12::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::Constant => d3d12::D3D12_BLEND_BLEND_FACTOR,
        Bf::OneMinusConstant => d3d12::D3D12_BLEND_INV_BLEND_FACTOR,
        Bf::SrcAlphaSaturated => d3d12::D3D12_BLEND_SRC_ALPHA_SAT,
        //Bf::Src1Color if is_alpha => d3d12::D3D12_BLEND_SRC1_ALPHA,
        //Bf::Src1Color => d3d12::D3D12_BLEND_SRC1_COLOR,
        //Bf::OneMinusSrc1Color if is_alpha => d3d12::D3D12_BLEND_INV_SRC1_ALPHA,
        //Bf::OneMinusSrc1Color => d3d12::D3D12_BLEND_INV_SRC1_COLOR,
        //Bf::Src1Alpha => d3d12::D3D12_BLEND_SRC1_ALPHA,
        //Bf::OneMinusSrc1Alpha => d3d12::D3D12_BLEND_INV_SRC1_ALPHA,
    }
}

fn map_blend_component(
    component: &wgt::BlendComponent,
    is_alpha: bool,
) -> (
    d3d12::D3D12_BLEND_OP,
    d3d12::D3D12_BLEND,
    d3d12::D3D12_BLEND,
) {
    let raw_op = match component.operation {
        wgt::BlendOperation::Add => d3d12::D3D12_BLEND_OP_ADD,
        wgt::BlendOperation::Subtract => d3d12::D3D12_BLEND_OP_SUBTRACT,
        wgt::BlendOperation::ReverseSubtract => d3d12::D3D12_BLEND_OP_REV_SUBTRACT,
        wgt::BlendOperation::Min => d3d12::D3D12_BLEND_OP_MIN,
        wgt::BlendOperation::Max => d3d12::D3D12_BLEND_OP_MAX,
    };
    let raw_src = map_blend_factor(component.src_factor, is_alpha);
    let raw_dst = map_blend_factor(component.dst_factor, is_alpha);
    (raw_op, raw_src, raw_dst)
}

pub fn map_render_targets(
    color_targets: &[wgt::ColorTargetState],
) -> [d3d12::D3D12_RENDER_TARGET_BLEND_DESC; d3d12::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize]
{
    let dummy_target = d3d12::D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: 0,
        LogicOpEnable: 0,
        SrcBlend: d3d12::D3D12_BLEND_ZERO,
        DestBlend: d3d12::D3D12_BLEND_ZERO,
        BlendOp: d3d12::D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: d3d12::D3D12_BLEND_ZERO,
        DestBlendAlpha: d3d12::D3D12_BLEND_ZERO,
        BlendOpAlpha: d3d12::D3D12_BLEND_OP_ADD,
        LogicOp: d3d12::D3D12_LOGIC_OP_CLEAR,
        RenderTargetWriteMask: 0,
    };
    let mut raw_targets = [dummy_target; d3d12::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize];

    for (raw, ct) in raw_targets.iter_mut().zip(color_targets.iter()) {
        raw.RenderTargetWriteMask = ct.write_mask.bits() as u8;
        if let Some(ref blend) = ct.blend {
            let (color_op, color_src, color_dst) = map_blend_component(&blend.color, false);
            let (alpha_op, alpha_src, alpha_dst) = map_blend_component(&blend.alpha, true);
            raw.BlendEnable = 1;
            raw.BlendOp = color_op;
            raw.SrcBlend = color_src;
            raw.DestBlend = color_dst;
            raw.BlendOpAlpha = alpha_op;
            raw.SrcBlendAlpha = alpha_src;
            raw.DestBlendAlpha = alpha_dst;
        }
    }

    raw_targets
}

fn map_stencil_op(op: wgt::StencilOperation) -> d3d12::D3D12_STENCIL_OP {
    use wgt::StencilOperation as So;
    match op {
        So::Keep => d3d12::D3D12_STENCIL_OP_KEEP,
        So::Zero => d3d12::D3D12_STENCIL_OP_ZERO,
        So::Replace => d3d12::D3D12_STENCIL_OP_REPLACE,
        So::IncrementClamp => d3d12::D3D12_STENCIL_OP_INCR_SAT,
        So::IncrementWrap => d3d12::D3D12_STENCIL_OP_INCR,
        So::DecrementClamp => d3d12::D3D12_STENCIL_OP_DECR_SAT,
        So::DecrementWrap => d3d12::D3D12_STENCIL_OP_DECR,
        So::Invert => d3d12::D3D12_STENCIL_OP_INVERT,
    }
}

fn map_stencil_face(face: &wgt::StencilFaceState) -> d3d12::D3D12_DEPTH_STENCILOP_DESC {
    d3d12::D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: map_stencil_op(face.fail_op),
        StencilDepthFailOp: map_stencil_op(face.depth_fail_op),
        StencilPassOp: map_stencil_op(face.pass_op),
        StencilFunc: map_comparison(face.compare),
    }
}

pub fn map_depth_stencil(ds: &wgt::DepthStencilState) -> d3d12::D3D12_DEPTH_STENCIL_DESC {
    d3d12::D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: if ds.is_depth_enabled() { 1 } else { 0 },
        DepthWriteMask: if ds.depth_write_enabled {
            d3d12::D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            d3d12::D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: map_comparison(ds.depth_compare),
        StencilEnable: if ds.stencil.is_enabled() { 1 } else { 0 },
        StencilReadMask: ds.stencil.read_mask as u8,
        StencilWriteMask: ds.stencil.write_mask as u8,
        FrontFace: map_stencil_face(&ds.stencil.front),
        BackFace: map_stencil_face(&ds.stencil.back),
    }
}
