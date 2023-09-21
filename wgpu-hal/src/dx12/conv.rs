use std::iter;
use winapi::{
    shared::minwindef::BOOL,
    um::{d3d12 as d3d12_ty, d3dcommon},
};

pub fn map_buffer_usage_to_resource_flags(
    usage: crate::BufferUses,
) -> d3d12_ty::D3D12_RESOURCE_FLAGS {
    let mut flags = 0;
    if usage.contains(crate::BufferUses::STORAGE_READ_WRITE) {
        flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    flags
}

pub fn map_texture_dimension(dim: wgt::TextureDimension) -> d3d12_ty::D3D12_RESOURCE_DIMENSION {
    match dim {
        wgt::TextureDimension::D1 => d3d12_ty::D3D12_RESOURCE_DIMENSION_TEXTURE1D,
        wgt::TextureDimension::D2 => d3d12_ty::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        wgt::TextureDimension::D3 => d3d12_ty::D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    }
}

pub fn map_texture_usage_to_resource_flags(
    usage: crate::TextureUses,
) -> d3d12_ty::D3D12_RESOURCE_FLAGS {
    let mut flags = 0;

    if usage.contains(crate::TextureUses::COLOR_TARGET) {
        flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if usage.intersects(
        crate::TextureUses::DEPTH_STENCIL_READ | crate::TextureUses::DEPTH_STENCIL_WRITE,
    ) {
        flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if !usage.contains(crate::TextureUses::RESOURCE) {
            flags |= d3d12_ty::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        }
    }
    if usage.contains(crate::TextureUses::STORAGE_READ_WRITE) {
        flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    flags
}

pub fn map_address_mode(mode: wgt::AddressMode) -> d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE {
    use wgt::AddressMode as Am;
    match mode {
        Am::Repeat => d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        Am::MirrorRepeat => d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        Am::ClampToEdge => d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        Am::ClampToBorder => d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE_BORDER,
        //Am::MirrorClamp => d3d12_ty::D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
    }
}

pub fn map_filter_mode(mode: wgt::FilterMode) -> d3d12_ty::D3D12_FILTER_TYPE {
    match mode {
        wgt::FilterMode::Nearest => d3d12_ty::D3D12_FILTER_TYPE_POINT,
        wgt::FilterMode::Linear => d3d12_ty::D3D12_FILTER_TYPE_LINEAR,
    }
}

pub fn map_comparison(func: wgt::CompareFunction) -> d3d12_ty::D3D12_COMPARISON_FUNC {
    use wgt::CompareFunction as Cf;
    match func {
        Cf::Never => d3d12_ty::D3D12_COMPARISON_FUNC_NEVER,
        Cf::Less => d3d12_ty::D3D12_COMPARISON_FUNC_LESS,
        Cf::LessEqual => d3d12_ty::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Cf::Equal => d3d12_ty::D3D12_COMPARISON_FUNC_EQUAL,
        Cf::GreaterEqual => d3d12_ty::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Cf::Greater => d3d12_ty::D3D12_COMPARISON_FUNC_GREATER,
        Cf::NotEqual => d3d12_ty::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Cf::Always => d3d12_ty::D3D12_COMPARISON_FUNC_ALWAYS,
    }
}

pub fn map_border_color(border_color: Option<wgt::SamplerBorderColor>) -> [f32; 4] {
    use wgt::SamplerBorderColor as Sbc;
    match border_color {
        Some(Sbc::TransparentBlack) | Some(Sbc::Zero) | None => [0.0; 4],
        Some(Sbc::OpaqueBlack) => [0.0, 0.0, 0.0, 1.0],
        Some(Sbc::OpaqueWhite) => [1.0; 4],
    }
}

pub fn map_visibility(visibility: wgt::ShaderStages) -> d3d12::ShaderVisibility {
    match visibility {
        wgt::ShaderStages::VERTEX => d3d12::ShaderVisibility::VS,
        wgt::ShaderStages::FRAGMENT => d3d12::ShaderVisibility::PS,
        _ => d3d12::ShaderVisibility::All,
    }
}

pub fn map_binding_type(ty: &wgt::BindingType) -> d3d12::DescriptorRangeType {
    use wgt::BindingType as Bt;
    match *ty {
        Bt::Sampler { .. } => d3d12::DescriptorRangeType::Sampler,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Uniform,
            ..
        } => d3d12::DescriptorRangeType::CBV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: true },
            ..
        }
        | Bt::Texture { .. } => d3d12::DescriptorRangeType::SRV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: false },
            ..
        }
        | Bt::StorageTexture { .. } => d3d12::DescriptorRangeType::UAV,
    }
}

pub fn map_label(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(iter::once(0)).collect()
}

pub fn map_buffer_usage_to_state(usage: crate::BufferUses) -> d3d12_ty::D3D12_RESOURCE_STATES {
    use crate::BufferUses as Bu;
    let mut state = d3d12_ty::D3D12_RESOURCE_STATE_COMMON;

    if usage.intersects(Bu::COPY_SRC) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Bu::COPY_DST) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Bu::INDEX) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_INDEX_BUFFER;
    }
    if usage.intersects(Bu::VERTEX | Bu::UNIFORM) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    }
    if usage.intersects(Bu::STORAGE_READ_WRITE) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else if usage.intersects(Bu::STORAGE_READ) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | d3d12_ty::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Bu::INDIRECT) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }
    state
}

pub fn map_texture_usage_to_state(usage: crate::TextureUses) -> d3d12_ty::D3D12_RESOURCE_STATES {
    use crate::TextureUses as Tu;
    let mut state = d3d12_ty::D3D12_RESOURCE_STATE_COMMON;
    //Note: `RESOLVE_SOURCE` and `RESOLVE_DEST` are not used here
    //Note: `PRESENT` is the same as `COMMON`
    if usage == crate::TextureUses::UNINITIALIZED {
        return state;
    }

    if usage.intersects(Tu::COPY_SRC) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Tu::COPY_DST) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Tu::RESOURCE) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | d3d12_ty::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Tu::COLOR_TARGET) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_READ) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_DEPTH_READ;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_WRITE) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    if usage.intersects(Tu::STORAGE_READ | Tu::STORAGE_READ_WRITE) {
        state |= d3d12_ty::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    state
}

pub fn map_topology(
    topology: wgt::PrimitiveTopology,
) -> (
    d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY,
) {
    match topology {
        wgt::PrimitiveTopology::PointList => (
            d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        ),
        wgt::PrimitiveTopology::LineList => (
            d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        ),
        wgt::PrimitiveTopology::LineStrip => (
            d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        ),
        wgt::PrimitiveTopology::TriangleList => (
            d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        ),
        wgt::PrimitiveTopology::TriangleStrip => (
            d3d12_ty::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            d3dcommon::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
        ),
    }
}

pub fn map_polygon_mode(mode: wgt::PolygonMode) -> d3d12_ty::D3D12_FILL_MODE {
    match mode {
        wgt::PolygonMode::Fill => d3d12_ty::D3D12_FILL_MODE_SOLID,
        wgt::PolygonMode::Line => d3d12_ty::D3D12_FILL_MODE_WIREFRAME,
        wgt::PolygonMode::Point => panic!(
            "{:?} is not enabled for this backend",
            wgt::Features::POLYGON_MODE_POINT
        ),
    }
}

/// D3D12 doesn't support passing factors ending in `_COLOR` for alpha blending
/// (see https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_render_target_blend_desc).
/// Therefore this function takes an additional `is_alpha` argument
/// which if set will return an equivalent `_ALPHA` factor.
fn map_blend_factor(factor: wgt::BlendFactor, is_alpha: bool) -> d3d12_ty::D3D12_BLEND {
    use wgt::BlendFactor as Bf;
    match factor {
        Bf::Zero => d3d12_ty::D3D12_BLEND_ZERO,
        Bf::One => d3d12_ty::D3D12_BLEND_ONE,
        Bf::Src if is_alpha => d3d12_ty::D3D12_BLEND_SRC_ALPHA,
        Bf::Src => d3d12_ty::D3D12_BLEND_SRC_COLOR,
        Bf::OneMinusSrc if is_alpha => d3d12_ty::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::OneMinusSrc => d3d12_ty::D3D12_BLEND_INV_SRC_COLOR,
        Bf::Dst if is_alpha => d3d12_ty::D3D12_BLEND_DEST_ALPHA,
        Bf::Dst => d3d12_ty::D3D12_BLEND_DEST_COLOR,
        Bf::OneMinusDst if is_alpha => d3d12_ty::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::OneMinusDst => d3d12_ty::D3D12_BLEND_INV_DEST_COLOR,
        Bf::SrcAlpha => d3d12_ty::D3D12_BLEND_SRC_ALPHA,
        Bf::OneMinusSrcAlpha => d3d12_ty::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::DstAlpha => d3d12_ty::D3D12_BLEND_DEST_ALPHA,
        Bf::OneMinusDstAlpha => d3d12_ty::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::Constant => d3d12_ty::D3D12_BLEND_BLEND_FACTOR,
        Bf::OneMinusConstant => d3d12_ty::D3D12_BLEND_INV_BLEND_FACTOR,
        Bf::SrcAlphaSaturated => d3d12_ty::D3D12_BLEND_SRC_ALPHA_SAT,
        Bf::Src1 if is_alpha => d3d12_ty::D3D12_BLEND_SRC1_ALPHA,
        Bf::Src1 => d3d12_ty::D3D12_BLEND_SRC1_COLOR,
        Bf::OneMinusSrc1 if is_alpha => d3d12_ty::D3D12_BLEND_INV_SRC1_ALPHA,
        Bf::OneMinusSrc1 => d3d12_ty::D3D12_BLEND_INV_SRC1_COLOR,
        Bf::Src1Alpha => d3d12_ty::D3D12_BLEND_SRC1_ALPHA,
        Bf::OneMinusSrc1Alpha => d3d12_ty::D3D12_BLEND_INV_SRC1_ALPHA,
    }
}

fn map_blend_component(
    component: &wgt::BlendComponent,
    is_alpha: bool,
) -> (
    d3d12_ty::D3D12_BLEND_OP,
    d3d12_ty::D3D12_BLEND,
    d3d12_ty::D3D12_BLEND,
) {
    let raw_op = match component.operation {
        wgt::BlendOperation::Add => d3d12_ty::D3D12_BLEND_OP_ADD,
        wgt::BlendOperation::Subtract => d3d12_ty::D3D12_BLEND_OP_SUBTRACT,
        wgt::BlendOperation::ReverseSubtract => d3d12_ty::D3D12_BLEND_OP_REV_SUBTRACT,
        wgt::BlendOperation::Min => d3d12_ty::D3D12_BLEND_OP_MIN,
        wgt::BlendOperation::Max => d3d12_ty::D3D12_BLEND_OP_MAX,
    };
    let raw_src = map_blend_factor(component.src_factor, is_alpha);
    let raw_dst = map_blend_factor(component.dst_factor, is_alpha);
    (raw_op, raw_src, raw_dst)
}

pub fn map_render_targets(
    color_targets: &[Option<wgt::ColorTargetState>],
) -> [d3d12_ty::D3D12_RENDER_TARGET_BLEND_DESC;
       d3d12_ty::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize] {
    let dummy_target = d3d12_ty::D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: 0,
        LogicOpEnable: 0,
        SrcBlend: d3d12_ty::D3D12_BLEND_ZERO,
        DestBlend: d3d12_ty::D3D12_BLEND_ZERO,
        BlendOp: d3d12_ty::D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: d3d12_ty::D3D12_BLEND_ZERO,
        DestBlendAlpha: d3d12_ty::D3D12_BLEND_ZERO,
        BlendOpAlpha: d3d12_ty::D3D12_BLEND_OP_ADD,
        LogicOp: d3d12_ty::D3D12_LOGIC_OP_CLEAR,
        RenderTargetWriteMask: 0,
    };
    let mut raw_targets = [dummy_target; d3d12_ty::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize];

    for (raw, ct) in raw_targets.iter_mut().zip(color_targets.iter()) {
        if let Some(ct) = ct.as_ref() {
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
    }

    raw_targets
}

fn map_stencil_op(op: wgt::StencilOperation) -> d3d12_ty::D3D12_STENCIL_OP {
    use wgt::StencilOperation as So;
    match op {
        So::Keep => d3d12_ty::D3D12_STENCIL_OP_KEEP,
        So::Zero => d3d12_ty::D3D12_STENCIL_OP_ZERO,
        So::Replace => d3d12_ty::D3D12_STENCIL_OP_REPLACE,
        So::IncrementClamp => d3d12_ty::D3D12_STENCIL_OP_INCR_SAT,
        So::IncrementWrap => d3d12_ty::D3D12_STENCIL_OP_INCR,
        So::DecrementClamp => d3d12_ty::D3D12_STENCIL_OP_DECR_SAT,
        So::DecrementWrap => d3d12_ty::D3D12_STENCIL_OP_DECR,
        So::Invert => d3d12_ty::D3D12_STENCIL_OP_INVERT,
    }
}

fn map_stencil_face(face: &wgt::StencilFaceState) -> d3d12_ty::D3D12_DEPTH_STENCILOP_DESC {
    d3d12_ty::D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: map_stencil_op(face.fail_op),
        StencilDepthFailOp: map_stencil_op(face.depth_fail_op),
        StencilPassOp: map_stencil_op(face.pass_op),
        StencilFunc: map_comparison(face.compare),
    }
}

pub fn map_depth_stencil(ds: &wgt::DepthStencilState) -> d3d12_ty::D3D12_DEPTH_STENCIL_DESC {
    d3d12_ty::D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: BOOL::from(ds.is_depth_enabled()),
        DepthWriteMask: if ds.depth_write_enabled {
            d3d12_ty::D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            d3d12_ty::D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: map_comparison(ds.depth_compare),
        StencilEnable: BOOL::from(ds.stencil.is_enabled()),
        StencilReadMask: ds.stencil.read_mask as u8,
        StencilWriteMask: ds.stencil.write_mask as u8,
        FrontFace: map_stencil_face(&ds.stencil.front),
        BackFace: map_stencil_face(&ds.stencil.back),
    }
}
