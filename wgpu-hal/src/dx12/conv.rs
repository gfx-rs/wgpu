use windows::Win32::Graphics::{Direct3D, Direct3D12};

pub fn map_buffer_usage_to_resource_flags(
    usage: crate::BufferUses,
) -> Direct3D12::D3D12_RESOURCE_FLAGS {
    let mut flags = Direct3D12::D3D12_RESOURCE_FLAG_NONE;
    if usage.contains(crate::BufferUses::STORAGE_READ_WRITE) {
        flags |= Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    flags
}

pub fn map_texture_dimension(dim: wgt::TextureDimension) -> Direct3D12::D3D12_RESOURCE_DIMENSION {
    match dim {
        wgt::TextureDimension::D1 => Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE1D,
        wgt::TextureDimension::D2 => Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        wgt::TextureDimension::D3 => Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    }
}

pub fn map_texture_usage_to_resource_flags(
    usage: crate::TextureUses,
) -> Direct3D12::D3D12_RESOURCE_FLAGS {
    let mut flags = Direct3D12::D3D12_RESOURCE_FLAG_NONE;

    if usage.contains(crate::TextureUses::COLOR_TARGET) {
        flags |= Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if usage.intersects(
        crate::TextureUses::DEPTH_STENCIL_READ | crate::TextureUses::DEPTH_STENCIL_WRITE,
    ) {
        flags |= Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if !usage.contains(crate::TextureUses::RESOURCE) {
            flags |= Direct3D12::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        }
    }
    if usage.contains(crate::TextureUses::STORAGE_READ_WRITE) {
        flags |= Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }

    flags
}

pub fn map_address_mode(mode: wgt::AddressMode) -> Direct3D12::D3D12_TEXTURE_ADDRESS_MODE {
    use wgt::AddressMode as Am;
    match mode {
        Am::Repeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        Am::MirrorRepeat => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        Am::ClampToEdge => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        Am::ClampToBorder => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_BORDER,
        //Am::MirrorClamp => Direct3D12::D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
    }
}

pub fn map_filter_mode(mode: wgt::FilterMode) -> Direct3D12::D3D12_FILTER_TYPE {
    match mode {
        wgt::FilterMode::Nearest => Direct3D12::D3D12_FILTER_TYPE_POINT,
        wgt::FilterMode::Linear => Direct3D12::D3D12_FILTER_TYPE_LINEAR,
    }
}

pub fn map_comparison(func: wgt::CompareFunction) -> Direct3D12::D3D12_COMPARISON_FUNC {
    use wgt::CompareFunction as Cf;
    match func {
        Cf::Never => Direct3D12::D3D12_COMPARISON_FUNC_NEVER,
        Cf::Less => Direct3D12::D3D12_COMPARISON_FUNC_LESS,
        Cf::LessEqual => Direct3D12::D3D12_COMPARISON_FUNC_LESS_EQUAL,
        Cf::Equal => Direct3D12::D3D12_COMPARISON_FUNC_EQUAL,
        Cf::GreaterEqual => Direct3D12::D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        Cf::Greater => Direct3D12::D3D12_COMPARISON_FUNC_GREATER,
        Cf::NotEqual => Direct3D12::D3D12_COMPARISON_FUNC_NOT_EQUAL,
        Cf::Always => Direct3D12::D3D12_COMPARISON_FUNC_ALWAYS,
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

pub fn map_visibility(visibility: wgt::ShaderStages) -> Direct3D12::D3D12_SHADER_VISIBILITY {
    match visibility {
        wgt::ShaderStages::VERTEX => Direct3D12::D3D12_SHADER_VISIBILITY_VERTEX,
        wgt::ShaderStages::FRAGMENT => Direct3D12::D3D12_SHADER_VISIBILITY_PIXEL,
        _ => Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
    }
}

pub fn map_binding_type(ty: &wgt::BindingType) -> Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE {
    use wgt::BindingType as Bt;
    match *ty {
        Bt::Sampler { .. } => Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Uniform,
            ..
        } => Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: true },
            ..
        }
        | Bt::Texture { .. } => Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        Bt::Buffer {
            ty: wgt::BufferBindingType::Storage { read_only: false },
            ..
        }
        | Bt::StorageTexture { .. } => Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        Bt::AccelerationStructure => todo!(),
    }
}

pub fn map_buffer_usage_to_state(usage: crate::BufferUses) -> Direct3D12::D3D12_RESOURCE_STATES {
    use crate::BufferUses as Bu;
    let mut state = Direct3D12::D3D12_RESOURCE_STATE_COMMON;

    if usage.intersects(Bu::COPY_SRC) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Bu::COPY_DST) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Bu::INDEX) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_INDEX_BUFFER;
    }
    if usage.intersects(Bu::VERTEX | Bu::UNIFORM) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    }
    if usage.intersects(Bu::STORAGE_READ_WRITE) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    } else if usage.intersects(Bu::STORAGE_READ) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | Direct3D12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Bu::INDIRECT) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    }
    state
}

pub fn map_texture_usage_to_state(usage: crate::TextureUses) -> Direct3D12::D3D12_RESOURCE_STATES {
    use crate::TextureUses as Tu;
    let mut state = Direct3D12::D3D12_RESOURCE_STATE_COMMON;
    //Note: `RESOLVE_SOURCE` and `RESOLVE_DEST` are not used here
    //Note: `PRESENT` is the same as `COMMON`
    if usage == crate::TextureUses::UNINITIALIZED {
        return state;
    }

    if usage.intersects(Tu::COPY_SRC) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE;
    }
    if usage.intersects(Tu::COPY_DST) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_COPY_DEST;
    }
    if usage.intersects(Tu::RESOURCE) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            | Direct3D12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    if usage.intersects(Tu::COLOR_TARGET) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_READ) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_DEPTH_READ;
    }
    if usage.intersects(Tu::DEPTH_STENCIL_WRITE) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_DEPTH_WRITE;
    }
    if usage.intersects(Tu::STORAGE_READ | Tu::STORAGE_READ_WRITE) {
        state |= Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    state
}

pub fn map_topology(
    topology: wgt::PrimitiveTopology,
) -> (
    Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    Direct3D::D3D_PRIMITIVE_TOPOLOGY,
) {
    match topology {
        wgt::PrimitiveTopology::PointList => (
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
        ),
        wgt::PrimitiveTopology::LineList => (
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINELIST,
        ),
        wgt::PrimitiveTopology::LineStrip => (
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
        ),
        wgt::PrimitiveTopology::TriangleList => (
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        ),
        wgt::PrimitiveTopology::TriangleStrip => (
            Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
        ),
    }
}

pub fn map_polygon_mode(mode: wgt::PolygonMode) -> Direct3D12::D3D12_FILL_MODE {
    match mode {
        wgt::PolygonMode::Fill => Direct3D12::D3D12_FILL_MODE_SOLID,
        wgt::PolygonMode::Line => Direct3D12::D3D12_FILL_MODE_WIREFRAME,
        wgt::PolygonMode::Point => panic!(
            "{:?} is not enabled for this backend",
            wgt::Features::POLYGON_MODE_POINT
        ),
    }
}

/// D3D12 doesn't support passing factors ending in `_COLOR` for alpha blending
/// (see <https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_render_target_blend_desc>).
/// Therefore this function takes an additional `is_alpha` argument
/// which if set will return an equivalent `_ALPHA` factor.
fn map_blend_factor(factor: wgt::BlendFactor, is_alpha: bool) -> Direct3D12::D3D12_BLEND {
    use wgt::BlendFactor as Bf;
    match factor {
        Bf::Zero => Direct3D12::D3D12_BLEND_ZERO,
        Bf::One => Direct3D12::D3D12_BLEND_ONE,
        Bf::Src if is_alpha => Direct3D12::D3D12_BLEND_SRC_ALPHA,
        Bf::Src => Direct3D12::D3D12_BLEND_SRC_COLOR,
        Bf::OneMinusSrc if is_alpha => Direct3D12::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::OneMinusSrc => Direct3D12::D3D12_BLEND_INV_SRC_COLOR,
        Bf::Dst if is_alpha => Direct3D12::D3D12_BLEND_DEST_ALPHA,
        Bf::Dst => Direct3D12::D3D12_BLEND_DEST_COLOR,
        Bf::OneMinusDst if is_alpha => Direct3D12::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::OneMinusDst => Direct3D12::D3D12_BLEND_INV_DEST_COLOR,
        Bf::SrcAlpha => Direct3D12::D3D12_BLEND_SRC_ALPHA,
        Bf::OneMinusSrcAlpha => Direct3D12::D3D12_BLEND_INV_SRC_ALPHA,
        Bf::DstAlpha => Direct3D12::D3D12_BLEND_DEST_ALPHA,
        Bf::OneMinusDstAlpha => Direct3D12::D3D12_BLEND_INV_DEST_ALPHA,
        Bf::Constant => Direct3D12::D3D12_BLEND_BLEND_FACTOR,
        Bf::OneMinusConstant => Direct3D12::D3D12_BLEND_INV_BLEND_FACTOR,
        Bf::SrcAlphaSaturated => Direct3D12::D3D12_BLEND_SRC_ALPHA_SAT,
        Bf::Src1 if is_alpha => Direct3D12::D3D12_BLEND_SRC1_ALPHA,
        Bf::Src1 => Direct3D12::D3D12_BLEND_SRC1_COLOR,
        Bf::OneMinusSrc1 if is_alpha => Direct3D12::D3D12_BLEND_INV_SRC1_ALPHA,
        Bf::OneMinusSrc1 => Direct3D12::D3D12_BLEND_INV_SRC1_COLOR,
        Bf::Src1Alpha => Direct3D12::D3D12_BLEND_SRC1_ALPHA,
        Bf::OneMinusSrc1Alpha => Direct3D12::D3D12_BLEND_INV_SRC1_ALPHA,
    }
}

fn map_blend_component(
    component: &wgt::BlendComponent,
    is_alpha: bool,
) -> (
    Direct3D12::D3D12_BLEND_OP,
    Direct3D12::D3D12_BLEND,
    Direct3D12::D3D12_BLEND,
) {
    let raw_op = match component.operation {
        wgt::BlendOperation::Add => Direct3D12::D3D12_BLEND_OP_ADD,
        wgt::BlendOperation::Subtract => Direct3D12::D3D12_BLEND_OP_SUBTRACT,
        wgt::BlendOperation::ReverseSubtract => Direct3D12::D3D12_BLEND_OP_REV_SUBTRACT,
        wgt::BlendOperation::Min => Direct3D12::D3D12_BLEND_OP_MIN,
        wgt::BlendOperation::Max => Direct3D12::D3D12_BLEND_OP_MAX,
    };
    let raw_src = map_blend_factor(component.src_factor, is_alpha);
    let raw_dst = map_blend_factor(component.dst_factor, is_alpha);
    (raw_op, raw_src, raw_dst)
}

pub fn map_render_targets(
    color_targets: &[Option<wgt::ColorTargetState>],
) -> [Direct3D12::D3D12_RENDER_TARGET_BLEND_DESC;
       Direct3D12::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize] {
    let dummy_target = Direct3D12::D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: false.into(),
        LogicOpEnable: false.into(),
        SrcBlend: Direct3D12::D3D12_BLEND_ZERO,
        DestBlend: Direct3D12::D3D12_BLEND_ZERO,
        BlendOp: Direct3D12::D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: Direct3D12::D3D12_BLEND_ZERO,
        DestBlendAlpha: Direct3D12::D3D12_BLEND_ZERO,
        BlendOpAlpha: Direct3D12::D3D12_BLEND_OP_ADD,
        LogicOp: Direct3D12::D3D12_LOGIC_OP_CLEAR,
        RenderTargetWriteMask: 0,
    };
    let mut raw_targets =
        [dummy_target; Direct3D12::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize];

    for (raw, ct) in raw_targets.iter_mut().zip(color_targets.iter()) {
        if let Some(ct) = ct.as_ref() {
            raw.RenderTargetWriteMask = ct.write_mask.bits() as u8;
            if let Some(ref blend) = ct.blend {
                let (color_op, color_src, color_dst) = map_blend_component(&blend.color, false);
                let (alpha_op, alpha_src, alpha_dst) = map_blend_component(&blend.alpha, true);
                raw.BlendEnable = true.into();
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

fn map_stencil_op(op: wgt::StencilOperation) -> Direct3D12::D3D12_STENCIL_OP {
    use wgt::StencilOperation as So;
    match op {
        So::Keep => Direct3D12::D3D12_STENCIL_OP_KEEP,
        So::Zero => Direct3D12::D3D12_STENCIL_OP_ZERO,
        So::Replace => Direct3D12::D3D12_STENCIL_OP_REPLACE,
        So::IncrementClamp => Direct3D12::D3D12_STENCIL_OP_INCR_SAT,
        So::IncrementWrap => Direct3D12::D3D12_STENCIL_OP_INCR,
        So::DecrementClamp => Direct3D12::D3D12_STENCIL_OP_DECR_SAT,
        So::DecrementWrap => Direct3D12::D3D12_STENCIL_OP_DECR,
        So::Invert => Direct3D12::D3D12_STENCIL_OP_INVERT,
    }
}

fn map_stencil_face(face: &wgt::StencilFaceState) -> Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
    Direct3D12::D3D12_DEPTH_STENCILOP_DESC {
        StencilFailOp: map_stencil_op(face.fail_op),
        StencilDepthFailOp: map_stencil_op(face.depth_fail_op),
        StencilPassOp: map_stencil_op(face.pass_op),
        StencilFunc: map_comparison(face.compare),
    }
}

pub fn map_depth_stencil(ds: &wgt::DepthStencilState) -> Direct3D12::D3D12_DEPTH_STENCIL_DESC {
    Direct3D12::D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: ds.is_depth_enabled().into(),
        DepthWriteMask: if ds.depth_write_enabled {
            Direct3D12::D3D12_DEPTH_WRITE_MASK_ALL
        } else {
            Direct3D12::D3D12_DEPTH_WRITE_MASK_ZERO
        },
        DepthFunc: map_comparison(ds.depth_compare),
        StencilEnable: ds.stencil.is_enabled().into(),
        StencilReadMask: ds.stencil.read_mask as u8,
        StencilWriteMask: ds.stencil.write_mask as u8,
        FrontFace: map_stencil_face(&ds.stencil.front),
        BackFace: map_stencil_face(&ds.stencil.back),
    }
}
