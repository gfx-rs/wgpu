pub fn map_texture_usage(usage: crate::TextureUses) -> mtl::MTLTextureUsage {
    use crate::TextureUses as Tu;

    let mut mtl_usage = mtl::MTLTextureUsage::Unknown;

    mtl_usage.set(
        mtl::MTLTextureUsage::RenderTarget,
        usage.intersects(Tu::COLOR_TARGET | Tu::DEPTH_STENCIL_READ | Tu::DEPTH_STENCIL_WRITE),
    );
    mtl_usage.set(
        mtl::MTLTextureUsage::ShaderRead,
        usage.intersects(
            Tu::RESOURCE | Tu::DEPTH_STENCIL_READ | Tu::STORAGE_READ | Tu::STORAGE_READ_WRITE,
        ),
    );
    mtl_usage.set(
        mtl::MTLTextureUsage::ShaderWrite,
        usage.intersects(Tu::STORAGE_READ_WRITE),
    );

    mtl_usage
}

pub fn map_texture_view_dimension(dim: wgt::TextureViewDimension) -> mtl::MTLTextureType {
    use mtl::MTLTextureType::*;
    use wgt::TextureViewDimension as Tvd;
    match dim {
        Tvd::D1 => D1,
        Tvd::D2 => D2,
        Tvd::D2Array => D2Array,
        Tvd::D3 => D3,
        Tvd::Cube => Cube,
        Tvd::CubeArray => CubeArray,
    }
}

pub fn map_compare_function(fun: wgt::CompareFunction) -> mtl::MTLCompareFunction {
    use mtl::MTLCompareFunction::*;
    use wgt::CompareFunction as Cf;
    match fun {
        Cf::Never => Never,
        Cf::Less => Less,
        Cf::LessEqual => LessEqual,
        Cf::Equal => Equal,
        Cf::GreaterEqual => GreaterEqual,
        Cf::Greater => Greater,
        Cf::NotEqual => NotEqual,
        Cf::Always => Always,
    }
}

pub fn map_filter_mode(filter: wgt::FilterMode) -> mtl::MTLSamplerMinMagFilter {
    use mtl::MTLSamplerMinMagFilter::*;
    match filter {
        wgt::FilterMode::Nearest => Nearest,
        wgt::FilterMode::Linear => Linear,
    }
}

pub fn map_address_mode(address: wgt::AddressMode) -> mtl::MTLSamplerAddressMode {
    use mtl::MTLSamplerAddressMode::*;
    use wgt::AddressMode as Fm;
    match address {
        Fm::Repeat => Repeat,
        Fm::MirrorRepeat => MirrorRepeat,
        Fm::ClampToEdge => ClampToEdge,
        Fm::ClampToBorder => ClampToBorderColor,
        //Fm::MirrorClamp => MirrorClampToEdge,
    }
}

pub fn map_border_color(border_color: wgt::SamplerBorderColor) -> mtl::MTLSamplerBorderColor {
    use mtl::MTLSamplerBorderColor::*;
    match border_color {
        wgt::SamplerBorderColor::TransparentBlack => TransparentBlack,
        wgt::SamplerBorderColor::OpaqueBlack => OpaqueBlack,
        wgt::SamplerBorderColor::OpaqueWhite => OpaqueWhite,
        wgt::SamplerBorderColor::Zero => unreachable!(),
    }
}

pub fn map_primitive_topology(
    topology: wgt::PrimitiveTopology,
) -> (mtl::MTLPrimitiveTopologyClass, mtl::MTLPrimitiveType) {
    use wgt::PrimitiveTopology as Pt;
    match topology {
        Pt::PointList => (
            mtl::MTLPrimitiveTopologyClass::Point,
            mtl::MTLPrimitiveType::Point,
        ),
        Pt::LineList => (
            mtl::MTLPrimitiveTopologyClass::Line,
            mtl::MTLPrimitiveType::Line,
        ),
        Pt::LineStrip => (
            mtl::MTLPrimitiveTopologyClass::Line,
            mtl::MTLPrimitiveType::LineStrip,
        ),
        Pt::TriangleList => (
            mtl::MTLPrimitiveTopologyClass::Triangle,
            mtl::MTLPrimitiveType::Triangle,
        ),
        Pt::TriangleStrip => (
            mtl::MTLPrimitiveTopologyClass::Triangle,
            mtl::MTLPrimitiveType::TriangleStrip,
        ),
    }
}

pub fn map_color_write(mask: wgt::ColorWrites) -> mtl::MTLColorWriteMask {
    let mut raw_mask = mtl::MTLColorWriteMask::empty();

    if mask.contains(wgt::ColorWrites::RED) {
        raw_mask |= mtl::MTLColorWriteMask::Red;
    }
    if mask.contains(wgt::ColorWrites::GREEN) {
        raw_mask |= mtl::MTLColorWriteMask::Green;
    }
    if mask.contains(wgt::ColorWrites::BLUE) {
        raw_mask |= mtl::MTLColorWriteMask::Blue;
    }
    if mask.contains(wgt::ColorWrites::ALPHA) {
        raw_mask |= mtl::MTLColorWriteMask::Alpha;
    }

    raw_mask
}

pub fn map_blend_factor(factor: wgt::BlendFactor) -> mtl::MTLBlendFactor {
    use mtl::MTLBlendFactor::*;
    use wgt::BlendFactor as Bf;

    match factor {
        Bf::Zero => Zero,
        Bf::One => One,
        Bf::Src => SourceColor,
        Bf::OneMinusSrc => OneMinusSourceColor,
        Bf::Dst => DestinationColor,
        Bf::OneMinusDst => OneMinusDestinationColor,
        Bf::SrcAlpha => SourceAlpha,
        Bf::OneMinusSrcAlpha => OneMinusSourceAlpha,
        Bf::DstAlpha => DestinationAlpha,
        Bf::OneMinusDstAlpha => OneMinusDestinationAlpha,
        Bf::Constant => BlendColor,
        Bf::OneMinusConstant => OneMinusBlendColor,
        //Bf::ConstantAlpha => BlendAlpha,
        //Bf::OneMinusConstantAlpha => OneMinusBlendAlpha,
        Bf::SrcAlphaSaturated => SourceAlphaSaturated,
        //Bf::Src1 => Source1Color,
        //Bf::OneMinusSrc1 => OneMinusSource1Color,
        //Bf::Src1Alpha => Source1Alpha,
        //Bf::OneMinusSrc1Alpha => OneMinusSource1Alpha,
    }
}

pub fn map_blend_op(operation: wgt::BlendOperation) -> mtl::MTLBlendOperation {
    use mtl::MTLBlendOperation::*;
    use wgt::BlendOperation as Bo;

    match operation {
        Bo::Add => Add,
        Bo::Subtract => Subtract,
        Bo::ReverseSubtract => ReverseSubtract,
        Bo::Min => Min,
        Bo::Max => Max,
    }
}

pub fn map_blend_component(
    component: &wgt::BlendComponent,
) -> (
    mtl::MTLBlendOperation,
    mtl::MTLBlendFactor,
    mtl::MTLBlendFactor,
) {
    (
        map_blend_op(component.operation),
        map_blend_factor(component.src_factor),
        map_blend_factor(component.dst_factor),
    )
}

pub fn map_vertex_format(format: wgt::VertexFormat) -> mtl::MTLVertexFormat {
    use mtl::MTLVertexFormat::*;
    use wgt::VertexFormat as Vf;

    match format {
        Vf::Unorm8x2 => UChar2Normalized,
        Vf::Snorm8x2 => Char2Normalized,
        Vf::Uint8x2 => UChar2,
        Vf::Sint8x2 => Char2,
        Vf::Unorm8x4 => UChar4Normalized,
        Vf::Snorm8x4 => Char4Normalized,
        Vf::Uint8x4 => UChar4,
        Vf::Sint8x4 => Char4,
        Vf::Unorm16x2 => UShort2Normalized,
        Vf::Snorm16x2 => Short2Normalized,
        Vf::Uint16x2 => UShort2,
        Vf::Sint16x2 => Short2,
        Vf::Float16x2 => Half2,
        Vf::Unorm16x4 => UShort4Normalized,
        Vf::Snorm16x4 => Short4Normalized,
        Vf::Uint16x4 => UShort4,
        Vf::Sint16x4 => Short4,
        Vf::Float16x4 => Half4,
        Vf::Uint32 => UInt,
        Vf::Sint32 => Int,
        Vf::Float32 => Float,
        Vf::Uint32x2 => UInt2,
        Vf::Sint32x2 => Int2,
        Vf::Float32x2 => Float2,
        Vf::Uint32x3 => UInt3,
        Vf::Sint32x3 => Int3,
        Vf::Float32x3 => Float3,
        Vf::Uint32x4 => UInt4,
        Vf::Sint32x4 => Int4,
        Vf::Float32x4 => Float4,
        Vf::Float64 | Vf::Float64x2 | Vf::Float64x3 | Vf::Float64x4 => unimplemented!(),
    }
}

pub fn map_step_mode(mode: wgt::VertexStepMode) -> mtl::MTLVertexStepFunction {
    match mode {
        wgt::VertexStepMode::Vertex => mtl::MTLVertexStepFunction::PerVertex,
        wgt::VertexStepMode::Instance => mtl::MTLVertexStepFunction::PerInstance,
    }
}

pub fn map_stencil_op(op: wgt::StencilOperation) -> mtl::MTLStencilOperation {
    use mtl::MTLStencilOperation::*;
    use wgt::StencilOperation as So;

    match op {
        So::Keep => Keep,
        So::Zero => Zero,
        So::Replace => Replace,
        So::IncrementClamp => IncrementClamp,
        So::IncrementWrap => IncrementWrap,
        So::DecrementClamp => DecrementClamp,
        So::DecrementWrap => DecrementWrap,
        So::Invert => Invert,
    }
}

pub fn map_winding(winding: wgt::FrontFace) -> mtl::MTLWinding {
    match winding {
        wgt::FrontFace::Cw => mtl::MTLWinding::Clockwise,
        wgt::FrontFace::Ccw => mtl::MTLWinding::CounterClockwise,
    }
}

pub fn map_cull_mode(face: Option<wgt::Face>) -> mtl::MTLCullMode {
    match face {
        None => mtl::MTLCullMode::None,
        Some(wgt::Face::Front) => mtl::MTLCullMode::Front,
        Some(wgt::Face::Back) => mtl::MTLCullMode::Back,
    }
}

pub fn map_range(range: &crate::MemoryRange) -> mtl::NSRange {
    mtl::NSRange {
        location: range.start,
        length: range.end - range.start,
    }
}

pub fn map_copy_extent(extent: &crate::CopyExtent) -> mtl::MTLSize {
    mtl::MTLSize {
        width: extent.width as u64,
        height: extent.height as u64,
        depth: extent.depth as u64,
    }
}

pub fn map_origin(origin: &wgt::Origin3d) -> mtl::MTLOrigin {
    mtl::MTLOrigin {
        x: origin.x as u64,
        y: origin.y as u64,
        z: origin.z as u64,
    }
}

pub fn map_store_action(store: bool, resolve: bool) -> mtl::MTLStoreAction {
    use mtl::MTLStoreAction::*;
    match (store, resolve) {
        (true, true) => StoreAndMultisampleResolve,
        (false, true) => MultisampleResolve,
        (true, false) => Store,
        (false, false) => DontCare,
    }
}

pub fn map_clear_color(color: &wgt::Color) -> mtl::MTLClearColor {
    mtl::MTLClearColor {
        red: color.r,
        green: color.g,
        blue: color.b,
        alpha: color.a,
    }
}
