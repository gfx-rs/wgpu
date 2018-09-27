use hal;

use {binding_model, pipeline, resource};

pub(crate) fn map_buffer_usage(
    usage: resource::BufferUsageFlags,
) -> (hal::buffer::Usage, hal::memory::Properties) {
    use hal::buffer::Usage as U;
    use hal::memory::Properties as P;
    use resource::BufferUsageFlags as W;

    let mut hal_memory = P::empty();
    if usage.contains(W::MAP_READ) {
        hal_memory |= P::CPU_VISIBLE | P::CPU_CACHED;
    }
    if usage.contains(W::MAP_WRITE) {
        hal_memory |= P::CPU_VISIBLE;
    }

    let mut hal_usage = U::empty();
    if usage.contains(W::TRANSFER_SRC) {
        hal_usage |= U::TRANSFER_SRC;
    }
    if usage.contains(W::TRANSFER_DST) {
        hal_usage |= U::TRANSFER_DST;
    }
    if usage.contains(W::INDEX) {
        hal_usage |= U::INDEX;
    }
    if usage.contains(W::VERTEX) {
        hal_usage |= U::VERTEX;
    }
    if usage.contains(W::UNIFORM) {
        hal_usage |= U::UNIFORM;
    }
    if usage.contains(W::STORAGE) {
        hal_usage |= U::STORAGE;
    }

    (hal_usage, hal_memory)
}

pub(crate) fn map_binding_type(
    binding_ty: &binding_model::BindingType,
) -> hal::pso::DescriptorType {
    use binding_model::BindingType::*;
    use hal::pso::DescriptorType as H;
    match binding_ty {
        UniformBuffer => H::UniformBuffer,
        Sampler => H::Sampler,
        SampledTexture => H::SampledImage,
        StorageBuffer => H::StorageBuffer,
    }
}

pub(crate) fn map_shader_stage_flags(
    shader_stage_flags: binding_model::ShaderStageFlags,
) -> hal::pso::ShaderStageFlags {
    use binding_model::{
        ShaderStageFlags_COMPUTE, ShaderStageFlags_FRAGMENT, ShaderStageFlags_VERTEX,
    };
    use hal::pso::ShaderStageFlags as H;
    let mut value = H::empty();
    if 0 != shader_stage_flags & ShaderStageFlags_VERTEX {
        value |= H::VERTEX;
    }
    if 0 != shader_stage_flags & ShaderStageFlags_FRAGMENT {
        value |= H::FRAGMENT;
    }
    if 0 != shader_stage_flags & ShaderStageFlags_COMPUTE {
        value |= H::COMPUTE;
    }
    value
}

pub(crate) fn map_primitive_topology(
    primitive_topology: pipeline::PrimitiveTopology,
) -> hal::Primitive {
    use pipeline::PrimitiveTopology::*;
    use hal::Primitive as H;
    match primitive_topology {
        PointList => H::PointList,
        LineList => H::LineList,
        LineStrip => H::LineStrip,
        TriangleList => H::TriangleList,
        TriangleStrip => H::TriangleStrip,
    }
}

pub(crate) fn map_blend_state_descriptor(
    desc: pipeline::BlendStateDescriptor,
) -> hal::pso::ColorBlendDesc {
    let color_mask = desc.write_mask;
    let blend_state = match desc.blend_enabled {
        true => hal::pso::BlendState::On {
            color: map_blend_descriptor(desc.color),
            alpha: map_blend_descriptor(desc.alpha),
        },
        false => hal::pso::BlendState::Off,
    };
    hal::pso::ColorBlendDesc(map_color_write_flags(color_mask), blend_state)
}

fn map_color_write_flags(flags: u32) -> hal::pso::ColorMask {
    use pipeline::{ColorWriteFlags_RED, ColorWriteFlags_GREEN, ColorWriteFlags_BLUE, ColorWriteFlags_ALPHA};
    use hal::pso::ColorMask as H;
    let mut value = H::empty();
    if 0 != flags & ColorWriteFlags_RED {
        value |= H::RED;
    }
    if 0 != flags & ColorWriteFlags_GREEN {
        value |= H::GREEN;
    }
    if 0 != flags & ColorWriteFlags_BLUE {
        value |= H::BLUE;
    }
    if 0 != flags & ColorWriteFlags_ALPHA {
        value |= H::ALPHA;
    }
    value
}

fn map_blend_descriptor(
    blend_desc: pipeline::BlendDescriptor,
) -> hal::pso::BlendOp {
    use pipeline::BlendOperation::*;
    use hal::pso::BlendOp as H;
    match blend_desc.operation {
        Add => H::Add { src: map_blend_factor(blend_desc.src_factor), dst: map_blend_factor(blend_desc.dst_factor) },
        Subtract => H::Sub { src: map_blend_factor(blend_desc.src_factor), dst: map_blend_factor(blend_desc.dst_factor) },
        ReverseSubtract => H::RevSub { src: map_blend_factor(blend_desc.src_factor), dst: map_blend_factor(blend_desc.dst_factor) },
        Min => H::Min,
        Max => H::Max,
    }
}

fn map_blend_factor(
    blend_factor: pipeline::BlendFactor,
) -> hal::pso::Factor {
    use pipeline::BlendFactor::*;
    use hal::pso::Factor as H;
    match blend_factor {
        Zero => H::Zero,
        One => H::One,
        SrcColor => H::SrcColor,
        OneMinusSrcColor => H::OneMinusSrcColor,
        SrcAlpha => H::SrcAlpha,
        OneMinusSrcAlpha => H::OneMinusSrcAlpha,
        DstColor => H::DstColor,
        OneMinusDstColor => H::OneMinusDstColor,
        DstAlpha => H::DstAlpha,
        OneMinusDstAlpha => H::OneMinusDstAlpha,
        SrcAlphaSaturated => H::SrcAlphaSaturate,
        BlendColor => H::ConstColor,
        OneMinusBlendColor => H::OneMinusConstColor,
    }
}

pub(crate) fn map_depth_stencil_state(
    desc: pipeline::DepthStencilStateDescriptor,
) -> hal::pso::DepthStencilDesc {
    hal::pso::DepthStencilDesc {
        // TODO DepthTest::Off?
        depth: hal::pso::DepthTest:: On {
            fun: map_compare_function(desc.depth_compare),
            write: desc.depth_write_enabled,
        },
        depth_bounds: false, // TODO
        // TODO StencilTest::Off?
        stencil: hal::pso::StencilTest::On {
            front: map_stencil_face(desc.front, desc.stencil_read_mask, desc.stencil_write_mask),
            back: map_stencil_face(desc.back, desc.stencil_read_mask, desc.stencil_write_mask),
        },
    }
}

fn map_stencil_face(
    stencil_state_face_desc: pipeline::StencilStateFaceDescriptor,
    stencil_read_mask: u32,
    stencil_write_mask: u32,
) -> hal::pso::StencilFace {
    hal::pso::StencilFace {
        fun: map_compare_function(stencil_state_face_desc.compare),
        mask_read: hal::pso::State::Static(stencil_read_mask), // TODO dynamic?
        mask_write: hal::pso::State::Static(stencil_read_mask), // TODO dynamic?
        op_fail: map_stencil_operation(stencil_state_face_desc.stencil_fail_op),
        op_depth_fail: map_stencil_operation(stencil_state_face_desc.depth_fail_op),
        op_pass: map_stencil_operation(stencil_state_face_desc.pass_op),
        reference: hal::pso::State::Static(0), // TODO can this be set?
    }
}

fn map_compare_function(
    compare_function: resource::CompareFunction
) -> hal::pso::Comparison {
    use resource::CompareFunction::*;
    use hal::pso::Comparison as H;
    match compare_function {
        Never => H::Never,
        Less => H::Less,
        Equal => H::Equal,
        LessEqual => H::LessEqual,
        Greater => H::Greater,
        NotEqual => H::NotEqual,
        GreaterEqual => H::GreaterEqual,
        Always => H::Always
    }
}

fn map_stencil_operation(
    stencil_operation: pipeline::StencilOperation,
) -> hal::pso::StencilOp {
    use pipeline::StencilOperation::*;
    use hal::pso::StencilOp as H;
    match stencil_operation {
        Keep => H::Keep,
        Zero => H::Zero,
        Replace => H::Replace,
        Invert => H::Invert,
        IncrementClamp => H::IncrementClamp,
        DecrementClamp => H::DecrementClamp,
        IncrementWrap => H::IncrementWrap,
        DecrementWrap => H::DecrementWrap,
    }
}

pub(crate) fn map_texture_format(
    texture_format: resource::TextureFormat,
) -> hal::format::Format {
    use resource::TextureFormat::*;
    use hal::format::Format as H;
    match texture_format {
        R8g8b8a8Unorm => H::Rgba8Unorm,
        R8g8b8a8Uint => H::Rgba8Uint,
        B8g8r8a8Unorm => H::Bgra8Unorm,
        D32FloatS8Uint => H::D32FloatS8Uint,
    }
}
