use crate::{binding_model, command, pipeline, resource, Color, Extent3d};
use log::warn;

pub fn map_buffer_usage(
    usage: resource::BufferUsageFlags,
) -> (hal::buffer::Usage, hal::memory::Properties) {
    use hal::buffer::Usage as U;
    use hal::memory::Properties as P;
    use crate::resource::BufferUsageFlags as W;

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

pub fn map_texture_usage(
    usage: resource::TextureUsageFlags,
    aspects: hal::format::Aspects,
) -> hal::image::Usage {
    use hal::image::Usage as U;
    use crate::resource::TextureUsageFlags as W;

    let mut value = U::empty();
    if usage.contains(W::TRANSFER_SRC) {
        value |= U::TRANSFER_SRC;
    }
    if usage.contains(W::TRANSFER_DST) {
        value |= U::TRANSFER_DST;
    }
    if usage.contains(W::SAMPLED) {
        value |= U::SAMPLED;
    }
    if usage.contains(W::STORAGE) {
        value |= U::STORAGE;
    }
    if usage.contains(W::OUTPUT_ATTACHMENT) {
        if aspects.intersects(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL) {
            value |= U::DEPTH_STENCIL_ATTACHMENT;
        } else {
            value |= U::COLOR_ATTACHMENT;
        }
    }
    // Note: TextureUsageFlags::Present does not need to be handled explicitly
    // TODO: HAL Transient Attachment, HAL Input Attachment
    value
}

pub fn map_binding_type(binding_ty: binding_model::BindingType) -> hal::pso::DescriptorType {
    use crate::binding_model::BindingType::*;
    use hal::pso::DescriptorType as H;
    match binding_ty {
        UniformBuffer => H::UniformBuffer,
        Sampler => H::Sampler,
        SampledTexture => H::SampledImage,
        StorageBuffer => H::StorageBuffer,
    }
}

pub fn map_shader_stage_flags(
    shader_stage_flags: binding_model::ShaderStageFlags,
) -> hal::pso::ShaderStageFlags {
    use crate::binding_model::ShaderStageFlags as F;
    use hal::pso::ShaderStageFlags as H;

    let mut value = H::empty();
    if shader_stage_flags.contains(F::VERTEX) {
        value |= H::VERTEX;
    }
    if shader_stage_flags.contains(F::FRAGMENT) {
        value |= H::FRAGMENT;
    }
    if shader_stage_flags.contains(F::COMPUTE) {
        value |= H::COMPUTE;
    }
    value
}

pub fn map_primitive_topology(primitive_topology: pipeline::PrimitiveTopology) -> hal::Primitive {
    use hal::Primitive as H;
    use crate::pipeline::PrimitiveTopology::*;
    match primitive_topology {
        PointList => H::PointList,
        LineList => H::LineList,
        LineStrip => H::LineStrip,
        TriangleList => H::TriangleList,
        TriangleStrip => H::TriangleStrip,
    }
}

pub fn map_blend_state_descriptor(
    desc: &pipeline::BlendStateDescriptor,
) -> hal::pso::ColorBlendDesc {
    let color_mask = desc.write_mask;
    let blend_state = if desc.blend_enabled {
        hal::pso::BlendState::On {
            color: map_blend_descriptor(&desc.color),
            alpha: map_blend_descriptor(&desc.alpha),
        }
    } else {
        hal::pso::BlendState::Off
    };
    hal::pso::ColorBlendDesc(map_color_write_flags(color_mask), blend_state)
}

fn map_color_write_flags(flags: pipeline::ColorWriteFlags) -> hal::pso::ColorMask {
    use hal::pso::ColorMask as H;
    use crate::pipeline::ColorWriteFlags as F;

    let mut value = H::empty();
    if flags.contains(F::RED) {
        value |= H::RED;
    }
    if flags.contains(F::GREEN) {
        value |= H::GREEN;
    }
    if flags.contains(F::BLUE) {
        value |= H::BLUE;
    }
    if flags.contains(F::ALPHA) {
        value |= H::ALPHA;
    }
    value
}

fn map_blend_descriptor(blend_desc: &pipeline::BlendDescriptor) -> hal::pso::BlendOp {
    use hal::pso::BlendOp as H;
    use crate::pipeline::BlendOperation::*;
    match blend_desc.operation {
        Add => H::Add {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        Subtract => H::Sub {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        ReverseSubtract => H::RevSub {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        Min => H::Min,
        Max => H::Max,
    }
}

fn map_blend_factor(blend_factor: pipeline::BlendFactor) -> hal::pso::Factor {
    use hal::pso::Factor as H;
    use crate::pipeline::BlendFactor::*;
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

pub fn map_depth_stencil_state(
    desc: &pipeline::DepthStencilStateDescriptor,
) -> hal::pso::DepthStencilDesc {
    hal::pso::DepthStencilDesc {
        // TODO DepthTest::Off?
        depth: hal::pso::DepthTest::On {
            fun: map_compare_function(desc.depth_compare),
            write: desc.depth_write_enabled,
        },
        depth_bounds: false, // TODO
        // TODO StencilTest::Off?
        stencil: hal::pso::StencilTest::On {
            front: map_stencil_face(&desc.front, desc.stencil_read_mask, desc.stencil_write_mask),
            back: map_stencil_face(&desc.back, desc.stencil_read_mask, desc.stencil_write_mask),
        },
    }
}

fn map_stencil_face(
    stencil_state_face_desc: &pipeline::StencilStateFaceDescriptor,
    stencil_read_mask: u32,
    stencil_write_mask: u32,
) -> hal::pso::StencilFace {
    hal::pso::StencilFace {
        fun: map_compare_function(stencil_state_face_desc.compare),
        mask_read: hal::pso::State::Static(stencil_read_mask), // TODO dynamic?
        mask_write: hal::pso::State::Static(stencil_write_mask), // TODO dynamic?
        op_fail: map_stencil_operation(stencil_state_face_desc.stencil_fail_op),
        op_depth_fail: map_stencil_operation(stencil_state_face_desc.depth_fail_op),
        op_pass: map_stencil_operation(stencil_state_face_desc.pass_op),
        reference: hal::pso::State::Static(0), // TODO can this be set?
    }
}

pub fn map_compare_function(compare_function: resource::CompareFunction) -> hal::pso::Comparison {
    use hal::pso::Comparison as H;
    use crate::resource::CompareFunction::*;
    match compare_function {
        Never => H::Never,
        Less => H::Less,
        Equal => H::Equal,
        LessEqual => H::LessEqual,
        Greater => H::Greater,
        NotEqual => H::NotEqual,
        GreaterEqual => H::GreaterEqual,
        Always => H::Always,
    }
}

fn map_stencil_operation(stencil_operation: pipeline::StencilOperation) -> hal::pso::StencilOp {
    use hal::pso::StencilOp as H;
    use crate::pipeline::StencilOperation::*;
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

pub fn map_texture_format(texture_format: resource::TextureFormat) -> hal::format::Format {
    use hal::format::Format as H;
    use crate::resource::TextureFormat::*;
    match texture_format {
        R8g8b8a8Unorm => H::Rgba8Unorm,
        R8g8b8a8Uint => H::Rgba8Uint,
        B8g8r8a8Unorm => H::Bgra8Unorm,
        D32FloatS8Uint => H::D32FloatS8Uint,
    }
}

fn checked_u32_as_u16(value: u32) -> u16 {
    assert!(value <= ::std::u16::MAX as u32);
    value as u16
}

pub fn map_texture_dimension_size(
    dimension: resource::TextureDimension,
    Extent3d {
        width,
        height,
        depth,
    }: Extent3d,
) -> hal::image::Kind {
    use hal::image::Kind as H;
    use crate::resource::TextureDimension::*;
    match dimension {
        D1 => {
            assert_eq!(height, 1);
            H::D1(width, checked_u32_as_u16(depth))
        }
        D2 => H::D2(width, height, checked_u32_as_u16(depth), 1), // TODO: Samples
        D3 => H::D3(width, height, depth),
    }
}

pub fn map_texture_view_dimension(
    dimension: resource::TextureViewDimension,
) -> hal::image::ViewKind {
    use hal::image::ViewKind as H;
    use crate::resource::TextureViewDimension::*;
    match dimension {
        D1 => H::D1,
        D2 => H::D2,
        D2Array => H::D2Array,
        Cube => H::Cube,
        CubeArray => H::CubeArray,
        D3 => H::D3,
    }
}

pub fn map_texture_aspect_flags(aspect: resource::TextureAspectFlags) -> hal::format::Aspects {
    use hal::format::Aspects;
    use crate::resource::TextureAspectFlags as Taf;

    let mut flags = Aspects::empty();
    if aspect.contains(Taf::COLOR) {
        flags |= Aspects::COLOR;
    }
    if aspect.contains(Taf::DEPTH) {
        flags |= Aspects::DEPTH;
    }
    if aspect.contains(Taf::STENCIL) {
        flags |= Aspects::STENCIL;
    }
    flags
}

pub fn map_buffer_state(usage: resource::BufferUsageFlags) -> hal::buffer::State {
    use hal::buffer::Access as A;
    use crate::resource::BufferUsageFlags as W;

    let mut access = A::empty();
    if usage.contains(W::TRANSFER_SRC) {
        access |= A::TRANSFER_READ;
    }
    if usage.contains(W::TRANSFER_DST) {
        access |= A::TRANSFER_WRITE;
    }
    if usage.contains(W::INDEX) {
        access |= A::INDEX_BUFFER_READ;
    }
    if usage.contains(W::VERTEX) {
        access |= A::VERTEX_BUFFER_READ;
    }
    if usage.contains(W::UNIFORM) {
        access |= A::CONSTANT_BUFFER_READ | A::SHADER_READ;
    }
    if usage.contains(W::STORAGE) {
        access |= A::SHADER_WRITE;
    }

    access
}

pub fn map_texture_state(
    usage: resource::TextureUsageFlags,
    aspects: hal::format::Aspects,
) -> hal::image::State {
    use hal::image::{Access as A, Layout as L};
    use crate::resource::TextureUsageFlags as W;

    let is_color = aspects.contains(hal::format::Aspects::COLOR);
    let layout = match usage {
        W::UNINITIALIZED => return (A::empty(), L::Undefined),
        W::TRANSFER_SRC => L::TransferSrcOptimal,
        W::TRANSFER_DST => L::TransferDstOptimal,
        W::SAMPLED => L::ShaderReadOnlyOptimal,
        W::OUTPUT_ATTACHMENT if is_color => L::ColorAttachmentOptimal,
        W::OUTPUT_ATTACHMENT => L::DepthStencilAttachmentOptimal, //TODO: read-only depth/stencil
        _ => L::General,
    };

    let mut access = A::empty();
    if usage.contains(W::TRANSFER_SRC) {
        access |= A::TRANSFER_READ;
    }
    if usage.contains(W::TRANSFER_DST) {
        access |= A::TRANSFER_WRITE;
    }
    if usage.contains(W::SAMPLED) {
        access |= A::SHADER_READ;
    }
    if usage.contains(W::STORAGE) {
        access |= A::SHADER_WRITE;
    }
    if usage.contains(W::OUTPUT_ATTACHMENT) {
        //TODO: read-only attachments
        access |= if is_color {
            A::COLOR_ATTACHMENT_WRITE
        } else {
            A::DEPTH_STENCIL_ATTACHMENT_WRITE
        };
    }

    (access, layout)
}

pub fn map_load_store_ops(
    load: command::LoadOp,
    store: command::StoreOp,
) -> hal::pass::AttachmentOps {
    hal::pass::AttachmentOps {
        load: match load {
            command::LoadOp::Clear => hal::pass::AttachmentLoadOp::Clear,
            command::LoadOp::Load => hal::pass::AttachmentLoadOp::Load,
        },
        store: match store {
            command::StoreOp::Store => hal::pass::AttachmentStoreOp::Store,
        },
    }
}

pub fn map_color(color: Color) -> hal::pso::ColorValue {
    [color.r, color.g, color.b, color.a]
}

pub fn map_filter(filter: resource::FilterMode) -> hal::image::Filter {
    match filter {
        resource::FilterMode::Nearest => hal::image::Filter::Nearest,
        resource::FilterMode::Linear => hal::image::Filter::Linear,
    }
}

pub fn map_wrap(address: resource::AddressMode) -> hal::image::WrapMode {
    use hal::image::WrapMode as W;
    use crate::resource::AddressMode as Am;
    match address {
        Am::ClampToEdge => W::Clamp,
        Am::Repeat => W::Tile,
        Am::MirrorRepeat => {
            warn!("MirrorRepeat isn't supported yet");
            W::Tile
        }
        Am::ClampToBorderColor => W::Border,
    }
}