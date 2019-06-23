use crate::{binding_model, command, pipeline, resource, Color, Extent3d, Origin3d};

pub fn map_buffer_usage(
    usage: resource::BufferUsage,
) -> (hal::buffer::Usage, hal::memory::Properties) {
    use crate::resource::BufferUsage as W;
    use hal::buffer::Usage as U;
    use hal::memory::Properties as P;

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
    usage: resource::TextureUsage,
    aspects: hal::format::Aspects,
) -> hal::image::Usage {
    use crate::resource::TextureUsage as W;
    use hal::image::Usage as U;

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
    // Note: TextureUsage::Present does not need to be handled explicitly
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
        StorageTexture => H::StorageImage,
        StorageBuffer => H::StorageBuffer,
        UniformBufferDynamic => H::UniformBufferDynamic,
        StorageBufferDynamic => H::StorageBufferDynamic,
    }
}

pub fn map_shader_stage_flags(
    shader_stage_flags: binding_model::ShaderStage,
) -> hal::pso::ShaderStageFlags {
    use crate::binding_model::ShaderStage as Ss;
    use hal::pso::ShaderStageFlags as H;

    let mut value = H::empty();
    if shader_stage_flags.contains(Ss::VERTEX) {
        value |= H::VERTEX;
    }
    if shader_stage_flags.contains(Ss::FRAGMENT) {
        value |= H::FRAGMENT;
    }
    if shader_stage_flags.contains(Ss::COMPUTE) {
        value |= H::COMPUTE;
    }
    value
}

pub fn map_origin(origin: Origin3d) -> hal::image::Offset {
    hal::image::Offset {
        x: origin.x as i32,
        y: origin.y as i32,
        z: origin.z as i32,
    }
}

pub fn map_extent(extent: Extent3d) -> hal::image::Extent {
    hal::image::Extent {
        width: extent.width,
        height: extent.height,
        depth: extent.depth,
    }
}

pub fn map_primitive_topology(primitive_topology: pipeline::PrimitiveTopology) -> hal::Primitive {
    use crate::pipeline::PrimitiveTopology::*;
    use hal::Primitive as H;
    match primitive_topology {
        PointList => H::PointList,
        LineList => H::LineList,
        LineStrip => H::LineStrip,
        TriangleList => H::TriangleList,
        TriangleStrip => H::TriangleStrip,
    }
}

pub fn map_color_state_descriptor(
    desc: &pipeline::ColorStateDescriptor,
) -> hal::pso::ColorBlendDesc {
    let color_mask = desc.write_mask;
    let blend_state = if desc.color_blend != pipeline::BlendDescriptor::REPLACE
        || desc.alpha_blend != pipeline::BlendDescriptor::REPLACE
    {
        hal::pso::BlendState::On {
            color: map_blend_descriptor(&desc.color_blend),
            alpha: map_blend_descriptor(&desc.alpha_blend),
        }
    } else {
        hal::pso::BlendState::Off
    };
    hal::pso::ColorBlendDesc(map_color_write_flags(color_mask), blend_state)
}

fn map_color_write_flags(flags: pipeline::ColorWrite) -> hal::pso::ColorMask {
    use crate::pipeline::ColorWrite as Cw;
    use hal::pso::ColorMask as H;

    let mut value = H::empty();
    if flags.contains(Cw::RED) {
        value |= H::RED;
    }
    if flags.contains(Cw::GREEN) {
        value |= H::GREEN;
    }
    if flags.contains(Cw::BLUE) {
        value |= H::BLUE;
    }
    if flags.contains(Cw::ALPHA) {
        value |= H::ALPHA;
    }
    value
}

fn map_blend_descriptor(blend_desc: &pipeline::BlendDescriptor) -> hal::pso::BlendOp {
    use crate::pipeline::BlendOperation::*;
    use hal::pso::BlendOp as H;
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
    use crate::pipeline::BlendFactor::*;
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

pub fn map_depth_stencil_state_descriptor(
    desc: &pipeline::DepthStencilStateDescriptor,
) -> hal::pso::DepthStencilDesc {
    hal::pso::DepthStencilDesc {
        depth: if desc.depth_write_enabled
            || desc.depth_compare != resource::CompareFunction::Always
        {
            hal::pso::DepthTest::On {
                fun: map_compare_function(desc.depth_compare),
                write: desc.depth_write_enabled,
            }
        } else {
            hal::pso::DepthTest::Off
        },
        depth_bounds: false, // TODO
        stencil: if desc.stencil_read_mask != !0
            || desc.stencil_write_mask != !0
            || desc.stencil_front != pipeline::StencilStateFaceDescriptor::IGNORE
            || desc.stencil_back != pipeline::StencilStateFaceDescriptor::IGNORE
        {
            hal::pso::StencilTest::On {
                front: map_stencil_face(
                    &desc.stencil_front,
                    desc.stencil_read_mask,
                    desc.stencil_write_mask,
                ),
                back: map_stencil_face(
                    &desc.stencil_back,
                    desc.stencil_read_mask,
                    desc.stencil_write_mask,
                ),
            }
        } else {
            hal::pso::StencilTest::Off
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
        op_fail: map_stencil_operation(stencil_state_face_desc.fail_op),
        op_depth_fail: map_stencil_operation(stencil_state_face_desc.depth_fail_op),
        op_pass: map_stencil_operation(stencil_state_face_desc.pass_op),
        reference: hal::pso::State::Static(0), // TODO can this be set?
    }
}

pub fn map_compare_function(compare_function: resource::CompareFunction) -> hal::pso::Comparison {
    use crate::resource::CompareFunction::*;
    use hal::pso::Comparison as H;
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
    use crate::pipeline::StencilOperation::*;
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

pub fn map_texture_format(texture_format: resource::TextureFormat) -> hal::format::Format {
    use crate::resource::TextureFormat::*;
    use hal::format::Format as H;
    match texture_format {
        // Normal 8 bit formats
        R8Unorm => H::R8Unorm,
        R8UnormSrgb => H::R8Srgb,
        R8Snorm => H::R8Snorm,
        R8Uint => H::R8Uint,
        R8Sint => H::R8Sint,

        // Normal 16 bit formats
        R16Unorm => H::R16Unorm,
        R16Snorm => H::R16Snorm,
        R16Uint => H::R16Uint,
        R16Sint => H::R16Sint,
        R16Float => H::R16Sfloat,

        Rg8Unorm => H::Rg8Unorm,
        Rg8UnormSrgb => H::Rg8Srgb,
        Rg8Snorm => H::Rg8Snorm,
        Rg8Uint => H::Rg8Uint,
        Rg8Sint => H::Rg8Sint,

        // Packed 16 bit formats
        B5g6r5Unorm => H::B5g6r5Unorm,

        // Normal 32 bit formats
        R32Uint => H::R32Uint,
        R32Sint => H::R32Sint,
        R32Float => H::R32Sfloat,
        Rg16Unorm => H::Rg16Unorm,
        Rg16Snorm => H::Rg16Snorm,
        Rg16Uint => H::Rg16Uint,
        Rg16Sint => H::Rg16Sint,
        Rg16Float => H::Rg16Sfloat,
        Rgba8Unorm => H::Rgba8Unorm,
        Rgba8UnormSrgb => H::Rgba8Srgb,
        Rgba8Snorm => H::Rgba8Snorm,
        Rgba8Uint => H::Rgba8Uint,
        Rgba8Sint => H::Rgba8Sint,
        Bgra8Unorm => H::Bgra8Unorm,
        Bgra8UnormSrgb => H::Bgra8Srgb,

        // Packed 32 bit formats
        Rgb10a2Unorm => H::A2r10g10b10Unorm,
        Rg11b10Float => H::B10g11r11Ufloat,

        // Normal 64 bit formats
        Rg32Uint => H::Rg32Uint,
        Rg32Sint => H::Rg32Sint,
        Rg32Float => H::Rg32Sfloat,
        Rgba16Unorm => H::Rgba16Unorm,
        Rgba16Snorm => H::Rgba16Snorm,
        Rgba16Uint => H::Rgba16Uint,
        Rgba16Sint => H::Rgba16Sint,
        Rgba16Float => H::Rgba16Sfloat,

        // Normal 128 bit formats
        Rgba32Uint => H::Rgba32Uint,
        Rgba32Sint => H::Rgba32Sint,
        Rgba32Float => H::Rgba32Sfloat,

        // Depth and stencil formats
        D16Unorm => H::D16Unorm,
        D32Float => H::D32Sfloat,
        D24UnormS8Uint => H::D24UnormS8Uint,
        D32FloatS8Uint => H::D32SfloatS8Uint,
    }
}

pub fn map_vertex_format(vertex_format: pipeline::VertexFormat) -> hal::format::Format {
    use crate::pipeline::VertexFormat::*;
    use hal::format::Format as H;
    match vertex_format {
        Uchar2 => H::Rg8Uint,
        Uchar4 => H::Rgba8Uint,
        Char2 => H::Rg8Sint,
        Char4 => H::Rgba8Sint,
        Uchar2Norm => H::Rg8Unorm,
        Uchar4Norm => H::Rgba8Unorm,
        Char2Norm => H::Rg8Snorm,
        Char4Norm => H::Rgba8Snorm,
        Ushort2 => H::Rg16Uint,
        Ushort4 => H::Rgba16Uint,
        Short2 => H::Rg16Sint,
        Short4 => H::Rgba16Sint,
        Ushort2Norm => H::Rg16Unorm,
        Ushort4Norm => H::Rgba16Unorm,
        Short2Norm => H::Rg16Snorm,
        Short4Norm => H::Rgba16Snorm,
        Half2 => H::Rg16Sfloat,
        Half4 => H::Rgba16Sfloat,
        Float => H::R32Sfloat,
        Float2 => H::Rg32Sfloat,
        Float3 => H::Rgb32Sfloat,
        Float4 => H::Rgba32Sfloat,
        Uint => H::R32Uint,
        Uint2 => H::Rg32Uint,
        Uint3 => H::Rgb32Uint,
        Uint4 => H::Rgba32Uint,
        Int => H::R32Sint,
        Int2 => H::Rg32Sint,
        Int3 => H::Rgb32Sint,
        Int4 => H::Rgba32Sint,
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
    array_size: u32,
    sample_size: u32,
) -> hal::image::Kind {
    use crate::resource::TextureDimension::*;
    use hal::image::Kind as H;
    match dimension {
        D1 => {
            assert_eq!(height, 1);
            assert_eq!(depth, 1);
            assert_eq!(sample_size, 1);
            H::D1(width, checked_u32_as_u16(array_size))
        }
        D2 => {
            assert_eq!(depth, 1);
            assert!(sample_size == 1 || sample_size == 2 || sample_size == 4
                || sample_size == 8 || sample_size == 16 || sample_size == 32,
                "Invalid sample_count of {}", sample_size);
            H::D2(width, height, checked_u32_as_u16(array_size), sample_size as u8)
        }
        D3 => {
            assert_eq!(array_size, 1);
            assert_eq!(sample_size, 1);
            H::D3(width, height, depth)
        }
    }
}

#[cfg(feature = "local")]
pub fn map_texture_view_dimension(
    dimension: resource::TextureViewDimension,
) -> hal::image::ViewKind {
    use crate::resource::TextureViewDimension::*;
    use hal::image::ViewKind as H;
    match dimension {
        D1 => H::D1,
        D2 => H::D2,
        D2Array => H::D2Array,
        Cube => H::Cube,
        CubeArray => H::CubeArray,
        D3 => H::D3,
    }
}

#[cfg(feature = "local")]
pub fn map_texture_aspect_flags(aspect: resource::TextureAspectFlags) -> hal::format::Aspects {
    use crate::resource::TextureAspectFlags as Taf;
    use hal::format::Aspects;

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

pub fn map_buffer_state(usage: resource::BufferUsage) -> hal::buffer::State {
    use crate::resource::BufferUsage as W;
    use hal::buffer::Access as A;

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
    usage: resource::TextureUsage,
    aspects: hal::format::Aspects,
) -> hal::image::State {
    use crate::resource::TextureUsage as W;
    use hal::image::{Access as A, Layout as L};

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

pub fn map_color_f32(color: &Color) -> hal::pso::ColorValue {
    [color.r, color.g, color.b, color.a]
}

pub fn map_color_i32(color: &Color) -> [i32; 4] {
    [color.r as i32, color.g as i32, color.b as i32, color.a as i32]
}

pub fn map_color_u32(color: &Color) -> [u32; 4] {
    [color.r as u32, color.g as u32, color.b as u32, color.a as u32]
}

pub fn map_filter(filter: resource::FilterMode) -> hal::image::Filter {
    match filter {
        resource::FilterMode::Nearest => hal::image::Filter::Nearest,
        resource::FilterMode::Linear => hal::image::Filter::Linear,
    }
}

pub fn map_wrap(address: resource::AddressMode) -> hal::image::WrapMode {
    use crate::resource::AddressMode as Am;
    use hal::image::WrapMode as W;
    match address {
        Am::ClampToEdge => W::Clamp,
        Am::Repeat => W::Tile,
        Am::MirrorRepeat => W::Mirror,
    }
}

pub fn map_rasterization_state_descriptor(
    desc: &pipeline::RasterizationStateDescriptor,
) -> hal::pso::Rasterizer {
    hal::pso::Rasterizer {
        depth_clamping: false,
        polygon_mode: hal::pso::PolygonMode::Fill,
        cull_face: match desc.cull_mode {
            pipeline::CullMode::None => hal::pso::Face::empty(),
            pipeline::CullMode::Front => hal::pso::Face::FRONT,
            pipeline::CullMode::Back => hal::pso::Face::BACK,
        },
        front_face: match desc.front_face {
            pipeline::FrontFace::Ccw => hal::pso::FrontFace::CounterClockwise,
            pipeline::FrontFace::Cw => hal::pso::FrontFace::Clockwise,
        },
        depth_bias: if desc.depth_bias != 0
            || desc.depth_bias_slope_scale != 0.0
            || desc.depth_bias_clamp != 0.0
        {
            Some(hal::pso::State::Static(hal::pso::DepthBias {
                const_factor: desc.depth_bias as f32,
                slope_factor: desc.depth_bias_slope_scale,
                clamp: desc.depth_bias_clamp,
            }))
        } else {
            None
        },
        conservative: false,
    }
}

pub fn map_index_format(index_format: pipeline::IndexFormat) -> hal::IndexType {
    match index_format {
        pipeline::IndexFormat::Uint16 => hal::IndexType::U16,
        pipeline::IndexFormat::Uint32 => hal::IndexType::U32,
    }
}
