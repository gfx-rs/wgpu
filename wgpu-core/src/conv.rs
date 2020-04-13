/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{binding_model, Features};
use wgt::{
    BlendDescriptor, BlendFactor, Color, ColorStateDescriptor, ColorWrite, CompareFunction,
    CullMode, DepthStencilStateDescriptor, Extent3d, FrontFace, IndexFormat, Origin3d,
    PrimitiveTopology, RasterizationStateDescriptor, StencilOperation, StencilStateFaceDescriptor,
    TextureFormat, VertexFormat,
};

pub fn map_buffer_usage(usage: wgt::BufferUsage) -> (hal::buffer::Usage, hal::memory::Properties) {
    use hal::buffer::Usage as U;
    use hal::memory::Properties as P;
    use wgt::BufferUsage as W;

    let mut hal_memory = P::empty();
    if usage.contains(W::MAP_READ) {
        hal_memory |= P::CPU_VISIBLE | P::CPU_CACHED;
    }
    if usage.contains(W::MAP_WRITE) {
        hal_memory |= P::CPU_VISIBLE;
    }

    let mut hal_usage = U::empty();
    if usage.contains(W::COPY_SRC) {
        hal_usage |= U::TRANSFER_SRC;
    }
    if usage.contains(W::COPY_DST) {
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
    if usage.intersects(W::STORAGE | W::STORAGE_READ) {
        hal_usage |= U::STORAGE;
    }
    if usage.contains(W::INDIRECT) {
        hal_usage |= U::INDIRECT;
    }

    (hal_usage, hal_memory)
}

pub fn map_texture_usage(
    usage: wgt::TextureUsage,
    aspects: hal::format::Aspects,
) -> hal::image::Usage {
    use hal::image::Usage as U;
    use wgt::TextureUsage as W;

    let mut value = U::empty();
    if usage.contains(W::COPY_SRC) {
        value |= U::TRANSFER_SRC;
    }
    if usage.contains(W::COPY_DST) {
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

pub fn map_binding_type(binding: &binding_model::BindGroupLayoutEntry) -> hal::pso::DescriptorType {
    use crate::binding_model::BindingType as Bt;
    use hal::pso;
    match binding.ty {
        Bt::UniformBuffer => pso::DescriptorType::Buffer {
            ty: pso::BufferDescriptorType::Uniform,
            format: pso::BufferDescriptorFormat::Structured {
                dynamic_offset: binding.has_dynamic_offset,
            },
        },
        Bt::StorageBuffer => pso::DescriptorType::Buffer {
            ty: pso::BufferDescriptorType::Storage { read_only: false },
            format: pso::BufferDescriptorFormat::Structured {
                dynamic_offset: binding.has_dynamic_offset,
            },
        },
        Bt::ReadonlyStorageBuffer => pso::DescriptorType::Buffer {
            ty: pso::BufferDescriptorType::Storage { read_only: true },
            format: pso::BufferDescriptorFormat::Structured {
                dynamic_offset: binding.has_dynamic_offset,
            },
        },
        Bt::Sampler | Bt::ComparisonSampler => pso::DescriptorType::Sampler,
        Bt::SampledTexture => pso::DescriptorType::Image {
            ty: pso::ImageDescriptorType::Sampled {
                with_sampler: false,
            },
        },
        Bt::ReadonlyStorageTexture => pso::DescriptorType::Image {
            ty: pso::ImageDescriptorType::Storage { read_only: true },
        },
        Bt::WriteonlyStorageTexture => pso::DescriptorType::Image {
            ty: pso::ImageDescriptorType::Storage { read_only: false },
        },
    }
}

pub fn map_shader_stage_flags(shader_stage_flags: wgt::ShaderStage) -> hal::pso::ShaderStageFlags {
    use hal::pso::ShaderStageFlags as H;
    use wgt::ShaderStage as Ss;

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

pub fn map_primitive_topology(primitive_topology: PrimitiveTopology) -> hal::pso::Primitive {
    use hal::pso::Primitive as H;
    use wgt::PrimitiveTopology as Pt;
    match primitive_topology {
        Pt::PointList => H::PointList,
        Pt::LineList => H::LineList,
        Pt::LineStrip => H::LineStrip,
        Pt::TriangleList => H::TriangleList,
        Pt::TriangleStrip => H::TriangleStrip,
    }
}

pub fn map_color_state_descriptor(desc: &ColorStateDescriptor) -> hal::pso::ColorBlendDesc {
    let color_mask = desc.write_mask;
    let blend_state = if desc.color_blend != BlendDescriptor::REPLACE
        || desc.alpha_blend != BlendDescriptor::REPLACE
    {
        Some(hal::pso::BlendState {
            color: map_blend_descriptor(&desc.color_blend),
            alpha: map_blend_descriptor(&desc.alpha_blend),
        })
    } else {
        None
    };
    hal::pso::ColorBlendDesc {
        mask: map_color_write_flags(color_mask),
        blend: blend_state,
    }
}

fn map_color_write_flags(flags: ColorWrite) -> hal::pso::ColorMask {
    use hal::pso::ColorMask as H;
    use wgt::ColorWrite as Cw;

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

fn map_blend_descriptor(blend_desc: &BlendDescriptor) -> hal::pso::BlendOp {
    use hal::pso::BlendOp as H;
    use wgt::BlendOperation as Bo;
    match blend_desc.operation {
        Bo::Add => H::Add {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        Bo::Subtract => H::Sub {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        Bo::ReverseSubtract => H::RevSub {
            src: map_blend_factor(blend_desc.src_factor),
            dst: map_blend_factor(blend_desc.dst_factor),
        },
        Bo::Min => H::Min,
        Bo::Max => H::Max,
    }
}

fn map_blend_factor(blend_factor: BlendFactor) -> hal::pso::Factor {
    use hal::pso::Factor as H;
    use wgt::BlendFactor as Bf;
    match blend_factor {
        Bf::Zero => H::Zero,
        Bf::One => H::One,
        Bf::SrcColor => H::SrcColor,
        Bf::OneMinusSrcColor => H::OneMinusSrcColor,
        Bf::SrcAlpha => H::SrcAlpha,
        Bf::OneMinusSrcAlpha => H::OneMinusSrcAlpha,
        Bf::DstColor => H::DstColor,
        Bf::OneMinusDstColor => H::OneMinusDstColor,
        Bf::DstAlpha => H::DstAlpha,
        Bf::OneMinusDstAlpha => H::OneMinusDstAlpha,
        Bf::SrcAlphaSaturated => H::SrcAlphaSaturate,
        Bf::BlendColor => H::ConstColor,
        Bf::OneMinusBlendColor => H::OneMinusConstColor,
    }
}

pub fn map_depth_stencil_state_descriptor(
    desc: &DepthStencilStateDescriptor,
) -> hal::pso::DepthStencilDesc {
    hal::pso::DepthStencilDesc {
        depth: if desc.depth_write_enabled || desc.depth_compare != CompareFunction::Always {
            Some(hal::pso::DepthTest {
                fun: map_compare_function(desc.depth_compare)
                    .expect("DepthStencilStateDescriptor has undefined compare function"),
                write: desc.depth_write_enabled,
            })
        } else {
            None
        },
        depth_bounds: false, // TODO
        stencil: if desc.stencil_read_mask != !0
            || desc.stencil_write_mask != !0
            || desc.stencil_front != StencilStateFaceDescriptor::IGNORE
            || desc.stencil_back != StencilStateFaceDescriptor::IGNORE
        {
            Some(hal::pso::StencilTest {
                faces: hal::pso::Sided {
                    front: map_stencil_face(&desc.stencil_front),
                    back: map_stencil_face(&desc.stencil_back),
                },
                read_masks: hal::pso::State::Static(hal::pso::Sided::new(desc.stencil_read_mask)),
                write_masks: hal::pso::State::Static(hal::pso::Sided::new(desc.stencil_write_mask)),
                reference_values: if desc.needs_stencil_reference() {
                    hal::pso::State::Dynamic
                } else {
                    hal::pso::State::Static(hal::pso::Sided::new(0))
                },
            })
        } else {
            None
        },
    }
}

fn map_stencil_face(stencil_state_face_desc: &StencilStateFaceDescriptor) -> hal::pso::StencilFace {
    hal::pso::StencilFace {
        fun: map_compare_function(stencil_state_face_desc.compare)
            .expect("StencilStateFaceDescriptor has undefined compare function"),
        op_fail: map_stencil_operation(stencil_state_face_desc.fail_op),
        op_depth_fail: map_stencil_operation(stencil_state_face_desc.depth_fail_op),
        op_pass: map_stencil_operation(stencil_state_face_desc.pass_op),
    }
}

pub fn map_compare_function(compare_function: CompareFunction) -> Option<hal::pso::Comparison> {
    use hal::pso::Comparison as H;
    use wgt::CompareFunction as Cf;
    match compare_function {
        Cf::Undefined => None,
        Cf::Never => Some(H::Never),
        Cf::Less => Some(H::Less),
        Cf::Equal => Some(H::Equal),
        Cf::LessEqual => Some(H::LessEqual),
        Cf::Greater => Some(H::Greater),
        Cf::NotEqual => Some(H::NotEqual),
        Cf::GreaterEqual => Some(H::GreaterEqual),
        Cf::Always => Some(H::Always),
    }
}

fn map_stencil_operation(stencil_operation: StencilOperation) -> hal::pso::StencilOp {
    use hal::pso::StencilOp as H;
    use wgt::StencilOperation as So;
    match stencil_operation {
        So::Keep => H::Keep,
        So::Zero => H::Zero,
        So::Replace => H::Replace,
        So::Invert => H::Invert,
        So::IncrementClamp => H::IncrementClamp,
        So::DecrementClamp => H::DecrementClamp,
        So::IncrementWrap => H::IncrementWrap,
        So::DecrementWrap => H::DecrementWrap,
    }
}

pub(crate) fn map_texture_format(
    texture_format: TextureFormat,
    features: Features,
) -> hal::format::Format {
    use hal::format::Format as H;
    use wgt::TextureFormat as Tf;
    match texture_format {
        // Normal 8 bit formats
        Tf::R8Unorm => H::R8Unorm,
        Tf::R8Snorm => H::R8Snorm,
        Tf::R8Uint => H::R8Uint,
        Tf::R8Sint => H::R8Sint,

        // Normal 16 bit formats
        Tf::R16Uint => H::R16Uint,
        Tf::R16Sint => H::R16Sint,
        Tf::R16Float => H::R16Sfloat,
        Tf::Rg8Unorm => H::Rg8Unorm,
        Tf::Rg8Snorm => H::Rg8Snorm,
        Tf::Rg8Uint => H::Rg8Uint,
        Tf::Rg8Sint => H::Rg8Sint,

        // Normal 32 bit formats
        Tf::R32Uint => H::R32Uint,
        Tf::R32Sint => H::R32Sint,
        Tf::R32Float => H::R32Sfloat,
        Tf::Rg16Uint => H::Rg16Uint,
        Tf::Rg16Sint => H::Rg16Sint,
        Tf::Rg16Float => H::Rg16Sfloat,
        Tf::Rgba8Unorm => H::Rgba8Unorm,
        Tf::Rgba8UnormSrgb => H::Rgba8Srgb,
        Tf::Rgba8Snorm => H::Rgba8Snorm,
        Tf::Rgba8Uint => H::Rgba8Uint,
        Tf::Rgba8Sint => H::Rgba8Sint,
        Tf::Bgra8Unorm => H::Bgra8Unorm,
        Tf::Bgra8UnormSrgb => H::Bgra8Srgb,

        // Packed 32 bit formats
        Tf::Rgb10a2Unorm => H::A2r10g10b10Unorm,
        Tf::Rg11b10Float => H::B10g11r11Ufloat,

        // Normal 64 bit formats
        Tf::Rg32Uint => H::Rg32Uint,
        Tf::Rg32Sint => H::Rg32Sint,
        Tf::Rg32Float => H::Rg32Sfloat,
        Tf::Rgba16Uint => H::Rgba16Uint,
        Tf::Rgba16Sint => H::Rgba16Sint,
        Tf::Rgba16Float => H::Rgba16Sfloat,

        // Normal 128 bit formats
        Tf::Rgba32Uint => H::Rgba32Uint,
        Tf::Rgba32Sint => H::Rgba32Sint,
        Tf::Rgba32Float => H::Rgba32Sfloat,

        // Depth and stencil formats
        Tf::Depth32Float => H::D32Sfloat,
        Tf::Depth24Plus => {
            if features.supports_texture_d24_s8 {
                H::D24UnormS8Uint
            } else {
                H::D32Sfloat
            }
        }
        Tf::Depth24PlusStencil8 => {
            if features.supports_texture_d24_s8 {
                H::D24UnormS8Uint
            } else {
                H::D32SfloatS8Uint
            }
        }
    }
}

pub fn map_vertex_format(vertex_format: VertexFormat) -> hal::format::Format {
    use hal::format::Format as H;
    use wgt::VertexFormat as Vf;
    match vertex_format {
        Vf::Uchar2 => H::Rg8Uint,
        Vf::Uchar4 => H::Rgba8Uint,
        Vf::Char2 => H::Rg8Sint,
        Vf::Char4 => H::Rgba8Sint,
        Vf::Uchar2Norm => H::Rg8Unorm,
        Vf::Uchar4Norm => H::Rgba8Unorm,
        Vf::Char2Norm => H::Rg8Snorm,
        Vf::Char4Norm => H::Rgba8Snorm,
        Vf::Ushort2 => H::Rg16Uint,
        Vf::Ushort4 => H::Rgba16Uint,
        Vf::Short2 => H::Rg16Sint,
        Vf::Short4 => H::Rgba16Sint,
        Vf::Ushort2Norm => H::Rg16Unorm,
        Vf::Ushort4Norm => H::Rgba16Unorm,
        Vf::Short2Norm => H::Rg16Snorm,
        Vf::Short4Norm => H::Rgba16Snorm,
        Vf::Half2 => H::Rg16Sfloat,
        Vf::Half4 => H::Rgba16Sfloat,
        Vf::Float => H::R32Sfloat,
        Vf::Float2 => H::Rg32Sfloat,
        Vf::Float3 => H::Rgb32Sfloat,
        Vf::Float4 => H::Rgba32Sfloat,
        Vf::Uint => H::R32Uint,
        Vf::Uint2 => H::Rg32Uint,
        Vf::Uint3 => H::Rgb32Uint,
        Vf::Uint4 => H::Rgba32Uint,
        Vf::Int => H::R32Sint,
        Vf::Int2 => H::Rg32Sint,
        Vf::Int3 => H::Rgb32Sint,
        Vf::Int4 => H::Rgba32Sint,
    }
}

fn checked_u32_as_u16(value: u32) -> u16 {
    assert!(value <= ::std::u16::MAX as u32);
    value as u16
}

pub fn map_texture_dimension_size(
    dimension: wgt::TextureDimension,
    Extent3d {
        width,
        height,
        depth,
    }: Extent3d,
    array_size: u32,
    sample_size: u32,
) -> hal::image::Kind {
    use hal::image::Kind as H;
    use wgt::TextureDimension::*;
    match dimension {
        D1 => {
            assert_eq!(height, 1);
            assert_eq!(depth, 1);
            assert_eq!(sample_size, 1);
            H::D1(width, checked_u32_as_u16(array_size))
        }
        D2 => {
            assert_eq!(depth, 1);
            assert!(
                sample_size == 1
                    || sample_size == 2
                    || sample_size == 4
                    || sample_size == 8
                    || sample_size == 16
                    || sample_size == 32,
                "Invalid sample_count of {}",
                sample_size
            );
            H::D2(
                width,
                height,
                checked_u32_as_u16(array_size),
                sample_size as u8,
            )
        }
        D3 => {
            assert_eq!(array_size, 1);
            assert_eq!(sample_size, 1);
            H::D3(width, height, depth)
        }
    }
}

pub fn map_texture_view_dimension(dimension: wgt::TextureViewDimension) -> hal::image::ViewKind {
    use hal::image::ViewKind as H;
    use wgt::TextureViewDimension::*;
    match dimension {
        D1 => H::D1,
        D2 => H::D2,
        D2Array => H::D2Array,
        Cube => H::Cube,
        CubeArray => H::CubeArray,
        D3 => H::D3,
    }
}

pub fn map_buffer_state(usage: wgt::BufferUsage) -> hal::buffer::State {
    use hal::buffer::Access as A;
    use wgt::BufferUsage as W;

    let mut access = A::empty();
    if usage.contains(W::COPY_SRC) {
        access |= A::TRANSFER_READ;
    }
    if usage.contains(W::COPY_DST) {
        access |= A::TRANSFER_WRITE;
    }
    if usage.contains(W::INDEX) {
        access |= A::INDEX_BUFFER_READ;
    }
    if usage.contains(W::VERTEX) {
        access |= A::VERTEX_BUFFER_READ;
    }
    if usage.contains(W::UNIFORM) {
        access |= A::UNIFORM_READ | A::SHADER_READ;
    }
    if usage.contains(W::STORAGE_READ) {
        access |= A::SHADER_READ;
    }
    if usage.contains(W::STORAGE) {
        access |= A::SHADER_WRITE;
    }

    access
}

pub fn map_texture_state(
    usage: wgt::TextureUsage,
    aspects: hal::format::Aspects,
) -> hal::image::State {
    use hal::image::{Access as A, Layout as L};
    use wgt::TextureUsage as W;

    let is_color = aspects.contains(hal::format::Aspects::COLOR);
    let layout = match usage {
        W::UNINITIALIZED => return (A::empty(), L::Undefined),
        W::COPY_SRC => L::TransferSrcOptimal,
        W::COPY_DST => L::TransferDstOptimal,
        W::SAMPLED => L::ShaderReadOnlyOptimal,
        W::OUTPUT_ATTACHMENT if is_color => L::ColorAttachmentOptimal,
        W::OUTPUT_ATTACHMENT => L::DepthStencilAttachmentOptimal, //TODO: read-only depth/stencil
        _ => L::General,
    };

    let mut access = A::empty();
    if usage.contains(W::COPY_SRC) {
        access |= A::TRANSFER_READ;
    }
    if usage.contains(W::COPY_DST) {
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

pub fn map_load_store_ops(load: wgt::LoadOp, store: wgt::StoreOp) -> hal::pass::AttachmentOps {
    hal::pass::AttachmentOps {
        load: match load {
            wgt::LoadOp::Clear => hal::pass::AttachmentLoadOp::Clear,
            wgt::LoadOp::Load => hal::pass::AttachmentLoadOp::Load,
        },
        store: match store {
            wgt::StoreOp::Clear => hal::pass::AttachmentStoreOp::DontCare, //TODO!
            wgt::StoreOp::Store => hal::pass::AttachmentStoreOp::Store,
        },
    }
}

pub fn map_color_f32(color: &Color) -> hal::pso::ColorValue {
    [
        color.r as f32,
        color.g as f32,
        color.b as f32,
        color.a as f32,
    ]
}
pub fn map_color_i32(color: &Color) -> [i32; 4] {
    [
        color.r as i32,
        color.g as i32,
        color.b as i32,
        color.a as i32,
    ]
}
pub fn map_color_u32(color: &Color) -> [u32; 4] {
    [
        color.r as u32,
        color.g as u32,
        color.b as u32,
        color.a as u32,
    ]
}

pub fn map_filter(filter: wgt::FilterMode) -> hal::image::Filter {
    match filter {
        wgt::FilterMode::Nearest => hal::image::Filter::Nearest,
        wgt::FilterMode::Linear => hal::image::Filter::Linear,
    }
}

pub fn map_wrap(address: wgt::AddressMode) -> hal::image::WrapMode {
    use hal::image::WrapMode as W;
    use wgt::AddressMode as Am;
    match address {
        Am::ClampToEdge => W::Clamp,
        Am::Repeat => W::Tile,
        Am::MirrorRepeat => W::Mirror,
    }
}

pub fn map_rasterization_state_descriptor(
    desc: &RasterizationStateDescriptor,
) -> hal::pso::Rasterizer {
    use hal::pso;
    pso::Rasterizer {
        depth_clamping: false,
        polygon_mode: pso::PolygonMode::Fill,
        cull_face: match desc.cull_mode {
            CullMode::None => pso::Face::empty(),
            CullMode::Front => pso::Face::FRONT,
            CullMode::Back => pso::Face::BACK,
        },
        front_face: match desc.front_face {
            FrontFace::Ccw => pso::FrontFace::CounterClockwise,
            FrontFace::Cw => pso::FrontFace::Clockwise,
        },
        depth_bias: if desc.depth_bias != 0
            || desc.depth_bias_slope_scale != 0.0
            || desc.depth_bias_clamp != 0.0
        {
            Some(pso::State::Static(pso::DepthBias {
                const_factor: desc.depth_bias as f32,
                slope_factor: desc.depth_bias_slope_scale,
                clamp: desc.depth_bias_clamp,
            }))
        } else {
            None
        },
        conservative: false,
        line_width: pso::State::Static(1.0),
    }
}

pub fn map_index_format(index_format: IndexFormat) -> hal::IndexType {
    match index_format {
        IndexFormat::Uint16 => hal::IndexType::U16,
        IndexFormat::Uint32 => hal::IndexType::U32,
    }
}
