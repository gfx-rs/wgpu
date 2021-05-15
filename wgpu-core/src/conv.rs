/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{LoadOp, PassChannel, StoreOp},
    pipeline::ColorStateError,
    resource, PrivateFeatures,
};

use std::convert::TryInto;

pub fn map_adapter_info(
    info: hal::adapter::AdapterInfo,
    backend: wgt::Backend,
) -> wgt::AdapterInfo {
    use hal::adapter::DeviceType as Dt;

    wgt::AdapterInfo {
        name: info.name,
        vendor: info.vendor,
        device: info.device,
        device_type: match info.device_type {
            Dt::Other => wgt::DeviceType::Other,
            Dt::IntegratedGpu => wgt::DeviceType::IntegratedGpu,
            Dt::DiscreteGpu => wgt::DeviceType::DiscreteGpu,
            Dt::VirtualGpu => wgt::DeviceType::VirtualGpu,
            Dt::Cpu => wgt::DeviceType::Cpu,
        },
        backend,
    }
}

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
    if usage.contains(W::STORAGE) {
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
    if usage.contains(W::RENDER_ATTACHMENT) {
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

pub fn map_binding_type(binding: &wgt::BindGroupLayoutEntry) -> hal::pso::DescriptorType {
    use hal::pso;
    use wgt::BindingType as Bt;
    match binding.ty {
        Bt::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size: _,
        } => pso::DescriptorType::Buffer {
            ty: match ty {
                wgt::BufferBindingType::Uniform => pso::BufferDescriptorType::Uniform,
                wgt::BufferBindingType::Storage { read_only } => {
                    pso::BufferDescriptorType::Storage { read_only }
                }
            },
            format: pso::BufferDescriptorFormat::Structured {
                dynamic_offset: has_dynamic_offset,
            },
        },
        Bt::Sampler { .. } => pso::DescriptorType::Sampler,
        Bt::Texture { .. } => pso::DescriptorType::Image {
            ty: pso::ImageDescriptorType::Sampled {
                with_sampler: false,
            },
        },
        Bt::StorageTexture { access, .. } => pso::DescriptorType::Image {
            ty: pso::ImageDescriptorType::Storage {
                read_only: match access {
                    wgt::StorageTextureAccess::ReadOnly => true,
                    _ => false,
                },
            },
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

pub fn map_hal_flags_to_shader_stage(
    shader_stage_flags: hal::pso::ShaderStageFlags,
) -> wgt::ShaderStage {
    use hal::pso::ShaderStageFlags as H;
    use wgt::ShaderStage as Ss;

    let mut value = Ss::empty();
    if shader_stage_flags.contains(H::VERTEX) {
        value |= Ss::VERTEX;
    }
    if shader_stage_flags.contains(H::FRAGMENT) {
        value |= Ss::FRAGMENT;
    }
    if shader_stage_flags.contains(H::COMPUTE) {
        value |= Ss::COMPUTE;
    }
    value
}

pub fn map_extent(extent: &wgt::Extent3d, dim: wgt::TextureDimension) -> hal::image::Extent {
    hal::image::Extent {
        width: extent.width,
        height: extent.height,
        depth: match dim {
            wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => 1,
            wgt::TextureDimension::D3 => extent.depth_or_array_layers,
        },
    }
}

pub fn map_primitive_topology(primitive_topology: wgt::PrimitiveTopology) -> hal::pso::Primitive {
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

pub fn map_color_target_state(
    desc: &wgt::ColorTargetState,
) -> Result<hal::pso::ColorBlendDesc, ColorStateError> {
    let color_mask = desc.write_mask;
    let blend = desc
        .blend
        .as_ref()
        .map(|bs| {
            Ok(hal::pso::BlendState {
                color: map_blend_component(&bs.color)?,
                alpha: map_blend_component(&bs.alpha)?,
            })
        })
        .transpose()?;
    Ok(hal::pso::ColorBlendDesc {
        mask: map_color_write_flags(color_mask),
        blend,
    })
}

fn map_color_write_flags(flags: wgt::ColorWrite) -> hal::pso::ColorMask {
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

fn map_blend_component(
    component: &wgt::BlendComponent,
) -> Result<hal::pso::BlendOp, ColorStateError> {
    use hal::pso::BlendOp as H;
    use wgt::BlendOperation as Bo;
    Ok(match *component {
        wgt::BlendComponent {
            operation: Bo::Add,
            src_factor,
            dst_factor,
        } => H::Add {
            src: map_blend_factor(src_factor),
            dst: map_blend_factor(dst_factor),
        },
        wgt::BlendComponent {
            operation: Bo::Subtract,
            src_factor,
            dst_factor,
        } => H::Sub {
            src: map_blend_factor(src_factor),
            dst: map_blend_factor(dst_factor),
        },
        wgt::BlendComponent {
            operation: Bo::ReverseSubtract,
            src_factor,
            dst_factor,
        } => H::RevSub {
            src: map_blend_factor(src_factor),
            dst: map_blend_factor(dst_factor),
        },
        wgt::BlendComponent {
            operation: Bo::Min,
            src_factor: wgt::BlendFactor::One,
            dst_factor: wgt::BlendFactor::One,
        } => H::Min,
        wgt::BlendComponent {
            operation: Bo::Max,
            src_factor: wgt::BlendFactor::One,
            dst_factor: wgt::BlendFactor::One,
        } => H::Max,
        _ => return Err(ColorStateError::InvalidMinMaxBlendFactors(*component)),
    })
}

fn map_blend_factor(blend_factor: wgt::BlendFactor) -> hal::pso::Factor {
    use hal::pso::Factor as H;
    use wgt::BlendFactor as Bf;
    match blend_factor {
        Bf::Zero => H::Zero,
        Bf::One => H::One,
        Bf::Src => H::SrcColor,
        Bf::OneMinusSrc => H::OneMinusSrcColor,
        Bf::SrcAlpha => H::SrcAlpha,
        Bf::OneMinusSrcAlpha => H::OneMinusSrcAlpha,
        Bf::Dst => H::DstColor,
        Bf::OneMinusDst => H::OneMinusDstColor,
        Bf::DstAlpha => H::DstAlpha,
        Bf::OneMinusDstAlpha => H::OneMinusDstAlpha,
        Bf::SrcAlphaSaturated => H::SrcAlphaSaturate,
        Bf::Constant => H::ConstColor,
        Bf::OneMinusConstant => H::OneMinusConstColor,
    }
}

pub fn map_depth_stencil_state(desc: &wgt::DepthStencilState) -> hal::pso::DepthStencilDesc {
    hal::pso::DepthStencilDesc {
        depth: if desc.is_depth_enabled() {
            Some(hal::pso::DepthTest {
                fun: map_compare_function(desc.depth_compare),
                write: desc.depth_write_enabled,
            })
        } else {
            None
        },
        depth_bounds: false, // TODO
        stencil: if desc.stencil.is_enabled() {
            let s = &desc.stencil;
            Some(hal::pso::StencilTest {
                faces: hal::pso::Sided {
                    front: map_stencil_face(&s.front),
                    back: map_stencil_face(&s.back),
                },
                read_masks: hal::pso::State::Static(hal::pso::Sided::new(s.read_mask)),
                write_masks: hal::pso::State::Static(hal::pso::Sided::new(s.write_mask)),
                reference_values: if s.needs_ref_value() {
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

fn map_stencil_face(stencil_state_face_desc: &wgt::StencilFaceState) -> hal::pso::StencilFace {
    hal::pso::StencilFace {
        fun: map_compare_function(stencil_state_face_desc.compare),
        op_fail: map_stencil_operation(stencil_state_face_desc.fail_op),
        op_depth_fail: map_stencil_operation(stencil_state_face_desc.depth_fail_op),
        op_pass: map_stencil_operation(stencil_state_face_desc.pass_op),
    }
}

pub fn map_compare_function(compare_function: wgt::CompareFunction) -> hal::pso::Comparison {
    use hal::pso::Comparison as H;
    use wgt::CompareFunction as Cf;
    match compare_function {
        Cf::Never => H::Never,
        Cf::Less => H::Less,
        Cf::Equal => H::Equal,
        Cf::LessEqual => H::LessEqual,
        Cf::Greater => H::Greater,
        Cf::NotEqual => H::NotEqual,
        Cf::GreaterEqual => H::GreaterEqual,
        Cf::Always => H::Always,
    }
}

fn map_stencil_operation(stencil_operation: wgt::StencilOperation) -> hal::pso::StencilOp {
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
    texture_format: wgt::TextureFormat,
    private_features: PrivateFeatures,
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
            if private_features.texture_d24 {
                H::X8D24Unorm
            } else {
                H::D32Sfloat
            }
        }
        Tf::Depth24PlusStencil8 => {
            if private_features.texture_d24_s8 {
                H::D24UnormS8Uint
            } else {
                H::D32SfloatS8Uint
            }
        }

        // BCn compressed formats
        Tf::Bc1RgbaUnorm => H::Bc1RgbaUnorm,
        Tf::Bc1RgbaUnormSrgb => H::Bc1RgbaSrgb,
        Tf::Bc2RgbaUnorm => H::Bc2Unorm,
        Tf::Bc2RgbaUnormSrgb => H::Bc2Srgb,
        Tf::Bc3RgbaUnorm => H::Bc3Unorm,
        Tf::Bc3RgbaUnormSrgb => H::Bc3Srgb,
        Tf::Bc4RUnorm => H::Bc4Unorm,
        Tf::Bc4RSnorm => H::Bc4Snorm,
        Tf::Bc5RgUnorm => H::Bc5Unorm,
        Tf::Bc5RgSnorm => H::Bc5Snorm,
        Tf::Bc6hRgbSfloat => H::Bc6hSfloat,
        Tf::Bc6hRgbUfloat => H::Bc6hUfloat,
        Tf::Bc7RgbaUnorm => H::Bc7Unorm,
        Tf::Bc7RgbaUnormSrgb => H::Bc7Srgb,

        // ETC compressed formats
        Tf::Etc2RgbUnorm => H::Etc2R8g8b8Unorm,
        Tf::Etc2RgbUnormSrgb => H::Etc2R8g8b8Srgb,
        Tf::Etc2RgbA1Unorm => H::Etc2R8g8b8a1Unorm,
        Tf::Etc2RgbA1UnormSrgb => H::Etc2R8g8b8a1Srgb,
        Tf::Etc2RgbA8Unorm => H::Etc2R8g8b8a8Unorm,
        Tf::Etc2RgbA8UnormSrgb => H::Etc2R8g8b8a8Unorm,
        Tf::EacRUnorm => H::EacR11Unorm,
        Tf::EacRSnorm => H::EacR11Snorm,
        Tf::EtcRgUnorm => H::EacR11g11Unorm,
        Tf::EtcRgSnorm => H::EacR11g11Snorm,

        // ASTC compressed formats
        Tf::Astc4x4RgbaUnorm => H::Astc4x4Srgb,
        Tf::Astc4x4RgbaUnormSrgb => H::Astc4x4Srgb,
        Tf::Astc5x4RgbaUnorm => H::Astc5x4Unorm,
        Tf::Astc5x4RgbaUnormSrgb => H::Astc5x4Srgb,
        Tf::Astc5x5RgbaUnorm => H::Astc5x5Unorm,
        Tf::Astc5x5RgbaUnormSrgb => H::Astc5x5Srgb,
        Tf::Astc6x5RgbaUnorm => H::Astc6x5Unorm,
        Tf::Astc6x5RgbaUnormSrgb => H::Astc6x5Srgb,
        Tf::Astc6x6RgbaUnorm => H::Astc6x6Unorm,
        Tf::Astc6x6RgbaUnormSrgb => H::Astc6x6Srgb,
        Tf::Astc8x5RgbaUnorm => H::Astc8x5Unorm,
        Tf::Astc8x5RgbaUnormSrgb => H::Astc8x5Srgb,
        Tf::Astc8x6RgbaUnorm => H::Astc8x6Unorm,
        Tf::Astc8x6RgbaUnormSrgb => H::Astc8x6Srgb,
        Tf::Astc10x5RgbaUnorm => H::Astc10x5Unorm,
        Tf::Astc10x5RgbaUnormSrgb => H::Astc10x5Srgb,
        Tf::Astc10x6RgbaUnorm => H::Astc10x6Unorm,
        Tf::Astc10x6RgbaUnormSrgb => H::Astc10x6Srgb,
        Tf::Astc8x8RgbaUnorm => H::Astc8x8Unorm,
        Tf::Astc8x8RgbaUnormSrgb => H::Astc8x8Srgb,
        Tf::Astc10x8RgbaUnorm => H::Astc10x8Unorm,
        Tf::Astc10x8RgbaUnormSrgb => H::Astc10x8Srgb,
        Tf::Astc10x10RgbaUnorm => H::Astc10x10Unorm,
        Tf::Astc10x10RgbaUnormSrgb => H::Astc10x10Srgb,
        Tf::Astc12x10RgbaUnorm => H::Astc12x10Unorm,
        Tf::Astc12x10RgbaUnormSrgb => H::Astc12x10Srgb,
        Tf::Astc12x12RgbaUnorm => H::Astc12x12Unorm,
        Tf::Astc12x12RgbaUnormSrgb => H::Astc12x12Srgb,
    }
}

pub fn map_vertex_format(vertex_format: wgt::VertexFormat) -> hal::format::Format {
    use hal::format::Format as H;
    use wgt::VertexFormat as Vf;
    match vertex_format {
        Vf::Uint8x2 => H::Rg8Uint,
        Vf::Uint8x4 => H::Rgba8Uint,
        Vf::Sint8x2 => H::Rg8Sint,
        Vf::Sint8x4 => H::Rgba8Sint,
        Vf::Unorm8x2 => H::Rg8Unorm,
        Vf::Unorm8x4 => H::Rgba8Unorm,
        Vf::Snorm8x2 => H::Rg8Snorm,
        Vf::Snorm8x4 => H::Rgba8Snorm,
        Vf::Uint16x2 => H::Rg16Uint,
        Vf::Uint16x4 => H::Rgba16Uint,
        Vf::Sint16x2 => H::Rg16Sint,
        Vf::Sint16x4 => H::Rgba16Sint,
        Vf::Unorm16x2 => H::Rg16Unorm,
        Vf::Unorm16x4 => H::Rgba16Unorm,
        Vf::Snorm16x2 => H::Rg16Snorm,
        Vf::Snorm16x4 => H::Rgba16Snorm,
        Vf::Float16x2 => H::Rg16Sfloat,
        Vf::Float16x4 => H::Rgba16Sfloat,
        Vf::Float32 => H::R32Sfloat,
        Vf::Float32x2 => H::Rg32Sfloat,
        Vf::Float32x3 => H::Rgb32Sfloat,
        Vf::Float32x4 => H::Rgba32Sfloat,
        Vf::Uint32 => H::R32Uint,
        Vf::Uint32x2 => H::Rg32Uint,
        Vf::Uint32x3 => H::Rgb32Uint,
        Vf::Uint32x4 => H::Rgba32Uint,
        Vf::Sint32 => H::R32Sint,
        Vf::Sint32x2 => H::Rg32Sint,
        Vf::Sint32x3 => H::Rgb32Sint,
        Vf::Sint32x4 => H::Rgba32Sint,
        Vf::Float64 => H::R64Sfloat,
        Vf::Float64x2 => H::Rg64Sfloat,
        Vf::Float64x3 => H::Rgb64Sfloat,
        Vf::Float64x4 => H::Rgba64Sfloat,
    }
}

pub fn is_power_of_two(val: u32) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_valid_copy_src_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth24Plus | Tf::Depth24PlusStencil8 => false,
        _ => true,
    }
}

pub fn is_valid_copy_dst_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth32Float | Tf::Depth24Plus | Tf::Depth24PlusStencil8 => false,
        _ => true,
    }
}

pub fn map_texture_dimension_size(
    dimension: wgt::TextureDimension,
    wgt::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }: wgt::Extent3d,
    sample_size: u32,
    limits: &wgt::Limits,
) -> Result<hal::image::Kind, resource::TextureDimensionError> {
    use hal::image::Kind as H;
    use resource::{TextureDimensionError as Tde, TextureErrorDimension as Ted};
    use wgt::TextureDimension::*;

    let layers = depth_or_array_layers.try_into().unwrap_or(!0);
    let (kind, extent_limits, sample_limit) = match dimension {
        D1 => (
            H::D1(width, layers),
            [
                limits.max_texture_dimension_1d,
                1,
                limits.max_texture_array_layers,
            ],
            1,
        ),
        D2 => (
            H::D2(width, height, layers, sample_size as u8),
            [
                limits.max_texture_dimension_2d,
                limits.max_texture_dimension_2d,
                limits.max_texture_array_layers,
            ],
            32,
        ),
        D3 => (
            H::D3(width, height, depth_or_array_layers),
            [
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
            ],
            1,
        ),
    };

    for (&dim, (&given, &limit)) in [Ted::X, Ted::Y, Ted::Z].iter().zip(
        [width, height, depth_or_array_layers]
            .iter()
            .zip(extent_limits.iter()),
    ) {
        if given == 0 {
            return Err(Tde::Zero(dim));
        }
        if given > limit {
            return Err(Tde::LimitExceeded { dim, given, limit });
        }
    }
    if sample_size == 0 || sample_size > sample_limit || !is_power_of_two(sample_size) {
        return Err(Tde::InvalidSampleCount(sample_size));
    }

    Ok(kind)
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

pub(crate) fn map_buffer_state(usage: resource::BufferUse) -> hal::buffer::State {
    use crate::resource::BufferUse as W;
    use hal::buffer::Access as A;

    let mut access = A::empty();
    if usage.contains(W::MAP_READ) {
        access |= A::HOST_READ;
    }
    if usage.contains(W::MAP_WRITE) {
        access |= A::HOST_WRITE;
    }
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
    if usage.contains(W::STORAGE_LOAD) {
        access |= A::SHADER_READ;
    }
    if usage.contains(W::STORAGE_STORE) {
        access |= A::SHADER_READ | A::SHADER_WRITE;
    }
    if usage.contains(W::INDIRECT) {
        access |= A::INDIRECT_COMMAND_READ;
    }

    access
}

pub(crate) fn map_texture_state(
    usage: resource::TextureUse,
    aspects: hal::format::Aspects,
) -> hal::image::State {
    use crate::resource::TextureUse as W;
    use hal::image::{Access as A, Layout as L};

    let is_color = aspects.contains(hal::format::Aspects::COLOR);
    let layout = match usage {
        W::UNINITIALIZED => return (A::empty(), L::Undefined),
        W::COPY_SRC => L::TransferSrcOptimal,
        W::COPY_DST => L::TransferDstOptimal,
        W::SAMPLED if is_color => L::ShaderReadOnlyOptimal,
        W::ATTACHMENT_READ | W::ATTACHMENT_WRITE if is_color => L::ColorAttachmentOptimal,
        _ if is_color => L::General,
        W::ATTACHMENT_WRITE => L::DepthStencilAttachmentOptimal,
        _ => L::DepthStencilReadOnlyOptimal,
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
    if usage.contains(W::ATTACHMENT_READ) {
        access |= if is_color {
            A::COLOR_ATTACHMENT_READ
        } else {
            A::DEPTH_STENCIL_ATTACHMENT_READ
        };
    }
    if usage.contains(W::ATTACHMENT_WRITE) {
        access |= if is_color {
            A::COLOR_ATTACHMENT_WRITE
        } else {
            A::DEPTH_STENCIL_ATTACHMENT_WRITE
        };
    }
    if usage.contains(W::STORAGE_LOAD) {
        access |= A::SHADER_READ;
    }
    if usage.contains(W::STORAGE_STORE) {
        access |= A::SHADER_WRITE;
    }

    (access, layout)
}

pub fn map_query_type(ty: &wgt::QueryType) -> (hal::query::Type, u32) {
    match *ty {
        wgt::QueryType::PipelineStatistics(pipeline_statistics) => {
            let mut ps = hal::query::PipelineStatistic::empty();
            ps.set(
                hal::query::PipelineStatistic::VERTEX_SHADER_INVOCATIONS,
                pipeline_statistics
                    .contains(wgt::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS),
            );
            ps.set(
                hal::query::PipelineStatistic::CLIPPING_INVOCATIONS,
                pipeline_statistics.contains(wgt::PipelineStatisticsTypes::CLIPPER_INVOCATIONS),
            );
            ps.set(
                hal::query::PipelineStatistic::CLIPPING_PRIMITIVES,
                pipeline_statistics.contains(wgt::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT),
            );
            ps.set(
                hal::query::PipelineStatistic::FRAGMENT_SHADER_INVOCATIONS,
                pipeline_statistics
                    .contains(wgt::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS),
            );
            ps.set(
                hal::query::PipelineStatistic::COMPUTE_SHADER_INVOCATIONS,
                pipeline_statistics
                    .contains(wgt::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS),
            );

            (
                hal::query::Type::PipelineStatistics(ps),
                pipeline_statistics.bits().count_ones(),
            )
        }
        wgt::QueryType::Timestamp => (hal::query::Type::Timestamp, 1),
    }
}

pub fn map_load_store_ops<V>(channel: &PassChannel<V>) -> hal::pass::AttachmentOps {
    hal::pass::AttachmentOps {
        load: match channel.load_op {
            LoadOp::Clear => hal::pass::AttachmentLoadOp::Clear,
            LoadOp::Load => hal::pass::AttachmentLoadOp::Load,
        },
        store: match channel.store_op {
            StoreOp::Clear => hal::pass::AttachmentStoreOp::DontCare, //TODO!
            StoreOp::Store => hal::pass::AttachmentStoreOp::Store,
        },
    }
}

pub fn map_color_f32(color: &wgt::Color) -> hal::pso::ColorValue {
    [
        color.r as f32,
        color.g as f32,
        color.b as f32,
        color.a as f32,
    ]
}
pub fn map_color_i32(color: &wgt::Color) -> [i32; 4] {
    [
        color.r as i32,
        color.g as i32,
        color.b as i32,
        color.a as i32,
    ]
}
pub fn map_color_u32(color: &wgt::Color) -> [u32; 4] {
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
        Am::ClampToBorder => W::Border,
    }
}

pub fn map_primitive_state_to_input_assembler(
    desc: &wgt::PrimitiveState,
) -> hal::pso::InputAssemblerDesc {
    hal::pso::InputAssemblerDesc {
        primitive: map_primitive_topology(desc.topology),
        with_adjacency: false,
        restart_index: desc.strip_index_format.map(map_index_format),
    }
}

pub fn map_primitive_state_to_rasterizer(
    desc: &wgt::PrimitiveState,
    depth_stencil: Option<&wgt::DepthStencilState>,
) -> hal::pso::Rasterizer {
    use hal::pso;
    let depth_bias = match depth_stencil {
        Some(dsd) if dsd.bias.is_enabled() => Some(pso::State::Static(pso::DepthBias {
            const_factor: dsd.bias.constant as f32,
            slope_factor: dsd.bias.slope_scale,
            clamp: dsd.bias.clamp,
        })),
        _ => None,
    };
    pso::Rasterizer {
        depth_clamping: desc.clamp_depth,
        polygon_mode: match desc.polygon_mode {
            wgt::PolygonMode::Fill => pso::PolygonMode::Fill,
            wgt::PolygonMode::Line => pso::PolygonMode::Line,
            wgt::PolygonMode::Point => pso::PolygonMode::Point,
        },
        cull_face: match desc.cull_mode {
            None => pso::Face::empty(),
            Some(wgt::Face::Front) => pso::Face::FRONT,
            Some(wgt::Face::Back) => pso::Face::BACK,
        },
        front_face: match desc.front_face {
            wgt::FrontFace::Ccw => pso::FrontFace::CounterClockwise,
            wgt::FrontFace::Cw => pso::FrontFace::Clockwise,
        },
        depth_bias,
        conservative: desc.conservative,
        line_width: pso::State::Static(1.0),
    }
}

pub fn map_multisample_state(desc: &wgt::MultisampleState) -> hal::pso::Multisampling {
    hal::pso::Multisampling {
        rasterization_samples: desc.count as _,
        sample_shading: None,
        sample_mask: desc.mask,
        alpha_coverage: desc.alpha_to_coverage_enabled,
        alpha_to_one: false,
    }
}

pub fn map_index_format(index_format: wgt::IndexFormat) -> hal::IndexType {
    match index_format {
        wgt::IndexFormat::Uint16 => hal::IndexType::U16,
        wgt::IndexFormat::Uint32 => hal::IndexType::U32,
    }
}

/// Take `value` and round it up to the nearest alignment `alignment`.
///
/// ```text
/// (0, 3) -> 0
/// (1, 3) -> 3
/// (2, 3) -> 3
/// (3, 3) -> 3
/// (4, 3) -> 6
/// ...
pub fn align_up(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}
