#![allow(clippy::type_complexity)]

mod defined_non_null_js_value;
mod ext_bindings;
mod webgpu_sys;

use js_sys::Promise;
use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    fmt,
    future::Future,
    ops::Range,
    pin::Pin,
    rc::Rc,
    task::{self, Poll},
};
use wasm_bindgen::{prelude::*, JsCast};

use crate::{
    context::{downcast_ref, QueueWriteBuffer},
    CompilationInfo, SurfaceTargetUnsafe, UncapturedErrorHandler,
};

use defined_non_null_js_value::DefinedNonNullJsValue;

// We need to make a wrapper for some of the handle types returned by the web backend to make them
// implement `Send` and `Sync` to match native.
//
// SAFETY: All webgpu handle types in wasm32 are internally a `JsValue`, and `JsValue` is neither
// Send nor Sync.  Currently, wasm32 has no threading support by default, so implementing `Send` or
// `Sync` for a type is harmless. However, nightly Rust supports compiling wasm with experimental
// threading support via `--target-features`. If `wgpu` is being compiled with those features, we do
// not implement `Send` and `Sync` on the webgpu handle types.

#[derive(Clone, Debug)]
pub(crate) struct Sendable<T>(T);
#[cfg(send_sync)]
unsafe impl<T> Send for Sendable<T> {}
#[cfg(send_sync)]
unsafe impl<T> Sync for Sendable<T> {}

pub(crate) struct ContextWebGpu {
    /// `None` if browser does not advertise support for WebGPU.
    gpu: Option<DefinedNonNullJsValue<webgpu_sys::Gpu>>,
}
#[cfg(send_sync)]
unsafe impl Send for ContextWebGpu {}
#[cfg(send_sync)]
unsafe impl Sync for ContextWebGpu {}
#[cfg(send_sync)]
unsafe impl Send for BufferMappedRange {}
#[cfg(send_sync)]
unsafe impl Sync for BufferMappedRange {}

impl fmt::Debug for ContextWebGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextWebGpu")
            .field("type", &"Web")
            .finish()
    }
}

impl crate::Error {
    fn from_js(js_error: js_sys::Object) -> Self {
        let source = Box::<dyn std::error::Error + Send + Sync>::from("<WebGPU Error>");
        if let Some(js_error) = js_error.dyn_ref::<webgpu_sys::GpuValidationError>() {
            crate::Error::Validation {
                source,
                description: js_error.message(),
            }
        } else if js_error.has_type::<webgpu_sys::GpuOutOfMemoryError>() {
            crate::Error::OutOfMemory { source }
        } else {
            panic!("Unexpected error");
        }
    }
}

#[derive(Debug)]
pub struct WebShaderModule {
    module: webgpu_sys::GpuShaderModule,
    compilation_info: WebShaderCompilationInfo,
}

#[derive(Debug, Clone)]
enum WebShaderCompilationInfo {
    /// WGSL shaders get their compilation info from a native WebGPU function.
    /// We need the source to be able to do UTF16 to UTF8 location remapping.
    Wgsl { source: String },
    /// Transformed shaders get their compilation info from the transformer.
    /// Further compilation errors are reported without a span.
    Transformed {
        compilation_info: crate::CompilationInfo,
    },
}

fn map_utf16_to_utf8_offset(utf16_offset: u32, text: &str) -> u32 {
    let mut utf16_i = 0;
    for (utf8_index, c) in text.char_indices() {
        if utf16_i >= utf16_offset {
            return utf8_index as u32;
        }
        utf16_i += c.len_utf16() as u32;
    }
    if utf16_i >= utf16_offset {
        text.len() as u32
    } else {
        log::error!(
            "UTF16 offset {} is out of bounds for string {}",
            utf16_offset,
            text
        );
        u32::MAX
    }
}

impl crate::CompilationMessage {
    fn from_js(
        js_message: webgpu_sys::GpuCompilationMessage,
        compilation_info: &WebShaderCompilationInfo,
    ) -> Self {
        let message_type = match js_message.type_() {
            webgpu_sys::GpuCompilationMessageType::Error => crate::CompilationMessageType::Error,
            webgpu_sys::GpuCompilationMessageType::Warning => {
                crate::CompilationMessageType::Warning
            }
            webgpu_sys::GpuCompilationMessageType::Info => crate::CompilationMessageType::Info,
            _ => crate::CompilationMessageType::Error,
        };
        let utf16_offset = js_message.offset() as u32;
        let utf16_length = js_message.length() as u32;
        let span = match compilation_info {
            WebShaderCompilationInfo::Wgsl { .. } if utf16_offset == 0 && utf16_length == 0 => None,
            WebShaderCompilationInfo::Wgsl { source } => {
                let offset = map_utf16_to_utf8_offset(utf16_offset, source);
                let length = map_utf16_to_utf8_offset(utf16_length, &source[offset as usize..]);
                let line_number = js_message.line_num() as u32; // That's legal, because we're counting lines the same way

                let prefix = &source[..offset as usize];
                let line_start = prefix.rfind('\n').map(|pos| pos + 1).unwrap_or(0) as u32;
                let line_position = offset - line_start + 1; // Counting UTF-8 byte indices

                Some(crate::SourceLocation {
                    offset,
                    length,
                    line_number,
                    line_position,
                })
            }
            WebShaderCompilationInfo::Transformed { .. } => None,
        };

        crate::CompilationMessage {
            message: js_message.message(),
            message_type,
            location: span,
        }
    }
}

// We need to assert that any future we return is Send to match the native API.
//
// This is safe on wasm32 *for now*, but similarly to the unsafe Send impls for the handle type
// wrappers, the full story for threading on wasm32 is still unfolding.

pub(crate) struct MakeSendFuture<F, M> {
    future: F,
    map: M,
}

impl<F: Future, M: Fn(F::Output) -> T, T> Future for MakeSendFuture<F, M> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        // This is safe because we have no Drop implementation to violate the Pin requirements and
        // do not provide any means of moving the inner future.
        unsafe {
            let this = self.get_unchecked_mut();
            match Pin::new_unchecked(&mut this.future).poll(cx) {
                task::Poll::Ready(value) => task::Poll::Ready((this.map)(value)),
                task::Poll::Pending => task::Poll::Pending,
            }
        }
    }
}

impl<F, M> MakeSendFuture<F, M> {
    fn new(future: F, map: M) -> Self {
        Self { future, map }
    }
}

#[cfg(send_sync)]
unsafe impl<F, M> Send for MakeSendFuture<F, M> {}

/// Wraps a future that returns `Option<T>` and adds the ability to immediately
/// return None.
pub(crate) struct OptionFuture<F>(Option<F>);

impl<F: Future<Output = Option<T>>, T> Future for OptionFuture<F> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        // This is safe because we have no Drop implementation to violate the Pin requirements and
        // do not provide any means of moving the inner future.
        unsafe {
            let this = self.get_unchecked_mut();
            match &mut this.0 {
                Some(future) => Pin::new_unchecked(future).poll(cx),
                None => task::Poll::Ready(None),
            }
        }
    }
}

impl<F> OptionFuture<F> {
    fn some(future: F) -> Self {
        Self(Some(future))
    }

    fn none() -> Self {
        Self(None)
    }
}

fn map_texture_format(texture_format: wgt::TextureFormat) -> webgpu_sys::GpuTextureFormat {
    use webgpu_sys::GpuTextureFormat as tf;
    use wgt::TextureFormat;
    match texture_format {
        // 8-bit formats
        TextureFormat::R8Unorm => tf::R8unorm,
        TextureFormat::R8Snorm => tf::R8snorm,
        TextureFormat::R8Uint => tf::R8uint,
        TextureFormat::R8Sint => tf::R8sint,
        // 16-bit formats
        TextureFormat::R16Uint => tf::R16uint,
        TextureFormat::R16Sint => tf::R16sint,
        TextureFormat::R16Float => tf::R16float,
        TextureFormat::Rg8Unorm => tf::Rg8unorm,
        TextureFormat::Rg8Snorm => tf::Rg8snorm,
        TextureFormat::Rg8Uint => tf::Rg8uint,
        TextureFormat::Rg8Sint => tf::Rg8sint,
        // 32-bit formats
        TextureFormat::R32Uint => tf::R32uint,
        TextureFormat::R32Sint => tf::R32sint,
        TextureFormat::R32Float => tf::R32float,
        TextureFormat::Rg16Uint => tf::Rg16uint,
        TextureFormat::Rg16Sint => tf::Rg16sint,
        TextureFormat::Rg16Float => tf::Rg16float,
        TextureFormat::Rgba8Unorm => tf::Rgba8unorm,
        TextureFormat::Rgba8UnormSrgb => tf::Rgba8unormSrgb,
        TextureFormat::Rgba8Snorm => tf::Rgba8snorm,
        TextureFormat::Rgba8Uint => tf::Rgba8uint,
        TextureFormat::Rgba8Sint => tf::Rgba8sint,
        TextureFormat::Bgra8Unorm => tf::Bgra8unorm,
        TextureFormat::Bgra8UnormSrgb => tf::Bgra8unormSrgb,
        // Packed 32-bit formats
        TextureFormat::Rgb9e5Ufloat => tf::Rgb9e5ufloat,
        TextureFormat::Rgb10a2Uint => {
            unimplemented!("Current version of web_sys is missing {texture_format:?}")
        }
        TextureFormat::Rgb10a2Unorm => tf::Rgb10a2unorm,
        TextureFormat::Rg11b10Ufloat => tf::Rg11b10ufloat,
        // 64-bit formats
        TextureFormat::Rg32Uint => tf::Rg32uint,
        TextureFormat::Rg32Sint => tf::Rg32sint,
        TextureFormat::Rg32Float => tf::Rg32float,
        TextureFormat::Rgba16Uint => tf::Rgba16uint,
        TextureFormat::Rgba16Sint => tf::Rgba16sint,
        TextureFormat::Rgba16Float => tf::Rgba16float,
        // 128-bit formats
        TextureFormat::Rgba32Uint => tf::Rgba32uint,
        TextureFormat::Rgba32Sint => tf::Rgba32sint,
        TextureFormat::Rgba32Float => tf::Rgba32float,
        // Depth/stencil formats
        TextureFormat::Stencil8 => tf::Stencil8,
        TextureFormat::Depth16Unorm => tf::Depth16unorm,
        TextureFormat::Depth24Plus => tf::Depth24plus,
        TextureFormat::Depth24PlusStencil8 => tf::Depth24plusStencil8,
        TextureFormat::Depth32Float => tf::Depth32float,
        // "depth32float-stencil8" feature
        TextureFormat::Depth32FloatStencil8 => tf::Depth32floatStencil8,

        TextureFormat::Bc1RgbaUnorm => tf::Bc1RgbaUnorm,
        TextureFormat::Bc1RgbaUnormSrgb => tf::Bc1RgbaUnormSrgb,
        TextureFormat::Bc2RgbaUnorm => tf::Bc2RgbaUnorm,
        TextureFormat::Bc2RgbaUnormSrgb => tf::Bc2RgbaUnormSrgb,
        TextureFormat::Bc3RgbaUnorm => tf::Bc3RgbaUnorm,
        TextureFormat::Bc3RgbaUnormSrgb => tf::Bc3RgbaUnormSrgb,
        TextureFormat::Bc4RUnorm => tf::Bc4RUnorm,
        TextureFormat::Bc4RSnorm => tf::Bc4RSnorm,
        TextureFormat::Bc5RgUnorm => tf::Bc5RgUnorm,
        TextureFormat::Bc5RgSnorm => tf::Bc5RgSnorm,
        TextureFormat::Bc6hRgbUfloat => tf::Bc6hRgbUfloat,
        TextureFormat::Bc6hRgbFloat => tf::Bc6hRgbFloat,
        TextureFormat::Bc7RgbaUnorm => tf::Bc7RgbaUnorm,
        TextureFormat::Bc7RgbaUnormSrgb => tf::Bc7RgbaUnormSrgb,
        TextureFormat::Etc2Rgb8Unorm => tf::Etc2Rgb8unorm,
        TextureFormat::Etc2Rgb8UnormSrgb => tf::Etc2Rgb8unormSrgb,
        TextureFormat::Etc2Rgb8A1Unorm => tf::Etc2Rgb8a1unorm,
        TextureFormat::Etc2Rgb8A1UnormSrgb => tf::Etc2Rgb8a1unormSrgb,
        TextureFormat::Etc2Rgba8Unorm => tf::Etc2Rgba8unorm,
        TextureFormat::Etc2Rgba8UnormSrgb => tf::Etc2Rgba8unormSrgb,
        TextureFormat::EacR11Unorm => tf::EacR11unorm,
        TextureFormat::EacR11Snorm => tf::EacR11snorm,
        TextureFormat::EacRg11Unorm => tf::EacRg11unorm,
        TextureFormat::EacRg11Snorm => tf::EacRg11snorm,
        TextureFormat::Astc { block, channel } => match channel {
            wgt::AstcChannel::Unorm => match block {
                wgt::AstcBlock::B4x4 => tf::Astc4x4Unorm,
                wgt::AstcBlock::B5x4 => tf::Astc5x4Unorm,
                wgt::AstcBlock::B5x5 => tf::Astc5x5Unorm,
                wgt::AstcBlock::B6x5 => tf::Astc6x5Unorm,
                wgt::AstcBlock::B6x6 => tf::Astc6x6Unorm,
                wgt::AstcBlock::B8x5 => tf::Astc8x5Unorm,
                wgt::AstcBlock::B8x6 => tf::Astc8x6Unorm,
                wgt::AstcBlock::B8x8 => tf::Astc8x8Unorm,
                wgt::AstcBlock::B10x5 => tf::Astc10x5Unorm,
                wgt::AstcBlock::B10x6 => tf::Astc10x6Unorm,
                wgt::AstcBlock::B10x8 => tf::Astc10x8Unorm,
                wgt::AstcBlock::B10x10 => tf::Astc10x10Unorm,
                wgt::AstcBlock::B12x10 => tf::Astc12x10Unorm,
                wgt::AstcBlock::B12x12 => tf::Astc12x12Unorm,
            },
            wgt::AstcChannel::UnormSrgb => match block {
                wgt::AstcBlock::B4x4 => tf::Astc4x4UnormSrgb,
                wgt::AstcBlock::B5x4 => tf::Astc5x4UnormSrgb,
                wgt::AstcBlock::B5x5 => tf::Astc5x5UnormSrgb,
                wgt::AstcBlock::B6x5 => tf::Astc6x5UnormSrgb,
                wgt::AstcBlock::B6x6 => tf::Astc6x6UnormSrgb,
                wgt::AstcBlock::B8x5 => tf::Astc8x5UnormSrgb,
                wgt::AstcBlock::B8x6 => tf::Astc8x6UnormSrgb,
                wgt::AstcBlock::B8x8 => tf::Astc8x8UnormSrgb,
                wgt::AstcBlock::B10x5 => tf::Astc10x5UnormSrgb,
                wgt::AstcBlock::B10x6 => tf::Astc10x6UnormSrgb,
                wgt::AstcBlock::B10x8 => tf::Astc10x8UnormSrgb,
                wgt::AstcBlock::B10x10 => tf::Astc10x10UnormSrgb,
                wgt::AstcBlock::B12x10 => tf::Astc12x10UnormSrgb,
                wgt::AstcBlock::B12x12 => tf::Astc12x12UnormSrgb,
            },
            wgt::AstcChannel::Hdr => {
                unimplemented!("Format {texture_format:?} has no WebGPU equivalent")
            }
        },
        _ => unimplemented!("Format {texture_format:?} has no WebGPU equivalent"),
    }
}

fn map_texture_component_type(
    sample_type: wgt::TextureSampleType,
) -> webgpu_sys::GpuTextureSampleType {
    use webgpu_sys::GpuTextureSampleType as ts;
    use wgt::TextureSampleType;
    match sample_type {
        TextureSampleType::Float { filterable: true } => ts::Float,
        TextureSampleType::Float { filterable: false } => ts::UnfilterableFloat,
        TextureSampleType::Sint => ts::Sint,
        TextureSampleType::Uint => ts::Uint,
        TextureSampleType::Depth => ts::Depth,
    }
}

fn map_cull_mode(cull_mode: Option<wgt::Face>) -> webgpu_sys::GpuCullMode {
    use webgpu_sys::GpuCullMode as cm;
    use wgt::Face;
    match cull_mode {
        None => cm::None,
        Some(Face::Front) => cm::Front,
        Some(Face::Back) => cm::Back,
    }
}

fn map_front_face(front_face: wgt::FrontFace) -> webgpu_sys::GpuFrontFace {
    use webgpu_sys::GpuFrontFace as ff;
    use wgt::FrontFace;
    match front_face {
        FrontFace::Ccw => ff::Ccw,
        FrontFace::Cw => ff::Cw,
    }
}

fn map_primitive_state(primitive: &wgt::PrimitiveState) -> webgpu_sys::GpuPrimitiveState {
    use webgpu_sys::GpuPrimitiveTopology as pt;
    use wgt::PrimitiveTopology;

    let mut mapped = webgpu_sys::GpuPrimitiveState::new();
    mapped.cull_mode(map_cull_mode(primitive.cull_mode));
    mapped.front_face(map_front_face(primitive.front_face));

    if let Some(format) = primitive.strip_index_format {
        mapped.strip_index_format(map_index_format(format));
    }

    mapped.topology(match primitive.topology {
        PrimitiveTopology::PointList => pt::PointList,
        PrimitiveTopology::LineList => pt::LineList,
        PrimitiveTopology::LineStrip => pt::LineStrip,
        PrimitiveTopology::TriangleList => pt::TriangleList,
        PrimitiveTopology::TriangleStrip => pt::TriangleStrip,
    });

    //TODO:
    //mapped.unclipped_depth(primitive.unclipped_depth);

    match primitive.polygon_mode {
        wgt::PolygonMode::Fill => {}
        wgt::PolygonMode::Line => panic!(
            "{:?} is not enabled for this backend",
            wgt::Features::POLYGON_MODE_LINE
        ),
        wgt::PolygonMode::Point => panic!(
            "{:?} is not enabled for this backend",
            wgt::Features::POLYGON_MODE_POINT
        ),
    }

    mapped
}

fn map_compare_function(compare_fn: wgt::CompareFunction) -> webgpu_sys::GpuCompareFunction {
    use webgpu_sys::GpuCompareFunction as cf;
    use wgt::CompareFunction;
    match compare_fn {
        CompareFunction::Never => cf::Never,
        CompareFunction::Less => cf::Less,
        CompareFunction::Equal => cf::Equal,
        CompareFunction::LessEqual => cf::LessEqual,
        CompareFunction::Greater => cf::Greater,
        CompareFunction::NotEqual => cf::NotEqual,
        CompareFunction::GreaterEqual => cf::GreaterEqual,
        CompareFunction::Always => cf::Always,
    }
}

fn map_stencil_operation(op: wgt::StencilOperation) -> webgpu_sys::GpuStencilOperation {
    use webgpu_sys::GpuStencilOperation as so;
    use wgt::StencilOperation;
    match op {
        StencilOperation::Keep => so::Keep,
        StencilOperation::Zero => so::Zero,
        StencilOperation::Replace => so::Replace,
        StencilOperation::Invert => so::Invert,
        StencilOperation::IncrementClamp => so::IncrementClamp,
        StencilOperation::DecrementClamp => so::DecrementClamp,
        StencilOperation::IncrementWrap => so::IncrementWrap,
        StencilOperation::DecrementWrap => so::DecrementWrap,
    }
}

fn map_stencil_state_face(desc: &wgt::StencilFaceState) -> webgpu_sys::GpuStencilFaceState {
    let mut mapped = webgpu_sys::GpuStencilFaceState::new();
    mapped.compare(map_compare_function(desc.compare));
    mapped.depth_fail_op(map_stencil_operation(desc.depth_fail_op));
    mapped.fail_op(map_stencil_operation(desc.fail_op));
    mapped.pass_op(map_stencil_operation(desc.pass_op));
    mapped
}

fn map_depth_stencil_state(desc: &wgt::DepthStencilState) -> webgpu_sys::GpuDepthStencilState {
    let mut mapped = webgpu_sys::GpuDepthStencilState::new(map_texture_format(desc.format));
    mapped.depth_compare(map_compare_function(desc.depth_compare));
    mapped.depth_write_enabled(desc.depth_write_enabled);
    mapped.depth_bias(desc.bias.constant);
    mapped.depth_bias_clamp(desc.bias.clamp);
    mapped.depth_bias_slope_scale(desc.bias.slope_scale);
    mapped.stencil_back(&map_stencil_state_face(&desc.stencil.back));
    mapped.stencil_front(&map_stencil_state_face(&desc.stencil.front));
    mapped.stencil_read_mask(desc.stencil.read_mask);
    mapped.stencil_write_mask(desc.stencil.write_mask);
    mapped
}

fn map_blend_component(desc: &wgt::BlendComponent) -> webgpu_sys::GpuBlendComponent {
    let mut mapped = webgpu_sys::GpuBlendComponent::new();
    mapped.dst_factor(map_blend_factor(desc.dst_factor));
    mapped.operation(map_blend_operation(desc.operation));
    mapped.src_factor(map_blend_factor(desc.src_factor));
    mapped
}

fn map_blend_factor(factor: wgt::BlendFactor) -> webgpu_sys::GpuBlendFactor {
    use webgpu_sys::GpuBlendFactor as bf;
    use wgt::BlendFactor;
    match factor {
        BlendFactor::Zero => bf::Zero,
        BlendFactor::One => bf::One,
        BlendFactor::Src => bf::Src,
        BlendFactor::OneMinusSrc => bf::OneMinusSrc,
        BlendFactor::SrcAlpha => bf::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => bf::OneMinusSrcAlpha,
        BlendFactor::Dst => bf::Dst,
        BlendFactor::OneMinusDst => bf::OneMinusDst,
        BlendFactor::DstAlpha => bf::DstAlpha,
        BlendFactor::OneMinusDstAlpha => bf::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => bf::SrcAlphaSaturated,
        BlendFactor::Constant => bf::Constant,
        BlendFactor::OneMinusConstant => bf::OneMinusConstant,
        BlendFactor::Src1
        | BlendFactor::OneMinusSrc1
        | BlendFactor::Src1Alpha
        | BlendFactor::OneMinusSrc1Alpha => {
            panic!(
                "{:?} is not enabled for this backend",
                wgt::Features::DUAL_SOURCE_BLENDING
            )
        }
    }
}

fn map_blend_operation(op: wgt::BlendOperation) -> webgpu_sys::GpuBlendOperation {
    use webgpu_sys::GpuBlendOperation as bo;
    use wgt::BlendOperation;
    match op {
        BlendOperation::Add => bo::Add,
        BlendOperation::Subtract => bo::Subtract,
        BlendOperation::ReverseSubtract => bo::ReverseSubtract,
        BlendOperation::Min => bo::Min,
        BlendOperation::Max => bo::Max,
    }
}

fn map_index_format(format: wgt::IndexFormat) -> webgpu_sys::GpuIndexFormat {
    use webgpu_sys::GpuIndexFormat as f;
    use wgt::IndexFormat;
    match format {
        IndexFormat::Uint16 => f::Uint16,
        IndexFormat::Uint32 => f::Uint32,
    }
}

fn map_vertex_format(format: wgt::VertexFormat) -> webgpu_sys::GpuVertexFormat {
    use webgpu_sys::GpuVertexFormat as vf;
    use wgt::VertexFormat;
    match format {
        VertexFormat::Uint8x2 => vf::Uint8x2,
        VertexFormat::Uint8x4 => vf::Uint8x4,
        VertexFormat::Sint8x2 => vf::Sint8x2,
        VertexFormat::Sint8x4 => vf::Sint8x4,
        VertexFormat::Unorm8x2 => vf::Unorm8x2,
        VertexFormat::Unorm8x4 => vf::Unorm8x4,
        VertexFormat::Snorm8x2 => vf::Snorm8x2,
        VertexFormat::Snorm8x4 => vf::Snorm8x4,
        VertexFormat::Uint16x2 => vf::Uint16x2,
        VertexFormat::Uint16x4 => vf::Uint16x4,
        VertexFormat::Sint16x2 => vf::Sint16x2,
        VertexFormat::Sint16x4 => vf::Sint16x4,
        VertexFormat::Unorm16x2 => vf::Unorm16x2,
        VertexFormat::Unorm16x4 => vf::Unorm16x4,
        VertexFormat::Snorm16x2 => vf::Snorm16x2,
        VertexFormat::Snorm16x4 => vf::Snorm16x4,
        VertexFormat::Float16x2 => vf::Float16x2,
        VertexFormat::Float16x4 => vf::Float16x4,
        VertexFormat::Float32 => vf::Float32,
        VertexFormat::Float32x2 => vf::Float32x2,
        VertexFormat::Float32x3 => vf::Float32x3,
        VertexFormat::Float32x4 => vf::Float32x4,
        VertexFormat::Uint32 => vf::Uint32,
        VertexFormat::Uint32x2 => vf::Uint32x2,
        VertexFormat::Uint32x3 => vf::Uint32x3,
        VertexFormat::Uint32x4 => vf::Uint32x4,
        VertexFormat::Sint32 => vf::Sint32,
        VertexFormat::Sint32x2 => vf::Sint32x2,
        VertexFormat::Sint32x3 => vf::Sint32x3,
        VertexFormat::Sint32x4 => vf::Sint32x4,
        VertexFormat::Unorm10_10_10_2 => vf::Unorm1010102,
        VertexFormat::Float64
        | VertexFormat::Float64x2
        | VertexFormat::Float64x3
        | VertexFormat::Float64x4 => {
            panic!("VERTEX_ATTRIBUTE_64BIT feature must be enabled to use Double formats")
        }
    }
}

fn map_vertex_step_mode(mode: wgt::VertexStepMode) -> webgpu_sys::GpuVertexStepMode {
    use webgpu_sys::GpuVertexStepMode as sm;
    use wgt::VertexStepMode;
    match mode {
        VertexStepMode::Vertex => sm::Vertex,
        VertexStepMode::Instance => sm::Instance,
    }
}

fn map_extent_3d(extent: wgt::Extent3d) -> webgpu_sys::GpuExtent3dDict {
    let mut mapped = webgpu_sys::GpuExtent3dDict::new(extent.width);
    mapped.height(extent.height);
    mapped.depth_or_array_layers(extent.depth_or_array_layers);
    mapped
}

fn map_origin_2d(extent: wgt::Origin2d) -> webgpu_sys::GpuOrigin2dDict {
    let mut mapped = webgpu_sys::GpuOrigin2dDict::new();
    mapped.x(extent.x);
    mapped.y(extent.y);
    mapped
}

fn map_origin_3d(origin: wgt::Origin3d) -> webgpu_sys::GpuOrigin3dDict {
    let mut mapped = webgpu_sys::GpuOrigin3dDict::new();
    mapped.x(origin.x);
    mapped.y(origin.y);
    mapped.z(origin.z);
    mapped
}

fn map_texture_dimension(
    texture_dimension: wgt::TextureDimension,
) -> webgpu_sys::GpuTextureDimension {
    match texture_dimension {
        wgt::TextureDimension::D1 => webgpu_sys::GpuTextureDimension::N1d,
        wgt::TextureDimension::D2 => webgpu_sys::GpuTextureDimension::N2d,
        wgt::TextureDimension::D3 => webgpu_sys::GpuTextureDimension::N3d,
    }
}

fn map_texture_view_dimension(
    texture_view_dimension: wgt::TextureViewDimension,
) -> webgpu_sys::GpuTextureViewDimension {
    use webgpu_sys::GpuTextureViewDimension as tvd;
    match texture_view_dimension {
        wgt::TextureViewDimension::D1 => tvd::N1d,
        wgt::TextureViewDimension::D2 => tvd::N2d,
        wgt::TextureViewDimension::D2Array => tvd::N2dArray,
        wgt::TextureViewDimension::Cube => tvd::Cube,
        wgt::TextureViewDimension::CubeArray => tvd::CubeArray,
        wgt::TextureViewDimension::D3 => tvd::N3d,
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer<'_>) -> webgpu_sys::GpuImageCopyBuffer {
    let buffer: &<ContextWebGpu as crate::Context>::BufferData =
        downcast_ref(view.buffer.data.as_ref());
    let mut mapped = webgpu_sys::GpuImageCopyBuffer::new(&buffer.0.buffer);
    if let Some(bytes_per_row) = view.layout.bytes_per_row {
        mapped.bytes_per_row(bytes_per_row);
    }
    if let Some(rows_per_image) = view.layout.rows_per_image {
        mapped.rows_per_image(rows_per_image);
    }
    mapped.offset(view.layout.offset as f64);
    mapped
}

fn map_texture_copy_view(view: crate::ImageCopyTexture<'_>) -> webgpu_sys::GpuImageCopyTexture {
    let texture: &<ContextWebGpu as crate::Context>::TextureData =
        downcast_ref(view.texture.data.as_ref());
    let mut mapped = webgpu_sys::GpuImageCopyTexture::new(&texture.0);
    mapped.mip_level(view.mip_level);
    mapped.origin(&map_origin_3d(view.origin));
    mapped
}

fn map_tagged_texture_copy_view(
    view: crate::ImageCopyTextureTagged<'_>,
) -> webgpu_sys::GpuImageCopyTextureTagged {
    let texture: &<ContextWebGpu as crate::Context>::TextureData =
        downcast_ref(view.texture.data.as_ref());
    let mut mapped = webgpu_sys::GpuImageCopyTextureTagged::new(&texture.0);
    mapped.mip_level(view.mip_level);
    mapped.origin(&map_origin_3d(view.origin));
    mapped.aspect(map_texture_aspect(view.aspect));
    // mapped.color_space(map_color_space(view.color_space));
    mapped.premultiplied_alpha(view.premultiplied_alpha);
    mapped
}

fn map_external_texture_copy_view(
    view: &crate::ImageCopyExternalImage,
) -> webgpu_sys::GpuImageCopyExternalImage {
    let mut mapped = webgpu_sys::GpuImageCopyExternalImage::new(&view.source);
    mapped.origin(&map_origin_2d(view.origin));
    mapped.flip_y(view.flip_y);
    mapped
}

fn map_texture_aspect(aspect: wgt::TextureAspect) -> webgpu_sys::GpuTextureAspect {
    match aspect {
        wgt::TextureAspect::All => webgpu_sys::GpuTextureAspect::All,
        wgt::TextureAspect::StencilOnly => webgpu_sys::GpuTextureAspect::StencilOnly,
        wgt::TextureAspect::DepthOnly => webgpu_sys::GpuTextureAspect::DepthOnly,
        wgt::TextureAspect::Plane0 | wgt::TextureAspect::Plane1 | wgt::TextureAspect::Plane2 => {
            panic!("multi-plane textures are not supported")
        }
    }
}

fn map_filter_mode(mode: wgt::FilterMode) -> webgpu_sys::GpuFilterMode {
    match mode {
        wgt::FilterMode::Nearest => webgpu_sys::GpuFilterMode::Nearest,
        wgt::FilterMode::Linear => webgpu_sys::GpuFilterMode::Linear,
    }
}

fn map_mipmap_filter_mode(mode: wgt::FilterMode) -> webgpu_sys::GpuMipmapFilterMode {
    match mode {
        wgt::FilterMode::Nearest => webgpu_sys::GpuMipmapFilterMode::Nearest,
        wgt::FilterMode::Linear => webgpu_sys::GpuMipmapFilterMode::Linear,
    }
}

fn map_address_mode(mode: wgt::AddressMode) -> webgpu_sys::GpuAddressMode {
    match mode {
        wgt::AddressMode::ClampToEdge => webgpu_sys::GpuAddressMode::ClampToEdge,
        wgt::AddressMode::Repeat => webgpu_sys::GpuAddressMode::Repeat,
        wgt::AddressMode::MirrorRepeat => webgpu_sys::GpuAddressMode::MirrorRepeat,
        wgt::AddressMode::ClampToBorder => panic!("Clamp to border is not supported"),
    }
}

fn map_color(color: wgt::Color) -> webgpu_sys::GpuColorDict {
    webgpu_sys::GpuColorDict::new(color.a, color.b, color.g, color.r)
}

fn map_store_op(store: crate::StoreOp) -> webgpu_sys::GpuStoreOp {
    match store {
        crate::StoreOp::Store => webgpu_sys::GpuStoreOp::Store,
        crate::StoreOp::Discard => webgpu_sys::GpuStoreOp::Discard,
    }
}

fn map_map_mode(mode: crate::MapMode) -> u32 {
    match mode {
        crate::MapMode::Read => webgpu_sys::gpu_map_mode::READ,
        crate::MapMode::Write => webgpu_sys::gpu_map_mode::WRITE,
    }
}

const FEATURES_MAPPING: [(wgt::Features, webgpu_sys::GpuFeatureName); 12] = [
    //TODO: update the name
    (
        wgt::Features::DEPTH_CLIP_CONTROL,
        webgpu_sys::GpuFeatureName::DepthClipControl,
    ),
    (
        wgt::Features::DEPTH32FLOAT_STENCIL8,
        webgpu_sys::GpuFeatureName::Depth32floatStencil8,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_BC,
        webgpu_sys::GpuFeatureName::TextureCompressionBc,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
        webgpu_sys::GpuFeatureName::TextureCompressionBcSliced3d,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_ETC2,
        webgpu_sys::GpuFeatureName::TextureCompressionEtc2,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_ASTC,
        webgpu_sys::GpuFeatureName::TextureCompressionAstc,
    ),
    (
        wgt::Features::TIMESTAMP_QUERY,
        webgpu_sys::GpuFeatureName::TimestampQuery,
    ),
    (
        wgt::Features::INDIRECT_FIRST_INSTANCE,
        webgpu_sys::GpuFeatureName::IndirectFirstInstance,
    ),
    (
        wgt::Features::SHADER_F16,
        webgpu_sys::GpuFeatureName::ShaderF16,
    ),
    (
        wgt::Features::RG11B10UFLOAT_RENDERABLE,
        webgpu_sys::GpuFeatureName::Rg11b10ufloatRenderable,
    ),
    (
        wgt::Features::BGRA8UNORM_STORAGE,
        webgpu_sys::GpuFeatureName::Bgra8unormStorage,
    ),
    (
        wgt::Features::FLOAT32_FILTERABLE,
        webgpu_sys::GpuFeatureName::Float32Filterable,
    ),
];

fn map_wgt_features(supported_features: webgpu_sys::GpuSupportedFeatures) -> wgt::Features {
    let mut features = wgt::Features::empty();
    for (wgpu_feat, web_feat) in FEATURES_MAPPING {
        match wasm_bindgen::JsValue::from(web_feat).as_string() {
            Some(value) if supported_features.has(&value) => features |= wgpu_feat,
            _ => {}
        }
    }
    features
}

fn map_wgt_limits(limits: webgpu_sys::GpuSupportedLimits) -> wgt::Limits {
    wgt::Limits {
        max_texture_dimension_1d: limits.max_texture_dimension_1d(),
        max_texture_dimension_2d: limits.max_texture_dimension_2d(),
        max_texture_dimension_3d: limits.max_texture_dimension_3d(),
        max_texture_array_layers: limits.max_texture_array_layers(),
        max_bind_groups: limits.max_bind_groups(),
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group(),
        max_dynamic_uniform_buffers_per_pipeline_layout: limits
            .max_dynamic_uniform_buffers_per_pipeline_layout(),
        max_dynamic_storage_buffers_per_pipeline_layout: limits
            .max_dynamic_storage_buffers_per_pipeline_layout(),
        max_sampled_textures_per_shader_stage: limits.max_sampled_textures_per_shader_stage(),
        max_samplers_per_shader_stage: limits.max_samplers_per_shader_stage(),
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage(),
        max_storage_textures_per_shader_stage: limits.max_storage_textures_per_shader_stage(),
        max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage(),
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size() as u32,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size() as u32,
        max_vertex_buffers: limits.max_vertex_buffers(),
        max_buffer_size: limits.max_buffer_size() as u64,
        max_vertex_attributes: limits.max_vertex_attributes(),
        max_vertex_buffer_array_stride: limits.max_vertex_buffer_array_stride(),
        min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment(),
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment(),
        max_color_attachments: limits.max_color_attachments(),
        max_color_attachment_bytes_per_sample: limits.max_color_attachment_bytes_per_sample(),
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size(),
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup(),
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x(),
        max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y(),
        max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z(),
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension(),
        // The following are not part of WebGPU
        min_subgroup_size: wgt::Limits::default().min_subgroup_size,
        max_subgroup_size: wgt::Limits::default().max_subgroup_size,
        max_push_constant_size: wgt::Limits::default().max_push_constant_size,
        max_non_sampler_bindings: wgt::Limits::default().max_non_sampler_bindings,
        max_inter_stage_shader_components: wgt::Limits::default().max_inter_stage_shader_components,
    }
}

fn map_js_sys_limits(limits: &wgt::Limits) -> js_sys::Object {
    let object = js_sys::Object::new();

    macro_rules! set_properties {
        (($from:expr) => ($on:expr) : $(($js_ident:ident, $rs_ident:ident)),* $(,)?) => {
            $(
                ::js_sys::Reflect::set(
                    &$on,
                    &::wasm_bindgen::JsValue::from(stringify!($js_ident)),
                    // Numbers may be u64, however using `from` on a u64 yields
                    // errors on the wasm side, since it uses an unsupported api.
                    // Wasm sends us things that need to fit into u64s by sending
                    // us f64s instead. So we just send them f64s back.
                    &::wasm_bindgen::JsValue::from($from.$rs_ident as f64)
                )
                    .expect("Setting Object properties should never fail.");
            )*
        }
    }

    set_properties![
        (limits) => (object):
        (maxTextureDimension1D, max_texture_dimension_1d),
        (maxTextureDimension2D, max_texture_dimension_2d),
        (maxTextureDimension3D, max_texture_dimension_3d),
        (maxTextureArrayLayers, max_texture_array_layers),
        (maxBindGroups, max_bind_groups),
        (maxBindingsPerBindGroup, max_bindings_per_bind_group),
        (maxDynamicUniformBuffersPerPipelineLayout, max_dynamic_uniform_buffers_per_pipeline_layout),
        (maxDynamicStorageBuffersPerPipelineLayout, max_dynamic_storage_buffers_per_pipeline_layout),
        (maxSampledTexturesPerShaderStage, max_sampled_textures_per_shader_stage),
        (maxSamplersPerShaderStage, max_samplers_per_shader_stage),
        (maxStorageBuffersPerShaderStage, max_storage_buffers_per_shader_stage),
        (maxStorageTexturesPerShaderStage, max_storage_textures_per_shader_stage),
        (maxUniformBuffersPerShaderStage, max_uniform_buffers_per_shader_stage),
        (maxUniformBufferBindingSize, max_uniform_buffer_binding_size),
        (maxStorageBufferBindingSize, max_storage_buffer_binding_size),
        (minUniformBufferOffsetAlignment, min_uniform_buffer_offset_alignment),
        (minStorageBufferOffsetAlignment, min_storage_buffer_offset_alignment),
        (maxVertexBuffers, max_vertex_buffers),
        (maxBufferSize, max_buffer_size),
        (maxVertexAttributes, max_vertex_attributes),
        (maxVertexBufferArrayStride, max_vertex_buffer_array_stride),
        (maxComputeWorkgroupStorageSize, max_compute_workgroup_storage_size),
        (maxComputeInvocationsPerWorkgroup, max_compute_invocations_per_workgroup),
        (maxComputeWorkgroupSizeX, max_compute_workgroup_size_x),
        (maxComputeWorkgroupSizeY, max_compute_workgroup_size_y),
        (maxComputeWorkgroupSizeZ, max_compute_workgroup_size_z),
        (maxComputeWorkgroupsPerDimension, max_compute_workgroups_per_dimension),
    ];

    object
}

type JsFutureResult = Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;

fn future_request_adapter(result: JsFutureResult) -> Option<Sendable<webgpu_sys::GpuAdapter>> {
    match result.and_then(wasm_bindgen::JsCast::dyn_into) {
        Ok(adapter) => Some(Sendable(adapter)),
        Err(_) => None,
    }
}

fn future_request_device(
    result: JsFutureResult,
) -> Result<
    (
        Sendable<webgpu_sys::GpuDevice>,
        Sendable<webgpu_sys::GpuQueue>,
    ),
    crate::RequestDeviceError,
> {
    result
        .map(|js_value| {
            let device_data = Sendable(webgpu_sys::GpuDevice::from(js_value));
            let queue_data = Sendable(device_data.0.queue());

            (device_data, queue_data)
        })
        .map_err(|error_value| crate::RequestDeviceError {
            inner: crate::RequestDeviceErrorKind::WebGpu(error_value),
        })
}

fn future_pop_error_scope(result: JsFutureResult) -> Option<crate::Error> {
    match result {
        Ok(js_value) if js_value.is_object() => {
            let js_error = wasm_bindgen::JsCast::dyn_into(js_value).unwrap();
            Some(crate::Error::from_js(js_error))
        }
        _ => None,
    }
}

fn future_compilation_info(
    result: JsFutureResult,
    base_compilation_info: &WebShaderCompilationInfo,
) -> crate::CompilationInfo {
    let base_messages = match base_compilation_info {
        WebShaderCompilationInfo::Transformed { compilation_info } => {
            compilation_info.messages.iter().cloned()
        }
        _ => [].iter().cloned(),
    };

    let messages = match result {
        Ok(js_value) => {
            let info = webgpu_sys::GpuCompilationInfo::from(js_value);
            base_messages
                .chain(info.messages().into_iter().map(|message| {
                    crate::CompilationMessage::from_js(
                        webgpu_sys::GpuCompilationMessage::from(message),
                        base_compilation_info,
                    )
                }))
                .collect()
        }
        Err(_v) => base_messages
            .chain(std::iter::once(crate::CompilationMessage {
                message: "Getting compilation info failed".to_string(),
                message_type: crate::CompilationMessageType::Error,
                location: None,
            }))
            .collect(),
    };

    crate::CompilationInfo { messages }
}

/// Calls `callback(success_value)` when the promise completes successfully, calls `callback(failure_value)`
/// when the promise completes unsuccessfully.
fn register_then_closures<F, T>(promise: &Promise, callback: F, success_value: T, failure_value: T)
where
    F: FnOnce(T) + 'static,
    T: 'static,
{
    // Both the 'success' and 'rejected' closures need access to callback, but only one
    // of them will ever run. We have them both hold a reference to a `Rc<RefCell<Option<impl FnOnce...>>>`,
    // and then take ownership of callback when invoked.
    //
    // We also only need Rc's because these will only ever be called on our thread.
    //
    // We also store the actual closure types inside this Rc, as the closures need to be kept alive
    // until they are actually called by the callback. It is valid to drop a closure inside of a callback.
    // This allows us to keep the closures alive without leaking them.
    let rc_callback: Rc<RefCell<Option<(_, _, F)>>> = Rc::new(RefCell::new(None));

    let rc_callback_clone1 = rc_callback.clone();
    let rc_callback_clone2 = rc_callback.clone();
    let closure_success = wasm_bindgen::closure::Closure::once(move |_| {
        let (success_closure, rejection_closure, callback) =
            rc_callback_clone1.borrow_mut().take().unwrap();
        callback(success_value);
        // drop the closures, including ourselves, which will free any captured memory.
        drop((success_closure, rejection_closure));
    });
    let closure_rejected = wasm_bindgen::closure::Closure::once(move |_| {
        let (success_closure, rejection_closure, callback) =
            rc_callback_clone2.borrow_mut().take().unwrap();
        callback(failure_value);
        // drop the closures, including ourselves, which will free any captured memory.
        drop((success_closure, rejection_closure));
    });

    // Calling then before setting the value in the Rc seems like a race, but it isn't
    // because the promise callback will run on this thread, so there is no race.
    let _ = promise.then2(&closure_success, &closure_rejected);

    *rc_callback.borrow_mut() = Some((closure_success, closure_rejected, callback));
}

impl ContextWebGpu {
    /// Common portion of the internal branches of the public `instance_create_surface` function.
    ///
    /// Note: Analogous code also exists in the WebGL2 backend at
    /// `wgpu_hal::gles::web::Instance`.
    fn create_surface_from_context(
        &self,
        canvas: Canvas,
        context_result: Result<Option<js_sys::Object>, wasm_bindgen::JsValue>,
    ) -> Result<<Self as crate::Context>::SurfaceData, crate::CreateSurfaceError> {
        let context: js_sys::Object = match context_result {
            Ok(Some(context)) => context,
            Ok(None) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext-dev>
                // A getContext() call “returns null if contextId is not supported, or if the
                // canvas has already been initialized with another context type”. Additionally,
                // “not supported” could include “insufficient GPU resources” or “the GPU process
                // previously crashed”. So, we must return it as an `Err` since it could occur
                // for circumstances outside the application author's control.
                return Err(crate::CreateSurfaceError {
                    inner: crate::CreateSurfaceErrorKind::Web(
                        String::from(
                            "canvas.getContext() returned null; webgpu not available or canvas already in use"
                        )
                    )
                });
            }
            Err(js_error) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext>
                // A thrown exception indicates misuse of the canvas state.
                return Err(crate::CreateSurfaceError {
                    inner: crate::CreateSurfaceErrorKind::Web(format!(
                        "canvas.getContext() threw exception {js_error:?}",
                    )),
                });
            }
        };

        // Not returning this error because it is a type error that shouldn't happen unless
        // the browser, JS builtin objects, or wasm bindings are misbehaving somehow.
        let context: webgpu_sys::GpuCanvasContext = context
            .dyn_into()
            .expect("canvas context is not a GPUCanvasContext");

        Ok(Sendable((canvas, context)))
    }

    /// Get mapped buffer range directly as a `js_sys::ArrayBuffer`.
    pub fn buffer_get_mapped_range_as_array_buffer(
        &self,
        buffer_data: &<ContextWebGpu as crate::Context>::BufferData,
        sub_range: Range<wgt::BufferAddress>,
    ) -> js_sys::ArrayBuffer {
        buffer_data.0.get_mapped_array_buffer(sub_range)
    }
}

// Represents the global object in the JavaScript context.
// It can be cast to from `webgpu_sys::global` and exposes two getters `window` and `worker` of which only one is defined depending on the caller's context.
// When called from the UI thread only `window` is defined whereas `worker` is only defined within a web worker context.
// See: https://github.com/rustwasm/gloo/blob/2c9e776701ecb90c53e62dec1abd19c2b70e47c7/crates/timers/src/callback.rs#L8-L40
#[wasm_bindgen]
extern "C" {
    type Global;

    #[wasm_bindgen(method, getter, js_name = Window)]
    fn window(this: &Global) -> JsValue;

    #[wasm_bindgen(method, getter, js_name = WorkerGlobalScope)]
    fn worker(this: &Global) -> JsValue;
}

#[derive(Debug)]
pub enum Canvas {
    Canvas(web_sys::HtmlCanvasElement),
    Offscreen(web_sys::OffscreenCanvas),
}

#[derive(Debug, Clone, Copy)]
pub struct BrowserGpuPropertyInaccessible;

/// Returns the browser's gpu object or `Err(BrowserGpuPropertyInaccessible)` if
/// the current context is neither the main thread nor a dedicated worker.
///
/// If WebGPU is not supported, the Gpu property may (!) be `undefined`,
/// and so this function will return `Ok(None)`.
/// Note that this check is insufficient to determine whether WebGPU is
/// supported, as the browser may define the Gpu property, but be unable to
/// create any WebGPU adapters.
/// To detect whether WebGPU is supported, use the [`crate::utils::is_browser_webgpu_supported`] function.
///
/// See:
/// * <https://developer.mozilla.org/en-US/docs/Web/API/Navigator/gpu>
/// * <https://developer.mozilla.org/en-US/docs/Web/API/WorkerNavigator/gpu>
pub fn get_browser_gpu_property(
) -> Result<Option<DefinedNonNullJsValue<webgpu_sys::Gpu>>, BrowserGpuPropertyInaccessible> {
    let global: Global = js_sys::global().unchecked_into();

    let maybe_undefined_gpu: webgpu_sys::Gpu = if !global.window().is_undefined() {
        let navigator = global.unchecked_into::<web_sys::Window>().navigator();
        ext_bindings::NavigatorGpu::gpu(&navigator)
    } else if !global.worker().is_undefined() {
        let navigator = global
            .unchecked_into::<web_sys::WorkerGlobalScope>()
            .navigator();
        ext_bindings::NavigatorGpu::gpu(&navigator)
    } else {
        return Err(BrowserGpuPropertyInaccessible);
    };
    Ok(DefinedNonNullJsValue::new(maybe_undefined_gpu))
}

impl crate::context::Context for ContextWebGpu {
    type AdapterData = Sendable<webgpu_sys::GpuAdapter>;
    type DeviceData = Sendable<webgpu_sys::GpuDevice>;
    type QueueData = Sendable<webgpu_sys::GpuQueue>;
    type ShaderModuleData = Sendable<WebShaderModule>;
    type BindGroupLayoutData = Sendable<webgpu_sys::GpuBindGroupLayout>;
    type BindGroupData = Sendable<webgpu_sys::GpuBindGroup>;
    type TextureViewData = Sendable<webgpu_sys::GpuTextureView>;
    type SamplerData = Sendable<webgpu_sys::GpuSampler>;
    type BufferData = Sendable<WebBuffer>;
    type TextureData = Sendable<webgpu_sys::GpuTexture>;
    type QuerySetData = Sendable<webgpu_sys::GpuQuerySet>;
    type PipelineLayoutData = Sendable<webgpu_sys::GpuPipelineLayout>;
    type RenderPipelineData = Sendable<webgpu_sys::GpuRenderPipeline>;
    type ComputePipelineData = Sendable<webgpu_sys::GpuComputePipeline>;
    type CommandEncoderData = Sendable<webgpu_sys::GpuCommandEncoder>;
    type ComputePassData = Sendable<webgpu_sys::GpuComputePassEncoder>;
    type RenderPassData = Sendable<webgpu_sys::GpuRenderPassEncoder>;
    type CommandBufferData = Sendable<webgpu_sys::GpuCommandBuffer>;
    type RenderBundleEncoderData = Sendable<webgpu_sys::GpuRenderBundleEncoder>;
    type RenderBundleData = Sendable<webgpu_sys::GpuRenderBundle>;
    type SurfaceData = Sendable<(Canvas, webgpu_sys::GpuCanvasContext)>;

    type SurfaceOutputDetail = SurfaceOutputDetail;
    type SubmissionIndexData = ();
    type PipelineCacheData = ();

    type RequestAdapterFuture = OptionFuture<
        MakeSendFuture<
            wasm_bindgen_futures::JsFuture,
            fn(JsFutureResult) -> Option<Self::AdapterData>,
        >,
    >;
    type RequestDeviceFuture = MakeSendFuture<
        wasm_bindgen_futures::JsFuture,
        fn(
            JsFutureResult,
        ) -> Result<(Self::DeviceData, Self::QueueData), crate::RequestDeviceError>,
    >;
    type PopErrorScopeFuture =
        MakeSendFuture<wasm_bindgen_futures::JsFuture, fn(JsFutureResult) -> Option<crate::Error>>;

    type CompilationInfoFuture = MakeSendFuture<
        wasm_bindgen_futures::JsFuture,
        Box<dyn Fn(JsFutureResult) -> CompilationInfo>,
    >;

    fn init(_instance_desc: wgt::InstanceDescriptor) -> Self {
        let Ok(gpu) = get_browser_gpu_property() else {
            panic!(
                "Accessing the GPU is only supported on the main thread or from a dedicated worker"
            );
        };

        ContextWebGpu { gpu }
    }

    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Self::SurfaceData, crate::CreateSurfaceError> {
        match target {
            SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: _,
                raw_window_handle,
            } => {
                let canvas_element: web_sys::HtmlCanvasElement = match raw_window_handle {
                    raw_window_handle::RawWindowHandle::Web(handle) => {
                        let canvas_node: wasm_bindgen::JsValue = web_sys::window()
                            .and_then(|win| win.document())
                            .and_then(|doc| {
                                doc.query_selector_all(&format!(
                                    "[data-raw-handle=\"{}\"]",
                                    handle.id
                                ))
                                .ok()
                            })
                            .and_then(|nodes| nodes.get(0))
                            .expect("expected to find single canvas")
                            .into();
                        canvas_node.into()
                    }
                    raw_window_handle::RawWindowHandle::WebCanvas(handle) => {
                        let value: &JsValue = unsafe { handle.obj.cast().as_ref() };
                        value.clone().unchecked_into()
                    }
                    raw_window_handle::RawWindowHandle::WebOffscreenCanvas(handle) => {
                        let value: &JsValue = unsafe { handle.obj.cast().as_ref() };
                        let canvas: web_sys::OffscreenCanvas = value.clone().unchecked_into();
                        let context_result = canvas.get_context("webgpu");

                        return self.create_surface_from_context(
                            Canvas::Offscreen(canvas),
                            context_result,
                        );
                    }
                    _ => panic!("expected valid handle for canvas"),
                };

                let context_result = canvas_element.get_context("webgpu");
                self.create_surface_from_context(Canvas::Canvas(canvas_element), context_result)
            }
        }
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture {
        //TODO: support this check, return `None` if the flag is not set.
        // It's not trivial, since we need the Future logic to have this check,
        // and currently the Future here has no room for extra parameter `backends`.
        //assert!(backends.contains(wgt::Backends::BROWSER_WEBGPU));
        let mut mapped_options = webgpu_sys::GpuRequestAdapterOptions::new();
        let mapped_power_preference = match options.power_preference {
            wgt::PowerPreference::None => None,
            wgt::PowerPreference::LowPower => Some(webgpu_sys::GpuPowerPreference::LowPower),
            wgt::PowerPreference::HighPerformance => {
                Some(webgpu_sys::GpuPowerPreference::HighPerformance)
            }
        };
        if let Some(mapped_pref) = mapped_power_preference {
            mapped_options.power_preference(mapped_pref);
        }
        if let Some(gpu) = &self.gpu {
            let adapter_promise = gpu.request_adapter_with_options(&mapped_options);
            OptionFuture::some(MakeSendFuture::new(
                wasm_bindgen_futures::JsFuture::from(adapter_promise),
                future_request_adapter,
            ))
        } else {
            // Gpu is undefined; WebGPU is not supported in this browser.
            OptionFuture::none()
        }
    }

    fn adapter_request_device(
        &self,
        adapter_data: &Self::AdapterData,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        if trace_dir.is_some() {
            //Error: Tracing isn't supported on the Web target
        }

        let mut mapped_desc = webgpu_sys::GpuDeviceDescriptor::new();

        // TODO: Migrate to a web_sys api.
        // See https://github.com/rustwasm/wasm-bindgen/issues/3587
        let limits_object = map_js_sys_limits(&desc.required_limits);

        js_sys::Reflect::set(
            &mapped_desc,
            &JsValue::from("requiredLimits"),
            &limits_object,
        )
        .expect("Setting Object properties should never fail.");

        let required_features = FEATURES_MAPPING
            .iter()
            .copied()
            .flat_map(|(flag, value)| {
                if desc.required_features.contains(flag) {
                    Some(JsValue::from(value))
                } else {
                    None
                }
            })
            .collect::<js_sys::Array>();
        mapped_desc.required_features(&required_features);

        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        let device_promise = adapter_data.0.request_device_with_descriptor(&mapped_desc);

        MakeSendFuture::new(
            wasm_bindgen_futures::JsFuture::from(device_promise),
            future_request_device,
        )
    }

    fn instance_poll_all_devices(&self, _force_wait: bool) -> bool {
        // Devices are automatically polled.
        true
    }

    fn adapter_is_surface_supported(
        &self,
        _adapter_data: &Self::AdapterData,
        _surface_data: &Self::SurfaceData,
    ) -> bool {
        true
    }

    fn adapter_features(&self, adapter_data: &Self::AdapterData) -> wgt::Features {
        map_wgt_features(adapter_data.0.features())
    }

    fn adapter_limits(&self, adapter_data: &Self::AdapterData) -> wgt::Limits {
        map_wgt_limits(adapter_data.0.limits())
    }

    fn adapter_downlevel_capabilities(
        &self,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::DownlevelCapabilities {
        // WebGPU is assumed to be fully compliant
        wgt::DownlevelCapabilities::default()
    }

    fn adapter_get_info(&self, _adapter_data: &Self::AdapterData) -> wgt::AdapterInfo {
        // TODO: web-sys has no way of getting information on adapters
        wgt::AdapterInfo {
            name: String::new(),
            vendor: 0,
            device: 0,
            device_type: wgt::DeviceType::Other,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgt::Backend::BrowserWebGpu,
        }
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter_data: &Self::AdapterData,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        format.guaranteed_format_features(self.adapter_features(adapter_data))
    }

    fn adapter_get_presentation_timestamp(
        &self,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        wgt::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn surface_get_capabilities(
        &self,
        _surface_data: &Self::SurfaceData,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        let mut formats = vec![
            wgt::TextureFormat::Rgba8Unorm,
            wgt::TextureFormat::Bgra8Unorm,
            wgt::TextureFormat::Rgba16Float,
        ];
        let mut mapped_formats = formats.iter().map(|format| map_texture_format(*format));
        // Preferred canvas format will only be either "rgba8unorm" or "bgra8unorm".
        // https://www.w3.org/TR/webgpu/#dom-gpu-getpreferredcanvasformat
        let gpu = self
            .gpu
            .as_ref()
            .expect("Caller could not have created an adapter if gpu is undefined.");
        let preferred_format = gpu.get_preferred_canvas_format();
        if let Some(index) = mapped_formats.position(|format| format == preferred_format) {
            formats.swap(0, index);
        }

        wgt::SurfaceCapabilities {
            // https://gpuweb.github.io/gpuweb/#supported-context-formats
            formats,
            // Doesn't really have meaning on the web.
            present_modes: vec![wgt::PresentMode::Fifo],
            alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
            // Statically set to RENDER_ATTACHMENT for now. See https://gpuweb.github.io/gpuweb/#dom-gpucanvasconfiguration-usage
            usages: wgt::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    fn surface_configure(
        &self,
        surface_data: &Self::SurfaceData,
        device_data: &Self::DeviceData,
        config: &crate::SurfaceConfiguration,
    ) {
        match surface_data.0 .0 {
            Canvas::Canvas(ref canvas) => {
                canvas.set_width(config.width);
                canvas.set_height(config.height);
            }
            Canvas::Offscreen(ref canvas) => {
                canvas.set_width(config.width);
                canvas.set_height(config.height);
            }
        }

        if let wgt::PresentMode::Mailbox | wgt::PresentMode::Immediate = config.present_mode {
            panic!("Only FIFO/Auto* is supported on web");
        }
        if let wgt::CompositeAlphaMode::PostMultiplied | wgt::CompositeAlphaMode::Inherit =
            config.alpha_mode
        {
            panic!("Only Opaque/Auto or PreMultiplied alpha mode are supported on web");
        }
        let alpha_mode = match config.alpha_mode {
            wgt::CompositeAlphaMode::PreMultiplied => webgpu_sys::GpuCanvasAlphaMode::Premultiplied,
            _ => webgpu_sys::GpuCanvasAlphaMode::Opaque,
        };
        let mut mapped = webgpu_sys::GpuCanvasConfiguration::new(
            &device_data.0,
            map_texture_format(config.format),
        );
        mapped.usage(config.usage.bits());
        mapped.alpha_mode(alpha_mode);
        let mapped_view_formats = config
            .view_formats
            .iter()
            .map(|format| JsValue::from(map_texture_format(*format)))
            .collect::<js_sys::Array>();
        mapped.view_formats(&mapped_view_formats);
        surface_data.0 .1.configure(&mapped);
    }

    fn surface_get_current_texture(
        &self,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureData>,
        wgt::SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        let surface_data = Sendable(surface_data.0 .1.get_current_texture());
        (Some(surface_data), wgt::SurfaceStatus::Good, ())
    }

    fn surface_present(&self, _detail: &Self::SurfaceOutputDetail) {
        // Swapchain is presented automatically
    }

    fn surface_texture_discard(&self, _detail: &Self::SurfaceOutputDetail) {
        // Can't really discard this on the Web
    }

    fn device_features(&self, device_data: &Self::DeviceData) -> wgt::Features {
        map_wgt_features(device_data.0.features())
    }

    fn device_limits(&self, device_data: &Self::DeviceData) -> wgt::Limits {
        map_wgt_limits(device_data.0.limits())
    }

    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga-ir"
        )),
        allow(unreachable_code, unused_variables)
    )]
    fn device_create_shader_module(
        &self,
        device_data: &Self::DeviceData,
        desc: crate::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Self::ShaderModuleData {
        let shader_module_result = match desc.source {
            #[cfg(feature = "spirv")]
            crate::ShaderSource::SpirV(ref spv) => {
                use naga::front;

                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false,
                    strict_capabilities: true,
                    block_ctx_dump_prefix: None,
                };
                let spv_parser = front::spv::Frontend::new(spv.iter().cloned(), &options);
                spv_parser
                    .parse()
                    .map_err(|inner| {
                        CompilationInfo::from(naga::error::ShaderError {
                            source: String::new(),
                            label: desc.label.map(|s| s.to_string()),
                            inner: Box::new(inner),
                        })
                    })
                    .and_then(|spv_module| {
                        validate_transformed_shader_module(&spv_module, "", &desc).map(|v| {
                            (
                                v,
                                WebShaderCompilationInfo::Transformed {
                                    compilation_info: CompilationInfo { messages: vec![] },
                                },
                            )
                        })
                    })
            }
            #[cfg(feature = "glsl")]
            crate::ShaderSource::Glsl {
                ref shader,
                stage,
                ref defines,
            } => {
                use naga::front;

                // Parse the given shader code and store its representation.
                let options = front::glsl::Options {
                    stage,
                    defines: defines.clone(),
                };
                let mut parser = front::glsl::Frontend::default();
                parser
                    .parse(&options, shader)
                    .map_err(|inner| {
                        CompilationInfo::from(naga::error::ShaderError {
                            source: shader.to_string(),
                            label: desc.label.map(|s| s.to_string()),
                            inner: Box::new(inner),
                        })
                    })
                    .and_then(|glsl_module| {
                        validate_transformed_shader_module(&glsl_module, shader, &desc).map(|v| {
                            (
                                v,
                                WebShaderCompilationInfo::Transformed {
                                    compilation_info: CompilationInfo { messages: vec![] },
                                },
                            )
                        })
                    })
            }
            #[cfg(feature = "wgsl")]
            crate::ShaderSource::Wgsl(ref code) => {
                let shader_module = webgpu_sys::GpuShaderModuleDescriptor::new(code);
                Ok((
                    shader_module,
                    WebShaderCompilationInfo::Wgsl {
                        source: code.to_string(),
                    },
                ))
            }
            #[cfg(feature = "naga-ir")]
            crate::ShaderSource::Naga(ref module) => {
                validate_transformed_shader_module(module, "", &desc).map(|v| {
                    (
                        v,
                        WebShaderCompilationInfo::Transformed {
                            compilation_info: CompilationInfo { messages: vec![] },
                        },
                    )
                })
            }
            crate::ShaderSource::Dummy(_) => {
                panic!("found `ShaderSource::Dummy`")
            }
        };

        #[cfg(naga)]
        fn validate_transformed_shader_module(
            module: &naga::Module,
            source: &str,
            desc: &crate::ShaderModuleDescriptor<'_>,
        ) -> Result<webgpu_sys::GpuShaderModuleDescriptor, crate::CompilationInfo> {
            use naga::{back, valid};
            let mut validator =
                valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all());
            let module_info = validator.validate(module).map_err(|err| {
                CompilationInfo::from(naga::error::ShaderError {
                    source: source.to_string(),
                    label: desc.label.map(|s| s.to_string()),
                    inner: Box::new(err),
                })
            })?;

            let writer_flags = naga::back::wgsl::WriterFlags::empty();
            let wgsl_text = back::wgsl::write_string(module, &module_info, writer_flags).unwrap();
            Ok(webgpu_sys::GpuShaderModuleDescriptor::new(
                wgsl_text.as_str(),
            ))
        }
        let (mut descriptor, compilation_info) = match shader_module_result {
            Ok(v) => v,
            Err(compilation_info) => (
                webgpu_sys::GpuShaderModuleDescriptor::new(""),
                WebShaderCompilationInfo::Transformed { compilation_info },
            ),
        };
        if let Some(label) = desc.label {
            descriptor.label(label);
        }
        let shader_module = WebShaderModule {
            module: device_data.0.create_shader_module(&descriptor),
            compilation_info,
        };
        Sendable(shader_module)
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        _device_data: &Self::DeviceData,
        _desc: &crate::ShaderModuleDescriptorSpirV<'_>,
    ) -> Self::ShaderModuleData {
        unreachable!("SPIRV_SHADER_PASSTHROUGH is not enabled for this backend")
    }

    fn device_create_bind_group_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::BindGroupLayoutDescriptor<'_>,
    ) -> Self::BindGroupLayoutData {
        let mapped_bindings = desc
            .entries
            .iter()
            .map(|bind| {
                let mut mapped_entry =
                    webgpu_sys::GpuBindGroupLayoutEntry::new(bind.binding, bind.visibility.bits());

                match bind.ty {
                    wgt::BindingType::Buffer {
                        ty,
                        has_dynamic_offset,
                        min_binding_size,
                    } => {
                        let mut buffer = webgpu_sys::GpuBufferBindingLayout::new();
                        buffer.has_dynamic_offset(has_dynamic_offset);
                        if let Some(size) = min_binding_size {
                            buffer.min_binding_size(size.get() as f64);
                        }
                        buffer.type_(match ty {
                            wgt::BufferBindingType::Uniform => {
                                webgpu_sys::GpuBufferBindingType::Uniform
                            }
                            wgt::BufferBindingType::Storage { read_only: false } => {
                                webgpu_sys::GpuBufferBindingType::Storage
                            }
                            wgt::BufferBindingType::Storage { read_only: true } => {
                                webgpu_sys::GpuBufferBindingType::ReadOnlyStorage
                            }
                        });
                        mapped_entry.buffer(&buffer);
                    }
                    wgt::BindingType::Sampler(ty) => {
                        let mut sampler = webgpu_sys::GpuSamplerBindingLayout::new();
                        sampler.type_(match ty {
                            wgt::SamplerBindingType::NonFiltering => {
                                webgpu_sys::GpuSamplerBindingType::NonFiltering
                            }
                            wgt::SamplerBindingType::Filtering => {
                                webgpu_sys::GpuSamplerBindingType::Filtering
                            }
                            wgt::SamplerBindingType::Comparison => {
                                webgpu_sys::GpuSamplerBindingType::Comparison
                            }
                        });
                        mapped_entry.sampler(&sampler);
                    }
                    wgt::BindingType::Texture {
                        multisampled,
                        sample_type,
                        view_dimension,
                    } => {
                        let mut texture = webgpu_sys::GpuTextureBindingLayout::new();
                        texture.multisampled(multisampled);
                        texture.sample_type(map_texture_component_type(sample_type));
                        texture.view_dimension(map_texture_view_dimension(view_dimension));
                        mapped_entry.texture(&texture);
                    }
                    wgt::BindingType::StorageTexture {
                        access,
                        format,
                        view_dimension,
                    } => {
                        let mapped_access = match access {
                            wgt::StorageTextureAccess::WriteOnly => {
                                webgpu_sys::GpuStorageTextureAccess::WriteOnly
                            }
                            wgt::StorageTextureAccess::ReadOnly => {
                                webgpu_sys::GpuStorageTextureAccess::ReadOnly
                            }
                            wgt::StorageTextureAccess::ReadWrite => {
                                webgpu_sys::GpuStorageTextureAccess::ReadWrite
                            }
                        };
                        let mut storage_texture = webgpu_sys::GpuStorageTextureBindingLayout::new(
                            map_texture_format(format),
                        );
                        storage_texture.access(mapped_access);
                        storage_texture.view_dimension(map_texture_view_dimension(view_dimension));
                        mapped_entry.storage_texture(&storage_texture);
                    }
                    wgt::BindingType::AccelerationStructure => todo!(),
                }

                mapped_entry
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = webgpu_sys::GpuBindGroupLayoutDescriptor::new(&mapped_bindings);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device_data.0.create_bind_group_layout(&mapped_desc))
    }

    fn device_create_bind_group(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::BindGroupDescriptor<'_>,
    ) -> Self::BindGroupData {
        let mapped_entries = desc
            .entries
            .iter()
            .map(|binding| {
                let mapped_resource = match binding.resource {
                    crate::BindingResource::Buffer(crate::BufferBinding {
                        buffer,
                        offset,
                        size,
                    }) => {
                        let buffer: &<ContextWebGpu as crate::Context>::BufferData =
                            downcast_ref(buffer.data.as_ref());
                        let mut mapped_buffer_binding =
                            webgpu_sys::GpuBufferBinding::new(&buffer.0.buffer);
                        mapped_buffer_binding.offset(offset as f64);
                        if let Some(s) = size {
                            mapped_buffer_binding.size(s.get() as f64);
                        }
                        JsValue::from(mapped_buffer_binding)
                    }
                    crate::BindingResource::BufferArray(..) => {
                        panic!("Web backend does not support arrays of buffers")
                    }
                    crate::BindingResource::Sampler(sampler) => {
                        let sampler: &<ContextWebGpu as crate::Context>::SamplerData =
                            downcast_ref(sampler.data.as_ref());
                        JsValue::from(&sampler.0)
                    }
                    crate::BindingResource::SamplerArray(..) => {
                        panic!("Web backend does not support arrays of samplers")
                    }
                    crate::BindingResource::TextureView(texture_view) => {
                        let texture_view: &<ContextWebGpu as crate::Context>::TextureViewData =
                            downcast_ref(texture_view.data.as_ref());
                        JsValue::from(&texture_view.0)
                    }
                    crate::BindingResource::TextureViewArray(..) => {
                        panic!("Web backend does not support BINDING_INDEXING extension")
                    }
                };

                webgpu_sys::GpuBindGroupEntry::new(binding.binding, &mapped_resource)
            })
            .collect::<js_sys::Array>();

        let bgl: &<ContextWebGpu as crate::Context>::BindGroupLayoutData =
            downcast_ref(desc.layout.data.as_ref());
        let mut mapped_desc = webgpu_sys::GpuBindGroupDescriptor::new(&mapped_entries, &bgl.0);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device_data.0.create_bind_group(&mapped_desc))
    }

    fn device_create_pipeline_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::PipelineLayoutDescriptor<'_>,
    ) -> Self::PipelineLayoutData {
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| {
                let bgl: &<ContextWebGpu as crate::Context>::BindGroupLayoutData =
                    downcast_ref(bgl.data.as_ref());
                &bgl.0
            })
            .collect::<js_sys::Array>();
        let mut mapped_desc = webgpu_sys::GpuPipelineLayoutDescriptor::new(&temp_layouts);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device_data.0.create_pipeline_layout(&mapped_desc))
    }

    fn device_create_render_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::RenderPipelineDescriptor<'_>,
    ) -> Self::RenderPipelineData {
        let module: &<ContextWebGpu as crate::Context>::ShaderModuleData =
            downcast_ref(desc.vertex.module.data.as_ref());
        let mut mapped_vertex_state = webgpu_sys::GpuVertexState::new(&module.0.module);
        insert_constants_map(
            &mapped_vertex_state,
            desc.vertex.compilation_options.constants,
        );
        if let Some(ep) = desc.vertex.entry_point {
            mapped_vertex_state.entry_point(ep);
        }

        let buffers = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| {
                let mapped_attributes = vbuf
                    .attributes
                    .iter()
                    .map(|attr| {
                        webgpu_sys::GpuVertexAttribute::new(
                            map_vertex_format(attr.format),
                            attr.offset as f64,
                            attr.shader_location,
                        )
                    })
                    .collect::<js_sys::Array>();

                let mut mapped_vbuf = webgpu_sys::GpuVertexBufferLayout::new(
                    vbuf.array_stride as f64,
                    &mapped_attributes,
                );
                mapped_vbuf.step_mode(map_vertex_step_mode(vbuf.step_mode));
                mapped_vbuf
            })
            .collect::<js_sys::Array>();

        mapped_vertex_state.buffers(&buffers);

        let auto_layout = wasm_bindgen::JsValue::from(webgpu_sys::GpuAutoLayoutMode::Auto);
        let mut mapped_desc = webgpu_sys::GpuRenderPipelineDescriptor::new(
            &match desc.layout {
                Some(layout) => {
                    let layout: &<ContextWebGpu as crate::Context>::PipelineLayoutData =
                        downcast_ref(layout.data.as_ref());
                    JsValue::from(&layout.0)
                }
                None => auto_layout,
            },
            &mapped_vertex_state,
        );

        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        if let Some(ref depth_stencil) = desc.depth_stencil {
            mapped_desc.depth_stencil(&map_depth_stencil_state(depth_stencil));
        }

        if let Some(ref frag) = desc.fragment {
            let targets = frag
                .targets
                .iter()
                .map(|target| match target {
                    Some(target) => {
                        let mapped_format = map_texture_format(target.format);
                        let mut mapped_color_state =
                            webgpu_sys::GpuColorTargetState::new(mapped_format);
                        if let Some(ref bs) = target.blend {
                            let alpha = map_blend_component(&bs.alpha);
                            let color = map_blend_component(&bs.color);
                            let mapped_blend_state = webgpu_sys::GpuBlendState::new(&alpha, &color);
                            mapped_color_state.blend(&mapped_blend_state);
                        }
                        mapped_color_state.write_mask(target.write_mask.bits());
                        wasm_bindgen::JsValue::from(mapped_color_state)
                    }
                    None => wasm_bindgen::JsValue::null(),
                })
                .collect::<js_sys::Array>();
            let module: &<ContextWebGpu as crate::Context>::ShaderModuleData =
                downcast_ref(frag.module.data.as_ref());
            let mut mapped_fragment_desc =
                webgpu_sys::GpuFragmentState::new(&module.0.module, &targets);
            insert_constants_map(&mapped_vertex_state, frag.compilation_options.constants);
            if let Some(ep) = frag.entry_point {
                mapped_fragment_desc.entry_point(ep);
            }
            mapped_desc.fragment(&mapped_fragment_desc);
        }

        let mut mapped_multisample = webgpu_sys::GpuMultisampleState::new();
        mapped_multisample.count(desc.multisample.count);
        mapped_multisample.mask(desc.multisample.mask as u32);
        mapped_multisample.alpha_to_coverage_enabled(desc.multisample.alpha_to_coverage_enabled);
        mapped_desc.multisample(&mapped_multisample);

        let mapped_primitive = map_primitive_state(&desc.primitive);
        mapped_desc.primitive(&mapped_primitive);

        Sendable(device_data.0.create_render_pipeline(&mapped_desc))
    }

    fn device_create_compute_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::ComputePipelineDescriptor<'_>,
    ) -> Self::ComputePipelineData {
        let shader_module: &<ContextWebGpu as crate::Context>::ShaderModuleData =
            downcast_ref(desc.module.data.as_ref());
        let mut mapped_compute_stage =
            webgpu_sys::GpuProgrammableStage::new(&shader_module.0.module);
        insert_constants_map(&mapped_compute_stage, desc.compilation_options.constants);
        if let Some(ep) = desc.entry_point {
            mapped_compute_stage.entry_point(ep);
        }
        let auto_layout = wasm_bindgen::JsValue::from(webgpu_sys::GpuAutoLayoutMode::Auto);
        let mut mapped_desc = webgpu_sys::GpuComputePipelineDescriptor::new(
            &match desc.layout {
                Some(layout) => {
                    let layout: &<ContextWebGpu as crate::Context>::PipelineLayoutData =
                        downcast_ref(layout.data.as_ref());
                    JsValue::from(&layout.0)
                }
                None => auto_layout,
            },
            &mapped_compute_stage,
        );
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        Sendable(device_data.0.create_compute_pipeline(&mapped_desc))
    }

    unsafe fn device_create_pipeline_cache(
        &self,
        _: &Self::DeviceData,
        _: &crate::PipelineCacheDescriptor<'_>,
    ) -> Self::PipelineCacheData {
    }
    fn pipeline_cache_drop(&self, _: &Self::PipelineCacheData) {}

    fn device_create_buffer(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::BufferDescriptor<'_>,
    ) -> Self::BufferData {
        let mut mapped_desc =
            webgpu_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
        mapped_desc.mapped_at_creation(desc.mapped_at_creation);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(WebBuffer::new(
            device_data.0.create_buffer(&mapped_desc),
            desc,
        ))
    }

    fn device_create_texture(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::TextureDescriptor<'_>,
    ) -> Self::TextureData {
        let mut mapped_desc = webgpu_sys::GpuTextureDescriptor::new(
            map_texture_format(desc.format),
            &map_extent_3d(desc.size),
            desc.usage.bits(),
        );
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        mapped_desc.dimension(map_texture_dimension(desc.dimension));
        mapped_desc.mip_level_count(desc.mip_level_count);
        mapped_desc.sample_count(desc.sample_count);
        let mapped_view_formats = desc
            .view_formats
            .iter()
            .map(|format| JsValue::from(map_texture_format(*format)))
            .collect::<js_sys::Array>();
        mapped_desc.view_formats(&mapped_view_formats);
        Sendable(device_data.0.create_texture(&mapped_desc))
    }

    fn device_create_sampler(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::SamplerDescriptor<'_>,
    ) -> Self::SamplerData {
        let mut mapped_desc = webgpu_sys::GpuSamplerDescriptor::new();
        mapped_desc.address_mode_u(map_address_mode(desc.address_mode_u));
        mapped_desc.address_mode_v(map_address_mode(desc.address_mode_v));
        mapped_desc.address_mode_w(map_address_mode(desc.address_mode_w));
        if let Some(compare) = desc.compare {
            mapped_desc.compare(map_compare_function(compare));
        }
        mapped_desc.lod_max_clamp(desc.lod_max_clamp);
        mapped_desc.lod_min_clamp(desc.lod_min_clamp);
        mapped_desc.mag_filter(map_filter_mode(desc.mag_filter));
        mapped_desc.min_filter(map_filter_mode(desc.min_filter));
        mapped_desc.mipmap_filter(map_mipmap_filter_mode(desc.mipmap_filter));
        // TODO: `max_anisotropy` is not available on `desc` yet
        // mapped_desc.max_anisotropy(desc.max_anisotropy);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device_data.0.create_sampler_with_descriptor(&mapped_desc))
    }

    fn device_create_query_set(
        &self,
        device_data: &Self::DeviceData,
        desc: &wgt::QuerySetDescriptor<crate::Label<'_>>,
    ) -> Self::QuerySetData {
        let ty = match desc.ty {
            wgt::QueryType::Occlusion => webgpu_sys::GpuQueryType::Occlusion,
            wgt::QueryType::Timestamp => webgpu_sys::GpuQueryType::Timestamp,
            wgt::QueryType::PipelineStatistics(_) => unreachable!(),
        };
        let mut mapped_desc = webgpu_sys::GpuQuerySetDescriptor::new(desc.count, ty);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device_data.0.create_query_set(&mapped_desc))
    }

    fn device_create_command_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::CommandEncoderDescriptor<'_>,
    ) -> Self::CommandEncoderData {
        let mut mapped_desc = webgpu_sys::GpuCommandEncoderDescriptor::new();
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(
            device_data
                .0
                .create_command_encoder_with_descriptor(&mapped_desc),
        )
    }

    fn device_create_render_bundle_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::RenderBundleEncoderDescriptor<'_>,
    ) -> Self::RenderBundleEncoderData {
        let mapped_color_formats = desc
            .color_formats
            .iter()
            .map(|cf| match cf {
                Some(cf) => wasm_bindgen::JsValue::from(map_texture_format(*cf)),
                None => wasm_bindgen::JsValue::null(),
            })
            .collect::<js_sys::Array>();
        let mut mapped_desc =
            webgpu_sys::GpuRenderBundleEncoderDescriptor::new(&mapped_color_formats);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        if let Some(ds) = desc.depth_stencil {
            mapped_desc.depth_stencil_format(map_texture_format(ds.format));
            mapped_desc.depth_read_only(ds.depth_read_only);
            mapped_desc.stencil_read_only(ds.stencil_read_only);
        }
        mapped_desc.sample_count(desc.sample_count);
        Sendable(device_data.0.create_render_bundle_encoder(&mapped_desc))
    }

    fn device_drop(&self, _device_data: &Self::DeviceData) {
        // Device is dropped automatically
    }

    fn device_destroy(&self, device_data: &Self::DeviceData) {
        device_data.0.destroy();
    }

    fn queue_drop(&self, _queue_data: &Self::QueueData) {
        // Queue is dropped automatically
    }

    fn device_set_device_lost_callback(
        &self,
        device_data: &Self::DeviceData,
        device_lost_callback: crate::context::DeviceLostCallback,
    ) {
        use webgpu_sys::{GpuDeviceLostInfo, GpuDeviceLostReason};

        let closure = Closure::once(move |info: JsValue| {
            let info = info.dyn_into::<GpuDeviceLostInfo>().unwrap();
            device_lost_callback(
                match info.reason() {
                    GpuDeviceLostReason::Destroyed => crate::DeviceLostReason::Destroyed,
                    GpuDeviceLostReason::Unknown => crate::DeviceLostReason::Unknown,
                    _ => crate::DeviceLostReason::Unknown,
                },
                info.message(),
            );
        });
        let _ = device_data.0.lost().then(&closure);
    }

    fn device_poll(
        &self,
        _device_data: &Self::DeviceData,
        _maintain: crate::Maintain,
    ) -> crate::MaintainResult {
        // Device is polled automatically
        crate::MaintainResult::SubmissionQueueEmpty
    }

    fn device_on_uncaptured_error(
        &self,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let f = Closure::wrap(Box::new(move |event: webgpu_sys::GpuUncapturedErrorEvent| {
            let error = crate::Error::from_js(event.error().value_of());
            handler(error);
        }) as Box<dyn FnMut(_)>);
        device_data
            .0
            .set_onuncapturederror(Some(f.as_ref().unchecked_ref()));
        // TODO: This will leak the memory associated with the error handler by default.
        f.forget();
    }

    fn device_push_error_scope(&self, device_data: &Self::DeviceData, filter: crate::ErrorFilter) {
        device_data.0.push_error_scope(match filter {
            crate::ErrorFilter::OutOfMemory => webgpu_sys::GpuErrorFilter::OutOfMemory,
            crate::ErrorFilter::Validation => webgpu_sys::GpuErrorFilter::Validation,
            crate::ErrorFilter::Internal => webgpu_sys::GpuErrorFilter::Internal,
        });
    }

    fn device_pop_error_scope(&self, device_data: &Self::DeviceData) -> Self::PopErrorScopeFuture {
        let error_promise = device_data.0.pop_error_scope();
        MakeSendFuture::new(
            wasm_bindgen_futures::JsFuture::from(error_promise),
            future_pop_error_scope,
        )
    }

    fn buffer_map_async(
        &self,
        buffer_data: &Self::BufferData,
        mode: crate::MapMode,
        range: Range<wgt::BufferAddress>,
        callback: crate::context::BufferMapCallback,
    ) {
        let map_promise = buffer_data.0.buffer.map_async_with_f64_and_f64(
            map_map_mode(mode),
            range.start as f64,
            (range.end - range.start) as f64,
        );

        buffer_data.0.set_mapped_range(range);

        register_then_closures(&map_promise, callback, Ok(()), Err(crate::BufferAsyncError));
    }

    fn buffer_get_mapped_range(
        &self,
        buffer_data: &Self::BufferData,
        sub_range: Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        let actual_mapping = buffer_data.0.get_mapped_range(sub_range);
        let temporary_mapping = actual_mapping.to_vec();
        Box::new(BufferMappedRange {
            actual_mapping,
            temporary_mapping,
        })
    }

    fn buffer_unmap(&self, buffer_data: &Self::BufferData) {
        buffer_data.0.buffer.unmap();
        buffer_data.0.mapping.borrow_mut().mapped_buffer = None;
    }

    fn shader_get_compilation_info(
        &self,
        shader_data: &Self::ShaderModuleData,
    ) -> Self::CompilationInfoFuture {
        let compilation_info_promise = shader_data.0.module.get_compilation_info();
        let map_future = Box::new({
            let compilation_info = shader_data.0.compilation_info.clone();
            move |result| future_compilation_info(result, &compilation_info)
        });
        MakeSendFuture::new(
            wasm_bindgen_futures::JsFuture::from(compilation_info_promise),
            map_future,
        )
    }

    fn texture_create_view(
        &self,
        texture_data: &Self::TextureData,
        desc: &crate::TextureViewDescriptor<'_>,
    ) -> Self::TextureViewData {
        let mut mapped = webgpu_sys::GpuTextureViewDescriptor::new();
        if let Some(dim) = desc.dimension {
            mapped.dimension(map_texture_view_dimension(dim));
        }
        if let Some(format) = desc.format {
            mapped.format(map_texture_format(format));
        }
        mapped.aspect(map_texture_aspect(desc.aspect));
        mapped.base_array_layer(desc.base_array_layer);
        if let Some(count) = desc.array_layer_count {
            mapped.array_layer_count(count);
        }
        mapped.base_mip_level(desc.base_mip_level);
        if let Some(count) = desc.mip_level_count {
            mapped.mip_level_count(count);
        }
        if let Some(label) = desc.label {
            mapped.label(label);
        }
        Sendable(texture_data.0.create_view_with_descriptor(&mapped))
    }

    fn surface_drop(&self, _surface_data: &Self::SurfaceData) {
        // Dropped automatically
    }

    fn adapter_drop(&self, _adapter_data: &Self::AdapterData) {
        // Dropped automatically
    }

    fn buffer_destroy(&self, buffer_data: &Self::BufferData) {
        buffer_data.0.buffer.destroy();
    }

    fn buffer_drop(&self, _buffer_data: &Self::BufferData) {
        // Dropped automatically
    }

    fn texture_destroy(&self, texture_data: &Self::TextureData) {
        texture_data.0.destroy();
    }

    fn texture_drop(&self, _texture_data: &Self::TextureData) {
        // Dropped automatically
    }

    fn texture_view_drop(&self, _texture_view_data: &Self::TextureViewData) {
        // Dropped automatically
    }

    fn sampler_drop(&self, _sampler_data: &Self::SamplerData) {
        // Dropped automatically
    }

    fn query_set_drop(&self, _query_set_data: &Self::QuerySetData) {
        // Dropped automatically
    }

    fn bind_group_drop(&self, _bind_group_data: &Self::BindGroupData) {
        // Dropped automatically
    }

    fn bind_group_layout_drop(&self, _bind_group_layout_data: &Self::BindGroupLayoutData) {
        // Dropped automatically
    }

    fn pipeline_layout_drop(&self, _pipeline_layout_data: &Self::PipelineLayoutData) {
        // Dropped automatically
    }

    fn shader_module_drop(&self, _shader_module_data: &Self::ShaderModuleData) {
        // Dropped automatically
    }

    fn command_encoder_drop(&self, _command_encoder_data: &Self::CommandEncoderData) {
        // Dropped automatically
    }

    fn command_buffer_drop(&self, _command_buffer_data: &Self::CommandBufferData) {
        // Dropped automatically
    }

    fn render_bundle_drop(&self, _render_bundle_data: &Self::RenderBundleData) {
        // Dropped automatically
    }

    fn compute_pipeline_drop(&self, _pipeline_data: &Self::ComputePipelineData) {
        // Dropped automatically
    }

    fn render_pipeline_drop(&self, _pipeline_data: &Self::RenderPipelineData) {
        // Dropped automatically
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData {
        Sendable(pipeline_data.0.get_bind_group_layout(index))
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData {
        Sendable(pipeline_data.0.get_bind_group_layout(index))
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source_data: &Self::BufferData,
        source_offset: wgt::BufferAddress,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .copy_buffer_to_buffer_with_f64_and_f64_and_f64(
                &source_data.0.buffer,
                source_offset as f64,
                &destination_data.0.buffer,
                destination_offset as f64,
                copy_size as f64,
            )
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyBuffer<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        encoder_data
            .0
            .copy_buffer_to_texture_with_gpu_extent_3d_dict(
                &map_buffer_copy_view(source),
                &map_texture_copy_view(destination),
                &map_extent_3d(copy_size),
            )
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyBuffer<'_>,
        copy_size: wgt::Extent3d,
    ) {
        encoder_data
            .0
            .copy_texture_to_buffer_with_gpu_extent_3d_dict(
                &map_texture_copy_view(source),
                &map_buffer_copy_view(destination),
                &map_extent_3d(copy_size),
            )
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        encoder_data
            .0
            .copy_texture_to_texture_with_gpu_extent_3d_dict(
                &map_texture_copy_view(source),
                &map_texture_copy_view(destination),
                &map_extent_3d(copy_size),
            )
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::ComputePassDescriptor<'_>,
    ) -> Self::ComputePassData {
        let mut mapped_desc = webgpu_sys::GpuComputePassDescriptor::new();
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        if let Some(ref timestamp_writes) = desc.timestamp_writes {
            let query_set: &<ContextWebGpu as crate::Context>::QuerySetData =
                downcast_ref(timestamp_writes.query_set.data.as_ref());
            let mut writes = webgpu_sys::GpuComputePassTimestampWrites::new(&query_set.0);
            if let Some(index) = timestamp_writes.beginning_of_pass_write_index {
                writes.beginning_of_pass_write_index(index);
            }
            if let Some(index) = timestamp_writes.end_of_pass_write_index {
                writes.end_of_pass_write_index(index);
            }
            mapped_desc.timestamp_writes(&writes);
        }

        Sendable(
            encoder_data
                .0
                .begin_compute_pass_with_descriptor(&mapped_desc),
        )
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::RenderPassDescriptor<'_>,
    ) -> Self::RenderPassData {
        let mapped_color_attachments = desc
            .color_attachments
            .iter()
            .map(|attachment| match attachment {
                Some(ca) => {
                    let mut clear_value: Option<wasm_bindgen::JsValue> = None;
                    let load_value = match ca.ops.load {
                        crate::LoadOp::Clear(color) => {
                            clear_value = Some(wasm_bindgen::JsValue::from(map_color(color)));
                            webgpu_sys::GpuLoadOp::Clear
                        }
                        crate::LoadOp::Load => webgpu_sys::GpuLoadOp::Load,
                    };

                    let view: &<ContextWebGpu as crate::Context>::TextureViewData =
                        downcast_ref(ca.view.data.as_ref());

                    let mut mapped_color_attachment = webgpu_sys::GpuRenderPassColorAttachment::new(
                        load_value,
                        map_store_op(ca.ops.store),
                        &view.0,
                    );
                    if let Some(cv) = clear_value {
                        mapped_color_attachment.clear_value(&cv);
                    }
                    if let Some(rt) = ca.resolve_target {
                        let resolve_target_view: &<ContextWebGpu as crate::Context>::TextureViewData =
                            downcast_ref(rt.data.as_ref());
                        mapped_color_attachment.resolve_target(&resolve_target_view.0);
                    }
                    mapped_color_attachment.store_op(map_store_op(ca.ops.store));

                    wasm_bindgen::JsValue::from(mapped_color_attachment)
                }
                None => wasm_bindgen::JsValue::null(),
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = webgpu_sys::GpuRenderPassDescriptor::new(&mapped_color_attachments);

        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        if let Some(dsa) = &desc.depth_stencil_attachment {
            let depth_stencil_attachment: &<ContextWebGpu as crate::Context>::TextureViewData =
                downcast_ref(dsa.view.data.as_ref());
            let mut mapped_depth_stencil_attachment =
                webgpu_sys::GpuRenderPassDepthStencilAttachment::new(&depth_stencil_attachment.0);
            if let Some(ref ops) = dsa.depth_ops {
                let load_op = match ops.load {
                    crate::LoadOp::Clear(v) => {
                        mapped_depth_stencil_attachment.depth_clear_value(v);
                        webgpu_sys::GpuLoadOp::Clear
                    }
                    crate::LoadOp::Load => webgpu_sys::GpuLoadOp::Load,
                };
                mapped_depth_stencil_attachment.depth_load_op(load_op);
                mapped_depth_stencil_attachment.depth_store_op(map_store_op(ops.store));
            }
            mapped_depth_stencil_attachment.depth_read_only(dsa.depth_ops.is_none());
            if let Some(ref ops) = dsa.stencil_ops {
                let load_op = match ops.load {
                    crate::LoadOp::Clear(v) => {
                        mapped_depth_stencil_attachment.stencil_clear_value(v);
                        webgpu_sys::GpuLoadOp::Clear
                    }
                    crate::LoadOp::Load => webgpu_sys::GpuLoadOp::Load,
                };
                mapped_depth_stencil_attachment.stencil_load_op(load_op);
                mapped_depth_stencil_attachment.stencil_store_op(map_store_op(ops.store));
            }
            mapped_depth_stencil_attachment.stencil_read_only(dsa.stencil_ops.is_none());
            mapped_desc.depth_stencil_attachment(&mapped_depth_stencil_attachment);
        }

        if let Some(ref timestamp_writes) = desc.timestamp_writes {
            let query_set: &<ContextWebGpu as crate::Context>::QuerySetData =
                downcast_ref(timestamp_writes.query_set.data.as_ref());
            let mut writes = webgpu_sys::GpuRenderPassTimestampWrites::new(&query_set.0);
            if let Some(index) = timestamp_writes.beginning_of_pass_write_index {
                writes.beginning_of_pass_write_index(index);
            }
            if let Some(index) = timestamp_writes.end_of_pass_write_index {
                writes.end_of_pass_write_index(index);
            }
            mapped_desc.timestamp_writes(&writes);
        }

        Sendable(encoder_data.0.begin_render_pass(&mapped_desc))
    }

    fn command_encoder_finish(
        &self,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> Self::CommandBufferData {
        let label = encoder_data.0.label();
        Sendable(if label.is_empty() {
            encoder_data.0.finish()
        } else {
            let mut mapped_desc = webgpu_sys::GpuCommandBufferDescriptor::new();
            mapped_desc.label(&label);
            encoder_data.0.finish_with_descriptor(&mapped_desc)
        })
    }

    fn command_encoder_clear_texture(
        &self,
        _encoder_data: &Self::CommandEncoderData,
        _texture_data: &Self::TextureData,
        _subresource_range: &wgt::ImageSubresourceRange,
    ) {
        //TODO
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) {
        match size {
            Some(size) => encoder_data.0.clear_buffer_with_f64_and_f64(
                &buffer_data.0.buffer,
                offset as f64,
                size as f64,
            ),
            None => encoder_data
                .0
                .clear_buffer_with_f64(&buffer_data.0.buffer, offset as f64),
        }
    }

    fn command_encoder_insert_debug_marker(
        &self,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // encoder.insert_debug_marker(label);
    }

    fn command_encoder_push_debug_group(
        &self,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // encoder.push_debug_group(label);
    }

    fn command_encoder_pop_debug_group(&self, _encoder_data: &Self::CommandEncoderData) {
        // Not available in gecko yet
        // encoder.pop_debug_group();
    }

    fn command_encoder_write_timestamp(
        &self,
        _encoder_data: &Self::CommandEncoderData,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        // Not available on WebGPU.
        // This was part of the spec originally but got removed, see https://github.com/gpuweb/gpuweb/pull/4370
        panic!("TIMESTAMP_QUERY_INSIDE_ENCODERS feature must be enabled to call write_timestamp on a command encoder.")
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder_data: &Self::CommandEncoderData,
        query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
    ) {
        encoder_data.0.resolve_query_set_with_u32(
            &query_set_data.0,
            first_query,
            query_count,
            &destination_data.0.buffer,
            destination_offset as u32,
        );
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &crate::RenderBundleDescriptor<'_>,
    ) -> Self::RenderBundleData {
        Sendable(match desc.label {
            Some(label) => {
                let mut mapped_desc = webgpu_sys::GpuRenderBundleDescriptor::new();
                mapped_desc.label(label);
                encoder_data.0.finish_with_descriptor(&mapped_desc)
            }
            None => encoder_data.0.finish(),
        })
    }

    fn queue_write_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        /* Skip the copy once gecko allows BufferSource instead of ArrayBuffer
        queue_data.0.write_buffer_with_f64_and_u8_array_and_f64_and_f64(
            &buffer_data.0,
            offset as f64,
            data,
            0f64,
            data.len() as f64,
        );
        */
        queue_data
            .0
            .write_buffer_with_f64_and_buffer_source_and_f64_and_f64(
                &buffer_data.0.buffer,
                offset as f64,
                &js_sys::Uint8Array::from(data).buffer(),
                0f64,
                data.len() as f64,
            );
    }

    fn queue_validate_write_buffer(
        &self,
        _queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let usage = wgt::BufferUsages::from_bits_truncate(buffer_data.0.buffer.usage());
        // TODO: actually send this down the error scope
        if !usage.contains(wgt::BufferUsages::COPY_DST) {
            log::error!("Destination buffer is missing the `COPY_DST` usage flag");
            return None;
        }
        let write_size = u64::from(size);
        if write_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            log::error!(
                "Copy size {} does not respect `COPY_BUFFER_ALIGNMENT`",
                size
            );
            return None;
        }
        if offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            log::error!(
                "Buffer offset {} is not aligned to block size or `COPY_BUFFER_ALIGNMENT`",
                offset
            );
            return None;
        }
        if write_size + offset > buffer_data.0.buffer.size() as u64 {
            log::error!("copy of {}..{} would end up overrunning the bounds of the destination buffer of size {}", offset, offset + write_size, buffer_data.0.buffer.size());
            return None;
        }
        Some(())
    }

    fn queue_create_staging_buffer(
        &self,
        _queue_data: &Self::QueueData,
        size: wgt::BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>> {
        Some(Box::new(WebQueueWriteBuffer(
            vec![0; size.get() as usize].into_boxed_slice(),
        )))
    }

    fn queue_write_staging_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    ) {
        let staging_buffer = staging_buffer
            .as_any()
            .downcast_ref::<WebQueueWriteBuffer>()
            .unwrap()
            .slice();
        self.queue_write_buffer(queue_data, buffer_data, offset, staging_buffer)
    }

    fn queue_write_texture(
        &self,
        queue_data: &Self::QueueData,
        texture: crate::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    ) {
        let mut mapped_data_layout = webgpu_sys::GpuImageDataLayout::new();
        if let Some(bytes_per_row) = data_layout.bytes_per_row {
            mapped_data_layout.bytes_per_row(bytes_per_row);
        }
        if let Some(rows_per_image) = data_layout.rows_per_image {
            mapped_data_layout.rows_per_image(rows_per_image);
        }
        mapped_data_layout.offset(data_layout.offset as f64);

        /* Skip the copy once gecko allows BufferSource instead of ArrayBuffer
        queue_data.0.write_texture_with_u8_array_and_gpu_extent_3d_dict(
            &map_texture_copy_view(texture),
            data,
            &mapped_data_layout,
            &map_extent_3d(size),
        );
        */
        queue_data
            .0
            .write_texture_with_buffer_source_and_gpu_extent_3d_dict(
                &map_texture_copy_view(texture),
                &js_sys::Uint8Array::from(data).buffer(),
                &mapped_data_layout,
                &map_extent_3d(size),
            );
    }

    fn queue_copy_external_image_to_texture(
        &self,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    ) {
        queue_data
            .0
            .copy_external_image_to_texture_with_gpu_extent_3d_dict(
                &map_external_texture_copy_view(source),
                &map_tagged_texture_copy_view(dest),
                &map_extent_3d(size),
            );
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferData>>(
        &self,
        queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> Self::SubmissionIndexData {
        let temp_command_buffers = command_buffers
            .map(|data| data.0)
            .collect::<js_sys::Array>();

        queue_data.0.submit(&temp_command_buffers);
    }

    fn queue_get_timestamp_period(&self, _queue_data: &Self::QueueData) -> f32 {
        // Timestamp values are always in nanoseconds, see https://gpuweb.github.io/gpuweb/#timestamp
        1.0
    }

    fn queue_on_submitted_work_done(
        &self,
        _queue_data: &Self::QueueData,
        _callback: crate::context::SubmittedWorkDoneCallback,
    ) {
        unimplemented!()
    }

    fn device_start_capture(&self, _device_data: &Self::DeviceData) {}
    fn device_stop_capture(&self, _device_data: &Self::DeviceData) {}

    fn device_get_internal_counters(
        &self,
        _device_data: &Self::DeviceData,
    ) -> wgt::InternalCounters {
        Default::default()
    }

    fn device_generate_allocator_report(
        &self,
        _device_data: &Self::DeviceData,
    ) -> Option<wgt::AllocatorReport> {
        None
    }

    fn pipeline_cache_get_data(&self, _: &Self::PipelineCacheData) -> Option<Vec<u8>> {
        None
    }

    fn compute_pass_set_pipeline(
        &self,
        pass_data: &mut Self::ComputePassData,
        pipeline_data: &Self::ComputePipelineData,
    ) {
        pass_data.0.set_pipeline(&pipeline_data.0)
    }

    fn compute_pass_set_bind_group(
        &self,
        pass_data: &mut Self::ComputePassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        if bind_group_data.is_none() {
            // TODO: Handle the None case.
            return;
        }
        let bind_group_data = bind_group_data.unwrap();
        if offsets.is_empty() {
            pass_data.0.set_bind_group(index, Some(&bind_group_data.0));
        } else {
            pass_data
                .0
                .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                    index,
                    Some(&bind_group_data.0),
                    offsets,
                    0f64,
                    offsets.len() as u32,
                );
        }
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass_data: &mut Self::ComputePassData,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass_data: &mut Self::ComputePassData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // self.0.insert_debug_marker(label);
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass_data: &mut Self::ComputePassData,
        _group_label: &str,
    ) {
        // Not available in gecko yet
        // self.0.push_debug_group(group_label);
    }

    fn compute_pass_pop_debug_group(&self, _pass_data: &mut Self::ComputePassData) {
        // Not available in gecko yet
        // self.0.pop_debug_group();
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass_data: &mut Self::ComputePassData,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        panic!("TIMESTAMP_QUERY_INSIDE_PASSES feature must be enabled to call write_timestamp in a compute pass.")
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass_data: &mut Self::ComputePassData,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        // Not available in gecko yet
    }

    fn compute_pass_end_pipeline_statistics_query(&self, _pass_data: &mut Self::ComputePassData) {
        // Not available in gecko yet
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    ) {
        pass_data
            .0
            .dispatch_workgroups_with_workgroup_count_y_and_workgroup_count_z(x, y, z);
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_data: &mut Self::ComputePassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data.0.dispatch_workgroups_indirect_with_f64(
            &indirect_buffer_data.0.buffer,
            indirect_offset as f64,
        );
    }

    fn compute_pass_end(&self, pass_data: &mut Self::ComputePassData) {
        pass_data.0.end();
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        encoder_data.0.set_pipeline(&pipeline_data.0);
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        if bind_group_data.is_none() {
            // TODO: Handle the None case.
            return;
        }
        let bind_group_data = bind_group_data.unwrap();
        if offsets.is_empty() {
            encoder_data
                .0
                .set_bind_group(index, Some(&bind_group_data.0));
        } else {
            encoder_data
                .0
                .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                    index,
                    Some(&bind_group_data.0),
                    offsets,
                    0f64,
                    offsets.len() as u32,
                );
        }
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                encoder_data.0.set_index_buffer_with_f64_and_f64(
                    &buffer_data.0.buffer,
                    map_index_format(index_format),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                encoder_data.0.set_index_buffer_with_f64(
                    &buffer_data.0.buffer,
                    map_index_format(index_format),
                    offset as f64,
                );
            }
        };
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                encoder_data.0.set_vertex_buffer_with_f64_and_f64(
                    slot,
                    Some(&buffer_data.0.buffer),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                encoder_data.0.set_vertex_buffer_with_f64(
                    slot,
                    Some(&buffer_data.0.buffer),
                    offset as f64,
                );
            }
        };
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_bundle_encoder_draw(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        encoder_data
            .0
            .draw_with_instance_count_and_first_vertex_and_first_instance(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        encoder_data
            .0
            .draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .draw_indirect_with_f64(&indirect_buffer_data.0.buffer, indirect_offset as f64);
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .draw_indexed_indirect_with_f64(&indirect_buffer_data.0.buffer, indirect_offset as f64);
    }

    fn render_pass_set_pipeline(
        &self,
        pass_data: &mut Self::RenderPassData,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        pass_data.0.set_pipeline(&pipeline_data.0);
    }

    fn render_pass_set_bind_group(
        &self,
        pass_data: &mut Self::RenderPassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        if bind_group_data.is_none() {
            // TODO: Handle the None case.
            return;
        }
        let bind_group_data = bind_group_data.unwrap();
        if offsets.is_empty() {
            pass_data.0.set_bind_group(index, Some(&bind_group_data.0));
        } else {
            pass_data
                .0
                .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                    index,
                    Some(&bind_group_data.0),
                    offsets,
                    0f64,
                    offsets.len() as u32,
                );
        }
    }

    fn render_pass_set_index_buffer(
        &self,
        pass_data: &mut Self::RenderPassData,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                pass_data.0.set_index_buffer_with_f64_and_f64(
                    &buffer_data.0.buffer,
                    map_index_format(index_format),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                pass_data.0.set_index_buffer_with_f64(
                    &buffer_data.0.buffer,
                    map_index_format(index_format),
                    offset as f64,
                );
            }
        };
    }

    fn render_pass_set_vertex_buffer(
        &self,
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                pass_data.0.set_vertex_buffer_with_f64_and_f64(
                    slot,
                    Some(&buffer_data.0.buffer),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                pass_data.0.set_vertex_buffer_with_f64(
                    slot,
                    Some(&buffer_data.0.buffer),
                    offset as f64,
                );
            }
        };
    }

    fn render_pass_set_push_constants(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_pass_draw(
        &self,
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        pass_data
            .0
            .draw_with_instance_count_and_first_vertex_and_first_instance(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
    }

    fn render_pass_draw_indexed(
        &self,
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        pass_data
            .0
            .draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
    }

    fn render_pass_draw_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data
            .0
            .draw_indirect_with_f64(&indirect_buffer_data.0.buffer, indirect_offset as f64);
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data
            .0
            .draw_indexed_indirect_with_f64(&indirect_buffer_data.0.buffer, indirect_offset as f64);
    }

    fn render_pass_multi_draw_indirect(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indirect")
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        panic!(
            "MULTI_DRAW_INDIRECT_COUNT feature must be enabled to call multi_draw_indirect_count"
        )
    }

    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT_COUNT feature must be enabled to call multi_draw_indexed_indirect_count")
    }

    fn render_pass_set_blend_constant(
        &self,
        pass_data: &mut Self::RenderPassData,
        color: wgt::Color,
    ) {
        pass_data
            .0
            .set_blend_constant_with_gpu_color_dict(&map_color(color));
    }

    fn render_pass_set_scissor_rect(
        &self,
        pass_data: &mut Self::RenderPassData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        pass_data.0.set_scissor_rect(x, y, width, height);
    }

    fn render_pass_set_viewport(
        &self,
        pass_data: &mut Self::RenderPassData,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        pass_data
            .0
            .set_viewport(x, y, width, height, min_depth, max_depth);
    }

    fn render_pass_set_stencil_reference(
        &self,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    ) {
        pass_data.0.set_stencil_reference(reference);
    }

    fn render_pass_insert_debug_marker(&self, _pass_data: &mut Self::RenderPassData, _label: &str) {
        // Not available in gecko yet
        // self.0.insert_debug_marker(label);
    }

    fn render_pass_push_debug_group(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _group_label: &str,
    ) {
        // Not available in gecko yet
        // self.0.push_debug_group(group_label);
    }

    fn render_pass_pop_debug_group(&self, _pass_data: &mut Self::RenderPassData) {
        // Not available in gecko yet
        // self.0.pop_debug_group();
    }

    fn render_pass_write_timestamp(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        panic!("TIMESTAMP_QUERY_INSIDE_PASSES feature must be enabled to call write_timestamp in a render pass.")
    }

    fn render_pass_begin_occlusion_query(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _query_index: u32,
    ) {
        // Not available in gecko yet
    }

    fn render_pass_end_occlusion_query(&self, _pass_data: &mut Self::RenderPassData) {
        // Not available in gecko yet
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass_data: &mut Self::RenderPassData,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        // Not available in gecko yet
    }

    fn render_pass_end_pipeline_statistics_query(&self, _pass_data: &mut Self::RenderPassData) {
        // Not available in gecko yet
    }

    fn render_pass_execute_bundles(
        &self,
        pass_data: &mut Self::RenderPassData,
        render_bundles: &mut dyn Iterator<Item = &Self::RenderBundleData>,
    ) {
        let mapped = render_bundles
            .map(|bundle_data| &bundle_data.0)
            .collect::<js_sys::Array>();
        pass_data.0.execute_bundles(&mapped);
    }

    fn render_pass_end(&self, pass_data: &mut Self::RenderPassData) {
        pass_data.0.end();
    }
}

pub(crate) type SurfaceOutputDetail = ();

#[derive(Debug)]
pub struct WebQueueWriteBuffer(Box<[u8]>);

impl QueueWriteBuffer for WebQueueWriteBuffer {
    fn slice(&self) -> &[u8] {
        &self.0
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Stores the state of a GPU buffer and a reference to its mapped `ArrayBuffer` (if any).
/// The WebGPU specification forbids calling `getMappedRange` on a `webgpu_sys::GpuBuffer` more than
/// once, so this struct stores the initial mapped range and re-uses it, allowing for multiple `get_mapped_range`
/// calls on the Rust-side.
#[derive(Debug)]
pub struct WebBuffer {
    /// The associated GPU buffer.
    buffer: webgpu_sys::GpuBuffer,
    /// The mapped array buffer and mapped range.
    mapping: RefCell<WebBufferMapState>,
}

impl WebBuffer {
    /// Creates a new web buffer for the given Javascript object and description.
    fn new(buffer: webgpu_sys::GpuBuffer, desc: &crate::BufferDescriptor<'_>) -> Self {
        Self {
            buffer,
            mapping: RefCell::new(WebBufferMapState {
                mapped_buffer: None,
                range: 0..desc.size,
            }),
        }
    }

    /// Creates a raw Javascript array buffer over the provided range.
    fn get_mapped_array_buffer(&self, sub_range: Range<wgt::BufferAddress>) -> js_sys::ArrayBuffer {
        self.buffer.get_mapped_range_with_f64_and_f64(
            sub_range.start as f64,
            (sub_range.end - sub_range.start) as f64,
        )
    }

    /// Obtains a reference to the re-usable buffer mapping as a Javascript array view.
    fn get_mapped_range(&self, sub_range: Range<wgt::BufferAddress>) -> js_sys::Uint8Array {
        let mut mapping = self.mapping.borrow_mut();
        let range = mapping.range.clone();
        let array_buffer = mapping.mapped_buffer.get_or_insert_with(|| {
            self.buffer.get_mapped_range_with_f64_and_f64(
                range.start as f64,
                (range.end - range.start) as f64,
            )
        });
        js_sys::Uint8Array::new_with_byte_offset_and_length(
            array_buffer,
            (sub_range.start - range.start) as u32,
            (sub_range.end - sub_range.start) as u32,
        )
    }

    /// Sets the range of the buffer which is presently mapped.
    fn set_mapped_range(&self, range: Range<wgt::BufferAddress>) {
        self.mapping.borrow_mut().range = range;
    }
}

/// Remembers which portion of a buffer has been mapped, along with a reference
/// to the mapped portion.
#[derive(Debug)]
struct WebBufferMapState {
    /// The mapped memory of the buffer.
    pub mapped_buffer: Option<js_sys::ArrayBuffer>,
    /// The total range which has been mapped in the buffer overall.
    pub range: Range<wgt::BufferAddress>,
}

#[derive(Debug)]
pub struct BufferMappedRange {
    actual_mapping: js_sys::Uint8Array,
    temporary_mapping: Vec<u8>,
}

impl crate::context::BufferMappedRange for BufferMappedRange {
    #[inline]
    fn slice(&self) -> &[u8] {
        &self.temporary_mapping
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.temporary_mapping
    }
}

impl Drop for BufferMappedRange {
    fn drop(&mut self) {
        // Copy from the temporary mapping back into the array buffer that was
        // originally provided by the browser
        let temporary_mapping_slice = self.temporary_mapping.as_slice();
        unsafe {
            // Note: no allocations can happen between `view` and `set`, or this
            // will break
            self.actual_mapping
                .set(&js_sys::Uint8Array::view(temporary_mapping_slice), 0);
        }
    }
}

/// Adds the constants map to the given pipeline descriptor if the map is nonempty.
/// Panics if the map cannot be set.
///
/// This function is necessary because the constants array is not currently
/// exposed by `wasm-bindgen`. See the following issues for details:
/// - [gfx-rs/wgpu#5688](https://github.com/gfx-rs/wgpu/pull/5688)
/// - [rustwasm/wasm-bindgen#3587](https://github.com/rustwasm/wasm-bindgen/issues/3587)
fn insert_constants_map(target: &JsValue, map: &HashMap<String, f64>) {
    if !map.is_empty() {
        js_sys::Reflect::set(target, &"constants".into(), &hashmap_to_jsvalue(map))
            .expect("Setting the values in a Javascript pipeline descriptor should never fail");
    }
}

/// Converts a hashmap to a Javascript object.
fn hashmap_to_jsvalue(map: &HashMap<String, f64>) -> JsValue {
    let obj = js_sys::Object::new();

    for (k, v) in map.iter() {
        js_sys::Reflect::set(&obj, &k.into(), &(*v).into())
            .expect("Setting the values in a Javascript map should never fail");
    }

    JsValue::from(obj)
}
