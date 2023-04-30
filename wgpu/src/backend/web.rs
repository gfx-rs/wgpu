#![allow(clippy::type_complexity)]

use js_sys::Promise;
use std::{
    any::Any,
    cell::RefCell,
    fmt,
    future::Future,
    marker::PhantomData,
    ops::Range,
    pin::Pin,
    rc::Rc,
    task::{self, Poll},
};
use wasm_bindgen::{prelude::*, JsCast};

use crate::{
    context::{downcast_ref, ObjectId, QueueWriteBuffer, Unused},
    UncapturedErrorHandler,
};

fn create_identified<T>(value: T) -> (Identified<T>, Sendable<T>) {
    cfg_if::cfg_if! {
        if #[cfg(feature = "expose-ids")] {
            static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            (Identified(core::num::NonZeroU64::new(id).unwrap(), PhantomData), Sendable(value))
        } else {
            (Identified(PhantomData), Sendable(value))
        }
    }
}

// We need to make a wrapper for some of the handle types returned by the web backend to make them
// implement `Send` and `Sync` to match native.
//
// SAFETY: All webgpu handle types in wasm32 are internally a `JsValue`, and `JsValue` is neither
// Send nor Sync.  Currently, wasm32 has no threading support so implementing `Send` or `Sync` for a
// type is (for now) harmless.  Eventually wasm32 will support threading, and depending on how this
// is integrated (or not integrated) with values like those in webgpu, this may become unsound.

#[allow(unused_variables)]
impl<T> From<ObjectId> for Identified<T> {
    fn from(object_id: ObjectId) -> Self {
        Self(
            #[cfg(feature = "expose-ids")]
            object_id.global_id(),
            PhantomData,
        )
    }
}

#[allow(unused_variables)]
impl<T> From<Identified<T>> for ObjectId {
    fn from(identified: Identified<T>) -> Self {
        Self::new(
            // TODO: the ID isn't used, so we hardcode it to 1 for now until we rework this
            // API.
            core::num::NonZeroU64::new(1).unwrap(),
            #[cfg(feature = "expose-ids")]
            identified.0,
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Sendable<T>(T);
unsafe impl<T> Send for Sendable<T> {}
unsafe impl<T> Sync for Sendable<T> {}

#[derive(Clone, Debug)]
pub(crate) struct Identified<T>(
    #[cfg(feature = "expose-ids")] std::num::NonZeroU64,
    PhantomData<T>,
);
unsafe impl<T> Send for Identified<T> {}
unsafe impl<T> Sync for Identified<T> {}

pub(crate) struct Context(web_sys::Gpu);
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("type", &"Web").finish()
    }
}

impl crate::Error {
    fn from_js(js_error: js_sys::Object) -> Self {
        let source = Box::<dyn std::error::Error + Send + Sync>::from("<WebGPU Error>");
        if let Some(js_error) = js_error.dyn_ref::<web_sys::GpuValidationError>() {
            crate::Error::Validation {
                source,
                description: js_error.message(),
            }
        } else if js_error.has_type::<web_sys::GpuOutOfMemoryError>() {
            crate::Error::OutOfMemory { source }
        } else {
            panic!("Unexpected error");
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

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context) -> Poll<Self::Output> {
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

unsafe impl<F, M> Send for MakeSendFuture<F, M> {}

fn map_texture_format(texture_format: wgt::TextureFormat) -> web_sys::GpuTextureFormat {
    use web_sys::GpuTextureFormat as tf;
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
        TextureFormat::Rgb10a2Unorm => tf::Rgb10a2unorm,
        TextureFormat::Rg11b10Float => tf::Rg11b10ufloat,
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
                unimplemented!("Format {texture_format:?} has no WebGPU equivilant")
            }
        },
        _ => unimplemented!("Format {texture_format:?} has no WebGPU equivilant"),
    }
}

fn map_texture_component_type(
    sample_type: wgt::TextureSampleType,
) -> web_sys::GpuTextureSampleType {
    use web_sys::GpuTextureSampleType as ts;
    use wgt::TextureSampleType;
    match sample_type {
        TextureSampleType::Float { filterable: true } => ts::Float,
        TextureSampleType::Float { filterable: false } => ts::UnfilterableFloat,
        TextureSampleType::Sint => ts::Sint,
        TextureSampleType::Uint => ts::Uint,
        TextureSampleType::Depth => ts::Depth,
    }
}

fn map_cull_mode(cull_mode: Option<wgt::Face>) -> web_sys::GpuCullMode {
    use web_sys::GpuCullMode as cm;
    use wgt::Face;
    match cull_mode {
        None => cm::None,
        Some(Face::Front) => cm::Front,
        Some(Face::Back) => cm::Back,
    }
}

fn map_front_face(front_face: wgt::FrontFace) -> web_sys::GpuFrontFace {
    use web_sys::GpuFrontFace as ff;
    use wgt::FrontFace;
    match front_face {
        FrontFace::Ccw => ff::Ccw,
        FrontFace::Cw => ff::Cw,
    }
}

fn map_primitive_state(primitive: &wgt::PrimitiveState) -> web_sys::GpuPrimitiveState {
    use web_sys::GpuPrimitiveTopology as pt;
    use wgt::PrimitiveTopology;

    let mut mapped = web_sys::GpuPrimitiveState::new();
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

    mapped
}

fn map_compare_function(compare_fn: wgt::CompareFunction) -> web_sys::GpuCompareFunction {
    use web_sys::GpuCompareFunction as cf;
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

fn map_stencil_operation(op: wgt::StencilOperation) -> web_sys::GpuStencilOperation {
    use web_sys::GpuStencilOperation as so;
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

fn map_stencil_state_face(desc: &wgt::StencilFaceState) -> web_sys::GpuStencilFaceState {
    let mut mapped = web_sys::GpuStencilFaceState::new();
    mapped.compare(map_compare_function(desc.compare));
    mapped.depth_fail_op(map_stencil_operation(desc.depth_fail_op));
    mapped.fail_op(map_stencil_operation(desc.fail_op));
    mapped.pass_op(map_stencil_operation(desc.pass_op));
    mapped
}

fn map_depth_stencil_state(desc: &wgt::DepthStencilState) -> web_sys::GpuDepthStencilState {
    let mut mapped = web_sys::GpuDepthStencilState::new(map_texture_format(desc.format));
    mapped.depth_bias(desc.bias.constant);
    mapped.depth_bias_clamp(desc.bias.clamp);
    mapped.depth_bias_slope_scale(desc.bias.slope_scale);
    mapped.depth_compare(map_compare_function(desc.depth_compare));
    mapped.depth_write_enabled(desc.depth_write_enabled);
    mapped.stencil_back(&map_stencil_state_face(&desc.stencil.back));
    mapped.stencil_front(&map_stencil_state_face(&desc.stencil.front));
    mapped.stencil_read_mask(desc.stencil.read_mask);
    mapped.stencil_write_mask(desc.stencil.write_mask);
    mapped
}

fn map_blend_component(desc: &wgt::BlendComponent) -> web_sys::GpuBlendComponent {
    let mut mapped = web_sys::GpuBlendComponent::new();
    mapped.dst_factor(map_blend_factor(desc.dst_factor));
    mapped.operation(map_blend_operation(desc.operation));
    mapped.src_factor(map_blend_factor(desc.src_factor));
    mapped
}

fn map_blend_factor(factor: wgt::BlendFactor) -> web_sys::GpuBlendFactor {
    use web_sys::GpuBlendFactor as bf;
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
    }
}

fn map_blend_operation(op: wgt::BlendOperation) -> web_sys::GpuBlendOperation {
    use web_sys::GpuBlendOperation as bo;
    use wgt::BlendOperation;
    match op {
        BlendOperation::Add => bo::Add,
        BlendOperation::Subtract => bo::Subtract,
        BlendOperation::ReverseSubtract => bo::ReverseSubtract,
        BlendOperation::Min => bo::Min,
        BlendOperation::Max => bo::Max,
    }
}

fn map_index_format(format: wgt::IndexFormat) -> web_sys::GpuIndexFormat {
    use web_sys::GpuIndexFormat as f;
    use wgt::IndexFormat;
    match format {
        IndexFormat::Uint16 => f::Uint16,
        IndexFormat::Uint32 => f::Uint32,
    }
}

fn map_vertex_format(format: wgt::VertexFormat) -> web_sys::GpuVertexFormat {
    use web_sys::GpuVertexFormat as vf;
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
        VertexFormat::Float64
        | VertexFormat::Float64x2
        | VertexFormat::Float64x3
        | VertexFormat::Float64x4 => {
            panic!("VERTEX_ATTRIBUTE_64BIT feature must be enabled to use Double formats")
        }
    }
}

fn map_vertex_step_mode(mode: wgt::VertexStepMode) -> web_sys::GpuVertexStepMode {
    use web_sys::GpuVertexStepMode as sm;
    use wgt::VertexStepMode;
    match mode {
        VertexStepMode::Vertex => sm::Vertex,
        VertexStepMode::Instance => sm::Instance,
    }
}

fn map_extent_3d(extent: wgt::Extent3d) -> web_sys::GpuExtent3dDict {
    let mut mapped = web_sys::GpuExtent3dDict::new(extent.width);
    mapped.height(extent.height);
    mapped.depth_or_array_layers(extent.depth_or_array_layers);
    mapped
}

fn map_origin_2d(extent: wgt::Origin2d) -> web_sys::GpuOrigin2dDict {
    let mut mapped = web_sys::GpuOrigin2dDict::new();
    mapped.x(extent.x);
    mapped.y(extent.y);
    mapped
}

fn map_origin_3d(origin: wgt::Origin3d) -> web_sys::GpuOrigin3dDict {
    let mut mapped = web_sys::GpuOrigin3dDict::new();
    mapped.x(origin.x);
    mapped.y(origin.y);
    mapped.z(origin.z);
    mapped
}

fn map_texture_dimension(texture_dimension: wgt::TextureDimension) -> web_sys::GpuTextureDimension {
    match texture_dimension {
        wgt::TextureDimension::D1 => web_sys::GpuTextureDimension::N1d,
        wgt::TextureDimension::D2 => web_sys::GpuTextureDimension::N2d,
        wgt::TextureDimension::D3 => web_sys::GpuTextureDimension::N3d,
    }
}

fn map_texture_view_dimension(
    texture_view_dimension: wgt::TextureViewDimension,
) -> web_sys::GpuTextureViewDimension {
    use web_sys::GpuTextureViewDimension as tvd;
    match texture_view_dimension {
        wgt::TextureViewDimension::D1 => tvd::N1d,
        wgt::TextureViewDimension::D2 => tvd::N2d,
        wgt::TextureViewDimension::D2Array => tvd::N2dArray,
        wgt::TextureViewDimension::Cube => tvd::Cube,
        wgt::TextureViewDimension::CubeArray => tvd::CubeArray,
        wgt::TextureViewDimension::D3 => tvd::N3d,
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer) -> web_sys::GpuImageCopyBuffer {
    let buffer: &<Context as crate::Context>::BufferData = downcast_ref(view.buffer.data.as_ref());
    let mut mapped = web_sys::GpuImageCopyBuffer::new(&buffer.0);
    if let Some(bytes_per_row) = view.layout.bytes_per_row {
        mapped.bytes_per_row(bytes_per_row);
    }
    if let Some(rows_per_image) = view.layout.rows_per_image {
        mapped.rows_per_image(rows_per_image);
    }
    mapped.offset(view.layout.offset as f64);
    mapped
}

fn map_texture_copy_view(view: crate::ImageCopyTexture) -> web_sys::GpuImageCopyTexture {
    let texture: &<Context as crate::Context>::TextureData =
        downcast_ref(view.texture.data.as_ref());
    let mut mapped = web_sys::GpuImageCopyTexture::new(&texture.0);
    mapped.mip_level(view.mip_level);
    mapped.origin(&map_origin_3d(view.origin));
    mapped
}

fn map_tagged_texture_copy_view(
    view: crate::ImageCopyTextureTagged,
) -> web_sys::GpuImageCopyTextureTagged {
    let texture: &<Context as crate::Context>::TextureData =
        downcast_ref(view.texture.data.as_ref());
    let mut mapped = web_sys::GpuImageCopyTextureTagged::new(&texture.0);
    mapped.mip_level(view.mip_level);
    mapped.origin(&map_origin_3d(view.origin));
    mapped.aspect(map_texture_aspect(view.aspect));
    // mapped.color_space(map_color_space(view.color_space));
    mapped.premultiplied_alpha(view.premultiplied_alpha);
    mapped
}

fn map_external_texture_copy_view(
    view: &crate::ImageCopyExternalImage,
) -> web_sys::GpuImageCopyExternalImage {
    let mut mapped = web_sys::GpuImageCopyExternalImage::new(&view.source);
    mapped.origin(&map_origin_2d(view.origin));
    mapped.flip_y(view.flip_y);
    mapped
}

fn map_texture_aspect(aspect: wgt::TextureAspect) -> web_sys::GpuTextureAspect {
    match aspect {
        wgt::TextureAspect::All => web_sys::GpuTextureAspect::All,
        wgt::TextureAspect::StencilOnly => web_sys::GpuTextureAspect::StencilOnly,
        wgt::TextureAspect::DepthOnly => web_sys::GpuTextureAspect::DepthOnly,
    }
}

fn map_filter_mode(mode: wgt::FilterMode) -> web_sys::GpuFilterMode {
    match mode {
        wgt::FilterMode::Nearest => web_sys::GpuFilterMode::Nearest,
        wgt::FilterMode::Linear => web_sys::GpuFilterMode::Linear,
    }
}

fn map_mipmap_filter_mode(mode: wgt::FilterMode) -> web_sys::GpuMipmapFilterMode {
    match mode {
        wgt::FilterMode::Nearest => web_sys::GpuMipmapFilterMode::Nearest,
        wgt::FilterMode::Linear => web_sys::GpuMipmapFilterMode::Linear,
    }
}

fn map_address_mode(mode: wgt::AddressMode) -> web_sys::GpuAddressMode {
    match mode {
        wgt::AddressMode::ClampToEdge => web_sys::GpuAddressMode::ClampToEdge,
        wgt::AddressMode::Repeat => web_sys::GpuAddressMode::Repeat,
        wgt::AddressMode::MirrorRepeat => web_sys::GpuAddressMode::MirrorRepeat,
        wgt::AddressMode::ClampToBorder => panic!("Clamp to border is not supported"),
    }
}

fn map_color(color: wgt::Color) -> web_sys::GpuColorDict {
    web_sys::GpuColorDict::new(color.a, color.b, color.g, color.r)
}

fn map_store_op(store: bool) -> web_sys::GpuStoreOp {
    if store {
        web_sys::GpuStoreOp::Store
    } else {
        web_sys::GpuStoreOp::Discard
    }
}

fn map_map_mode(mode: crate::MapMode) -> u32 {
    match mode {
        crate::MapMode::Read => web_sys::gpu_map_mode::READ,
        crate::MapMode::Write => web_sys::gpu_map_mode::WRITE,
    }
}

const FEATURES_MAPPING: [(wgt::Features, web_sys::GpuFeatureName); 9] = [
    //TODO: update the name
    (
        wgt::Features::DEPTH_CLIP_CONTROL,
        web_sys::GpuFeatureName::DepthClipControl,
    ),
    (
        wgt::Features::DEPTH32FLOAT_STENCIL8,
        web_sys::GpuFeatureName::Depth32floatStencil8,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_BC,
        web_sys::GpuFeatureName::TextureCompressionBc,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_ETC2,
        web_sys::GpuFeatureName::TextureCompressionEtc2,
    ),
    (
        wgt::Features::TEXTURE_COMPRESSION_ASTC,
        web_sys::GpuFeatureName::TextureCompressionAstc,
    ),
    (
        wgt::Features::TIMESTAMP_QUERY,
        web_sys::GpuFeatureName::TimestampQuery,
    ),
    (
        wgt::Features::INDIRECT_FIRST_INSTANCE,
        web_sys::GpuFeatureName::IndirectFirstInstance,
    ),
    (
        wgt::Features::SHADER_F16,
        web_sys::GpuFeatureName::ShaderF16,
    ),
    (
        wgt::Features::RG11B10UFLOAT_RENDERABLE,
        web_sys::GpuFeatureName::Rg11b10ufloatRenderable,
    ),
];

fn map_wgt_features(supported_features: web_sys::GpuSupportedFeatures) -> wgt::Features {
    let mut features = wgt::Features::empty();
    for (wgpu_feat, web_feat) in FEATURES_MAPPING {
        match wasm_bindgen::JsValue::from(web_feat).as_string() {
            Some(value) if supported_features.has(&value) => features |= wgpu_feat,
            _ => {}
        }
    }
    features
}

type JsFutureResult = Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;

fn future_request_adapter(
    result: JsFutureResult,
) -> Option<(
    Identified<web_sys::GpuAdapter>,
    Sendable<web_sys::GpuAdapter>,
)> {
    match result.and_then(wasm_bindgen::JsCast::dyn_into) {
        Ok(adapter) => Some(create_identified(adapter)),
        Err(_) => None,
    }
}

fn future_request_device(
    result: JsFutureResult,
) -> Result<
    (
        Identified<web_sys::GpuDevice>,
        Sendable<web_sys::GpuDevice>,
        Identified<web_sys::GpuQueue>,
        Sendable<web_sys::GpuQueue>,
    ),
    crate::RequestDeviceError,
> {
    result
        .map(|js_value| {
            let (device_id, device_data) = create_identified(web_sys::GpuDevice::from(js_value));
            let (queue_id, queue_data) = create_identified(device_data.0.queue());

            (device_id, device_data, queue_id, queue_data)
        })
        .map_err(|_| crate::RequestDeviceError)
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

impl Context {
    pub fn instance_create_surface_from_canvas(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<
        (
            <Self as crate::Context>::SurfaceId,
            <Self as crate::Context>::SurfaceData,
        ),
        crate::CreateSurfaceError,
    > {
        let result = canvas.get_context("webgpu");
        self.create_surface_from_context(Canvas::Canvas(canvas), result)
    }

    pub fn instance_create_surface_from_offscreen_canvas(
        &self,
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<
        (
            <Self as crate::Context>::SurfaceId,
            <Self as crate::Context>::SurfaceData,
        ),
        crate::CreateSurfaceError,
    > {
        let result = canvas.get_context("webgpu");
        self.create_surface_from_context(Canvas::Offscreen(canvas), result)
    }

    /// Common portion of public `instance_create_surface_from_*` functions.
    ///
    /// Note: Analogous code also exists in the WebGL2 backend at
    /// `wgpu_hal::gles::web::Instance`.
    fn create_surface_from_context(
        &self,
        canvas: Canvas,
        context_result: Result<Option<js_sys::Object>, wasm_bindgen::JsValue>,
    ) -> Result<
        (
            <Self as crate::Context>::SurfaceId,
            <Self as crate::Context>::SurfaceData,
        ),
        crate::CreateSurfaceError,
    > {
        let context: js_sys::Object = match context_result {
            Ok(Some(context)) => context,
            Ok(None) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext-dev>
                // A getContext() call “returns null if contextId is not supported, or if the
                // canvas has already been initialized with another context type”. Additionally,
                // “not supported” could include “insufficient GPU resources” or “the GPU process
                // previously crashed”. So, we must return it as an `Err` since it could occur
                // for circumstances outside the application author's control.
                return Err(crate::CreateSurfaceError {});
            }
            Err(js_error) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext>
                // A thrown exception indicates misuse of the canvas state. Ideally we wouldn't
                // panic in this case ... TODO
                panic!("canvas.getContext() threw {js_error:?}")
            }
        };

        // Not returning this error because it is a type error that shouldn't happen unless
        // the browser, JS builtin objects, or wasm bindings are misbehaving somehow.
        let context: web_sys::GpuCanvasContext = context
            .dyn_into()
            .expect("canvas context is not a GPUCanvasContext");

        Ok(create_identified((canvas, context)))
    }
}

// Represents the global object in the JavaScript context.
// It can be cast to from `web_sys::global` and exposes two getters `window` and `worker` of which only one is defined depending on the caller's context.
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

impl crate::context::Context for Context {
    type AdapterId = Identified<web_sys::GpuAdapter>;
    type AdapterData = Sendable<web_sys::GpuAdapter>;
    type DeviceId = Identified<web_sys::GpuDevice>;
    type DeviceData = Sendable<web_sys::GpuDevice>;
    type QueueId = Identified<web_sys::GpuQueue>;
    type QueueData = Sendable<web_sys::GpuQueue>;
    type ShaderModuleId = Identified<web_sys::GpuShaderModule>;
    type ShaderModuleData = Sendable<web_sys::GpuShaderModule>;
    type BindGroupLayoutId = Identified<web_sys::GpuBindGroupLayout>;
    type BindGroupLayoutData = Sendable<web_sys::GpuBindGroupLayout>;
    type BindGroupId = Identified<web_sys::GpuBindGroup>;
    type BindGroupData = Sendable<web_sys::GpuBindGroup>;
    type TextureViewId = Identified<web_sys::GpuTextureView>;
    type TextureViewData = Sendable<web_sys::GpuTextureView>;
    type SamplerId = Identified<web_sys::GpuSampler>;
    type SamplerData = Sendable<web_sys::GpuSampler>;
    type BufferId = Identified<web_sys::GpuBuffer>;
    type BufferData = Sendable<web_sys::GpuBuffer>;
    type TextureId = Identified<web_sys::GpuTexture>;
    type TextureData = Sendable<web_sys::GpuTexture>;
    type QuerySetId = Identified<web_sys::GpuQuerySet>;
    type QuerySetData = Sendable<web_sys::GpuQuerySet>;
    type PipelineLayoutId = Identified<web_sys::GpuPipelineLayout>;
    type PipelineLayoutData = Sendable<web_sys::GpuPipelineLayout>;
    type RenderPipelineId = Identified<web_sys::GpuRenderPipeline>;
    type RenderPipelineData = Sendable<web_sys::GpuRenderPipeline>;
    type ComputePipelineId = Identified<web_sys::GpuComputePipeline>;
    type ComputePipelineData = Sendable<web_sys::GpuComputePipeline>;
    type CommandEncoderId = Identified<web_sys::GpuCommandEncoder>;
    type CommandEncoderData = Sendable<web_sys::GpuCommandEncoder>;
    type ComputePassId = Identified<web_sys::GpuComputePassEncoder>;
    type ComputePassData = Sendable<web_sys::GpuComputePassEncoder>;
    type RenderPassId = Identified<web_sys::GpuRenderPassEncoder>;
    type RenderPassData = Sendable<web_sys::GpuRenderPassEncoder>;
    type CommandBufferId = Identified<web_sys::GpuCommandBuffer>;
    type CommandBufferData = Sendable<web_sys::GpuCommandBuffer>;
    type RenderBundleEncoderId = Identified<web_sys::GpuRenderBundleEncoder>;
    type RenderBundleEncoderData = Sendable<web_sys::GpuRenderBundleEncoder>;
    type RenderBundleId = Identified<web_sys::GpuRenderBundle>;
    type RenderBundleData = Sendable<web_sys::GpuRenderBundle>;
    type SurfaceId = Identified<(Canvas, web_sys::GpuCanvasContext)>;
    type SurfaceData = Sendable<(Canvas, web_sys::GpuCanvasContext)>;

    type SurfaceOutputDetail = SurfaceOutputDetail;
    type SubmissionIndex = Unused;
    type SubmissionIndexData = ();

    type RequestAdapterFuture = MakeSendFuture<
        wasm_bindgen_futures::JsFuture,
        fn(JsFutureResult) -> Option<(Self::AdapterId, Self::AdapterData)>,
    >;
    type RequestDeviceFuture = MakeSendFuture<
        wasm_bindgen_futures::JsFuture,
        fn(
            JsFutureResult,
        ) -> Result<
            (
                Self::DeviceId,
                Self::DeviceData,
                Self::QueueId,
                Self::QueueData,
            ),
            crate::RequestDeviceError,
        >,
    >;
    type PopErrorScopeFuture =
        MakeSendFuture<wasm_bindgen_futures::JsFuture, fn(JsFutureResult) -> Option<crate::Error>>;

    fn init(_instance_desc: wgt::InstanceDescriptor) -> Self {
        let global: Global = js_sys::global().unchecked_into();
        let gpu = if !global.window().is_undefined() {
            global.unchecked_into::<web_sys::Window>().navigator().gpu()
        } else if !global.worker().is_undefined() {
            global
                .unchecked_into::<web_sys::WorkerGlobalScope>()
                .navigator()
                .gpu()
        } else {
            panic!(
                "Accessing the GPU is only supported on the main thread or from a dedicated worker"
            );
        };
        Context(gpu)
    }

    fn instance_create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError> {
        let canvas_attribute = match window_handle {
            raw_window_handle::RawWindowHandle::Web(web_handle) => web_handle.id,
            _ => panic!("expected valid handle for canvas"),
        };
        let canvas_node: wasm_bindgen::JsValue = web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.query_selector_all(&format!("[data-raw-handle=\"{canvas_attribute}\"]"))
                    .ok()
            })
            .and_then(|nodes| nodes.get(0))
            .expect("expected to find single canvas")
            .into();
        let canvas_element: web_sys::HtmlCanvasElement = canvas_node.into();
        self.instance_create_surface_from_canvas(canvas_element)
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_>,
    ) -> Self::RequestAdapterFuture {
        //TODO: support this check, return `None` if the flag is not set.
        // It's not trivial, since we need the Future logic to have this check,
        // and currently the Future here has no room for extra parameter `backends`.
        //assert!(backends.contains(wgt::Backends::BROWSER_WEBGPU));
        let mut mapped_options = web_sys::GpuRequestAdapterOptions::new();
        let mapped_power_preference = match options.power_preference {
            wgt::PowerPreference::LowPower => web_sys::GpuPowerPreference::LowPower,
            wgt::PowerPreference::HighPerformance => web_sys::GpuPowerPreference::HighPerformance,
        };
        mapped_options.power_preference(mapped_power_preference);
        let adapter_promise = self.0.request_adapter_with_options(&mapped_options);

        MakeSendFuture::new(
            wasm_bindgen_futures::JsFuture::from(adapter_promise),
            future_request_adapter,
        )
    }

    fn adapter_request_device(
        &self,
        _adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        if trace_dir.is_some() {
            //Error: Tracing isn't supported on the Web target
        }

        // TODO: non-guaranteed limits
        let mut mapped_desc = web_sys::GpuDeviceDescriptor::new();

        let required_features = FEATURES_MAPPING
            .iter()
            .copied()
            .flat_map(|(flag, value)| {
                if desc.features.contains(flag) {
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
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> bool {
        true
    }

    fn adapter_features(
        &self,
        _adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> wgt::Features {
        map_wgt_features(adapter_data.0.features())
    }

    fn adapter_limits(
        &self,
        _adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> wgt::Limits {
        let limits = adapter_data.0.limits();
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
            max_vertex_attributes: limits.max_vertex_attributes(),
            max_vertex_buffer_array_stride: limits.max_vertex_buffer_array_stride(),
            ..wgt::Limits::default()
        }
    }

    fn adapter_downlevel_capabilities(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::DownlevelCapabilities {
        // WebGPU is assumed to be fully compliant
        wgt::DownlevelCapabilities::default()
    }

    fn adapter_get_info(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::AdapterInfo {
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
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        format.guaranteed_format_features(self.adapter_features(adapter, adapter_data))
    }

    fn adapter_get_presentation_timestamp(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        wgt::PresentationTimestamp::INVALID_TIMESTAMP
    }

    fn surface_get_capabilities(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        wgt::SurfaceCapabilities {
            // https://gpuweb.github.io/gpuweb/#supported-context-formats
            formats: vec![
                wgt::TextureFormat::Bgra8Unorm,
                wgt::TextureFormat::Rgba8Unorm,
                wgt::TextureFormat::Rgba16Float,
            ],
            // Doesn't really have meaning on the web.
            present_modes: vec![wgt::PresentMode::Fifo],
            alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
        }
    }

    fn surface_configure(
        &self,
        _surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
        _device: &Self::DeviceId,
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
            wgt::CompositeAlphaMode::PreMultiplied => web_sys::GpuCanvasAlphaMode::Premultiplied,
            _ => web_sys::GpuCanvasAlphaMode::Opaque,
        };
        let mut mapped =
            web_sys::GpuCanvasConfiguration::new(&device_data.0, map_texture_format(config.format));
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
        _surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureId>,
        Option<Self::TextureData>,
        wgt::SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        let (surface_id, surface_data) = create_identified(surface_data.0 .1.get_current_texture());
        (
            Some(surface_id),
            Some(surface_data),
            wgt::SurfaceStatus::Good,
            (),
        )
    }

    fn surface_present(&self, _texture: &Self::TextureId, _detail: &Self::SurfaceOutputDetail) {
        // Swapchain is presented automatically
    }

    fn surface_texture_discard(
        &self,
        _texture: &Self::TextureId,
        _detail: &Self::SurfaceOutputDetail,
    ) {
        // Can't really discard this on the Web
    }

    fn device_features(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
    ) -> wgt::Features {
        map_wgt_features(device_data.0.features())
    }

    fn device_limits(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::Limits {
        // TODO
        wgt::Limits::default()
    }

    fn device_downlevel_properties(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::DownlevelCapabilities {
        // WebGPU is assumed to be fully compliant
        wgt::DownlevelCapabilities::default()
    }

    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga"
        )),
        allow(unreachable_code, unused_variables)
    )]
    fn device_create_shader_module(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: crate::ShaderModuleDescriptor,
        _shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        let mut descriptor = match desc.source {
            #[cfg(feature = "spirv")]
            crate::ShaderSource::SpirV(ref spv) => {
                use naga::{back, front, valid};

                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false,
                    strict_capabilities: true,
                    block_ctx_dump_prefix: None,
                };
                let spv_parser = front::spv::Frontend::new(spv.iter().cloned(), &options);
                let spv_module = spv_parser.parse().unwrap();

                let mut validator = valid::Validator::new(
                    valid::ValidationFlags::all(),
                    valid::Capabilities::all(),
                );
                let spv_module_info = validator.validate(&spv_module).unwrap();

                let writer_flags = naga::back::wgsl::WriterFlags::empty();
                let wgsl_text =
                    back::wgsl::write_string(&spv_module, &spv_module_info, writer_flags).unwrap();
                web_sys::GpuShaderModuleDescriptor::new(wgsl_text.as_str())
            }
            #[cfg(feature = "glsl")]
            crate::ShaderSource::Glsl {
                ref shader,
                stage,
                ref defines,
            } => {
                use naga::{back, front, valid};

                // Parse the given shader code and store its representation.
                let options = front::glsl::Options {
                    stage,
                    defines: defines.clone(),
                };
                let mut parser = front::glsl::Frontend::default();
                let glsl_module = parser.parse(&options, shader).unwrap();

                let mut validator = valid::Validator::new(
                    valid::ValidationFlags::all(),
                    valid::Capabilities::all(),
                );
                let glsl_module_info = validator.validate(&glsl_module).unwrap();

                let writer_flags = naga::back::wgsl::WriterFlags::empty();
                let wgsl_text =
                    back::wgsl::write_string(&glsl_module, &glsl_module_info, writer_flags)
                        .unwrap();
                web_sys::GpuShaderModuleDescriptor::new(wgsl_text.as_str())
            }
            #[cfg(feature = "wgsl")]
            crate::ShaderSource::Wgsl(ref code) => web_sys::GpuShaderModuleDescriptor::new(code),
            #[cfg(feature = "naga")]
            crate::ShaderSource::Naga(module) => {
                use naga::{back, valid};

                let mut validator = valid::Validator::new(
                    valid::ValidationFlags::all(),
                    valid::Capabilities::all(),
                );
                let module_info = validator.validate(&module).unwrap();

                let writer_flags = naga::back::wgsl::WriterFlags::empty();
                let wgsl_text =
                    back::wgsl::write_string(&module, &module_info, writer_flags).unwrap();
                web_sys::GpuShaderModuleDescriptor::new(wgsl_text.as_str())
            }
            crate::ShaderSource::Dummy(_) => {
                panic!("found `ShaderSource::Dummy`")
            }
        };
        if let Some(label) = desc.label {
            descriptor.label(label);
        }
        create_identified(device_data.0.create_shader_module(&descriptor))
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::ShaderModuleDescriptorSpirV,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        unreachable!("SPIRV_SHADER_PASSTHROUGH is not enabled for this backend")
    }

    fn device_create_bind_group_layout(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let mapped_bindings = desc
            .entries
            .iter()
            .map(|bind| {
                let mut mapped_entry =
                    web_sys::GpuBindGroupLayoutEntry::new(bind.binding, bind.visibility.bits());

                match bind.ty {
                    wgt::BindingType::Buffer {
                        ty,
                        has_dynamic_offset,
                        min_binding_size,
                    } => {
                        let mut buffer = web_sys::GpuBufferBindingLayout::new();
                        buffer.has_dynamic_offset(has_dynamic_offset);
                        if let Some(size) = min_binding_size {
                            buffer.min_binding_size(size.get() as f64);
                        }
                        buffer.type_(match ty {
                            wgt::BufferBindingType::Uniform => {
                                web_sys::GpuBufferBindingType::Uniform
                            }
                            wgt::BufferBindingType::Storage { read_only: false } => {
                                web_sys::GpuBufferBindingType::Storage
                            }
                            wgt::BufferBindingType::Storage { read_only: true } => {
                                web_sys::GpuBufferBindingType::ReadOnlyStorage
                            }
                        });
                        mapped_entry.buffer(&buffer);
                    }
                    wgt::BindingType::Sampler(ty) => {
                        let mut sampler = web_sys::GpuSamplerBindingLayout::new();
                        sampler.type_(match ty {
                            wgt::SamplerBindingType::NonFiltering => {
                                web_sys::GpuSamplerBindingType::NonFiltering
                            }
                            wgt::SamplerBindingType::Filtering => {
                                web_sys::GpuSamplerBindingType::Filtering
                            }
                            wgt::SamplerBindingType::Comparison => {
                                web_sys::GpuSamplerBindingType::Comparison
                            }
                        });
                        mapped_entry.sampler(&sampler);
                    }
                    wgt::BindingType::Texture {
                        multisampled,
                        sample_type,
                        view_dimension,
                    } => {
                        let mut texture = web_sys::GpuTextureBindingLayout::new();
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
                                web_sys::GpuStorageTextureAccess::WriteOnly
                            }
                            wgt::StorageTextureAccess::ReadOnly => {
                                panic!("ReadOnly is not available")
                            }
                            wgt::StorageTextureAccess::ReadWrite => {
                                panic!("ReadWrite is not available")
                            }
                        };
                        let mut storage_texture = web_sys::GpuStorageTextureBindingLayout::new(
                            map_texture_format(format),
                        );
                        storage_texture.access(mapped_access);
                        storage_texture.view_dimension(map_texture_view_dimension(view_dimension));
                        mapped_entry.storage_texture(&storage_texture);
                    }
                }

                mapped_entry
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = web_sys::GpuBindGroupLayoutDescriptor::new(&mapped_bindings);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(device_data.0.create_bind_group_layout(&mapped_desc))
    }

    fn device_create_bind_group(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::BindGroupDescriptor,
    ) -> (Self::BindGroupId, Self::BindGroupData) {
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
                        let buffer: &<Context as crate::Context>::BufferData =
                            downcast_ref(buffer.data.as_ref());
                        let mut mapped_buffer_binding = web_sys::GpuBufferBinding::new(&buffer.0);
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
                        let sampler: &<Context as crate::Context>::SamplerData =
                            downcast_ref(sampler.data.as_ref());
                        JsValue::from(&sampler.0)
                    }
                    crate::BindingResource::SamplerArray(..) => {
                        panic!("Web backend does not support arrays of samplers")
                    }
                    crate::BindingResource::TextureView(texture_view) => {
                        let texture_view: &<Context as crate::Context>::TextureViewData =
                            downcast_ref(texture_view.data.as_ref());
                        JsValue::from(&texture_view.0)
                    }
                    crate::BindingResource::TextureViewArray(..) => {
                        panic!("Web backend does not support BINDING_INDEXING extension")
                    }
                };

                web_sys::GpuBindGroupEntry::new(binding.binding, &mapped_resource)
            })
            .collect::<js_sys::Array>();

        let bgl: &<Context as crate::Context>::BindGroupLayoutData =
            downcast_ref(desc.layout.data.as_ref());
        let mut mapped_desc = web_sys::GpuBindGroupDescriptor::new(&mapped_entries, &bgl.0);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(device_data.0.create_bind_group(&mapped_desc))
    }

    fn device_create_pipeline_layout(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::PipelineLayoutDescriptor,
    ) -> (Self::PipelineLayoutId, Self::PipelineLayoutData) {
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| {
                let bgl: &<Context as crate::Context>::BindGroupLayoutData =
                    downcast_ref(bgl.data.as_ref());
                &bgl.0
            })
            .collect::<js_sys::Array>();
        let mut mapped_desc = web_sys::GpuPipelineLayoutDescriptor::new(&temp_layouts);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(device_data.0.create_pipeline_layout(&mapped_desc))
    }

    fn device_create_render_pipeline(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::RenderPipelineDescriptor,
    ) -> (Self::RenderPipelineId, Self::RenderPipelineData) {
        let module: &<Context as crate::Context>::ShaderModuleData =
            downcast_ref(desc.vertex.module.data.as_ref());
        let mut mapped_vertex_state =
            web_sys::GpuVertexState::new(desc.vertex.entry_point, &module.0);

        let buffers = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| {
                let mapped_attributes = vbuf
                    .attributes
                    .iter()
                    .map(|attr| {
                        web_sys::GpuVertexAttribute::new(
                            map_vertex_format(attr.format),
                            attr.offset as f64,
                            attr.shader_location,
                        )
                    })
                    .collect::<js_sys::Array>();

                let mut mapped_vbuf = web_sys::GpuVertexBufferLayout::new(
                    vbuf.array_stride as f64,
                    &mapped_attributes,
                );
                mapped_vbuf.step_mode(map_vertex_step_mode(vbuf.step_mode));
                mapped_vbuf
            })
            .collect::<js_sys::Array>();

        mapped_vertex_state.buffers(&buffers);

        let auto_layout = wasm_bindgen::JsValue::from(web_sys::GpuAutoLayoutMode::Auto);
        let mut mapped_desc = web_sys::GpuRenderPipelineDescriptor::new(
            &match desc.layout {
                Some(layout) => {
                    let layout: &<Context as crate::Context>::PipelineLayoutData =
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
                            web_sys::GpuColorTargetState::new(mapped_format);
                        if let Some(ref bs) = target.blend {
                            let alpha = map_blend_component(&bs.alpha);
                            let color = map_blend_component(&bs.color);
                            let mapped_blend_state = web_sys::GpuBlendState::new(&alpha, &color);
                            mapped_color_state.blend(&mapped_blend_state);
                        }
                        mapped_color_state.write_mask(target.write_mask.bits());
                        wasm_bindgen::JsValue::from(mapped_color_state)
                    }
                    None => wasm_bindgen::JsValue::null(),
                })
                .collect::<js_sys::Array>();
            let module: &<Context as crate::Context>::ShaderModuleData =
                downcast_ref(frag.module.data.as_ref());
            let mapped_fragment_desc =
                web_sys::GpuFragmentState::new(frag.entry_point, &module.0, &targets);
            mapped_desc.fragment(&mapped_fragment_desc);
        }

        let mut mapped_multisample = web_sys::GpuMultisampleState::new();
        mapped_multisample.count(desc.multisample.count);
        mapped_multisample.mask(desc.multisample.mask as u32);
        mapped_multisample.alpha_to_coverage_enabled(desc.multisample.alpha_to_coverage_enabled);
        mapped_desc.multisample(&mapped_multisample);

        let mapped_primitive = map_primitive_state(&desc.primitive);
        mapped_desc.primitive(&mapped_primitive);

        create_identified(device_data.0.create_render_pipeline(&mapped_desc))
    }

    fn device_create_compute_pipeline(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::ComputePipelineDescriptor,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData) {
        let shader_module: &<Context as crate::Context>::ShaderModuleData =
            downcast_ref(desc.module.data.as_ref());
        let mapped_compute_stage =
            web_sys::GpuProgrammableStage::new(desc.entry_point, &shader_module.0);
        let auto_layout = wasm_bindgen::JsValue::from(web_sys::GpuAutoLayoutMode::Auto);
        let mut mapped_desc = web_sys::GpuComputePipelineDescriptor::new(
            &match desc.layout {
                Some(layout) => {
                    let layout: &<Context as crate::Context>::PipelineLayoutData =
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
        create_identified(device_data.0.create_compute_pipeline(&mapped_desc))
    }

    fn device_create_buffer(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::BufferDescriptor,
    ) -> (Self::BufferId, Self::BufferData) {
        let mut mapped_desc =
            web_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
        mapped_desc.mapped_at_creation(desc.mapped_at_creation);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(device_data.0.create_buffer(&mapped_desc))
    }

    fn device_create_texture(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::TextureDescriptor,
    ) -> (Self::TextureId, Self::TextureData) {
        let mut mapped_desc = web_sys::GpuTextureDescriptor::new(
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
        create_identified(device_data.0.create_texture(&mapped_desc))
    }

    fn device_create_sampler(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::SamplerDescriptor,
    ) -> (Self::SamplerId, Self::SamplerData) {
        let mut mapped_desc = web_sys::GpuSamplerDescriptor::new();
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
        create_identified(device_data.0.create_sampler_with_descriptor(&mapped_desc))
    }

    fn device_create_query_set(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> (Self::QuerySetId, Self::QuerySetData) {
        let ty = match desc.ty {
            wgt::QueryType::Occlusion => web_sys::GpuQueryType::Occlusion,
            wgt::QueryType::Timestamp => web_sys::GpuQueryType::Timestamp,
            wgt::QueryType::PipelineStatistics(_) => unreachable!(),
        };
        let mut mapped_desc = web_sys::GpuQuerySetDescriptor::new(desc.count, ty);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(device_data.0.create_query_set(&mapped_desc))
    }

    fn device_create_command_encoder(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::CommandEncoderDescriptor,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData) {
        let mut mapped_desc = web_sys::GpuCommandEncoderDescriptor::new();
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(
            device_data
                .0
                .create_command_encoder_with_descriptor(&mapped_desc),
        )
    }

    fn device_create_render_bundle_encoder(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::RenderBundleEncoderDescriptor,
    ) -> (Self::RenderBundleEncoderId, Self::RenderBundleEncoderData) {
        let mapped_color_formats = desc
            .color_formats
            .iter()
            .map(|cf| match cf {
                Some(cf) => wasm_bindgen::JsValue::from(map_texture_format(*cf)),
                None => wasm_bindgen::JsValue::null(),
            })
            .collect::<js_sys::Array>();
        let mut mapped_desc = web_sys::GpuRenderBundleEncoderDescriptor::new(&mapped_color_formats);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        if let Some(ds) = desc.depth_stencil {
            mapped_desc.depth_stencil_format(map_texture_format(ds.format));
            mapped_desc.depth_read_only(ds.depth_read_only);
            mapped_desc.stencil_read_only(ds.stencil_read_only);
        }
        mapped_desc.sample_count(desc.sample_count);
        create_identified(device_data.0.create_render_bundle_encoder(&mapped_desc))
    }

    fn device_drop(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        // Device is dropped automatically
    }

    fn device_poll(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _maintain: crate::Maintain,
    ) -> bool {
        // Device is polled automatically
        true
    }

    fn device_on_uncaptured_error(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let f = Closure::wrap(Box::new(move |event: web_sys::GpuUncapturedErrorEvent| {
            let error = crate::Error::from_js(event.error().value_of());
            handler(error);
        }) as Box<dyn FnMut(_)>);
        device_data
            .0
            .set_onuncapturederror(Some(f.as_ref().unchecked_ref()));
        // TODO: This will leak the memory associated with the error handler by default.
        f.forget();
    }

    fn device_push_error_scope(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        filter: crate::ErrorFilter,
    ) {
        device_data.0.push_error_scope(match filter {
            crate::ErrorFilter::OutOfMemory => web_sys::GpuErrorFilter::OutOfMemory,
            crate::ErrorFilter::Validation => web_sys::GpuErrorFilter::Validation,
        });
    }

    fn device_pop_error_scope(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
    ) -> Self::PopErrorScopeFuture {
        let error_promise = device_data.0.pop_error_scope();
        MakeSendFuture::new(
            wasm_bindgen_futures::JsFuture::from(error_promise),
            future_pop_error_scope,
        )
    }

    fn buffer_map_async(
        &self,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        mode: crate::MapMode,
        range: Range<wgt::BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), crate::BufferAsyncError>) + Send + 'static>,
    ) {
        let map_promise = buffer_data.0.map_async_with_f64_and_f64(
            map_map_mode(mode),
            range.start as f64,
            (range.end - range.start) as f64,
        );

        register_then_closures(&map_promise, callback, Ok(()), Err(crate::BufferAsyncError));
    }

    fn buffer_get_mapped_range(
        &self,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        sub_range: Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        let array_buffer = buffer_data.0.get_mapped_range_with_f64_and_f64(
            sub_range.start as f64,
            (sub_range.end - sub_range.start) as f64,
        );
        let actual_mapping = js_sys::Uint8Array::new(&array_buffer);
        let temporary_mapping = actual_mapping.to_vec();
        Box::new(BufferMappedRange {
            actual_mapping,
            temporary_mapping,
        })
    }

    fn buffer_unmap(&self, _buffer: &Self::BufferId, buffer_data: &Self::BufferData) {
        buffer_data.0.unmap();
    }

    fn texture_create_view(
        &self,
        _texture: &Self::TextureId,
        texture_data: &Self::TextureData,
        desc: &crate::TextureViewDescriptor,
    ) -> (Self::TextureViewId, Self::TextureViewData) {
        let mut mapped = web_sys::GpuTextureViewDescriptor::new();
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
        create_identified(texture_data.0.create_view_with_descriptor(&mapped))
    }

    fn surface_drop(&self, _surface: &Self::SurfaceId, _surface_data: &Self::SurfaceData) {
        // Dropped automatically
    }

    fn adapter_drop(&self, _adapter: &Self::AdapterId, _adapter_data: &Self::AdapterData) {
        // Dropped automatically
    }

    fn buffer_destroy(&self, _buffer: &Self::BufferId, buffer_data: &Self::BufferData) {
        buffer_data.0.destroy();
    }

    fn buffer_drop(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        // Dropped automatically
    }

    fn texture_destroy(&self, _texture: &Self::TextureId, texture_data: &Self::TextureData) {
        texture_data.0.destroy();
    }

    fn texture_drop(&self, _texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        // Dropped automatically
    }

    fn texture_view_drop(
        &self,
        _texture_view: &Self::TextureViewId,
        _texture_view_data: &Self::TextureViewData,
    ) {
        // Dropped automatically
    }

    fn sampler_drop(&self, _sampler: &Self::SamplerId, _sampler_data: &Self::SamplerData) {
        // Dropped automatically
    }

    fn query_set_drop(&self, _query_set: &Self::QuerySetId, _query_set_data: &Self::QuerySetData) {
        // Dropped automatically
    }

    fn bind_group_drop(
        &self,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
    ) {
        // Dropped automatically
    }

    fn bind_group_layout_drop(
        &self,
        _bind_group_layout: &Self::BindGroupLayoutId,
        _bind_group_layout_data: &Self::BindGroupLayoutData,
    ) {
        // Dropped automatically
    }

    fn pipeline_layout_drop(
        &self,
        _pipeline_layout: &Self::PipelineLayoutId,
        _pipeline_layout_data: &Self::PipelineLayoutData,
    ) {
        // Dropped automatically
    }

    fn shader_module_drop(
        &self,
        _shader_module: &Self::ShaderModuleId,
        _shader_module_data: &Self::ShaderModuleData,
    ) {
        // Dropped automatically
    }

    fn command_encoder_drop(
        &self,
        _command_encoder: &Self::CommandEncoderId,
        _command_encoder_data: &Self::CommandEncoderData,
    ) {
        // Dropped automatically
    }

    fn command_buffer_drop(
        &self,
        _command_buffer: &Self::CommandBufferId,
        _command_buffer_data: &Self::CommandBufferData,
    ) {
        // Dropped automatically
    }

    fn render_bundle_drop(
        &self,
        _render_bundle: &Self::RenderBundleId,
        _render_bundle_data: &Self::RenderBundleData,
    ) {
        // Dropped automatically
    }

    fn compute_pipeline_drop(
        &self,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        // Dropped automatically
    }

    fn render_pipeline_drop(
        &self,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        // Dropped automatically
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::ComputePipelineId,
        pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        create_identified(pipeline_data.0.get_bind_group_layout(index))
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        create_identified(pipeline_data.0.get_bind_group_layout(index))
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        _source: &Self::BufferId,
        source_data: &Self::BufferData,
        source_offset: wgt::BufferAddress,
        _destination: &Self::BufferId,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .copy_buffer_to_buffer_with_f64_and_f64_and_f64(
                &source_data.0,
                source_offset as f64,
                &destination_data.0,
                destination_offset as f64,
                copy_size as f64,
            )
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyBuffer,
        destination: crate::ImageCopyTexture,
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
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyBuffer,
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
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyTexture,
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
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::ComputePassDescriptor,
    ) -> (Self::ComputePassId, Self::ComputePassData) {
        let mut mapped_desc = web_sys::GpuComputePassDescriptor::new();
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        create_identified(
            encoder_data
                .0
                .begin_compute_pass_with_descriptor(&mapped_desc),
        )
    }

    fn command_encoder_end_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        pass_data.0.end();
    }

    fn command_encoder_begin_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::RenderPassDescriptor<'_, '_>,
    ) -> (Self::RenderPassId, Self::RenderPassData) {
        let mapped_color_attachments = desc
            .color_attachments
            .iter()
            .map(|attachment| match attachment {
                Some(ca) => {
                    let mut clear_value: Option<wasm_bindgen::JsValue> = None;
                    let load_value = match ca.ops.load {
                        crate::LoadOp::Clear(color) => {
                            clear_value = Some(wasm_bindgen::JsValue::from(map_color(color)));
                            web_sys::GpuLoadOp::Clear
                        }
                        crate::LoadOp::Load => web_sys::GpuLoadOp::Load,
                    };

                    let view: &<Context as crate::Context>::TextureViewData =
                        downcast_ref(ca.view.data.as_ref());

                    let mut mapped_color_attachment = web_sys::GpuRenderPassColorAttachment::new(
                        load_value,
                        map_store_op(ca.ops.store),
                        &view.0,
                    );
                    if let Some(cv) = clear_value {
                        mapped_color_attachment.clear_value(&cv);
                    }
                    if let Some(rt) = ca.resolve_target {
                        let resolve_target_view: &<Context as crate::Context>::TextureViewData =
                            downcast_ref(rt.data.as_ref());
                        mapped_color_attachment.resolve_target(&resolve_target_view.0);
                    }
                    mapped_color_attachment.store_op(map_store_op(ca.ops.store));

                    wasm_bindgen::JsValue::from(mapped_color_attachment)
                }
                None => wasm_bindgen::JsValue::null(),
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = web_sys::GpuRenderPassDescriptor::new(&mapped_color_attachments);

        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }

        if let Some(dsa) = &desc.depth_stencil_attachment {
            let depth_stencil_attachment: &<Context as crate::Context>::TextureViewData =
                downcast_ref(dsa.view.data.as_ref());
            let mut mapped_depth_stencil_attachment =
                web_sys::GpuRenderPassDepthStencilAttachment::new(&depth_stencil_attachment.0);
            if let Some(ref ops) = dsa.depth_ops {
                let load_op = match ops.load {
                    crate::LoadOp::Clear(v) => {
                        mapped_depth_stencil_attachment.depth_clear_value(v);
                        web_sys::GpuLoadOp::Clear
                    }
                    crate::LoadOp::Load => web_sys::GpuLoadOp::Load,
                };
                mapped_depth_stencil_attachment.depth_load_op(load_op);
                mapped_depth_stencil_attachment.depth_store_op(map_store_op(ops.store));
            }
            mapped_depth_stencil_attachment.depth_read_only(dsa.depth_ops.is_none());
            if let Some(ref ops) = dsa.stencil_ops {
                let load_op = match ops.load {
                    crate::LoadOp::Clear(v) => {
                        mapped_depth_stencil_attachment.stencil_clear_value(v);
                        web_sys::GpuLoadOp::Clear
                    }
                    crate::LoadOp::Load => web_sys::GpuLoadOp::Load,
                };
                mapped_depth_stencil_attachment.stencil_load_op(load_op);
                mapped_depth_stencil_attachment.stencil_store_op(map_store_op(ops.store));
            }
            mapped_depth_stencil_attachment.stencil_read_only(dsa.stencil_ops.is_none());
            mapped_desc.depth_stencil_attachment(&mapped_depth_stencil_attachment);
        }

        create_identified(encoder_data.0.begin_render_pass(&mapped_desc))
    }

    fn command_encoder_end_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        pass_data.0.end();
    }

    fn command_encoder_finish(
        &self,
        _encoder: Self::CommandEncoderId,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData) {
        let label = encoder_data.0.label();
        create_identified(if label.is_empty() {
            encoder_data.0.finish()
        } else {
            let mut mapped_desc = web_sys::GpuCommandBufferDescriptor::new();
            mapped_desc.label(&label);
            encoder_data.0.finish_with_descriptor(&mapped_desc)
        })
    }

    fn command_encoder_clear_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _texture: &crate::Texture,
        _subresource_range: &wgt::ImageSubresourceRange,
    ) {
        //TODO
    }

    fn command_encoder_clear_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        buffer: &crate::Buffer,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let buffer: &<Context as crate::Context>::BufferData = downcast_ref(buffer.data.as_ref());
        match size {
            Some(size) => encoder_data.0.clear_buffer_with_f64_and_f64(
                &buffer.0,
                offset as f64,
                size.get() as f64,
            ),
            None => encoder_data
                .0
                .clear_buffer_with_f64(&buffer.0, offset as f64),
        }
    }

    fn command_encoder_insert_debug_marker(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // encoder.insert_debug_marker(label);
    }

    fn command_encoder_push_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // encoder.push_debug_group(label);
    }

    fn command_encoder_pop_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
    ) {
        // Not available in gecko yet
        // encoder.pop_debug_group();
    }

    fn command_encoder_write_timestamp(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        _query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        encoder_data
            .0
            .write_timestamp(&query_set_data.0, query_index);
    }

    fn command_encoder_resolve_query_set(
        &self,
        _encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        _query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        _destination: &Self::BufferId,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
    ) {
        encoder_data.0.resolve_query_set_with_u32(
            &query_set_data.0,
            first_query,
            query_count,
            &destination_data.0,
            destination_offset as u32,
        );
    }

    fn render_bundle_encoder_finish(
        &self,
        _encoder: Self::RenderBundleEncoderId,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &crate::RenderBundleDescriptor,
    ) -> (Self::RenderBundleId, Self::RenderBundleData) {
        create_identified(match desc.label {
            Some(label) => {
                let mut mapped_desc = web_sys::GpuRenderBundleDescriptor::new();
                mapped_desc.label(label);
                encoder_data.0.finish_with_descriptor(&mapped_desc)
            }
            None => encoder_data.0.finish(),
        })
    }

    fn queue_write_buffer(
        &self,
        _queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        _buffer: &Self::BufferId,
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
                &buffer_data.0,
                offset as f64,
                &js_sys::Uint8Array::from(data).buffer(),
                0f64,
                data.len() as f64,
            );
    }

    fn queue_validate_write_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let usage = wgt::BufferUsages::from_bits_truncate(buffer_data.0.usage());
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
        if write_size + offset > buffer_data.0.size() as u64 {
            log::error!("copy of {}..{} would end up overrunning the bounds of the destination buffer of size {}", offset, offset + write_size, buffer_data.0.size());
            return None;
        }
        Some(())
    }

    fn queue_create_staging_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        size: wgt::BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>> {
        Some(Box::new(WebQueueWriteBuffer(
            vec![0; size.get() as usize].into_boxed_slice(),
        )))
    }

    fn queue_write_staging_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    ) {
        let staging_buffer = staging_buffer
            .as_any()
            .downcast_ref::<WebQueueWriteBuffer>()
            .unwrap()
            .slice();
        self.queue_write_buffer(
            queue,
            queue_data,
            buffer,
            buffer_data,
            offset,
            staging_buffer,
        )
    }

    fn queue_write_texture(
        &self,
        _queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        texture: crate::ImageCopyTexture,
        data: &[u8],
        data_layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    ) {
        let mut mapped_data_layout = web_sys::GpuImageDataLayout::new();
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
        _queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged,
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

    fn queue_submit<I: Iterator<Item = (Self::CommandBufferId, Self::CommandBufferData)>>(
        &self,
        _queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> (Self::SubmissionIndex, Self::SubmissionIndexData) {
        let temp_command_buffers = command_buffers
            .map(|(_, data)| data.0)
            .collect::<js_sys::Array>();

        queue_data.0.submit(&temp_command_buffers);

        (Unused, ())
    }

    fn queue_get_timestamp_period(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
    ) -> f32 {
        1.0 //TODO
    }

    fn queue_on_submitted_work_done(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _callback: Box<dyn FnOnce() + Send + 'static>,
    ) {
        unimplemented!()
    }

    fn device_start_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {}
    fn device_stop_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {}

    fn compute_pass_set_pipeline(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        _pipeline: &Self::ComputePipelineId,
        pipeline_data: &Self::ComputePipelineData,
    ) {
        pass_data.0.set_pipeline(&pipeline_data.0)
    }

    fn compute_pass_set_bind_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        index: u32,
        _bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        pass_data
            .0
            .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                index,
                &bind_group_data.0,
                offsets,
                0f64,
                offsets.len() as u32,
            );
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // self.0.insert_debug_marker(label);
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _group_label: &str,
    ) {
        // Not available in gecko yet
        // self.0.push_debug_group(group_label);
    }

    fn compute_pass_pop_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        // Not available in gecko yet
        // self.0.pop_debug_group();
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        panic!("TIMESTAMP_QUERY_INSIDE_PASSES feature must be enabled to call write_timestamp in a compute pass")
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        // Not available in gecko yet
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        // Not available in gecko yet
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        _pass: &mut Self::ComputePassId,
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
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        _indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data
            .0
            .dispatch_workgroups_indirect_with_f64(&indirect_buffer_data.0, indirect_offset as f64);
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        _pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        encoder_data.0.set_pipeline(&pipeline_data.0);
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        _bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        encoder_data
            .0
            .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                index,
                &bind_group_data.0,
                offsets,
                0f64,
                offsets.len() as u32,
            );
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                encoder_data.0.set_index_buffer_with_f64_and_f64(
                    &buffer_data.0,
                    map_index_format(index_format),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                encoder_data.0.set_index_buffer_with_f64(
                    &buffer_data.0,
                    map_index_format(index_format),
                    offset as f64,
                );
            }
        };
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                encoder_data.0.set_vertex_buffer_with_f64_and_f64(
                    slot,
                    &buffer_data.0,
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                encoder_data
                    .0
                    .set_vertex_buffer_with_f64(slot, &buffer_data.0, offset as f64);
            }
        };
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_bundle_encoder_draw(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
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
        _encoder: &mut Self::RenderBundleEncoderId,
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
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .draw_indirect_with_f64(&indirect_buffer_data.0, indirect_offset as f64);
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        encoder_data
            .0
            .draw_indexed_indirect_with_f64(&indirect_buffer_data.0, indirect_offset as f64);
    }

    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indirect")
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        panic!(
            "MULTI_DRAW_INDIRECT_COUNT feature must be enabled to call multi_draw_indirect_count"
        )
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT_COUNT feature must be enabled to call multi_draw_indexed_indirect_count")
    }

    fn render_pass_set_pipeline(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        _pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        pass_data.0.set_pipeline(&pipeline_data.0);
    }

    fn render_pass_set_bind_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        index: u32,
        _bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        pass_data
            .0
            .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                index,
                &bind_group_data.0,
                offsets,
                0f64,
                offsets.len() as u32,
            );
    }

    fn render_pass_set_index_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                pass_data.0.set_index_buffer_with_f64_and_f64(
                    &buffer_data.0,
                    map_index_format(index_format),
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                pass_data.0.set_index_buffer_with_f64(
                    &buffer_data.0,
                    map_index_format(index_format),
                    offset as f64,
                );
            }
        };
    }

    fn render_pass_set_vertex_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        _buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        match size {
            Some(s) => {
                pass_data.0.set_vertex_buffer_with_f64_and_f64(
                    slot,
                    &buffer_data.0,
                    offset as f64,
                    s.get() as f64,
                );
            }
            None => {
                pass_data
                    .0
                    .set_vertex_buffer_with_f64(slot, &buffer_data.0, offset as f64);
            }
        };
    }

    fn render_pass_set_push_constants(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        panic!("PUSH_CONSTANTS feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_pass_draw(
        &self,
        _pass: &mut Self::RenderPassId,
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
        _pass: &mut Self::RenderPassId,
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
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data
            .0
            .draw_indirect_with_f64(&indirect_buffer_data.0, indirect_offset as f64);
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        pass_data
            .0
            .draw_indexed_indirect_with_f64(&indirect_buffer_data.0, indirect_offset as f64);
    }

    fn render_pass_multi_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indirect")
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT feature must be enabled to call multi_draw_indexed_indirect")
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
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
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        panic!("MULTI_DRAW_INDIRECT_COUNT feature must be enabled to call multi_draw_indexed_indirect_count")
    }

    fn render_pass_set_blend_constant(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        color: wgt::Color,
    ) {
        pass_data
            .0
            .set_blend_constant_with_gpu_color_dict(&map_color(color));
    }

    fn render_pass_set_scissor_rect(
        &self,
        _pass: &mut Self::RenderPassId,
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
        _pass: &mut Self::RenderPassId,
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
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    ) {
        pass_data.0.set_stencil_reference(reference);
    }

    fn render_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _label: &str,
    ) {
        // Not available in gecko yet
        // self.0.insert_debug_marker(label);
    }

    fn render_pass_push_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _group_label: &str,
    ) {
        // Not available in gecko yet
        // self.0.push_debug_group(group_label);
    }

    fn render_pass_pop_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        // Not available in gecko yet
        // self.0.pop_debug_group();
    }

    fn render_pass_write_timestamp(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        panic!("TIMESTAMP_QUERY_INSIDE_PASSES feature must be enabled to call write_timestamp in a compute pass")
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        // Not available in gecko yet
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        // Not available in gecko yet
    }

    fn render_pass_execute_bundles<'a>(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        render_bundles: Box<
            dyn Iterator<Item = (Self::RenderBundleId, &'a Self::RenderBundleData)> + 'a,
        >,
    ) {
        let mapped = render_bundles
            .map(|(_, bundle_data)| &bundle_data.0)
            .collect::<js_sys::Array>();
        pass_data.0.execute_bundles(&mapped);
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
