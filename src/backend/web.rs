use crate::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, BindingType, BufferDescriptor,
    CommandEncoderDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ProgrammableStageDescriptor, RenderPipelineDescriptor, SamplerDescriptor, TextureDescriptor,
    TextureViewDescriptor, TextureViewDimension,
};

use futures::FutureExt;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    pin::Pin,
    task::{self, Poll},
};
use wasm_bindgen::prelude::*;

// We need to make a wrapper for some of the handle types returned by the web backend to make them
// implement `Send` and `Sync` to match native.
//
// SAFETY: All webgpu handle types in wasm32 are internally a `JsValue`, and `JsValue` is neither
// Send nor Sync.  Currently, wasm32 has no threading support so implementing `Send` or `Sync` for a
// type is (for now) harmless.  Eventually wasm32 will support threading, and depending on how this
// is integrated (or not integrated) with values like those in webgpu, this may become unsound.

#[derive(Debug, Clone)]
pub(crate) struct Sendable<T>(T);
unsafe impl<T> Send for Sendable<T> {}
unsafe impl<T> Sync for Sendable<T> {}

pub(crate) type Context = Sendable<web_sys::Gpu>;

pub(crate) struct ComputePass(web_sys::GpuComputePassEncoder);
pub(crate) struct RenderPass(web_sys::GpuRenderPassEncoder);

// We need to assert that any future we return is Send to match the native API.
//
// This is safe on wasm32 *for now*, but similarly to the unsafe Send impls for the handle type
// wrappers, the full story for threading on wasm32 is still unfolding.

pub(crate) struct MakeSendFuture<F>(F);

impl<F: Future> Future for MakeSendFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context) -> Poll<Self::Output> {
        // This is safe because we have no Drop implementation to violate the Pin requirements and
        // do not provide any means of moving the inner future.
        unsafe { self.map_unchecked_mut(|s| &mut s.0) }.poll(cx)
    }
}

unsafe impl<F> Send for MakeSendFuture<F> {}

impl crate::ComputePassInner<Context> for ComputePass {
    fn set_pipeline(&mut self, pipeline: &Sendable<web_sys::GpuComputePipeline>) {
        self.0.set_pipeline(&pipeline.0);
    }
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &Sendable<web_sys::GpuBindGroup>,
        offsets: &[wgt::DynamicOffset],
    ) {
        self.0
            .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                index,
                &bind_group.0,
                offsets,
                0f64,
                offsets.len() as u32,
            );
    }
    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        self.0.dispatch_with_y_and_z(x, y, z);
    }
    fn dispatch_indirect(
        &mut self,
        indirect_buffer: &Sendable<web_sys::GpuBuffer>,
        indirect_offset: wgt::BufferAddress,
    ) {
        self.0
            .dispatch_indirect_with_f64(&indirect_buffer.0, indirect_offset as f64);
    }
}

impl crate::RenderPassInner<Context> for RenderPass {
    fn set_pipeline(&mut self, pipeline: &Sendable<web_sys::GpuRenderPipeline>) {
        self.0.set_pipeline(&pipeline.0);
    }
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &Sendable<web_sys::GpuBindGroup>,
        offsets: &[wgt::DynamicOffset],
    ) {
        self.0
            .set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
                index,
                &bind_group.0,
                offsets,
                0f64,
                offsets.len() as u32,
            );
    }
    fn set_index_buffer(
        &mut self,
        buffer: &Sendable<web_sys::GpuBuffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) {
        self.0
            .set_index_buffer_with_f64_and_f64(&buffer.0, offset as f64, size as f64);
    }
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Sendable<web_sys::GpuBuffer>,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    ) {
        self.0
            .set_vertex_buffer_with_f64_and_f64(slot, &buffer.0, offset as f64, size as f64);
    }
    fn set_blend_color(&mut self, color: wgt::Color) {
        self.0
            .set_blend_color_with_gpu_color_dict(&map_color(color));
    }
    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.0.set_scissor_rect(x, y, width, height);
    }
    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.0
            .set_viewport(x, y, width, height, min_depth, max_depth);
    }
    fn set_stencil_reference(&mut self, reference: u32) {
        self.0.set_stencil_reference(reference);
    }
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.0
            .draw_with_instance_count_and_first_vertex_and_first_instance(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
    }
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.0
            .draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
    }
    fn draw_indirect(
        &mut self,
        indirect_buffer: &Sendable<web_sys::GpuBuffer>,
        indirect_offset: wgt::BufferAddress,
    ) {
        self.0
            .draw_indirect_with_f64(&indirect_buffer.0, indirect_offset as f64);
    }
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Sendable<web_sys::GpuBuffer>,
        indirect_offset: wgt::BufferAddress,
    ) {
        self.0
            .draw_indexed_indirect_with_f64(&indirect_buffer.0, indirect_offset as f64);
    }
}

fn map_texture_format(texture_format: wgt::TextureFormat) -> web_sys::GpuTextureFormat {
    use web_sys::GpuTextureFormat as tf;
    use wgt::TextureFormat;
    match texture_format {
        TextureFormat::R8Unorm => tf::R8unorm,
        TextureFormat::R8Snorm => tf::R8snorm,
        TextureFormat::R8Uint => tf::R8uint,
        TextureFormat::R8Sint => tf::R8sint,
        TextureFormat::R16Uint => tf::R16uint,
        TextureFormat::R16Sint => tf::R16sint,
        TextureFormat::R16Float => tf::R16float,
        TextureFormat::Rg8Unorm => tf::Rg8unorm,
        TextureFormat::Rg8Snorm => tf::Rg8snorm,
        TextureFormat::Rg8Uint => tf::Rg8uint,
        TextureFormat::Rg8Sint => tf::Rg8sint,
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
        TextureFormat::Rgb10a2Unorm => tf::Rgb10a2unorm,
        TextureFormat::Rg11b10Float => tf::Rg11b10float,
        TextureFormat::Rg32Uint => tf::Rg32uint,
        TextureFormat::Rg32Sint => tf::Rg32sint,
        TextureFormat::Rg32Float => tf::Rg32float,
        TextureFormat::Rgba16Uint => tf::Rgba16uint,
        TextureFormat::Rgba16Sint => tf::Rgba16sint,
        TextureFormat::Rgba16Float => tf::Rgba16float,
        TextureFormat::Rgba32Uint => tf::Rgba32uint,
        TextureFormat::Rgba32Sint => tf::Rgba32sint,
        TextureFormat::Rgba32Float => tf::Rgba32float,
        TextureFormat::Depth32Float => tf::Depth32float,
        TextureFormat::Depth24Plus => tf::Depth24plus,
        TextureFormat::Depth24PlusStencil8 => tf::Depth24plusStencil8,
    }
}

fn map_texture_component_type(
    texture_component_type: wgt::TextureComponentType,
) -> web_sys::GpuTextureComponentType {
    match texture_component_type {
        wgt::TextureComponentType::Float => web_sys::GpuTextureComponentType::Float,
        wgt::TextureComponentType::Sint => web_sys::GpuTextureComponentType::Sint,
        wgt::TextureComponentType::Uint => web_sys::GpuTextureComponentType::Uint,
    }
}

fn map_stage_descriptor(
    desc: &ProgrammableStageDescriptor,
) -> web_sys::GpuProgrammableStageDescriptor {
    web_sys::GpuProgrammableStageDescriptor::new(desc.entry_point, &desc.module.id.0)
}

fn map_cull_mode(cull_mode: wgt::CullMode) -> web_sys::GpuCullMode {
    use web_sys::GpuCullMode as cm;
    use wgt::CullMode;
    match cull_mode {
        CullMode::None => cm::None,
        CullMode::Front => cm::Front,
        CullMode::Back => cm::Back,
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

fn map_rasterization_state_descriptor(
    desc: &wgt::RasterizationStateDescriptor,
) -> web_sys::GpuRasterizationStateDescriptor {
    let mut mapped = web_sys::GpuRasterizationStateDescriptor::new();
    mapped.cull_mode(map_cull_mode(desc.cull_mode));
    mapped.depth_bias(desc.depth_bias);
    mapped.depth_bias_clamp(desc.depth_bias_clamp);
    mapped.depth_bias_slope_scale(desc.depth_bias_slope_scale);
    mapped.front_face(map_front_face(desc.front_face));
    mapped
}

fn map_compare_function(compare_fn: wgt::CompareFunction) -> Option<web_sys::GpuCompareFunction> {
    use web_sys::GpuCompareFunction as cf;
    use wgt::CompareFunction;
    match compare_fn {
        CompareFunction::Undefined => None,
        CompareFunction::Never => Some(cf::Never),
        CompareFunction::Less => Some(cf::Less),
        CompareFunction::Equal => Some(cf::Equal),
        CompareFunction::LessEqual => Some(cf::LessEqual),
        CompareFunction::Greater => Some(cf::Greater),
        CompareFunction::NotEqual => Some(cf::NotEqual),
        CompareFunction::GreaterEqual => Some(cf::GreaterEqual),
        CompareFunction::Always => Some(cf::Always),
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

fn map_stencil_state_face_descriptor(
    desc: &wgt::StencilStateFaceDescriptor,
) -> web_sys::GpuStencilStateFaceDescriptor {
    let mut mapped = web_sys::GpuStencilStateFaceDescriptor::new();
    if let Some(compare) = map_compare_function(desc.compare) {
        mapped.compare(compare);
    }
    mapped.depth_fail_op(map_stencil_operation(desc.depth_fail_op));
    mapped.fail_op(map_stencil_operation(desc.fail_op));
    mapped.pass_op(map_stencil_operation(desc.pass_op));
    mapped
}

fn map_depth_stencil_state_descriptor(
    desc: &wgt::DepthStencilStateDescriptor,
) -> web_sys::GpuDepthStencilStateDescriptor {
    let mut mapped = web_sys::GpuDepthStencilStateDescriptor::new(map_texture_format(desc.format));
    if let Some(depth_compare) = map_compare_function(desc.depth_compare) {
        mapped.depth_compare(depth_compare);
    }
    mapped.depth_write_enabled(desc.depth_write_enabled);
    mapped.stencil_back(&map_stencil_state_face_descriptor(&desc.stencil_back));
    mapped.stencil_front(&map_stencil_state_face_descriptor(&desc.stencil_front));
    mapped.stencil_read_mask(desc.stencil_read_mask);
    mapped.stencil_write_mask(desc.stencil_write_mask);
    mapped
}

fn map_blend_descriptor(desc: &wgt::BlendDescriptor) -> web_sys::GpuBlendDescriptor {
    let mut mapped = web_sys::GpuBlendDescriptor::new();
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
        BlendFactor::SrcColor => bf::SrcColor,
        BlendFactor::OneMinusSrcColor => bf::OneMinusSrcColor,
        BlendFactor::SrcAlpha => bf::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => bf::OneMinusSrcAlpha,
        BlendFactor::DstColor => bf::DstColor,
        BlendFactor::OneMinusDstColor => bf::OneMinusDstColor,
        BlendFactor::DstAlpha => bf::DstAlpha,
        BlendFactor::OneMinusDstAlpha => bf::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => bf::SrcAlphaSaturated,
        BlendFactor::BlendColor => bf::BlendColor,
        BlendFactor::OneMinusBlendColor => bf::OneMinusBlendColor,
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
        VertexFormat::Uchar2 => vf::Uchar2,
        VertexFormat::Uchar4 => vf::Uchar4,
        VertexFormat::Char2 => vf::Char2,
        VertexFormat::Char4 => vf::Char4,
        VertexFormat::Uchar2Norm => vf::Uchar2norm,
        VertexFormat::Uchar4Norm => vf::Uchar4norm,
        VertexFormat::Char2Norm => vf::Char2norm,
        VertexFormat::Char4Norm => vf::Char4norm,
        VertexFormat::Ushort2 => vf::Ushort2,
        VertexFormat::Ushort4 => vf::Ushort4,
        VertexFormat::Short2 => vf::Short2,
        VertexFormat::Short4 => vf::Short4,
        VertexFormat::Ushort2Norm => vf::Ushort2norm,
        VertexFormat::Ushort4Norm => vf::Ushort4norm,
        VertexFormat::Short2Norm => vf::Short2norm,
        VertexFormat::Short4Norm => vf::Short4norm,
        VertexFormat::Half2 => vf::Half2,
        VertexFormat::Half4 => vf::Half4,
        VertexFormat::Float => vf::Float,
        VertexFormat::Float2 => vf::Float2,
        VertexFormat::Float3 => vf::Float3,
        VertexFormat::Float4 => vf::Float4,
        VertexFormat::Uint => vf::Uint,
        VertexFormat::Uint2 => vf::Uint2,
        VertexFormat::Uint3 => vf::Uint3,
        VertexFormat::Uint4 => vf::Uint4,
        VertexFormat::Int => vf::Int,
        VertexFormat::Int2 => vf::Int2,
        VertexFormat::Int3 => vf::Int3,
        VertexFormat::Int4 => vf::Int4,
    }
}

fn map_input_step_mode(mode: wgt::InputStepMode) -> web_sys::GpuInputStepMode {
    use web_sys::GpuInputStepMode as sm;
    use wgt::InputStepMode;
    match mode {
        InputStepMode::Vertex => sm::Vertex,
        InputStepMode::Instance => sm::Instance,
    }
}

fn map_vertex_state_descriptor(
    desc: &RenderPipelineDescriptor,
) -> web_sys::GpuVertexStateDescriptor {
    let mapped_vertex_buffers = desc
        .vertex_state
        .vertex_buffers
        .iter()
        .map(|vbuf| {
            let mapped_attributes = vbuf
                .attributes
                .iter()
                .map(|attr| {
                    web_sys::GpuVertexAttributeDescriptor::new(
                        map_vertex_format(attr.format),
                        attr.offset as f64,
                        attr.shader_location,
                    )
                })
                .collect::<js_sys::Array>();

            let mut mapped_vbuf = web_sys::GpuVertexBufferLayoutDescriptor::new(
                vbuf.stride as f64,
                &mapped_attributes,
            );
            mapped_vbuf.step_mode(map_input_step_mode(vbuf.step_mode));
            mapped_vbuf
        })
        .collect::<js_sys::Array>();

    let mut mapped = web_sys::GpuVertexStateDescriptor::new();
    mapped.index_format(map_index_format(desc.vertex_state.index_format));
    mapped.vertex_buffers(&mapped_vertex_buffers);
    mapped
}

fn map_extent_3d(extent: wgt::Extent3d) -> web_sys::GpuExtent3dDict {
    web_sys::GpuExtent3dDict::new(extent.depth, extent.height, extent.width)
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
        TextureViewDimension::D1 => tvd::N1d,
        TextureViewDimension::D2 => tvd::N2d,
        TextureViewDimension::D2Array => tvd::N2dArray,
        TextureViewDimension::Cube => tvd::Cube,
        TextureViewDimension::CubeArray => tvd::CubeArray,
        TextureViewDimension::D3 => tvd::N3d,
    }
}

fn map_buffer_copy_view(view: crate::BufferCopyView<'_>) -> web_sys::GpuBufferCopyView {
    let mut mapped = web_sys::GpuBufferCopyView::new(&view.buffer.id.0, view.bytes_per_row);
    mapped.rows_per_image(view.rows_per_image);
    mapped.offset(view.offset as f64);
    mapped
}

fn map_texture_copy_view<'a>(view: crate::TextureCopyView<'a>) -> web_sys::GpuTextureCopyView {
    let mut mapped = web_sys::GpuTextureCopyView::new(&view.texture.id.0);
    mapped.array_layer(view.array_layer);
    mapped.mip_level(view.mip_level);
    mapped.origin(&map_origin_3d(view.origin));
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

fn map_address_mode(mode: wgt::AddressMode) -> web_sys::GpuAddressMode {
    match mode {
        wgt::AddressMode::ClampToEdge => web_sys::GpuAddressMode::ClampToEdge,
        wgt::AddressMode::Repeat => web_sys::GpuAddressMode::Repeat,
        wgt::AddressMode::MirrorRepeat => web_sys::GpuAddressMode::MirrorRepeat,
    }
}

fn map_color(color: wgt::Color) -> web_sys::GpuColorDict {
    web_sys::GpuColorDict::new(color.a, color.b, color.g, color.r)
}

fn map_store_op(op: wgt::StoreOp) -> web_sys::GpuStoreOp {
    match op {
        wgt::StoreOp::Clear => web_sys::GpuStoreOp::Clear,
        wgt::StoreOp::Store => web_sys::GpuStoreOp::Store,
    }
}

type JsFutureResult = Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
type FutureMap<T> = futures::future::Map<wasm_bindgen_futures::JsFuture, fn(JsFutureResult) -> T>;

fn future_request_adapter(result: JsFutureResult) -> Option<Sendable<web_sys::GpuAdapter>> {
    match result {
        Ok(js_value) => Some(Sendable(web_sys::GpuAdapter::from(js_value))),
        Err(_) => None,
    }
}
fn future_request_device(
    result: JsFutureResult,
) -> Result<(Sendable<web_sys::GpuDevice>, Sendable<web_sys::GpuQueue>), crate::RequestDeviceError>
{
    result
        .map(|js_value| {
            let device_id = web_sys::GpuDevice::from(js_value);
            let queue_id = device_id.default_queue();
            (Sendable(device_id), Sendable(queue_id))
        })
        .map_err(|_| crate::RequestDeviceError)
}

pub(crate) struct MapFuture<T> {
    child: wasm_bindgen_futures::JsFuture,
    buffer: Option<web_sys::GpuBuffer>,
    marker: PhantomData<T>,
}
impl<T> Unpin for MapFuture<T> {}
type MapData = (web_sys::GpuBuffer, Vec<u8>);
impl From<MapData> for BufferReadMappingDetail {
    fn from((buffer_id, mapped): MapData) -> Self {
        BufferReadMappingDetail {
            buffer_id: Sendable(buffer_id),
            mapped,
        }
    }
}
impl From<MapData> for BufferWriteMappingDetail {
    fn from((buffer_id, mapped): MapData) -> Self {
        BufferWriteMappingDetail {
            buffer_id: Sendable(buffer_id),
            mapped,
        }
    }
}
impl<T: From<MapData>> std::future::Future for MapFuture<T> {
    type Output = Result<T, crate::BufferAsyncError>;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context,
    ) -> std::task::Poll<Self::Output> {
        std::future::Future::poll(
            std::pin::Pin::new(&mut self.as_mut().get_mut().child),
            context,
        )
        .map(|result| {
            let buffer = self.buffer.take().unwrap();
            result
                .map(|js_value| {
                    let array_buffer = js_sys::ArrayBuffer::from(js_value);
                    let view = js_sys::Uint8Array::new(&array_buffer);
                    T::from((buffer, view.to_vec()))
                })
                .map_err(|_| crate::BufferAsyncError)
        })
    }
}

impl crate::Context for Context {
    type AdapterId = Sendable<web_sys::GpuAdapter>;
    type DeviceId = Sendable<web_sys::GpuDevice>;
    type QueueId = Sendable<web_sys::GpuQueue>;
    type ShaderModuleId = Sendable<web_sys::GpuShaderModule>;
    type BindGroupLayoutId = Sendable<web_sys::GpuBindGroupLayout>;
    type BindGroupId = Sendable<web_sys::GpuBindGroup>;
    type TextureViewId = Sendable<web_sys::GpuTextureView>;
    type SamplerId = Sendable<web_sys::GpuSampler>;
    type BufferId = Sendable<web_sys::GpuBuffer>;
    type TextureId = Sendable<web_sys::GpuTexture>;
    type PipelineLayoutId = Sendable<web_sys::GpuPipelineLayout>;
    type RenderPipelineId = Sendable<web_sys::GpuRenderPipeline>;
    type ComputePipelineId = Sendable<web_sys::GpuComputePipeline>;
    type CommandEncoderId = web_sys::GpuCommandEncoder;
    type ComputePassId = ComputePass;
    type CommandBufferId = Sendable<web_sys::GpuCommandBuffer>;
    type SurfaceId = Sendable<web_sys::GpuCanvasContext>;
    type SwapChainId = Sendable<web_sys::GpuSwapChain>;
    type RenderPassId = RenderPass;

    type CreateBufferMappedDetail = CreateBufferMappedDetail;
    type BufferReadMappingDetail = BufferReadMappingDetail;
    type BufferWriteMappingDetail = BufferWriteMappingDetail;
    type SwapChainOutputDetail = SwapChainOutputDetail;

    type RequestAdapterFuture = MakeSendFuture<FutureMap<Option<Self::AdapterId>>>;
    type RequestDeviceFuture = MakeSendFuture<
        FutureMap<Result<(Self::DeviceId, Self::QueueId), crate::RequestDeviceError>>,
    >;
    type MapReadFuture = MakeSendFuture<MapFuture<BufferReadMappingDetail>>;
    type MapWriteFuture = MakeSendFuture<MapFuture<BufferWriteMappingDetail>>;

    fn init() -> Self {
        Sendable(web_sys::window().unwrap().navigator().gpu())
    }

    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId {
        let canvas_attribute = match handle.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Web(web_handle) => web_handle.id,
            _ => panic!("expected valid handle for canvas"),
        };
        let canvas_node: wasm_bindgen::JsValue = web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.query_selector_all(&format!("[data-raw-handle=\"{}\"]", canvas_attribute))
                    .ok()
            })
            .and_then(|nodes| nodes.get(0))
            .expect("expected to find single canvas")
            .into();
        let canvas_element: web_sys::HtmlCanvasElement = canvas_node.into();
        let context: wasm_bindgen::JsValue = match canvas_element.get_context("gpupresent") {
            Ok(Some(ctx)) => ctx.into(),
            _ => panic!("expected to get context from canvas"),
        };
        Sendable(context.into())
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_>,
        _backends: wgt::BackendBit,
    ) -> Self::RequestAdapterFuture {
        //TODO: support this check, return `None` if the flag is not set.
        // It's not trivial, since we need the Future logic to have this check,
        // and currently the Future her has no room for extra parameter `backends`.
        //assert!(backends.contains(wgt::BackendBit::BROWSER_WEBGPU));
        let mut mapped_options = web_sys::GpuRequestAdapterOptions::new();
        let mapped_power_preference = match options.power_preference {
            wgt::PowerPreference::LowPower => web_sys::GpuPowerPreference::LowPower,
            wgt::PowerPreference::HighPerformance | wgt::PowerPreference::Default => {
                web_sys::GpuPowerPreference::HighPerformance
            }
        };
        mapped_options.power_preference(mapped_power_preference);
        let adapter_promise = self.0.request_adapter_with_options(&mapped_options);
        MakeSendFuture(
            wasm_bindgen_futures::JsFuture::from(adapter_promise).map(future_request_adapter),
        )
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        if trace_dir.is_some() {
            //Error: Tracing isn't supported on the Web target
        }
        let mut mapped_desc = web_sys::GpuDeviceDescriptor::new();
        // TODO: label, extensions
        let mut mapped_limits = web_sys::GpuLimits::new();
        mapped_limits.max_bind_groups(desc.limits.max_bind_groups);
        mapped_desc.limits(&mapped_limits);
        let device_promise = adapter.0.request_device_with_descriptor(&mapped_desc);

        MakeSendFuture(
            wasm_bindgen_futures::JsFuture::from(device_promise).map(future_request_device),
        )
    }

    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &wgt::SwapChainDescriptor,
    ) -> Self::SwapChainId {
        let mut mapped =
            web_sys::GpuSwapChainDescriptor::new(&device.0, map_texture_format(desc.format));
        mapped.usage(desc.usage.bits());
        Sendable(surface.0.configure_swap_chain(&mapped))
    }

    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        spv: &[u32],
    ) -> Self::ShaderModuleId {
        let desc = web_sys::GpuShaderModuleDescriptor::new(&js_sys::Uint32Array::from(spv));
        // TODO: label
        Sendable(device.0.create_shader_module(&desc))
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId {
        use web_sys::GpuBindingType as bt;

        let mapped_bindings = desc
            .bindings
            .iter()
            .map(|bind| {
                let mapped_type = match bind.ty {
                    BindingType::UniformBuffer { .. } => bt::UniformBuffer,
                    BindingType::StorageBuffer {
                        readonly: false, ..
                    } => bt::StorageBuffer,
                    BindingType::StorageBuffer { readonly: true, .. } => bt::ReadonlyStorageBuffer,
                    BindingType::Sampler { comparison: false } => bt::Sampler,
                    BindingType::Sampler { .. } => bt::ComparisonSampler,
                    BindingType::SampledTexture { .. } => bt::SampledTexture,
                    BindingType::StorageTexture { readonly: true, .. } => {
                        bt::ReadonlyStorageTexture
                    }
                    BindingType::StorageTexture { .. } => bt::WriteonlyStorageTexture,
                };

                let mut mapped_entry = web_sys::GpuBindGroupLayoutEntry::new(
                    bind.binding,
                    mapped_type,
                    bind.visibility.bits(),
                );

                match bind.ty {
                    BindingType::UniformBuffer { dynamic }
                    | BindingType::StorageBuffer { dynamic, .. } => {
                        mapped_entry.has_dynamic_offset(dynamic);
                    }
                    _ => {}
                }

                if let BindingType::SampledTexture { multisampled, .. } = bind.ty {
                    mapped_entry.multisampled(multisampled);
                }

                match bind.ty {
                    BindingType::SampledTexture { dimension, .. }
                    | BindingType::StorageTexture { dimension, .. } => {
                        mapped_entry.view_dimension(map_texture_view_dimension(dimension));
                    }
                    _ => {}
                }

                if let BindingType::StorageTexture { format, .. } = bind.ty {
                    mapped_entry.storage_texture_format(map_texture_format(format));
                }

                match bind.ty {
                    BindingType::SampledTexture { component_type, .. }
                    | BindingType::StorageTexture { component_type, .. } => {
                        mapped_entry
                            .texture_component_type(map_texture_component_type(component_type));
                    }
                    _ => {}
                }

                mapped_entry
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = web_sys::GpuBindGroupLayoutDescriptor::new(&mapped_bindings);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device.0.create_bind_group_layout(&mapped_desc))
    }

    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId {
        let mapped_entries = desc
            .bindings
            .iter()
            .map(|binding| {
                let mapped_resource = match &binding.resource {
                    BindingResource::Buffer(buffer_slice) => {
                        let mut mapped_buffer_binding =
                            web_sys::GpuBufferBinding::new(&buffer_slice.buffer.id.0);
                        mapped_buffer_binding.offset(buffer_slice.offset as f64);
                        mapped_buffer_binding.size(buffer_slice.size_or_0() as f64);
                        JsValue::from(mapped_buffer_binding.clone())
                    }
                    BindingResource::Sampler(ref sampler) => JsValue::from(sampler.id.0.clone()),
                    BindingResource::TextureView(ref texture_view) => {
                        JsValue::from(texture_view.id.0.clone())
                    }
                };

                web_sys::GpuBindGroupEntry::new(binding.binding, &mapped_resource)
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc =
            web_sys::GpuBindGroupDescriptor::new(&mapped_entries, &desc.layout.id.0);
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device.0.create_bind_group(&mapped_desc))
    }

    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId {
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id.0.clone())
            .collect::<js_sys::Array>();
        let mapped_desc = web_sys::GpuPipelineLayoutDescriptor::new(&temp_layouts);
        // TODO: label
        Sendable(device.0.create_pipeline_layout(&mapped_desc))
    }

    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId {
        use web_sys::GpuPrimitiveTopology as pt;

        let mapped_color_states = desc
            .color_states
            .iter()
            .map(|color_state_desc| {
                let mapped_format = map_texture_format(color_state_desc.format);
                let mut mapped_color_state_desc =
                    web_sys::GpuColorStateDescriptor::new(mapped_format);
                mapped_color_state_desc
                    .alpha_blend(&map_blend_descriptor(&color_state_desc.alpha_blend));
                mapped_color_state_desc
                    .color_blend(&map_blend_descriptor(&color_state_desc.color_blend));
                mapped_color_state_desc.write_mask(color_state_desc.write_mask.bits());
                mapped_color_state_desc
            })
            .collect::<js_sys::Array>();

        let mapped_primitive_topology = match desc.primitive_topology {
            wgt::PrimitiveTopology::PointList => pt::PointList,
            wgt::PrimitiveTopology::LineList => pt::LineList,
            wgt::PrimitiveTopology::LineStrip => pt::LineStrip,
            wgt::PrimitiveTopology::TriangleList => pt::TriangleList,
            wgt::PrimitiveTopology::TriangleStrip => pt::TriangleStrip,
        };

        let mapped_vertex_stage = map_stage_descriptor(&desc.vertex_stage);

        let mut mapped_desc = web_sys::GpuRenderPipelineDescriptor::new(
            &desc.layout.id.0,
            &mapped_color_states,
            mapped_primitive_topology,
            &mapped_vertex_stage,
        );

        // TODO: label

        if let Some(ref frag) = desc.fragment_stage {
            mapped_desc.fragment_stage(&map_stage_descriptor(frag));
        }

        if let Some(ref rasterization) = desc.rasterization_state {
            mapped_desc.rasterization_state(&map_rasterization_state_descriptor(rasterization));
        }

        if let Some(ref depth_stencil) = desc.depth_stencil_state {
            mapped_desc.depth_stencil_state(&map_depth_stencil_state_descriptor(depth_stencil));
        }

        mapped_desc.vertex_state(&map_vertex_state_descriptor(&desc));
        mapped_desc.sample_count(desc.sample_count);
        mapped_desc.sample_mask(desc.sample_mask);
        mapped_desc.alpha_to_coverage_enabled(desc.alpha_to_coverage_enabled);

        Sendable(device.0.create_render_pipeline(&mapped_desc))
    }

    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId {
        let mapped_compute_stage = map_stage_descriptor(&desc.compute_stage);
        let mapped_desc =
            web_sys::GpuComputePipelineDescriptor::new(&desc.layout.id.0, &mapped_compute_stage);
        // TODO: label
        Sendable(device.0.create_compute_pipeline(&mapped_desc))
    }

    fn device_create_buffer_mapped<'a>(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> (Self::BufferId, &'a mut [u8], Self::CreateBufferMappedDetail) {
        let mut mapped_desc =
            web_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        unsafe {
            let pair = device.0.create_buffer_mapped(&mapped_desc);
            let id = pair.get(0).into();
            let array_buffer = pair.get(1).into();
            // TODO: Use `Vec::from_raw_parts` once it's stable
            let memory = vec![0; desc.size as usize].into_boxed_slice();
            let mapped_data = std::slice::from_raw_parts_mut(
                Box::into_raw(memory) as *mut u8,
                desc.size as usize,
            );
            (
                Sendable(id),
                mapped_data,
                CreateBufferMappedDetail { array_buffer },
            )
        }
    }

    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> Self::BufferId {
        let mut mapped_desc =
            web_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        Sendable(device.0.create_buffer(&mapped_desc))
    }

    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        desc: &TextureDescriptor,
    ) -> Self::TextureId {
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
        Sendable(device.0.create_texture(&mapped_desc))
    }

    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        desc: &SamplerDescriptor,
    ) -> Self::SamplerId {
        let mut mapped_desc = web_sys::GpuSamplerDescriptor::new();
        // TODO: label
        mapped_desc.address_mode_u(map_address_mode(desc.address_mode_u));
        mapped_desc.address_mode_v(map_address_mode(desc.address_mode_v));
        mapped_desc.address_mode_w(map_address_mode(desc.address_mode_w));
        if let Some(compare) = map_compare_function(desc.compare) {
            mapped_desc.compare(compare);
        }
        mapped_desc.lod_max_clamp(desc.lod_max_clamp);
        mapped_desc.lod_min_clamp(desc.lod_min_clamp);
        mapped_desc.mag_filter(map_filter_mode(desc.mag_filter));
        mapped_desc.min_filter(map_filter_mode(desc.min_filter));
        mapped_desc.mipmap_filter(map_filter_mode(desc.mipmap_filter));
        Sendable(device.0.create_sampler_with_descriptor(&mapped_desc))
    }

    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &CommandEncoderDescriptor,
    ) -> Self::CommandEncoderId {
        let mut mapped_desc = web_sys::GpuCommandEncoderDescriptor::new();
        if let Some(label) = desc.label {
            mapped_desc.label(label);
        }
        device
            .0
            .create_command_encoder_with_descriptor(&mapped_desc)
    }

    fn device_drop(&self, _device: &Self::DeviceId) {
        // Device is dropped automatically
    }

    fn device_poll(&self, _device: &Self::DeviceId, _maintain: crate::Maintain) {
        // Device is polled automatically
    }

    fn buffer_map_read(
        &self,
        buffer: &Self::BufferId,
        _start: wgt::BufferAddress,
        _size: wgt::BufferAddress,
    ) -> Self::MapReadFuture {
        MakeSendFuture(MapFuture {
            child: wasm_bindgen_futures::JsFuture::from(buffer.0.map_read_async()),
            buffer: Some(buffer.0.clone()),
            marker: PhantomData,
        })
    }

    fn buffer_map_write(
        &self,
        buffer: &Self::BufferId,
        _start: wgt::BufferAddress,
        _size: wgt::BufferAddress,
    ) -> Self::MapWriteFuture {
        MakeSendFuture(MapFuture {
            child: wasm_bindgen_futures::JsFuture::from(buffer.0.map_write_async()),
            buffer: Some(buffer.0.clone()),
            marker: PhantomData,
        })
    }

    fn buffer_unmap(&self, buffer: &Self::BufferId) {
        buffer.0.unmap();
    }

    fn swap_chain_get_next_texture(
        &self,
        swap_chain: &Self::SwapChainId,
    ) -> Result<(Self::TextureViewId, Self::SwapChainOutputDetail), crate::TimeOut> {
        // TODO: Should we pass a descriptor here?
        // Or is the default view always correct?
        Ok((
            Sendable(swap_chain.0.get_current_texture().create_view()),
            (),
        ))
    }

    fn swap_chain_present(
        &self,
        _view: &Self::TextureViewId,
        _detail: &Self::SwapChainOutputDetail,
    ) {
        // Swapchain is presented automatically
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        desc: Option<&TextureViewDescriptor>,
    ) -> Self::TextureViewId {
        Sendable(match desc {
            Some(d) => {
                let mut mapped_desc = web_sys::GpuTextureViewDescriptor::new();
                mapped_desc.array_layer_count(d.array_layer_count);
                mapped_desc.aspect(map_texture_aspect(d.aspect));
                mapped_desc.base_array_layer(d.base_array_layer);
                mapped_desc.base_mip_level(d.base_mip_level);
                mapped_desc.dimension(map_texture_view_dimension(d.dimension));
                mapped_desc.format(map_texture_format(d.format));
                mapped_desc.mip_level_count(d.level_count);
                // TODO: label
                texture.0.create_view_with_descriptor(&mapped_desc)
            }
            None => texture.0.create_view(),
        })
    }

    fn texture_drop(&self, _texture: &Self::TextureId) {
        // Buffer is dropped automatically
    }
    fn texture_view_drop(&self, _texture_view: &Self::TextureViewId) {
        // Buffer is dropped automatically
    }
    fn sampler_drop(&self, _sampler: &Self::SamplerId) {
        // Buffer is dropped automatically
    }
    fn buffer_drop(&self, _buffer: &Self::BufferId) {
        // Buffer is dropped automatically
    }
    fn bind_group_drop(&self, _bind_group: &Self::BindGroupId) {
        // Buffer is dropped automatically
    }
    fn bind_group_layout_drop(&self, _bind_group_layout: &Self::BindGroupLayoutId) {
        // Buffer is dropped automatically
    }
    fn pipeline_layout_drop(&self, _pipeline_layout: &Self::PipelineLayoutId) {
        // Buffer is dropped automatically
    }
    fn shader_module_drop(&self, _shader_module: &Self::ShaderModuleId) {
        // Buffer is dropped automatically
    }
    fn command_buffer_drop(&self, _command_buffer: &Self::CommandBufferId) {
        // Buffer is dropped automatically
    }
    fn compute_pipeline_drop(&self, _pipeline: &Self::ComputePipelineId) {
        // Buffer is dropped automatically
    }
    fn render_pipeline_drop(&self, _pipeline: &Self::RenderPipelineId) {
        // Buffer is dropped automatically
    }

    fn flush_mapped_data(data: &mut [u8], detail: CreateBufferMappedDetail) {
        unsafe {
            // Convert the `mapped_data` slice back into a `Vec`. This should be
            // safe because `mapped_data` is no longer accessible beyond this
            // function.
            let memory: Vec<u8> = Box::<[u8]>::from_raw(data).into();

            // Create a view into the mapped `ArrayBuffer` that was provided by the
            // browser
            let mapped = js_sys::Uint8Array::new(&detail.array_buffer);

            // Convert `memory` into a temporary `Uint8Array` view. This should be
            // safe as long as the backing wasm memory is not resized.
            let memory_view = js_sys::Uint8Array::view(&memory[..]);

            // Finally copy into `mapped` and let `memory` drop
            mapped.set(&memory_view, 0);
        }
    }

    fn encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: wgt::BufferAddress,
        destination: &Self::BufferId,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        encoder.copy_buffer_to_buffer_with_f64_and_f64_and_f64(
            &source.0,
            source_offset as f64,
            &destination.0,
            destination_offset as f64,
            copy_size as f64,
        )
    }

    fn encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::BufferCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        encoder.copy_buffer_to_texture_with_gpu_extent_3d_dict(
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &map_extent_3d(copy_size),
        )
    }

    fn encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::BufferCopyView,
        copy_size: wgt::Extent3d,
    ) {
        encoder.copy_texture_to_buffer_with_gpu_extent_3d_dict(
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &map_extent_3d(copy_size),
        )
    }

    fn encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::TextureCopyView,
        destination: crate::TextureCopyView,
        copy_size: wgt::Extent3d,
    ) {
        encoder.copy_texture_to_texture_with_gpu_extent_3d_dict(
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &map_extent_3d(copy_size),
        )
    }

    fn encoder_begin_compute_pass(&self, encoder: &Self::CommandEncoderId) -> Self::ComputePassId {
        let mut mapped_desc = web_sys::GpuComputePassDescriptor::new();
        if let Some(ref label) = encoder.label() {
            mapped_desc.label(label);
        }
        ComputePass(encoder.begin_compute_pass_with_descriptor(&mapped_desc))
    }

    fn encoder_end_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    ) {
        pass.0.end_pass();
    }

    fn encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &crate::RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId {
        let mapped_color_attachments = desc
            .color_attachments
            .iter()
            .map(|ca| {
                let mut mapped_color_attachment =
                    web_sys::GpuRenderPassColorAttachmentDescriptor::new(
                        &ca.attachment.id.0,
                        &match ca.load_op {
                            wgt::LoadOp::Clear => {
                                wasm_bindgen::JsValue::from(map_color(ca.clear_color))
                            }
                            wgt::LoadOp::Load => {
                                wasm_bindgen::JsValue::from(web_sys::GpuLoadOp::Load)
                            }
                        },
                    );

                if let Some(rt) = ca.resolve_target {
                    mapped_color_attachment.resolve_target(&rt.id.0);
                }

                mapped_color_attachment.store_op(map_store_op(ca.store_op));

                mapped_color_attachment
            })
            .collect::<js_sys::Array>();

        let mut mapped_desc = web_sys::GpuRenderPassDescriptor::new(&mapped_color_attachments);

        // TODO: label

        if let Some(dsa) = &desc.depth_stencil_attachment {
            let mapped_depth_stencil_attachment =
                web_sys::GpuRenderPassDepthStencilAttachmentDescriptor::new(
                    &dsa.attachment.id.0,
                    &match dsa.depth_load_op {
                        wgt::LoadOp::Clear => wasm_bindgen::JsValue::from(dsa.clear_depth),
                        wgt::LoadOp::Load => wasm_bindgen::JsValue::from(web_sys::GpuLoadOp::Load),
                    },
                    map_store_op(dsa.depth_store_op),
                    &match dsa.stencil_load_op {
                        wgt::LoadOp::Clear => wasm_bindgen::JsValue::from(dsa.clear_stencil),
                        wgt::LoadOp::Load => wasm_bindgen::JsValue::from(web_sys::GpuLoadOp::Load),
                    },
                    map_store_op(dsa.stencil_store_op),
                );

            mapped_desc.depth_stencil_attachment(&mapped_depth_stencil_attachment);
        }

        RenderPass(encoder.begin_render_pass(&mapped_desc))
    }

    fn encoder_end_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    ) {
        pass.0.end_pass();
    }

    fn encoder_finish(&self, encoder: &Self::CommandEncoderId) -> Self::CommandBufferId {
        let mut mapped_desc = web_sys::GpuCommandBufferDescriptor::new();
        if let Some(ref label) = encoder.label() {
            mapped_desc.label(label);
        }
        Sendable(encoder.finish_with_descriptor(&mapped_desc))
    }

    fn queue_write_buffer(
        &self,
        _queue: &Self::QueueId,
        _data: &[u8],
        _buffer: &Self::BufferId,
        _offset: wgt::BufferAddress,
    ) {
        unimplemented!()
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    ) {
        let temp_command_buffers = command_buffers.map(|i| i.0).collect::<js_sys::Array>();

        queue.0.submit(&temp_command_buffers);
    }
}

pub(crate) struct CreateBufferMappedDetail {
    /// On wasm we need to allocate our own temporary storage for `data`. Later
    /// we copy this temporary storage into the `Uint8Array` which was returned
    /// by the browser originally.
    array_buffer: js_sys::ArrayBuffer,
}

// `CreateBufferMappedDetail` must be `Send` to match native.
//
// SAFETY: This is safe on wasm32 *for now*, but similarly to the unsafe Send impls for the handle
// type wrappers, the full story for threading on wasm32 is still unfolding.
unsafe impl Send for CreateBufferMappedDetail {}

pub(crate) struct BufferReadMappingDetail {
    pub(crate) buffer_id: Sendable<web_sys::GpuBuffer>,
    mapped: Vec<u8>,
}

impl BufferReadMappingDetail {
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.mapped[..]
    }
}

pub(crate) struct BufferWriteMappingDetail {
    pub(crate) buffer_id: Sendable<web_sys::GpuBuffer>,
    mapped: Vec<u8>,
}

impl BufferWriteMappingDetail {
    pub(crate) fn as_slice(&mut self) -> &mut [u8] {
        &mut self.mapped[..]
    }
}

pub(crate) type SwapChainOutputDetail = ();
