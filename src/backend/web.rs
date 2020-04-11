use crate::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, BindingType, BufferDescriptor,
    CommandEncoderDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ProgrammableStageDescriptor, RenderPipelineDescriptor, SamplerDescriptor, TextureDescriptor,
    TextureViewDescriptor, TextureViewDimension,
};

use std::ops::Range;
use wasm_bindgen::prelude::*;

pub type AdapterId = web_sys::GpuAdapter;
pub type DeviceId = web_sys::GpuDevice;
pub type QueueId = web_sys::GpuQueue;
pub type ShaderModuleId = web_sys::GpuShaderModule;
pub type BindGroupLayoutId = web_sys::GpuBindGroupLayout;
pub type BindGroupId = web_sys::GpuBindGroup;
pub type TextureViewId = web_sys::GpuTextureView;
pub type SamplerId = web_sys::GpuSampler;
pub type BufferId = web_sys::GpuBuffer;
pub type TextureId = web_sys::GpuTexture;
pub type PipelineLayoutId = web_sys::GpuPipelineLayout;
pub type RenderPipelineId = web_sys::GpuRenderPipeline;
pub type ComputePipelineId = web_sys::GpuComputePipeline;
pub type CommandEncoderId = web_sys::GpuCommandEncoder;
pub type ComputePassId = web_sys::GpuComputePassEncoder;
pub type CommandBufferId = web_sys::GpuCommandBuffer;
pub type SurfaceId = web_sys::GpuCanvasContext;
pub type SwapChainId = web_sys::GpuSwapChain;
pub type RenderPassEncoderId = web_sys::GpuRenderPassEncoder;

fn gpu() -> web_sys::Gpu {
    web_sys::window().unwrap().navigator().gpu()
}

pub(crate) async fn request_adapter(
    options: &crate::RequestAdapterOptions<'_>,
    backends: wgt::BackendBit,
) -> Option<AdapterId> {
    if !backends.contains(wgt::BackendBit::BROWSER_WEBGPU) {
        return None;
    }

    let mut mapped_options = web_sys::GpuRequestAdapterOptions::new();
    let mapped_power_preference = match options.power_preference {
        wgt::PowerPreference::LowPower => web_sys::GpuPowerPreference::LowPower,
        wgt::PowerPreference::HighPerformance | wgt::PowerPreference::Default => {
            web_sys::GpuPowerPreference::HighPerformance
        }
    };
    mapped_options.power_preference(mapped_power_preference);
    let adapter_promise = gpu().request_adapter_with_options(&mapped_options);
    Some(
        wasm_bindgen_futures::JsFuture::from(adapter_promise)
            .await
            .expect("Unable to get adapter")
            .into(),
    )
}

pub(crate) async fn request_device_and_queue(
    adapter: &AdapterId,
    desc: Option<&wgt::DeviceDescriptor>,
) -> (DeviceId, QueueId) {
    let device_promise = match desc {
        Some(d) => {
            let mut mapped_descriptor = web_sys::GpuDeviceDescriptor::new();
            // TODO: Extensions
            let mut mapped_limits = web_sys::GpuLimits::new();
            mapped_limits.max_bind_groups(d.limits.max_bind_groups);
            // TODO: Other fields
            mapped_descriptor.limits(&mapped_limits);
            adapter.request_device_with_descriptor(&mapped_descriptor)
        }
        None => adapter.request_device(),
    };
    let js_value = wasm_bindgen_futures::JsFuture::from(device_promise)
        .await
        .expect("Unable to get device");
    let device_id = DeviceId::from(js_value);
    let queue_id = device_id.default_queue();
    (device_id, queue_id)
}

pub(crate) fn create_shader_module(device: &DeviceId, spv: &[u32]) -> ShaderModuleId {
    let desc = web_sys::GpuShaderModuleDescriptor::new(&js_sys::Uint32Array::from(spv));
    device.create_shader_module(&desc)
}

pub(crate) fn create_bind_group_layout(
    device: &DeviceId,
    desc: &BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
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
                BindingType::Sampler { .. } => unimplemented!(), // TODO: bt::ComparisonSampler,
                BindingType::SampledTexture { .. } => bt::SampledTexture,
                BindingType::StorageTexture { readonly: true, .. } => {
                    unimplemented!() // TODO: bt::ReadonlyStorageTexture
                }
                BindingType::StorageTexture { .. } => {
                    unimplemented!() // TODO: bt::WriteonlyStorageTexture
                }
            };

            let mapped_dynamic = match bind.ty {
                BindingType::UniformBuffer { dynamic }
                | BindingType::StorageBuffer { dynamic, .. } => dynamic,
                _ => false,
            };

            let mapped_multisampled = match bind.ty {
                BindingType::SampledTexture { multisampled, .. } => multisampled,
                _ => false,
            };

            let mapped_view_dimension = match bind.ty {
                BindingType::SampledTexture { dimension, .. }
                | BindingType::StorageTexture { dimension, .. } => {
                    map_texture_view_dimension(dimension)
                }
                _ => web_sys::GpuTextureViewDimension::N2d,
            };

            let mut mapped_entry = web_sys::GpuBindGroupLayoutEntry::new(
                bind.binding,
                mapped_type,
                bind.visibility.bits(),
            );
            mapped_entry.has_dynamic_offset(mapped_dynamic);
            mapped_entry.multisampled(mapped_multisampled);
            mapped_entry.view_dimension(mapped_view_dimension);

            // TODO: Texture component type, storage texture format

            mapped_entry
        })
        .collect::<js_sys::Array>();

    let mapped_desc = web_sys::GpuBindGroupLayoutDescriptor::new(&mapped_bindings);
    device.create_bind_group_layout(&mapped_desc)
}

pub(crate) fn create_bind_group(device: &DeviceId, desc: &BindGroupDescriptor) -> BindGroupId {
    let mapped_entries = desc
        .bindings
        .iter()
        .map(|binding| {
            let mapped_resource = match binding.resource {
                BindingResource::Buffer {
                    ref buffer,
                    ref range,
                } => {
                    let mut mapped_buffer_binding = web_sys::GpuBufferBinding::new(&buffer.id);
                    mapped_buffer_binding.offset(range.start as f64);
                    mapped_buffer_binding.size((range.end - range.start) as f64);
                    JsValue::from(mapped_buffer_binding.clone())
                }
                BindingResource::Sampler(ref sampler) => JsValue::from(sampler.id.clone()),
                BindingResource::TextureView(ref texture_view) => {
                    JsValue::from(texture_view.id.clone())
                }
            };

            web_sys::GpuBindGroupEntry::new(binding.binding, &mapped_resource)
        })
        .collect::<js_sys::Array>();

    let mapped_desc = web_sys::GpuBindGroupDescriptor::new(&mapped_entries, &desc.layout.id);
    device.create_bind_group(&mapped_desc)
}

pub(crate) fn create_pipeline_layout(
    device: &DeviceId,
    desc: &PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    //TODO: avoid allocation here
    let temp_layouts = desc
        .bind_group_layouts
        .iter()
        .map(|bgl| bgl.id.clone())
        .collect::<js_sys::Array>();
    let mapped_desc = web_sys::GpuPipelineLayoutDescriptor::new(&temp_layouts);
    device.create_pipeline_layout(&mapped_desc)
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

fn map_stage_descriptor(
    desc: &ProgrammableStageDescriptor,
) -> web_sys::GpuProgrammableStageDescriptor {
    web_sys::GpuProgrammableStageDescriptor::new(desc.entry_point, &desc.module.id)
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
    let mut mapped = web_sys::GpuBufferCopyView::new(&view.buffer.id, view.bytes_per_row);
    mapped.rows_per_image(view.rows_per_image);
    mapped.offset(view.offset as f64);
    mapped
}

fn map_texture_copy_view<'a>(view: crate::TextureCopyView<'a>) -> web_sys::GpuTextureCopyView {
    let mut mapped = web_sys::GpuTextureCopyView::new(&view.texture.id);
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

pub(crate) fn create_render_pipeline(
    device: &DeviceId,
    desc: &RenderPipelineDescriptor,
) -> RenderPipelineId {
    use web_sys::GpuPrimitiveTopology as pt;

    let mapped_color_states = desc
        .color_states
        .iter()
        .map(|color_state_desc| {
            let mapped_format = map_texture_format(color_state_desc.format);
            let mut mapped_color_state_desc = web_sys::GpuColorStateDescriptor::new(mapped_format);
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
        &desc.layout.id,
        &mapped_color_states,
        mapped_primitive_topology,
        &mapped_vertex_stage,
    );

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

    device.create_render_pipeline(&mapped_desc)
}

pub(crate) fn create_compute_pipeline(
    device: &DeviceId,
    desc: &ComputePipelineDescriptor,
) -> ComputePipelineId {
    let mapped_compute_stage = map_stage_descriptor(&desc.compute_stage);
    let mapped_desc =
        web_sys::GpuComputePipelineDescriptor::new(&desc.layout.id, &mapped_compute_stage);

    device.create_compute_pipeline(&mapped_desc)
}

pub(crate) struct CreateBufferMappedDetail {
    /// On wasm we need to allocate our own temporary storage for `data`. Later
    /// we copy this temporary storage into the `Uint8Array` which was returned
    /// by the browser originally.
    array_buffer: js_sys::ArrayBuffer,
}

pub(crate) fn device_create_buffer_mapped<'a>(
    device: &DeviceId,
    desc: &BufferDescriptor,
) -> crate::CreateBufferMapped<'a> {
    let mapped_desc = web_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
    unsafe {
        let pair = device.create_buffer_mapped(&mapped_desc);
        let id = pair.get(0).into();
        let array_buffer = pair.get(1).into();
        // TODO: Use `Vec::from_raw_parts` once it's stable
        let memory = vec![0; desc.size as usize].into_boxed_slice();
        let mapped_data =
            std::slice::from_raw_parts_mut(Box::into_raw(memory) as *mut u8, desc.size as usize);
        crate::CreateBufferMapped {
            id,
            mapped_data,
            detail: CreateBufferMappedDetail { array_buffer },
        }
    }
}

pub type BufferDetail = ();

pub(crate) fn device_create_buffer_mapped_finish(
    create_buffer_mapped: crate::CreateBufferMapped<'_>,
) -> crate::Buffer {
    unsafe {
        // Convert the `mapped_data` slice back into a `Vec`. This should be
        // safe because `mapped_data` is no longer accessible beyond this
        // function.
        let memory: Vec<u8> = Box::<[u8]>::from_raw(create_buffer_mapped.mapped_data).into();

        // Create a view into the mapped `ArrayBuffer` that was provided by the
        // browser
        let mapped = js_sys::Uint8Array::new(&create_buffer_mapped.detail.array_buffer);

        // Convert `memory` into a temporary `Uint8Array` view. This should be
        // safe as long as the backing wasm memory is not resized.
        let memory_view = js_sys::Uint8Array::view(&memory[..]);

        // Finally copy into `mapped` and let `memory` drop
        mapped.set(&memory_view, 0);
    }

    buffer_unmap(&create_buffer_mapped.id);

    crate::Buffer {
        id: create_buffer_mapped.id,
        detail: (),
    }
}

pub(crate) fn buffer_unmap(buffer: &BufferId) {
    buffer.unmap();
}

pub(crate) fn device_create_buffer(device: &DeviceId, desc: &BufferDescriptor) -> crate::Buffer {
    let mapped_desc = web_sys::GpuBufferDescriptor::new(desc.size as f64, desc.usage.bits());
    crate::Buffer {
        id: device.create_buffer(&mapped_desc),
        detail: (),
    }
}

pub(crate) fn device_create_texture(device: &DeviceId, desc: &TextureDescriptor) -> TextureId {
    let mut mapped_desc = web_sys::GpuTextureDescriptor::new(
        map_texture_format(desc.format),
        &map_extent_3d(desc.size),
        desc.usage.bits(),
    );
    mapped_desc.dimension(map_texture_dimension(desc.dimension));
    mapped_desc.mip_level_count(desc.mip_level_count);
    mapped_desc.sample_count(desc.sample_count);
    device.create_texture(&mapped_desc)
}

pub(crate) fn device_create_sampler(device: &DeviceId, desc: &SamplerDescriptor) -> SamplerId {
    let mut mapped_desc = web_sys::GpuSamplerDescriptor::new();
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
    device.create_sampler_with_descriptor(&mapped_desc)
}

pub(crate) fn create_command_encoder(
    device: &DeviceId,
    _desc: &CommandEncoderDescriptor,
) -> CommandEncoderId {
    let mapped_desc = web_sys::GpuCommandEncoderDescriptor::new();
    device.create_command_encoder_with_descriptor(&mapped_desc)
}

pub(crate) fn command_encoder_copy_buffer_to_buffer(
    command_encoder: &CommandEncoderId,
    source: &crate::Buffer,
    source_offset: wgt::BufferAddress,
    destination: &crate::Buffer,
    destination_offset: wgt::BufferAddress,
    copy_size: wgt::BufferAddress,
) {
    command_encoder.copy_buffer_to_buffer_with_f64_and_f64_and_f64(
        &source.id,
        source_offset as f64,
        &destination.id,
        destination_offset as f64,
        copy_size as f64,
    );
}

pub(crate) fn command_encoder_copy_buffer_to_texture(
    command_encoder: &CommandEncoderId,
    source: crate::BufferCopyView,
    destination: crate::TextureCopyView,
    copy_size: wgt::Extent3d,
) {
    command_encoder.copy_buffer_to_texture_with_gpu_extent_3d_dict(
        &map_buffer_copy_view(source),
        &map_texture_copy_view(destination),
        &map_extent_3d(copy_size),
    );
}

pub(crate) fn command_encoder_copy_texture_to_buffer(
    command_encoder: &CommandEncoderId,
    source: crate::TextureCopyView,
    destination: crate::BufferCopyView,
    copy_size: wgt::Extent3d,
) {
    command_encoder.copy_texture_to_buffer_with_gpu_extent_3d_dict(
        &map_texture_copy_view(source),
        &map_buffer_copy_view(destination),
        &map_extent_3d(copy_size),
    );
}

pub(crate) fn begin_compute_pass(command_encoder: &CommandEncoderId) -> ComputePassId {
    let mapped_desc = web_sys::GpuComputePassDescriptor::new();
    command_encoder.begin_compute_pass_with_descriptor(&mapped_desc)
}

pub(crate) fn compute_pass_set_pipeline(
    compute_pass: &ComputePassId,
    pipeline: &ComputePipelineId,
) {
    compute_pass.set_pipeline(&pipeline);
}

pub(crate) fn compute_pass_set_bind_group<'a>(
    compute_pass: &ComputePassId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[wgt::DynamicOffset],
) {
    compute_pass.set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
        index,
        bind_group,
        // TODO: `offsets` currently requires `&mut` so we have to clone it
        // here, but this should be fixed upstream in web-sys in the future
        &mut offsets.to_vec(),
        0f64,
        offsets.len() as u32,
    );
}

pub(crate) fn compute_pass_dispatch(compute_pass: &ComputePassId, x: u32, y: u32, z: u32) {
    compute_pass.dispatch_with_y_and_z(x, y, z);
}

pub(crate) fn compute_pass_dispatch_indirect(
    compute_pass: &ComputePassId,
    indirect_buffer: &BufferId,
    indirect_offset: wgt::BufferAddress,
) {
    compute_pass.dispatch_indirect_with_f64(indirect_buffer, indirect_offset as f64);
}

pub(crate) fn compute_pass_end_pass(compute_pass: &ComputePassId) {
    compute_pass.end_pass();
}

pub(crate) fn command_encoder_finish(command_encoder: &CommandEncoderId) -> CommandBufferId {
    let mapped_desc = web_sys::GpuCommandBufferDescriptor::new();
    command_encoder.finish_with_descriptor(&mapped_desc)
}

pub(crate) fn queue_submit(queue: &QueueId, command_buffers: &[crate::CommandBuffer]) {
    let temp_command_buffers = command_buffers
        .iter()
        .map(|cb| &cb.id)
        .collect::<js_sys::Array>();

    queue.submit(&temp_command_buffers);
}

pub(crate) async fn buffer_map_read(
    buffer: &crate::Buffer,
    _start: wgt::BufferAddress,
    _size: wgt::BufferAddress,
) -> Result<crate::BufferReadMapping, crate::BufferAsyncErr> {
    let array_buffer_promise = buffer.id.map_read_async();
    let array_buffer: js_sys::ArrayBuffer =
        wasm_bindgen_futures::JsFuture::from(array_buffer_promise)
            .await
            .expect("Unable to map buffer")
            .into();
    let view = js_sys::Uint8Array::new(&array_buffer);
    Ok(crate::BufferReadMapping {
        detail: BufferReadMappingDetail {
            buffer_id: buffer.id.clone(),
            mapped: view.to_vec(),
        },
    })
}

pub(crate) struct BufferReadMappingDetail {
    pub(crate) buffer_id: BufferId,
    mapped: Vec<u8>,
}

impl BufferReadMappingDetail {
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.mapped[..]
    }
}

pub(crate) fn device_create_surface<W: raw_window_handle::HasRawWindowHandle>(
    window: &W,
) -> SurfaceId {
    let handle = window.raw_window_handle();
    let canvas_attribute = match handle {
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
    context.into()
}

pub(crate) fn device_create_swap_chain(
    device: &DeviceId,
    surface: &SurfaceId,
    desc: &wgt::SwapChainDescriptor,
) -> SwapChainId {
    let mut mapped = web_sys::GpuSwapChainDescriptor::new(device, map_texture_format(desc.format));
    mapped.usage(desc.usage.bits());
    surface.configure_swap_chain(&mapped)
}

pub(crate) fn swap_chain_get_next_texture(swap_chain: &SwapChainId) -> Option<TextureViewId> {
    // TODO: Should we pass a descriptor here?
    // Or is the default view always correct?
    Some(swap_chain.get_current_texture().create_view())
}

fn map_store_op(op: wgt::StoreOp) -> web_sys::GpuStoreOp {
    match op {
        wgt::StoreOp::Clear => web_sys::GpuStoreOp::Clear,
        wgt::StoreOp::Store => web_sys::GpuStoreOp::Store,
    }
}

pub(crate) fn command_encoder_begin_render_pass<'a>(
    command_encoder: &CommandEncoderId,
    desc: &crate::RenderPassDescriptor<'a, '_>,
) -> RenderPassEncoderId {
    let mapped_color_attachments = desc
        .color_attachments
        .iter()
        .map(|ca| {
            let mut mapped_color_attachment = web_sys::GpuRenderPassColorAttachmentDescriptor::new(
                &ca.attachment.id,
                &match ca.load_op {
                    wgt::LoadOp::Clear => {
                        let color = ca.clear_color;
                        let mapped_color =
                            web_sys::GpuColorDict::new(color.a, color.b, color.g, color.r);
                        wasm_bindgen::JsValue::from(mapped_color)
                    }
                    wgt::LoadOp::Load => wasm_bindgen::JsValue::from(web_sys::GpuLoadOp::Load),
                },
            );

            if let Some(rt) = ca.resolve_target {
                mapped_color_attachment.resolve_target(&rt.id);
            }

            mapped_color_attachment.store_op(map_store_op(ca.store_op));

            mapped_color_attachment
        })
        .collect::<js_sys::Array>();

    let mut mapped = web_sys::GpuRenderPassDescriptor::new(&mapped_color_attachments);

    if let Some(dsa) = &desc.depth_stencil_attachment {
        let mapped_depth_stencil_attachment =
            web_sys::GpuRenderPassDepthStencilAttachmentDescriptor::new(
                &dsa.attachment.id,
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

        mapped.depth_stencil_attachment(&mapped_depth_stencil_attachment);
    }

    command_encoder.begin_render_pass(&mapped)
}

pub(crate) fn render_pass_set_pipeline(
    render_pass: &RenderPassEncoderId,
    pipeline: &RenderPipelineId,
) {
    render_pass.set_pipeline(&pipeline);
}

pub(crate) fn render_pass_set_bind_group(
    render_pass: &RenderPassEncoderId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[wgt::DynamicOffset],
) {
    render_pass.set_bind_group_with_u32_array_and_f64_and_dynamic_offsets_data_length(
        index,
        bind_group,
        // TODO: `offsets` currently requires `&mut` so we have to clone it
        // here, but this should be fixed upstream in web-sys in the future
        &mut offsets.to_vec(),
        0f64,
        offsets.len() as u32,
    );
}

pub(crate) fn render_pass_set_index_buffer<'a>(
    render_pass: &RenderPassEncoderId,
    buffer: &'a crate::Buffer,
    offset: wgt::BufferAddress,
    _size: wgt::BufferAddress,
) {
    render_pass.set_index_buffer_with_f64(
        &buffer.id,
        offset as f64,
        // TODO: size,
    );
}

pub(crate) fn render_pass_set_vertex_buffer<'a>(
    render_pass: &RenderPassEncoderId,
    slot: u32,
    buffer: &'a crate::Buffer,
    offset: wgt::BufferAddress,
    _size: wgt::BufferAddress,
) {
    render_pass.set_vertex_buffer_with_f64(
        slot,
        &buffer.id,
        offset as f64,
        // TODO: size,
    );
}

pub(crate) fn render_pass_draw(
    render_pass: &RenderPassEncoderId,
    vertices: Range<u32>,
    instances: Range<u32>,
) {
    render_pass.draw_with_instance_count_and_first_vertex_and_first_instance(
        vertices.end - vertices.start,
        instances.end - instances.start,
        vertices.start,
        instances.start,
    )
}

pub(crate) fn render_pass_draw_indexed(
    render_pass: &RenderPassEncoderId,
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32>,
) {
    render_pass
        .draw_indexed_with_instance_count_and_first_index_and_base_vertex_and_first_instance(
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        );
}

pub(crate) fn render_pass_end_pass(render_pass: &RenderPassEncoderId) {
    render_pass.end_pass();
}

pub(crate) fn texture_create_view(
    texture: &TextureId,
    desc: Option<&TextureViewDescriptor>,
) -> TextureViewId {
    match desc {
        Some(d) => {
            let mut mapped_desc = web_sys::GpuTextureViewDescriptor::new();
            mapped_desc.array_layer_count(d.array_layer_count);
            mapped_desc.aspect(map_texture_aspect(d.aspect));
            mapped_desc.base_array_layer(d.base_array_layer);
            mapped_desc.base_mip_level(d.base_mip_level);
            mapped_desc.dimension(map_texture_view_dimension(d.dimension));
            mapped_desc.format(map_texture_format(d.format));
            mapped_desc.mip_level_count(d.level_count);
            texture.create_view_with_descriptor(&mapped_desc)
        }
        None => texture.create_view(),
    }
}

pub(crate) fn swap_chain_present(_swap_chain: &SwapChainId) {
    // Swapchain is presented automatically
}

pub(crate) fn device_poll(_device: &DeviceId, _maintain: crate::Maintain) {
    // Device is polled automatically
}
