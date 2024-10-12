use std::{any::Any, fmt::Debug, future::Future, ops::Range, pin::Pin, sync::Arc};

use wgt::{
    strict_assert, AdapterInfo, BufferAddress, BufferSize, Color, DeviceLostReason,
    DownlevelCapabilities, DynamicOffset, Extent3d, Features, ImageDataLayout,
    ImageSubresourceRange, IndexFormat, Limits, ShaderStages, SurfaceStatus, TextureFormat,
    TextureFormatFeatures, WasmNotSend, WasmNotSendSync,
};

use crate::{
    AnyWasmNotSendSync, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferAsyncError,
    BufferDescriptor, CommandEncoderDescriptor, CompilationInfo, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Error, ErrorFilter, ImageCopyBuffer,
    ImageCopyTexture, Maintain, MaintainResult, MapMode, PipelineCacheDescriptor,
    PipelineLayoutDescriptor, QuerySetDescriptor, RenderBundleDescriptor,
    RenderBundleEncoderDescriptor, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, RequestDeviceError, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderModuleDescriptorSpirV, SurfaceTargetUnsafe, TextureDescriptor, TextureViewDescriptor,
    UncapturedErrorHandler,
};
/// Meta trait for an data associated with an id tracked by a context.
///
/// There is no need to manually implement this trait since there is a blanket implementation for this trait.
#[cfg_attr(target_os = "emscripten", allow(dead_code))]
pub trait ContextData: Debug + WasmNotSendSync + 'static {}
impl<T: Debug + WasmNotSendSync + 'static> ContextData for T {}

pub trait Context: Debug + WasmNotSendSync + Sized {
    type AdapterData: ContextData;
    type DeviceData: ContextData;
    type QueueData: ContextData;
    type ShaderModuleData: ContextData;
    type BindGroupLayoutData: ContextData;
    type BindGroupData: ContextData;
    type TextureViewData: ContextData;
    type SamplerData: ContextData;
    type BufferData: ContextData;
    type TextureData: ContextData;
    type QuerySetData: ContextData;
    type PipelineLayoutData: ContextData;
    type RenderPipelineData: ContextData;
    type ComputePipelineData: ContextData;
    type PipelineCacheData: ContextData;
    type CommandEncoderData: ContextData;
    type ComputePassData: ContextData;
    type RenderPassData: ContextData;
    type CommandBufferData: ContextData;
    type RenderBundleEncoderData: ContextData;
    type RenderBundleData: ContextData;
    type SurfaceData: ContextData;

    type SurfaceOutputDetail: WasmNotSendSync + 'static;
    type SubmissionIndexData: ContextData + Copy;

    type RequestAdapterFuture: Future<Output = Option<Self::AdapterData>> + WasmNotSend + 'static;
    type RequestDeviceFuture: Future<Output = Result<(Self::DeviceData, Self::QueueData), RequestDeviceError>>
        + WasmNotSend
        + 'static;
    type PopErrorScopeFuture: Future<Output = Option<Error>> + WasmNotSend + 'static;

    type CompilationInfoFuture: Future<Output = CompilationInfo> + WasmNotSend + 'static;

    #[cfg(not(target_os = "emscripten"))]
    fn init(instance_desc: wgt::InstanceDescriptor) -> Self;
    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Self::SurfaceData, crate::CreateSurfaceError>;
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture;
    fn adapter_request_device(
        &self,
        adapter_data: &Self::AdapterData,
        desc: &DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture;
    fn instance_poll_all_devices(&self, force_wait: bool) -> bool;
    fn adapter_is_surface_supported(
        &self,
        adapter_data: &Self::AdapterData,
        surface_data: &Self::SurfaceData,
    ) -> bool;
    fn adapter_features(&self, adapter_data: &Self::AdapterData) -> Features;
    fn adapter_limits(&self, adapter_data: &Self::AdapterData) -> Limits;
    fn adapter_downlevel_capabilities(
        &self,
        adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities;
    fn adapter_get_info(&self, adapter_data: &Self::AdapterData) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter_data: &Self::AdapterData,
        format: TextureFormat,
    ) -> TextureFormatFeatures;
    fn adapter_get_presentation_timestamp(
        &self,
        adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp;

    fn surface_get_capabilities(
        &self,
        surface_data: &Self::SurfaceData,
        adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities;
    fn surface_configure(
        &self,
        surface_data: &Self::SurfaceData,
        device_data: &Self::DeviceData,
        config: &crate::SurfaceConfiguration,
    );
    #[allow(clippy::type_complexity)]
    fn surface_get_current_texture(
        &self,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureData>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    );
    fn surface_present(&self, detail: &Self::SurfaceOutputDetail);
    fn surface_texture_discard(&self, detail: &Self::SurfaceOutputDetail);

    fn device_features(&self, device_data: &Self::DeviceData) -> Features;
    fn device_limits(&self, device_data: &Self::DeviceData) -> Limits;
    fn device_create_shader_module(
        &self,
        device_data: &Self::DeviceData,
        desc: ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Self::ShaderModuleData;
    unsafe fn device_create_shader_module_spirv(
        &self,
        device_data: &Self::DeviceData,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> Self::ShaderModuleData;
    fn device_create_bind_group_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> Self::BindGroupLayoutData;
    fn device_create_bind_group(
        &self,
        device_data: &Self::DeviceData,
        desc: &BindGroupDescriptor<'_>,
    ) -> Self::BindGroupData;
    fn device_create_pipeline_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &PipelineLayoutDescriptor<'_>,
    ) -> Self::PipelineLayoutData;
    fn device_create_render_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Self::RenderPipelineData;
    fn device_create_compute_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Self::ComputePipelineData;
    unsafe fn device_create_pipeline_cache(
        &self,
        device_data: &Self::DeviceData,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> Self::PipelineCacheData;
    fn device_create_buffer(
        &self,
        device_data: &Self::DeviceData,
        desc: &BufferDescriptor<'_>,
    ) -> Self::BufferData;
    fn device_create_texture(
        &self,
        device_data: &Self::DeviceData,
        desc: &TextureDescriptor<'_>,
    ) -> Self::TextureData;
    fn device_create_sampler(
        &self,
        device_data: &Self::DeviceData,
        desc: &SamplerDescriptor<'_>,
    ) -> Self::SamplerData;
    fn device_create_query_set(
        &self,
        device_data: &Self::DeviceData,
        desc: &QuerySetDescriptor<'_>,
    ) -> Self::QuerySetData;
    fn device_create_command_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &CommandEncoderDescriptor<'_>,
    ) -> Self::CommandEncoderData;
    fn device_create_render_bundle_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> Self::RenderBundleEncoderData;
    fn device_drop(&self, device_data: &Self::DeviceData);
    fn device_set_device_lost_callback(
        &self,
        device_data: &Self::DeviceData,
        device_lost_callback: DeviceLostCallback,
    );
    fn device_destroy(&self, device_data: &Self::DeviceData);
    fn queue_drop(&self, queue_data: &Self::QueueData);
    fn device_poll(&self, device_data: &Self::DeviceData, maintain: Maintain) -> MaintainResult;
    fn device_on_uncaptured_error(
        &self,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    );
    fn device_push_error_scope(&self, device_data: &Self::DeviceData, filter: ErrorFilter);
    fn device_pop_error_scope(&self, device_data: &Self::DeviceData) -> Self::PopErrorScopeFuture;

    fn buffer_map_async(
        &self,
        buffer_data: &Self::BufferData,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: BufferMapCallback,
    );
    fn buffer_get_mapped_range(
        &self,
        buffer_data: &Self::BufferData,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange>;
    fn buffer_unmap(&self, buffer_data: &Self::BufferData);
    fn shader_get_compilation_info(
        &self,
        shader_data: &Self::ShaderModuleData,
    ) -> Self::CompilationInfoFuture;
    fn texture_create_view(
        &self,
        texture_data: &Self::TextureData,
        desc: &TextureViewDescriptor<'_>,
    ) -> Self::TextureViewData;

    fn surface_drop(&self, surface_data: &Self::SurfaceData);
    fn adapter_drop(&self, adapter_data: &Self::AdapterData);
    fn buffer_destroy(&self, buffer_data: &Self::BufferData);
    fn buffer_drop(&self, buffer_data: &Self::BufferData);
    fn texture_destroy(&self, texture_data: &Self::TextureData);
    fn texture_drop(&self, texture_data: &Self::TextureData);
    fn texture_view_drop(&self, texture_view_data: &Self::TextureViewData);
    fn sampler_drop(&self, sampler_data: &Self::SamplerData);
    fn query_set_drop(&self, query_set_data: &Self::QuerySetData);
    fn bind_group_drop(&self, bind_group_data: &Self::BindGroupData);
    fn bind_group_layout_drop(&self, bind_group_layout_data: &Self::BindGroupLayoutData);
    fn pipeline_layout_drop(&self, pipeline_layout_data: &Self::PipelineLayoutData);
    fn shader_module_drop(&self, shader_module_data: &Self::ShaderModuleData);
    fn command_encoder_drop(&self, command_encoder_data: &Self::CommandEncoderData);
    fn command_buffer_drop(&self, command_buffer_data: &Self::CommandBufferData);
    fn render_bundle_drop(&self, render_bundle_data: &Self::RenderBundleData);
    fn compute_pipeline_drop(&self, pipeline_data: &Self::ComputePipelineData);
    fn render_pipeline_drop(&self, pipeline_data: &Self::RenderPipelineData);
    fn pipeline_cache_drop(&self, cache_data: &Self::PipelineCacheData);

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData;
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData;

    #[allow(clippy::too_many_arguments)]
    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source_data: &Self::BufferData,
        source_offset: BufferAddress,
        destination_data: &Self::BufferData,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &ComputePassDescriptor<'_>,
    ) -> Self::ComputePassData;
    fn command_encoder_begin_render_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &RenderPassDescriptor<'_>,
    ) -> Self::RenderPassData;
    fn command_encoder_finish(
        &self,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> Self::CommandBufferData;

    fn command_encoder_clear_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        texture_data: &Self::TextureData,
        subresource_range: &ImageSubresourceRange,
    );
    fn command_encoder_clear_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    );

    fn command_encoder_insert_debug_marker(
        &self,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    );
    fn command_encoder_push_debug_group(
        &self,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    );
    fn command_encoder_pop_debug_group(&self, encoder_data: &Self::CommandEncoderData);

    fn command_encoder_write_timestamp(
        &self,
        encoder_data: &Self::CommandEncoderData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn command_encoder_resolve_query_set(
        &self,
        encoder_data: &Self::CommandEncoderData,
        query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        destination_data: &Self::BufferData,
        destination_offset: BufferAddress,
    );

    fn render_bundle_encoder_finish(
        &self,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &RenderBundleDescriptor<'_>,
    ) -> Self::RenderBundleData;
    fn queue_write_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_validate_write_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()>;
    fn queue_create_staging_buffer(
        &self,
        queue_data: &Self::QueueData,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>>;
    fn queue_write_staging_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    );
    fn queue_write_texture(
        &self,
        queue_data: &Self::QueueData,
        texture: ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    #[cfg(any(webgl, webgpu))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    );
    fn queue_submit<I: Iterator<Item = Self::CommandBufferData>>(
        &self,
        queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> Self::SubmissionIndexData;
    fn queue_get_timestamp_period(&self, queue_data: &Self::QueueData) -> f32;
    fn queue_on_submitted_work_done(
        &self,
        queue_data: &Self::QueueData,
        callback: SubmittedWorkDoneCallback,
    );

    fn device_start_capture(&self, device_data: &Self::DeviceData);
    fn device_stop_capture(&self, device_data: &Self::DeviceData);

    fn device_get_internal_counters(
        &self,
        _device_data: &Self::DeviceData,
    ) -> wgt::InternalCounters;

    fn device_generate_allocator_report(
        &self,
        _device_data: &Self::DeviceData,
    ) -> Option<wgt::AllocatorReport>;

    fn pipeline_cache_get_data(&self, cache_data: &Self::PipelineCacheData) -> Option<Vec<u8>>;

    fn compute_pass_set_pipeline(
        &self,
        pass_data: &mut Self::ComputePassData,
        pipeline_data: &Self::ComputePipelineData,
    );
    fn compute_pass_set_bind_group(
        &self,
        pass_data: &mut Self::ComputePassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[DynamicOffset],
    );
    fn compute_pass_set_push_constants(
        &self,
        pass_data: &mut Self::ComputePassData,
        offset: u32,
        data: &[u8],
    );
    fn compute_pass_insert_debug_marker(&self, pass_data: &mut Self::ComputePassData, label: &str);
    fn compute_pass_push_debug_group(
        &self,
        pass_data: &mut Self::ComputePassData,
        group_label: &str,
    );
    fn compute_pass_pop_debug_group(&self, pass_data: &mut Self::ComputePassData);
    fn compute_pass_write_timestamp(
        &self,
        pass_data: &mut Self::ComputePassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut Self::ComputePassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn compute_pass_end_pipeline_statistics_query(&self, pass_data: &mut Self::ComputePassData);
    fn compute_pass_dispatch_workgroups(
        &self,
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    );
    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_data: &mut Self::ComputePassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn compute_pass_end(&self, pass_data: &mut Self::ComputePassData);

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        pipeline_data: &Self::RenderPipelineData,
    );
    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        buffer_data: &Self::BufferData,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_bundle_encoder_draw(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );

    fn render_pass_set_pipeline(
        &self,
        pass_data: &mut Self::RenderPassData,
        pipeline_data: &Self::RenderPipelineData,
    );
    fn render_pass_set_bind_group(
        &self,
        pass_data: &mut Self::RenderPassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_index_buffer(
        &self,
        pass_data: &mut Self::RenderPassData,
        buffer_data: &Self::BufferData,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_vertex_buffer(
        &self,
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_push_constants(
        &self,
        pass_data: &mut Self::RenderPassData,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_pass_draw(
        &self,
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_pass_draw_indexed(
        &self,
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_pass_draw_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_pass_draw_indexed_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_pass_multi_draw_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indirect_count(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_set_blend_constant(&self, pass_data: &mut Self::RenderPassData, color: Color);
    fn render_pass_set_scissor_rect(
        &self,
        pass_data: &mut Self::RenderPassData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_viewport(
        &self,
        pass_data: &mut Self::RenderPassData,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    fn render_pass_set_stencil_reference(
        &self,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    );
    fn render_pass_insert_debug_marker(&self, pass_data: &mut Self::RenderPassData, label: &str);
    fn render_pass_push_debug_group(&self, pass_data: &mut Self::RenderPassData, group_label: &str);
    fn render_pass_pop_debug_group(&self, pass_data: &mut Self::RenderPassData);
    fn render_pass_write_timestamp(
        &self,
        pass_data: &mut Self::RenderPassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn render_pass_begin_occlusion_query(
        &self,
        pass_data: &mut Self::RenderPassData,
        query_index: u32,
    );
    fn render_pass_end_occlusion_query(&self, pass_data: &mut Self::RenderPassData);
    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut Self::RenderPassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn render_pass_end_pipeline_statistics_query(&self, pass_data: &mut Self::RenderPassData);
    fn render_pass_execute_bundles(
        &self,
        pass_data: &mut Self::RenderPassData,
        render_bundles: &mut dyn Iterator<Item = &Self::RenderBundleData>,
    );
    fn render_pass_end(&self, pass_data: &mut Self::RenderPassData);
}

pub(crate) fn downcast_ref<T: Debug + WasmNotSendSync + 'static>(data: &crate::Data) -> &T {
    strict_assert!(data.is::<T>());
    // Copied from std.
    unsafe { &*(data as *const dyn Any as *const T) }
}

fn downcast_mut<T: Debug + WasmNotSendSync + 'static>(data: &mut crate::Data) -> &mut T {
    strict_assert!(data.is::<T>());
    // Copied from std.
    unsafe { &mut *(data as *mut dyn Any as *mut T) }
}

pub(crate) struct DeviceRequest {
    pub device_data: Box<crate::Data>,
    pub queue_data: Box<crate::Data>,
}

#[cfg(send_sync)]
pub type BufferMapCallback = Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>;
#[cfg(not(send_sync))]
pub type BufferMapCallback = Box<dyn FnOnce(Result<(), BufferAsyncError>) + 'static>;

#[cfg(send_sync)]
pub(crate) type AdapterRequestDeviceFuture =
    Box<dyn Future<Output = Result<DeviceRequest, RequestDeviceError>> + Send>;
#[cfg(not(send_sync))]
pub(crate) type AdapterRequestDeviceFuture =
    Box<dyn Future<Output = Result<DeviceRequest, RequestDeviceError>>>;

#[cfg(send_sync)]
pub type InstanceRequestAdapterFuture = Box<dyn Future<Output = Option<Box<crate::Data>>> + Send>;
#[cfg(not(send_sync))]
pub type InstanceRequestAdapterFuture = Box<dyn Future<Output = Option<Box<crate::Data>>>>;

#[cfg(send_sync)]
pub type DevicePopErrorFuture = Box<dyn Future<Output = Option<Error>> + Send>;
#[cfg(not(send_sync))]
pub type DevicePopErrorFuture = Box<dyn Future<Output = Option<Error>>>;

#[cfg(send_sync)]
pub type ShaderCompilationInfoFuture = Box<dyn Future<Output = CompilationInfo> + Send>;
#[cfg(not(send_sync))]
pub type ShaderCompilationInfoFuture = Box<dyn Future<Output = CompilationInfo>>;

#[cfg(send_sync)]
pub type SubmittedWorkDoneCallback = Box<dyn FnOnce() + Send + 'static>;
#[cfg(not(send_sync))]
pub type SubmittedWorkDoneCallback = Box<dyn FnOnce() + 'static>;
#[cfg(send_sync)]
pub type DeviceLostCallback = Box<dyn Fn(DeviceLostReason, String) + Send + 'static>;
#[cfg(not(send_sync))]
pub type DeviceLostCallback = Box<dyn Fn(DeviceLostReason, String) + 'static>;

/// An object safe variant of [`Context`] implemented by all types that implement [`Context`].
pub(crate) trait DynContext: Debug + WasmNotSendSync {
    #[cfg(not(target_os = "emscripten"))]
    fn as_any(&self) -> &dyn Any;

    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Box<crate::Data>, crate::CreateSurfaceError>;
    #[allow(clippy::type_complexity)]
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> Pin<InstanceRequestAdapterFuture>;
    fn adapter_request_device(
        &self,
        adapter_data: &crate::Data,
        desc: &DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<AdapterRequestDeviceFuture>;

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool;
    fn adapter_is_surface_supported(
        &self,
        adapter_data: &crate::Data,
        surface_data: &crate::Data,
    ) -> bool;
    fn adapter_features(&self, adapter_data: &crate::Data) -> Features;
    fn adapter_limits(&self, adapter_data: &crate::Data) -> Limits;
    fn adapter_downlevel_capabilities(&self, adapter_data: &crate::Data) -> DownlevelCapabilities;
    fn adapter_get_info(&self, adapter_data: &crate::Data) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter_data: &crate::Data,
        format: TextureFormat,
    ) -> TextureFormatFeatures;
    fn adapter_get_presentation_timestamp(
        &self,
        adapter_data: &crate::Data,
    ) -> wgt::PresentationTimestamp;

    fn surface_get_capabilities(
        &self,
        surface_data: &crate::Data,
        adapter_data: &crate::Data,
    ) -> wgt::SurfaceCapabilities;
    fn surface_configure(
        &self,
        surface_data: &crate::Data,
        device_data: &crate::Data,
        config: &crate::SurfaceConfiguration,
    );
    fn surface_get_current_texture(
        &self,
        surface_data: &crate::Data,
    ) -> (
        Option<Box<crate::Data>>,
        SurfaceStatus,
        Box<dyn AnyWasmNotSendSync>,
    );
    fn surface_present(&self, detail: &dyn AnyWasmNotSendSync);
    fn surface_texture_discard(&self, detail: &dyn AnyWasmNotSendSync);

    fn device_features(&self, device_data: &crate::Data) -> Features;
    fn device_limits(&self, device_data: &crate::Data) -> Limits;
    fn device_create_shader_module(
        &self,
        device_data: &crate::Data,
        desc: ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Box<crate::Data>;
    unsafe fn device_create_shader_module_spirv(
        &self,
        device_data: &crate::Data,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> Box<crate::Data>;
    fn device_create_bind_group_layout(
        &self,
        device_data: &crate::Data,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_bind_group(
        &self,
        device_data: &crate::Data,
        desc: &BindGroupDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_pipeline_layout(
        &self,
        device_data: &crate::Data,
        desc: &PipelineLayoutDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_render_pipeline(
        &self,
        device_data: &crate::Data,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_compute_pipeline(
        &self,
        device_data: &crate::Data,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Box<crate::Data>;
    unsafe fn device_create_pipeline_cache(
        &self,
        device_data: &crate::Data,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_buffer(
        &self,
        device_data: &crate::Data,
        desc: &BufferDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_texture(
        &self,
        device_data: &crate::Data,
        desc: &TextureDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_sampler(
        &self,
        device_data: &crate::Data,
        desc: &SamplerDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_query_set(
        &self,
        device_data: &crate::Data,
        desc: &QuerySetDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_command_encoder(
        &self,
        device_data: &crate::Data,
        desc: &CommandEncoderDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_create_render_bundle_encoder(
        &self,
        device_data: &crate::Data,
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn device_drop(&self, device_data: &crate::Data);
    fn device_set_device_lost_callback(
        &self,
        device_data: &crate::Data,
        device_lost_callback: DeviceLostCallback,
    );
    fn device_destroy(&self, device_data: &crate::Data);
    fn queue_drop(&self, queue_data: &crate::Data);
    fn device_poll(&self, device_data: &crate::Data, maintain: Maintain) -> MaintainResult;
    fn device_on_uncaptured_error(
        &self,
        device_data: &crate::Data,
        handler: Box<dyn UncapturedErrorHandler>,
    );
    fn device_push_error_scope(&self, device_data: &crate::Data, filter: ErrorFilter);
    fn device_pop_error_scope(&self, device_data: &crate::Data) -> Pin<DevicePopErrorFuture>;
    fn buffer_map_async(
        &self,
        buffer_data: &crate::Data,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: BufferMapCallback,
    );
    fn buffer_get_mapped_range(
        &self,
        buffer_data: &crate::Data,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange>;
    fn buffer_unmap(&self, buffer_data: &crate::Data);
    fn shader_get_compilation_info(
        &self,
        shader_data: &crate::Data,
    ) -> Pin<ShaderCompilationInfoFuture>;
    fn texture_create_view(
        &self,
        texture_data: &crate::Data,
        desc: &TextureViewDescriptor<'_>,
    ) -> Box<crate::Data>;

    fn surface_drop(&self, surface_data: &crate::Data);
    fn adapter_drop(&self, adapter_data: &crate::Data);
    fn buffer_destroy(&self, buffer_data: &crate::Data);
    fn buffer_drop(&self, buffer_data: &crate::Data);
    fn texture_destroy(&self, buffer_data: &crate::Data);
    fn texture_drop(&self, texture_data: &crate::Data);
    fn texture_view_drop(&self, texture_view_data: &crate::Data);
    fn sampler_drop(&self, sampler_data: &crate::Data);
    fn query_set_drop(&self, query_set_data: &crate::Data);
    fn bind_group_drop(&self, bind_group_data: &crate::Data);
    fn bind_group_layout_drop(&self, bind_group_layout_data: &crate::Data);
    fn pipeline_layout_drop(&self, pipeline_layout_data: &crate::Data);
    fn shader_module_drop(&self, shader_module_data: &crate::Data);
    fn command_encoder_drop(&self, command_encoder_data: &crate::Data);
    fn command_buffer_drop(&self, command_buffer_data: &crate::Data);
    fn render_bundle_drop(&self, render_bundle_data: &crate::Data);
    fn compute_pipeline_drop(&self, pipeline_data: &crate::Data);
    fn render_pipeline_drop(&self, pipeline_data: &crate::Data);
    fn pipeline_cache_drop(&self, _cache_data: &crate::Data);

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> Box<crate::Data>;
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> Box<crate::Data>;

    #[allow(clippy::too_many_arguments)]
    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder_data: &crate::Data,
        source_data: &crate::Data,
        source_offset: BufferAddress,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder_data: &crate::Data,
        desc: &ComputePassDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn command_encoder_begin_render_pass(
        &self,
        encoder_data: &crate::Data,
        desc: &RenderPassDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn command_encoder_finish(&self, encoder_data: &mut crate::Data) -> Box<crate::Data>;

    fn command_encoder_clear_texture(
        &self,
        encoder_data: &crate::Data,
        texture_data: &crate::Data,
        subresource_range: &ImageSubresourceRange,
    );
    fn command_encoder_clear_buffer(
        &self,
        encoder_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    );

    fn command_encoder_insert_debug_marker(&self, encoder_data: &crate::Data, label: &str);
    fn command_encoder_push_debug_group(&self, encoder_data: &crate::Data, label: &str);
    fn command_encoder_pop_debug_group(&self, encoder_data: &crate::Data);

    fn command_encoder_write_timestamp(
        &self,
        encoder_data: &crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn command_encoder_resolve_query_set(
        &self,
        encoder_data: &crate::Data,
        query_set_data: &crate::Data,
        first_query: u32,
        query_count: u32,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
    );

    fn render_bundle_encoder_finish(
        &self,
        encoder_data: Box<crate::Data>,
        desc: &RenderBundleDescriptor<'_>,
    ) -> Box<crate::Data>;
    fn queue_write_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_validate_write_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()>;
    fn queue_create_staging_buffer(
        &self,
        queue_data: &crate::Data,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>>;
    fn queue_write_staging_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    );
    fn queue_write_texture(
        &self,
        queue_data: &crate::Data,
        texture: ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    #[cfg(any(webgpu, webgl))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue_data: &crate::Data,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    );
    fn queue_submit(
        &self,
        queue_data: &crate::Data,
        command_buffers: &mut dyn Iterator<Item = Box<crate::Data>>,
    ) -> Arc<crate::Data>;
    fn queue_get_timestamp_period(&self, queue_data: &crate::Data) -> f32;
    fn queue_on_submitted_work_done(
        &self,
        queue_data: &crate::Data,
        callback: SubmittedWorkDoneCallback,
    );

    fn device_start_capture(&self, data: &crate::Data);
    fn device_stop_capture(&self, data: &crate::Data);

    fn device_get_internal_counters(&self, device_data: &crate::Data) -> wgt::InternalCounters;

    fn generate_allocator_report(&self, device_data: &crate::Data) -> Option<wgt::AllocatorReport>;

    fn pipeline_cache_get_data(&self, cache_data: &crate::Data) -> Option<Vec<u8>>;

    fn compute_pass_set_pipeline(&self, pass_data: &mut crate::Data, pipeline_data: &crate::Data);
    fn compute_pass_set_bind_group(
        &self,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    );
    fn compute_pass_set_push_constants(
        &self,
        pass_data: &mut crate::Data,
        offset: u32,
        data: &[u8],
    );
    fn compute_pass_insert_debug_marker(&self, pass_data: &mut crate::Data, label: &str);
    fn compute_pass_push_debug_group(&self, pass_data: &mut crate::Data, group_label: &str);
    fn compute_pass_pop_debug_group(&self, pass_data: &mut crate::Data);
    fn compute_pass_write_timestamp(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn compute_pass_end_pipeline_statistics_query(&self, pass_data: &mut crate::Data);
    fn compute_pass_dispatch_workgroups(&self, pass_data: &mut crate::Data, x: u32, y: u32, z: u32);
    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn compute_pass_end(&self, pass_data: &mut crate::Data);

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder_data: &mut crate::Data,
        pipeline_data: &crate::Data,
    );
    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder_data: &mut crate::Data,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder_data: &mut crate::Data,
        slot: u32,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_bundle_encoder_draw(
        &self,
        encoder_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );

    fn render_pass_set_pipeline(&self, pass_data: &mut crate::Data, pipeline_data: &crate::Data);
    fn render_pass_set_bind_group(
        &self,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_index_buffer(
        &self,
        pass_data: &mut crate::Data,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_vertex_buffer(
        &self,
        pass_data: &mut crate::Data,
        slot: u32,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_push_constants(
        &self,
        pass_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_pass_draw(
        &self,
        pass_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_pass_draw_indexed(
        &self,
        pass_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_pass_draw_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_pass_draw_indexed_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_pass_multi_draw_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indirect_count(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        command_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_set_blend_constant(&self, pass_data: &mut crate::Data, color: Color);
    fn render_pass_set_scissor_rect(
        &self,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_viewport(
        &self,
        pass_data: &mut crate::Data,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    fn render_pass_set_stencil_reference(&self, pass_data: &mut crate::Data, reference: u32);
    fn render_pass_insert_debug_marker(&self, pass_data: &mut crate::Data, label: &str);
    fn render_pass_push_debug_group(&self, pass_data: &mut crate::Data, group_label: &str);
    fn render_pass_pop_debug_group(&self, pass_data: &mut crate::Data);
    fn render_pass_write_timestamp(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn render_pass_begin_occlusion_query(&self, pass_data: &mut crate::Data, query_index: u32);
    fn render_pass_end_occlusion_query(&self, pass_data: &mut crate::Data);
    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn render_pass_end_pipeline_statistics_query(&self, pass_data: &mut crate::Data);
    fn render_pass_execute_bundles(
        &self,
        pass_data: &mut crate::Data,
        render_bundles: &mut dyn Iterator<Item = &crate::Data>,
    );
    fn render_pass_end(&self, pass_data: &mut crate::Data);
}

// Blanket impl of DynContext for all types which implement Context.
impl<T> DynContext for T
where
    T: Context + 'static,
{
    #[cfg(not(target_os = "emscripten"))]
    fn as_any(&self) -> &dyn Any {
        self
    }

    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Box<crate::Data>, crate::CreateSurfaceError> {
        let data = unsafe { Context::instance_create_surface(self, target) }?;
        Ok(Box::new(data) as _)
    }

    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> Pin<InstanceRequestAdapterFuture> {
        let future: T::RequestAdapterFuture = Context::instance_request_adapter(self, options);
        Box::pin(async move { future.await.map(|data| Box::new(data) as _) })
    }

    fn adapter_request_device(
        &self,
        adapter_data: &crate::Data,
        desc: &DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<AdapterRequestDeviceFuture> {
        let adapter_data = downcast_ref(adapter_data);
        let future = Context::adapter_request_device(self, adapter_data, desc, trace_dir);

        Box::pin(async move {
            let (device_data, queue_data) = future.await?;
            Ok(DeviceRequest {
                device_data: Box::new(device_data) as _,
                queue_data: Box::new(queue_data) as _,
            })
        })
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        Context::instance_poll_all_devices(self, force_wait)
    }

    fn adapter_is_surface_supported(
        &self,
        adapter_data: &crate::Data,
        surface_data: &crate::Data,
    ) -> bool {
        let adapter_data = downcast_ref(adapter_data);
        let surface_data = downcast_ref(surface_data);
        Context::adapter_is_surface_supported(self, adapter_data, surface_data)
    }

    fn adapter_features(&self, adapter_data: &crate::Data) -> Features {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_features(self, adapter_data)
    }

    fn adapter_limits(&self, adapter_data: &crate::Data) -> Limits {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_limits(self, adapter_data)
    }

    fn adapter_downlevel_capabilities(&self, adapter_data: &crate::Data) -> DownlevelCapabilities {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_downlevel_capabilities(self, adapter_data)
    }

    fn adapter_get_info(&self, adapter_data: &crate::Data) -> AdapterInfo {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_info(self, adapter_data)
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter_data: &crate::Data,
        format: TextureFormat,
    ) -> TextureFormatFeatures {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_texture_format_features(self, adapter_data, format)
    }
    fn adapter_get_presentation_timestamp(
        &self,
        adapter_data: &crate::Data,
    ) -> wgt::PresentationTimestamp {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_presentation_timestamp(self, adapter_data)
    }

    fn surface_get_capabilities(
        &self,
        surface_data: &crate::Data,
        adapter_data: &crate::Data,
    ) -> wgt::SurfaceCapabilities {
        let surface_data = downcast_ref(surface_data);
        let adapter_data = downcast_ref(adapter_data);
        Context::surface_get_capabilities(self, surface_data, adapter_data)
    }

    fn surface_configure(
        &self,
        surface_data: &crate::Data,
        device_data: &crate::Data,
        config: &crate::SurfaceConfiguration,
    ) {
        let surface_data = downcast_ref(surface_data);
        let device_data = downcast_ref(device_data);
        Context::surface_configure(self, surface_data, device_data, config)
    }

    fn surface_get_current_texture(
        &self,
        surface_data: &crate::Data,
    ) -> (
        Option<Box<crate::Data>>,
        SurfaceStatus,
        Box<dyn AnyWasmNotSendSync>,
    ) {
        let surface_data = downcast_ref(surface_data);
        let (texture_data, status, detail) =
            Context::surface_get_current_texture(self, surface_data);
        let detail = Box::new(detail) as Box<dyn AnyWasmNotSendSync>;
        (texture_data.map(|b| Box::new(b) as _), status, detail)
    }

    fn surface_present(&self, detail: &dyn AnyWasmNotSendSync) {
        Context::surface_present(self, detail.downcast_ref().unwrap())
    }

    fn surface_texture_discard(&self, detail: &dyn AnyWasmNotSendSync) {
        Context::surface_texture_discard(self, detail.downcast_ref().unwrap())
    }

    fn device_features(&self, device_data: &crate::Data) -> Features {
        let device_data = downcast_ref(device_data);
        Context::device_features(self, device_data)
    }

    fn device_limits(&self, device_data: &crate::Data) -> Limits {
        let device_data = downcast_ref(device_data);
        Context::device_limits(self, device_data)
    }

    fn device_create_shader_module(
        &self,
        device_data: &crate::Data,
        desc: ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data =
            Context::device_create_shader_module(self, device_data, desc, shader_bound_checks);
        Box::new(data) as _
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device_data: &crate::Data,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = unsafe { Context::device_create_shader_module_spirv(self, device_data, desc) };
        Box::new(data) as _
    }

    fn device_create_bind_group_layout(
        &self,
        device_data: &crate::Data,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_bind_group_layout(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_bind_group(
        &self,
        device_data: &crate::Data,
        desc: &BindGroupDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_bind_group(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_pipeline_layout(
        &self,
        device_data: &crate::Data,
        desc: &PipelineLayoutDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_pipeline_layout(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_render_pipeline(
        &self,
        device_data: &crate::Data,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_render_pipeline(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_compute_pipeline(
        &self,
        device_data: &crate::Data,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_compute_pipeline(self, device_data, desc);
        Box::new(data) as _
    }

    unsafe fn device_create_pipeline_cache(
        &self,
        device_data: &crate::Data,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = unsafe { Context::device_create_pipeline_cache(self, device_data, desc) };
        Box::new(data) as _
    }

    fn device_create_buffer(
        &self,
        device_data: &crate::Data,
        desc: &BufferDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_buffer(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_texture(
        &self,
        device_data: &crate::Data,
        desc: &TextureDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_texture(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_sampler(
        &self,
        device_data: &crate::Data,
        desc: &SamplerDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_sampler(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_query_set(
        &self,
        device_data: &crate::Data,
        desc: &QuerySetDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_query_set(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_command_encoder(
        &self,
        device_data: &crate::Data,
        desc: &CommandEncoderDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_command_encoder(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_create_render_bundle_encoder(
        &self,
        device_data: &crate::Data,
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> Box<crate::Data> {
        let device_data = downcast_ref(device_data);
        let data = Context::device_create_render_bundle_encoder(self, device_data, desc);
        Box::new(data) as _
    }

    fn device_drop(&self, device_data: &crate::Data) {
        let device_data = downcast_ref(device_data);
        Context::device_drop(self, device_data)
    }

    fn device_set_device_lost_callback(
        &self,
        device_data: &crate::Data,
        device_lost_callback: DeviceLostCallback,
    ) {
        let device_data = downcast_ref(device_data);
        Context::device_set_device_lost_callback(self, device_data, device_lost_callback)
    }

    fn device_destroy(&self, device_data: &crate::Data) {
        let device_data = downcast_ref(device_data);
        Context::device_destroy(self, device_data)
    }

    fn queue_drop(&self, queue_data: &crate::Data) {
        let queue_data = downcast_ref(queue_data);
        Context::queue_drop(self, queue_data)
    }

    fn device_poll(&self, device_data: &crate::Data, maintain: Maintain) -> MaintainResult {
        let device_data = downcast_ref(device_data);
        Context::device_poll(self, device_data, maintain)
    }

    fn device_on_uncaptured_error(
        &self,
        device_data: &crate::Data,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let device_data = downcast_ref(device_data);
        Context::device_on_uncaptured_error(self, device_data, handler)
    }

    fn device_push_error_scope(&self, device_data: &crate::Data, filter: ErrorFilter) {
        let device_data = downcast_ref(device_data);
        Context::device_push_error_scope(self, device_data, filter)
    }

    fn device_pop_error_scope(&self, device_data: &crate::Data) -> Pin<DevicePopErrorFuture> {
        let device_data = downcast_ref(device_data);
        Box::pin(Context::device_pop_error_scope(self, device_data))
    }

    fn buffer_map_async(
        &self,
        buffer_data: &crate::Data,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: BufferMapCallback,
    ) {
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_map_async(self, buffer_data, mode, range, callback)
    }

    fn buffer_get_mapped_range(
        &self,
        buffer_data: &crate::Data,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange> {
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_get_mapped_range(self, buffer_data, sub_range)
    }

    fn buffer_unmap(&self, buffer_data: &crate::Data) {
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_unmap(self, buffer_data)
    }

    fn shader_get_compilation_info(
        &self,
        shader_data: &crate::Data,
    ) -> Pin<ShaderCompilationInfoFuture> {
        let shader_data = downcast_ref(shader_data);
        let future = Context::shader_get_compilation_info(self, shader_data);
        Box::pin(future)
    }

    fn texture_create_view(
        &self,
        texture_data: &crate::Data,
        desc: &TextureViewDescriptor<'_>,
    ) -> Box<crate::Data> {
        let texture_data = downcast_ref(texture_data);
        let data = Context::texture_create_view(self, texture_data, desc);
        Box::new(data) as _
    }

    fn surface_drop(&self, surface_data: &crate::Data) {
        let surface_data = downcast_ref(surface_data);
        Context::surface_drop(self, surface_data)
    }

    fn adapter_drop(&self, adapter_data: &crate::Data) {
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_drop(self, adapter_data)
    }

    fn buffer_destroy(&self, buffer_data: &crate::Data) {
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_destroy(self, buffer_data)
    }

    fn buffer_drop(&self, buffer_data: &crate::Data) {
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_drop(self, buffer_data)
    }

    fn texture_destroy(&self, texture_data: &crate::Data) {
        let texture_data = downcast_ref(texture_data);
        Context::texture_destroy(self, texture_data)
    }

    fn texture_drop(&self, texture_data: &crate::Data) {
        let texture_data = downcast_ref(texture_data);
        Context::texture_drop(self, texture_data)
    }

    fn texture_view_drop(&self, texture_view_data: &crate::Data) {
        let texture_view_data = downcast_ref(texture_view_data);
        Context::texture_view_drop(self, texture_view_data)
    }

    fn sampler_drop(&self, sampler_data: &crate::Data) {
        let sampler_data = downcast_ref(sampler_data);
        Context::sampler_drop(self, sampler_data)
    }

    fn query_set_drop(&self, query_set_data: &crate::Data) {
        let query_set_data = downcast_ref(query_set_data);
        Context::query_set_drop(self, query_set_data)
    }

    fn bind_group_drop(&self, bind_group_data: &crate::Data) {
        let bind_group_data = downcast_ref(bind_group_data);
        Context::bind_group_drop(self, bind_group_data)
    }

    fn bind_group_layout_drop(&self, bind_group_layout_data: &crate::Data) {
        let bind_group_layout_data = downcast_ref(bind_group_layout_data);
        Context::bind_group_layout_drop(self, bind_group_layout_data)
    }

    fn pipeline_layout_drop(&self, pipeline_layout_data: &crate::Data) {
        let pipeline_layout_data = downcast_ref(pipeline_layout_data);
        Context::pipeline_layout_drop(self, pipeline_layout_data)
    }

    fn shader_module_drop(&self, shader_module_data: &crate::Data) {
        let shader_module_data = downcast_ref(shader_module_data);
        Context::shader_module_drop(self, shader_module_data)
    }

    fn command_encoder_drop(&self, command_encoder_data: &crate::Data) {
        let command_encoder_data = downcast_ref(command_encoder_data);
        Context::command_encoder_drop(self, command_encoder_data)
    }

    fn command_buffer_drop(&self, command_buffer_data: &crate::Data) {
        let command_buffer_data = downcast_ref(command_buffer_data);
        Context::command_buffer_drop(self, command_buffer_data)
    }

    fn render_bundle_drop(&self, render_bundle_data: &crate::Data) {
        let render_bundle_data = downcast_ref(render_bundle_data);
        Context::render_bundle_drop(self, render_bundle_data)
    }

    fn compute_pipeline_drop(&self, pipeline_data: &crate::Data) {
        let pipeline_data = downcast_ref(pipeline_data);
        Context::compute_pipeline_drop(self, pipeline_data)
    }

    fn render_pipeline_drop(&self, pipeline_data: &crate::Data) {
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_pipeline_drop(self, pipeline_data)
    }

    fn pipeline_cache_drop(&self, cache_data: &crate::Data) {
        let cache_data = downcast_ref(cache_data);
        Context::pipeline_cache_drop(self, cache_data)
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> Box<crate::Data> {
        let pipeline_data = downcast_ref(pipeline_data);
        let data = Context::compute_pipeline_get_bind_group_layout(self, pipeline_data, index);
        Box::new(data) as _
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> Box<crate::Data> {
        let pipeline_data = downcast_ref(pipeline_data);
        let data = Context::render_pipeline_get_bind_group_layout(self, pipeline_data, index);
        Box::new(data) as _
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder_data: &crate::Data,
        source_data: &crate::Data,
        source_offset: BufferAddress,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        let source_data = downcast_ref(source_data);
        let destination_data = downcast_ref(destination_data);
        Context::command_encoder_copy_buffer_to_buffer(
            self,
            encoder_data,
            source_data,
            source_offset,
            destination_data,
            destination_offset,
            copy_size,
        )
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_buffer_to_texture(
            self,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Extent3d,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_texture_to_buffer(
            self,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder_data: &crate::Data,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_texture_to_texture(
            self,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder_data: &crate::Data,
        desc: &ComputePassDescriptor<'_>,
    ) -> Box<crate::Data> {
        let encoder_data = downcast_ref(encoder_data);
        let data = Context::command_encoder_begin_compute_pass(self, encoder_data, desc);
        Box::new(data) as _
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder_data: &crate::Data,
        desc: &RenderPassDescriptor<'_>,
    ) -> Box<crate::Data> {
        let encoder_data = downcast_ref(encoder_data);
        let data = Context::command_encoder_begin_render_pass(self, encoder_data, desc);
        Box::new(data) as _
    }

    fn command_encoder_finish(&self, encoder_data: &mut crate::Data) -> Box<crate::Data> {
        let data = Context::command_encoder_finish(self, downcast_mut(encoder_data));
        Box::new(data) as _
    }

    fn command_encoder_clear_texture(
        &self,
        encoder_data: &crate::Data,
        texture_data: &crate::Data,
        subresource_range: &ImageSubresourceRange,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        let texture_data = downcast_ref(texture_data);
        Context::command_encoder_clear_texture(self, encoder_data, texture_data, subresource_range)
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::command_encoder_clear_buffer(self, encoder_data, buffer_data, offset, size)
    }

    fn command_encoder_insert_debug_marker(&self, encoder_data: &crate::Data, label: &str) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_insert_debug_marker(self, encoder_data, label)
    }

    fn command_encoder_push_debug_group(&self, encoder_data: &crate::Data, label: &str) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_push_debug_group(self, encoder_data, label)
    }

    fn command_encoder_pop_debug_group(&self, encoder_data: &crate::Data) {
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_pop_debug_group(self, encoder_data)
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder_data: &crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        let query_set_data = downcast_ref(query_set_data);
        Context::command_encoder_write_timestamp(self, encoder_data, query_set_data, query_index)
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder_data: &crate::Data,
        query_set_data: &crate::Data,
        first_query: u32,
        query_count: u32,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
    ) {
        let encoder_data = downcast_ref(encoder_data);
        let query_set_data = downcast_ref(query_set_data);
        let destination_data = downcast_ref(destination_data);
        Context::command_encoder_resolve_query_set(
            self,
            encoder_data,
            query_set_data,
            first_query,
            query_count,
            destination_data,
            destination_offset,
        )
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder_data: Box<crate::Data>,
        desc: &RenderBundleDescriptor<'_>,
    ) -> Box<crate::Data> {
        let encoder_data = *encoder_data.downcast().unwrap();
        let data = Context::render_bundle_encoder_finish(self, encoder_data, desc);
        Box::new(data) as _
    }

    fn queue_write_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        data: &[u8],
    ) {
        let queue_data = downcast_ref(queue_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_write_buffer(self, queue_data, buffer_data, offset, data)
    }

    fn queue_validate_write_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let queue_data = downcast_ref(queue_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_validate_write_buffer(self, queue_data, buffer_data, offset, size)
    }

    fn queue_create_staging_buffer(
        &self,
        queue_data: &crate::Data,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>> {
        let queue_data = downcast_ref(queue_data);
        Context::queue_create_staging_buffer(self, queue_data, size)
    }

    fn queue_write_staging_buffer(
        &self,
        queue_data: &crate::Data,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    ) {
        let queue_data = downcast_ref(queue_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_write_staging_buffer(self, queue_data, buffer_data, offset, staging_buffer)
    }

    fn queue_write_texture(
        &self,
        queue_data: &crate::Data,
        texture: ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    ) {
        let queue_data = downcast_ref(queue_data);
        Context::queue_write_texture(self, queue_data, texture, data, data_layout, size)
    }

    #[cfg(any(webgpu, webgl))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue_data: &crate::Data,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    ) {
        let queue_data = downcast_ref(queue_data);
        Context::queue_copy_external_image_to_texture(self, queue_data, source, dest, size)
    }

    fn queue_submit(
        &self,
        queue_data: &crate::Data,
        command_buffers: &mut dyn Iterator<Item = Box<crate::Data>>,
    ) -> Arc<crate::Data> {
        let queue_data = downcast_ref(queue_data);
        let command_buffers = command_buffers.map(|data| *data.downcast().unwrap());
        let data = Context::queue_submit(self, queue_data, command_buffers);
        Arc::new(data) as _
    }

    fn queue_get_timestamp_period(&self, queue_data: &crate::Data) -> f32 {
        let queue_data = downcast_ref(queue_data);
        Context::queue_get_timestamp_period(self, queue_data)
    }

    fn queue_on_submitted_work_done(
        &self,
        queue_data: &crate::Data,
        callback: SubmittedWorkDoneCallback,
    ) {
        let queue_data = downcast_ref(queue_data);
        Context::queue_on_submitted_work_done(self, queue_data, callback)
    }

    fn device_start_capture(&self, device_data: &crate::Data) {
        let device_data = downcast_ref(device_data);
        Context::device_start_capture(self, device_data)
    }

    fn device_stop_capture(&self, device_data: &crate::Data) {
        let device_data = downcast_ref(device_data);
        Context::device_stop_capture(self, device_data)
    }

    fn device_get_internal_counters(&self, device_data: &crate::Data) -> wgt::InternalCounters {
        let device_data = downcast_ref(device_data);
        Context::device_get_internal_counters(self, device_data)
    }

    fn generate_allocator_report(&self, device_data: &crate::Data) -> Option<wgt::AllocatorReport> {
        let device_data = downcast_ref(device_data);
        Context::device_generate_allocator_report(self, device_data)
    }

    fn pipeline_cache_get_data(&self, cache_data: &crate::Data) -> Option<Vec<u8>> {
        let cache_data = downcast_ref::<T::PipelineCacheData>(cache_data);
        Context::pipeline_cache_get_data(self, cache_data)
    }

    fn compute_pass_set_pipeline(&self, pass_data: &mut crate::Data, pipeline_data: &crate::Data) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::compute_pass_set_pipeline(self, pass_data, pipeline_data)
    }

    fn compute_pass_set_bind_group(
        &self,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let bg = bind_group_data.map(downcast_ref);
        Context::compute_pass_set_bind_group(self, pass_data, index, bg, offsets)
    }

    fn compute_pass_set_push_constants(
        &self,
        pass_data: &mut crate::Data,
        offset: u32,
        data: &[u8],
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_set_push_constants(self, pass_data, offset, data)
    }

    fn compute_pass_insert_debug_marker(&self, pass_data: &mut crate::Data, label: &str) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_insert_debug_marker(self, pass_data, label)
    }

    fn compute_pass_push_debug_group(&self, pass_data: &mut crate::Data, group_label: &str) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_push_debug_group(self, pass_data, group_label)
    }

    fn compute_pass_pop_debug_group(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_pop_debug_group(self, pass_data)
    }

    fn compute_pass_write_timestamp(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let query_set_data = downcast_ref(query_set_data);
        Context::compute_pass_write_timestamp(self, pass_data, query_set_data, query_index)
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let query_set_data = downcast_ref(query_set_data);
        Context::compute_pass_begin_pipeline_statistics_query(
            self,
            pass_data,
            query_set_data,
            query_index,
        )
    }

    fn compute_pass_end_pipeline_statistics_query(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_end_pipeline_statistics_query(self, pass_data)
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_dispatch_workgroups(self, pass_data, x, y, z)
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::compute_pass_dispatch_workgroups_indirect(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn compute_pass_end(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut(pass_data);
        Context::compute_pass_end(self, pass_data)
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder_data: &mut crate::Data,
        pipeline_data: &crate::Data,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_bundle_encoder_set_pipeline(self, encoder_data, pipeline_data)
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let bg = bind_group_data.map(downcast_ref);
        Context::render_bundle_encoder_set_bind_group(self, encoder_data, index, bg, offsets)
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder_data: &mut crate::Data,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_bundle_encoder_set_index_buffer(
            self,
            encoder_data,
            buffer_data,
            index_format,
            offset,
            size,
        )
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder_data: &mut crate::Data,
        slot: u32,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_bundle_encoder_set_vertex_buffer(
            self,
            encoder_data,
            slot,
            buffer_data,
            offset,
            size,
        )
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_set_push_constants(self, encoder_data, stages, offset, data)
    }

    fn render_bundle_encoder_draw(
        &self,
        encoder_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_draw(self, encoder_data, vertices, instances)
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_draw_indexed(
            self,
            encoder_data,
            indices,
            base_vertex,
            instances,
        )
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_draw_indirect(
            self,
            encoder_data,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_draw_indexed_indirect(
            self,
            encoder_data,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_pass_set_pipeline(&self, pass_data: &mut crate::Data, pipeline_data: &crate::Data) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_pass_set_pipeline(self, pass_data, pipeline_data)
    }

    fn render_pass_set_bind_group(
        &self,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group_data: Option<&crate::Data>,
        offsets: &[DynamicOffset],
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let bg = bind_group_data.map(downcast_ref);
        Context::render_pass_set_bind_group(self, pass_data, index, bg, offsets)
    }

    fn render_pass_set_index_buffer(
        &self,
        pass_data: &mut crate::Data,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_pass_set_index_buffer(
            self,
            pass_data,
            buffer_data,
            index_format,
            offset,
            size,
        )
    }

    fn render_pass_set_vertex_buffer(
        &self,
        pass_data: &mut crate::Data,
        slot: u32,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_pass_set_vertex_buffer(self, pass_data, slot, buffer_data, offset, size)
    }

    fn render_pass_set_push_constants(
        &self,
        pass_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_push_constants(self, pass_data, stages, offset, data)
    }

    fn render_pass_draw(
        &self,
        pass_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_draw(self, pass_data, vertices, instances)
    }

    fn render_pass_draw_indexed(
        &self,
        pass_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_draw_indexed(self, pass_data, indices, base_vertex, instances)
    }

    fn render_pass_draw_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_draw_indirect(self, pass_data, indirect_buffer_data, indirect_offset)
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_draw_indexed_indirect(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_pass_multi_draw_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_multi_draw_indirect(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_multi_draw_indexed_indirect(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_pass_multi_draw_indirect_count(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass_data: &mut crate::Data,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_pass_multi_draw_indexed_indirect_count(
            self,
            pass_data,
            indirect_buffer_data,
            indirect_offset,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_blend_constant(&self, pass_data: &mut crate::Data, color: Color) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_blend_constant(self, pass_data, color)
    }

    fn render_pass_set_scissor_rect(
        &self,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_scissor_rect(self, pass_data, x, y, width, height)
    }

    fn render_pass_set_viewport(
        &self,
        pass_data: &mut crate::Data,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_viewport(
            self, pass_data, x, y, width, height, min_depth, max_depth,
        )
    }

    fn render_pass_set_stencil_reference(&self, pass_data: &mut crate::Data, reference: u32) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_stencil_reference(self, pass_data, reference)
    }

    fn render_pass_insert_debug_marker(&self, pass_data: &mut crate::Data, label: &str) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_insert_debug_marker(self, pass_data, label)
    }

    fn render_pass_push_debug_group(&self, pass_data: &mut crate::Data, group_label: &str) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_push_debug_group(self, pass_data, group_label)
    }

    fn render_pass_pop_debug_group(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_pop_debug_group(self, pass_data)
    }

    fn render_pass_write_timestamp(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let query_set_data = downcast_ref(query_set_data);
        Context::render_pass_write_timestamp(self, pass_data, query_set_data, query_index)
    }

    fn render_pass_begin_occlusion_query(&self, pass_data: &mut crate::Data, query_index: u32) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_begin_occlusion_query(self, pass_data, query_index)
    }

    fn render_pass_end_occlusion_query(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_end_occlusion_query(self, pass_data)
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass_data: &mut crate::Data,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let query_set_data = downcast_ref(query_set_data);
        Context::render_pass_begin_pipeline_statistics_query(
            self,
            pass_data,
            query_set_data,
            query_index,
        )
    }

    fn render_pass_end_pipeline_statistics_query(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_end_pipeline_statistics_query(self, pass_data)
    }

    fn render_pass_execute_bundles(
        &self,
        pass_data: &mut crate::Data,
        render_bundles: &mut dyn Iterator<Item = &crate::Data>,
    ) {
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let mut render_bundles = render_bundles.map(downcast_ref);
        Context::render_pass_execute_bundles(self, pass_data, &mut render_bundles)
    }

    fn render_pass_end(&self, pass_data: &mut crate::Data) {
        let pass_data = downcast_mut(pass_data);
        Context::render_pass_end(self, pass_data)
    }
}

pub trait QueueWriteBuffer: WasmNotSendSync + Debug {
    fn slice(&self) -> &[u8];

    fn slice_mut(&mut self) -> &mut [u8];

    #[cfg(not(target_os = "emscripten"))]
    fn as_any(&self) -> &dyn Any;
}

pub trait BufferMappedRange: WasmNotSendSync + Debug {
    fn slice(&self) -> &[u8];
    fn slice_mut(&mut self) -> &mut [u8];
}

#[cfg(test)]
mod tests {
    use super::DynContext;

    fn compiles<T>() {}

    /// Assert that DynContext is object safe.
    #[test]
    fn object_safe() {
        compiles::<Box<dyn DynContext>>();
    }
}
