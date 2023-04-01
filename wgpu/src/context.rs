use std::{any::Any, fmt::Debug, future::Future, num::NonZeroU64, ops::Range, pin::Pin, sync::Arc};

use wgt::{
    strict_assert, strict_assert_eq, AdapterInfo, BufferAddress, BufferSize, Color,
    DownlevelCapabilities, DynamicOffset, Extent3d, Features, ImageDataLayout,
    ImageSubresourceRange, IndexFormat, Limits, ShaderStages, SurfaceStatus, TextureFormat,
    TextureFormatFeatures,
};

use crate::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, Buffer, BufferAsyncError, BufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Error, ErrorFilter, ImageCopyBuffer, ImageCopyTexture, Maintain, MapMode,
    PipelineLayoutDescriptor, QuerySetDescriptor, RenderBundleDescriptor,
    RenderBundleEncoderDescriptor, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, RequestDeviceError, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderModuleDescriptorSpirV, Texture, TextureDescriptor, TextureViewDescriptor,
    UncapturedErrorHandler,
};

/// Meta trait for an id tracked by a context.
///
/// There is no need to manually implement this trait since there is a blanket implementation for this trait.
pub trait ContextId: Into<ObjectId> + From<ObjectId> + Debug + 'static {}
impl<T: Into<ObjectId> + From<ObjectId> + Debug + 'static> ContextId for T {}

/// Meta trait for an data associated with an id tracked by a context.
///
/// There is no need to manually implement this trait since there is a blanket implementation for this trait.
pub trait ContextData: Debug + Send + Sync + 'static {}
impl<T: Debug + Send + Sync + 'static> ContextData for T {}

pub trait Context: Debug + Send + Sized + Sync {
    type AdapterId: ContextId + Send + Sync;
    type AdapterData: ContextData;
    type DeviceId: ContextId + Send + Sync;
    type DeviceData: ContextData;
    type QueueId: ContextId + Send + Sync;
    type QueueData: ContextData;
    type ShaderModuleId: ContextId + Send + Sync;
    type ShaderModuleData: ContextData;
    type BindGroupLayoutId: ContextId + Send + Sync;
    type BindGroupLayoutData: ContextData;
    type BindGroupId: ContextId + Send + Sync;
    type BindGroupData: ContextData;
    type TextureViewId: ContextId + Send + Sync;
    type TextureViewData: ContextData;
    type SamplerId: ContextId + Send + Sync;
    type SamplerData: ContextData;
    type BufferId: ContextId + Send + Sync;
    type BufferData: ContextData;
    type TextureId: ContextId + Send + Sync;
    type TextureData: ContextData;
    type QuerySetId: ContextId + Send + Sync;
    type QuerySetData: ContextData;
    type PipelineLayoutId: ContextId + Send + Sync;
    type PipelineLayoutData: ContextData;
    type RenderPipelineId: ContextId + Send + Sync;
    type RenderPipelineData: ContextData;
    type ComputePipelineId: ContextId + Send + Sync;
    type ComputePipelineData: ContextData;
    type CommandEncoderId: ContextId + Send + Sync;
    type CommandEncoderData: ContextData;
    type ComputePassId: ContextId;
    type ComputePassData: ContextData;
    type RenderPassId: ContextId;
    type RenderPassData: ContextData;
    type CommandBufferId: ContextId + Send + Sync;
    type CommandBufferData: ContextData;
    type RenderBundleEncoderId: ContextId;
    type RenderBundleEncoderData: ContextData;
    type RenderBundleId: ContextId + Send + Sync;
    type RenderBundleData: ContextData;
    type SurfaceId: ContextId + Send + Sync;
    type SurfaceData: ContextData;

    type SurfaceOutputDetail: Send + Sync + 'static;
    type SubmissionIndex: ContextId + Clone + Copy + Send + Sync;
    type SubmissionIndexData: ContextData + Copy;

    type RequestAdapterFuture: Future<Output = Option<(Self::AdapterId, Self::AdapterData)>>
        + Send
        + 'static;
    type RequestDeviceFuture: Future<
            Output = Result<
                (
                    Self::DeviceId,
                    Self::DeviceData,
                    Self::QueueId,
                    Self::QueueData,
                ),
                RequestDeviceError,
            >,
        > + Send
        + 'static;
    type PopErrorScopeFuture: Future<Output = Option<Error>> + Send + 'static;

    fn init(instance_desc: wgt::InstanceDescriptor) -> Self;
    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError>;
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
    ) -> Self::RequestAdapterFuture;
    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture;
    fn instance_poll_all_devices(&self, force_wait: bool) -> bool;
    fn adapter_is_surface_supported(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
    ) -> bool;
    fn adapter_features(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> Features;
    fn adapter_limits(&self, adapter: &Self::AdapterId, adapter_data: &Self::AdapterData)
        -> Limits;
    fn adapter_downlevel_capabilities(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities;
    fn adapter_get_info(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
        format: TextureFormat,
    ) -> TextureFormatFeatures;
    fn adapter_get_presentation_timestamp(
        &self,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp;

    fn surface_get_capabilities(
        &self,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
        adapter: &Self::AdapterId,
        adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities;
    fn surface_configure(
        &self,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        config: &crate::SurfaceConfiguration,
    );
    #[allow(clippy::type_complexity)]
    fn surface_get_current_texture(
        &self,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureId>,
        Option<Self::TextureData>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    );
    fn surface_present(&self, texture: &Self::TextureId, detail: &Self::SurfaceOutputDetail);
    fn surface_texture_discard(
        &self,
        texture: &Self::TextureId,
        detail: &Self::SurfaceOutputDetail,
    );

    fn device_features(&self, device: &Self::DeviceId, device_data: &Self::DeviceData) -> Features;
    fn device_limits(&self, device: &Self::DeviceId, device_data: &Self::DeviceData) -> Limits;
    fn device_downlevel_properties(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
    ) -> DownlevelCapabilities;
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData);
    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData);
    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BindGroupLayoutDescriptor,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData);
    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BindGroupDescriptor,
    ) -> (Self::BindGroupId, Self::BindGroupData);
    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &PipelineLayoutDescriptor,
    ) -> (Self::PipelineLayoutId, Self::PipelineLayoutData);
    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &RenderPipelineDescriptor,
    ) -> (Self::RenderPipelineId, Self::RenderPipelineData);
    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &ComputePipelineDescriptor,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData);
    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BufferDescriptor,
    ) -> (Self::BufferId, Self::BufferData);
    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &TextureDescriptor,
    ) -> (Self::TextureId, Self::TextureData);
    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &SamplerDescriptor,
    ) -> (Self::SamplerId, Self::SamplerData);
    fn device_create_query_set(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &QuerySetDescriptor,
    ) -> (Self::QuerySetId, Self::QuerySetData);
    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &CommandEncoderDescriptor,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData);
    fn device_create_render_bundle_encoder(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &RenderBundleEncoderDescriptor,
    ) -> (Self::RenderBundleEncoderId, Self::RenderBundleEncoderData);
    fn device_drop(&self, device: &Self::DeviceId, device_data: &Self::DeviceData);
    fn device_poll(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        maintain: Maintain,
    ) -> bool;
    fn device_on_uncaptured_error(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    );
    fn device_push_error_scope(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        filter: ErrorFilter,
    );
    fn device_pop_error_scope(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
    ) -> Self::PopErrorScopeFuture;

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>,
    );
    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange>;
    fn buffer_unmap(&self, buffer: &Self::BufferId, buffer_data: &Self::BufferData);
    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        texture_data: &Self::TextureData,
        desc: &TextureViewDescriptor,
    ) -> (Self::TextureViewId, Self::TextureViewData);

    fn surface_drop(&self, surface: &Self::SurfaceId, surface_data: &Self::SurfaceData);
    fn adapter_drop(&self, adapter: &Self::AdapterId, adapter_data: &Self::AdapterData);
    fn buffer_destroy(&self, buffer: &Self::BufferId, buffer_data: &Self::BufferData);
    fn buffer_drop(&self, buffer: &Self::BufferId, buffer_data: &Self::BufferData);
    fn texture_destroy(&self, texture: &Self::TextureId, texture_data: &Self::TextureData);
    fn texture_drop(&self, texture: &Self::TextureId, texture_data: &Self::TextureData);
    fn texture_view_drop(
        &self,
        texture_view: &Self::TextureViewId,
        texture_view_data: &Self::TextureViewData,
    );
    fn sampler_drop(&self, sampler: &Self::SamplerId, sampler_data: &Self::SamplerData);
    fn query_set_drop(&self, query_set: &Self::QuerySetId, query_set_data: &Self::QuerySetData);
    fn bind_group_drop(
        &self,
        bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
    );
    fn bind_group_layout_drop(
        &self,
        bind_group_layout: &Self::BindGroupLayoutId,
        bind_group_layout_data: &Self::BindGroupLayoutData,
    );
    fn pipeline_layout_drop(
        &self,
        pipeline_layout: &Self::PipelineLayoutId,
        pipeline_layout_data: &Self::PipelineLayoutData,
    );
    fn shader_module_drop(
        &self,
        shader_module: &Self::ShaderModuleId,
        shader_module_data: &Self::ShaderModuleData,
    );
    fn command_encoder_drop(
        &self,
        command_encoder: &Self::CommandEncoderId,
        command_encoder_data: &Self::CommandEncoderData,
    );
    fn command_buffer_drop(
        &self,
        command_buffer: &Self::CommandBufferId,
        command_buffer_data: &Self::CommandBufferData,
    );
    fn render_bundle_drop(
        &self,
        render_bundle: &Self::RenderBundleId,
        render_bundle_data: &Self::RenderBundleData,
    );
    fn compute_pipeline_drop(
        &self,
        pipeline: &Self::ComputePipelineId,
        pipeline_data: &Self::ComputePipelineData,
    );
    fn render_pipeline_drop(
        &self,
        pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
    );

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::ComputePipelineId,
        pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData);
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData);

    #[allow(clippy::too_many_arguments)]
    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: &Self::BufferId,
        source_data: &Self::BufferData,
        source_offset: BufferAddress,
        destination: &Self::BufferId,
        destination_data: &Self::BufferData,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        desc: &ComputePassDescriptor,
    ) -> (Self::ComputePassId, Self::ComputePassData);
    fn command_encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    );
    fn command_encoder_begin_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        desc: &RenderPassDescriptor<'_, '_>,
    ) -> (Self::RenderPassId, Self::RenderPassData);
    fn command_encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    );
    fn command_encoder_finish(
        &self,
        encoder: Self::CommandEncoderId,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData);

    fn command_encoder_clear_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        texture: &Texture, // TODO: Decompose?
        subresource_range: &ImageSubresourceRange,
    );
    fn command_encoder_clear_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );

    fn command_encoder_insert_debug_marker(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    );
    fn command_encoder_push_debug_group(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    );
    fn command_encoder_pop_debug_group(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
    );

    fn command_encoder_write_timestamp(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn command_encoder_resolve_query_set(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        destination: &Self::BufferId,
        destination_data: &Self::BufferData,
        destination_offset: BufferAddress,
    );

    fn render_bundle_encoder_finish(
        &self,
        encoder: Self::RenderBundleEncoderId,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &RenderBundleDescriptor,
    ) -> (Self::RenderBundleId, Self::RenderBundleData);
    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_validate_write_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()>;
    fn queue_create_staging_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>>;
    fn queue_write_staging_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    );
    fn queue_write_texture(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    );
    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> (Self::SubmissionIndex, Self::SubmissionIndexData);
    fn queue_get_timestamp_period(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
    ) -> f32;
    fn queue_on_submitted_work_done(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        callback: Box<dyn FnOnce() + Send + 'static>,
    );

    fn device_start_capture(&self, device: &Self::DeviceId, device_data: &Self::DeviceData);
    fn device_stop_capture(&self, device: &Self::DeviceId, device_data: &Self::DeviceData);

    fn compute_pass_set_pipeline(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        pipeline: &Self::ComputePipelineId,
        pipeline_data: &Self::ComputePipelineData,
    );
    fn compute_pass_set_bind_group(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        index: u32,
        bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[DynamicOffset],
    );
    fn compute_pass_set_push_constants(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        offset: u32,
        data: &[u8],
    );
    fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        label: &str,
    );
    fn compute_pass_push_debug_group(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        group_label: &str,
    );
    fn compute_pass_pop_debug_group(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    );
    fn compute_pass_write_timestamp(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    );
    fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    );
    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
    );
    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_bundle_encoder_draw(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer: &Self::BufferId,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer: &Self::BufferId,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );

    fn render_pass_set_pipeline(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        pipeline: &Self::RenderPipelineId,
        pipeline_data: &Self::RenderPipelineData,
    );
    fn render_pass_set_bind_group(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        index: u32,
        bind_group: &Self::BindGroupId,
        bind_group_data: &Self::BindGroupData,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_index_buffer(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_push_constants(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_pass_draw(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_pass_draw_indexed(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_pass_draw_indirect(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
    );
    fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer: &Self::BufferId,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: BufferAddress,
        count_buffer: &Self::BufferId,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_set_blend_constant(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        color: Color,
    );
    fn render_pass_set_scissor_rect(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_viewport(
        &self,
        pass: &mut Self::RenderPassId,
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
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    );
    fn render_pass_insert_debug_marker(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        label: &str,
    );
    fn render_pass_push_debug_group(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        group_label: &str,
    );
    fn render_pass_pop_debug_group(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    );
    fn render_pass_write_timestamp(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    );
    fn render_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    );
    fn render_pass_execute_bundles<'a>(
        &self,
        pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        render_bundles: Box<dyn Iterator<Item = Self::RenderBundleId> + 'a>,
    );
}

/// Object id.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ObjectId {
    /// ID that is unique at any given time
    id: Option<NonZeroU64>,
    #[cfg(feature = "expose-ids")]
    /// ID that is unique at all times
    global_id: Option<NonZeroU64>,
}

impl ObjectId {
    const UNUSED: Self = ObjectId {
        id: None,
        #[cfg(feature = "expose-ids")]
        global_id: None,
    };

    pub fn new(id: NonZeroU64, #[cfg(feature = "expose-ids")] global_id: NonZeroU64) -> Self {
        Self {
            id: Some(id),
            #[cfg(feature = "expose-ids")]
            global_id: Some(global_id),
        }
    }

    #[allow(dead_code)]
    pub fn from_global_id(global_id: NonZeroU64) -> Self {
        Self {
            id: Some(global_id),
            #[cfg(feature = "expose-ids")]
            global_id: Some(global_id),
        }
    }

    pub fn id(&self) -> NonZeroU64 {
        self.id.unwrap()
    }

    #[cfg(feature = "expose-ids")]
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> NonZeroU64 {
        self.global_id.unwrap()
    }
}

static_assertions::assert_impl_all!(ObjectId: Send, Sync);

fn downcast_ref<T: Debug + Send + Sync + 'static>(data: &crate::Data) -> &T {
    strict_assert!(data.is::<T>());
    // Copied from std.
    unsafe { &*(data as *const dyn Any as *const T) }
}

fn downcast_mut<T: Debug + Send + Sync + 'static>(data: &mut crate::Data) -> &mut T {
    strict_assert!(data.is::<T>());
    // Copied from std.
    unsafe { &mut *(data as *mut dyn Any as *mut T) }
}

/// Representation of an object id that is not used.
///
/// This may be used as the id type when only a the data associated type is used for a specific type of object.
#[derive(Debug, Clone, Copy)]
pub struct Unused;

impl From<ObjectId> for Unused {
    fn from(id: ObjectId) -> Self {
        strict_assert_eq!(id, ObjectId::UNUSED);
        Self
    }
}

impl From<Unused> for ObjectId {
    fn from(_: Unused) -> Self {
        ObjectId::UNUSED
    }
}

pub(crate) struct DeviceRequest {
    pub device_id: ObjectId,
    pub device_data: Box<crate::Data>,
    pub queue_id: ObjectId,
    pub queue_data: Box<crate::Data>,
}

/// An object safe variant of [`Context`] implemented by all types that implement [`Context`].
pub(crate) trait DynContext: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(ObjectId, Box<crate::Data>), crate::CreateSurfaceError>;
    #[allow(clippy::type_complexity)]
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
    ) -> Pin<Box<dyn Future<Output = Option<(ObjectId, Box<crate::Data>)>> + Send>>;
    fn adapter_request_device(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn Future<Output = Result<DeviceRequest, RequestDeviceError>> + Send>>;

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool;
    fn adapter_is_surface_supported(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        surface: &ObjectId,
        surface_data: &crate::Data,
    ) -> bool;
    fn adapter_features(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> Features;
    fn adapter_limits(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> Limits;
    fn adapter_downlevel_capabilities(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> DownlevelCapabilities;
    fn adapter_get_info(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        format: TextureFormat,
    ) -> TextureFormatFeatures;
    fn adapter_get_presentation_timestamp(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> wgt::PresentationTimestamp;

    fn surface_get_capabilities(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> wgt::SurfaceCapabilities;
    fn surface_configure(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
        device: &ObjectId,
        device_data: &crate::Data,
        config: &crate::SurfaceConfiguration,
    );
    fn surface_get_current_texture(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
    ) -> (
        Option<ObjectId>,
        Option<Box<crate::Data>>,
        SurfaceStatus,
        Box<dyn Any + Send + Sync>,
    );
    fn surface_present(&self, texture: &ObjectId, detail: &(dyn Any + Send + Sync));
    fn surface_texture_discard(&self, texture: &ObjectId, detail: &(dyn Any + Send + Sync));

    fn device_features(&self, device: &ObjectId, device_data: &crate::Data) -> Features;
    fn device_limits(&self, device: &ObjectId, device_data: &crate::Data) -> Limits;
    fn device_downlevel_properties(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
    ) -> DownlevelCapabilities;
    fn device_create_shader_module(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (ObjectId, Box<crate::Data>);
    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_bind_group_layout(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BindGroupLayoutDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_bind_group(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BindGroupDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_pipeline_layout(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &PipelineLayoutDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_render_pipeline(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &RenderPipelineDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_compute_pipeline(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &ComputePipelineDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_buffer(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BufferDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_texture(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &TextureDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_sampler(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &SamplerDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_query_set(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &QuerySetDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_command_encoder(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &CommandEncoderDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_create_render_bundle_encoder(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &RenderBundleEncoderDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn device_drop(&self, device: &ObjectId, device_data: &crate::Data);
    fn device_poll(&self, device: &ObjectId, device_data: &crate::Data, maintain: Maintain)
        -> bool;
    fn device_on_uncaptured_error(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        handler: Box<dyn UncapturedErrorHandler>,
    );
    fn device_push_error_scope(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        filter: ErrorFilter,
    );
    fn device_pop_error_scope(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
    ) -> Pin<Box<dyn Future<Output = Option<Error>> + Send + 'static>>;
    fn buffer_map_async(
        &self,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>,
    );
    fn buffer_get_mapped_range(
        &self,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange>;
    fn buffer_unmap(&self, buffer: &ObjectId, buffer_data: &crate::Data);
    fn texture_create_view(
        &self,
        texture: &ObjectId,
        texture_data: &crate::Data,
        desc: &TextureViewDescriptor,
    ) -> (ObjectId, Box<crate::Data>);

    fn surface_drop(&self, surface: &ObjectId, surface_data: &crate::Data);
    fn adapter_drop(&self, adapter: &ObjectId, adapter_data: &crate::Data);
    fn buffer_destroy(&self, buffer: &ObjectId, buffer_data: &crate::Data);
    fn buffer_drop(&self, buffer: &ObjectId, buffer_data: &crate::Data);
    fn texture_destroy(&self, buffer: &ObjectId, buffer_data: &crate::Data);
    fn texture_drop(&self, texture: &ObjectId, texture_data: &crate::Data);
    fn texture_view_drop(&self, texture_view: &ObjectId, texture_view_data: &crate::Data);
    fn sampler_drop(&self, sampler: &ObjectId, sampler_data: &crate::Data);
    fn query_set_drop(&self, query_set: &ObjectId, query_set_data: &crate::Data);
    fn bind_group_drop(&self, bind_group: &ObjectId, bind_group_data: &crate::Data);
    fn bind_group_layout_drop(
        &self,
        bind_group_layout: &ObjectId,
        bind_group_layout_data: &crate::Data,
    );
    fn pipeline_layout_drop(&self, pipeline_layout: &ObjectId, pipeline_layout_data: &crate::Data);
    fn shader_module_drop(&self, shader_module: &ObjectId, shader_module_data: &crate::Data);
    fn command_encoder_drop(&self, command_encoder: &ObjectId, command_encoder_data: &crate::Data);
    fn command_buffer_drop(&self, command_buffer: &ObjectId, command_buffer_data: &crate::Data);
    fn render_bundle_drop(&self, render_bundle: &ObjectId, render_bundle_data: &crate::Data);
    fn compute_pipeline_drop(&self, pipeline: &ObjectId, pipeline_data: &crate::Data);
    fn render_pipeline_drop(&self, pipeline: &ObjectId, pipeline_data: &crate::Data);

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> (ObjectId, Box<crate::Data>);
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> (ObjectId, Box<crate::Data>);

    #[allow(clippy::too_many_arguments)]
    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: &ObjectId,
        source_data: &crate::Data,
        source_offset: BufferAddress,
        destination: &ObjectId,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        desc: &ComputePassDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn command_encoder_end_compute_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    );
    fn command_encoder_begin_render_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        desc: &RenderPassDescriptor<'_, '_>,
    ) -> (ObjectId, Box<crate::Data>);
    fn command_encoder_end_render_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    );
    fn command_encoder_finish(
        &self,
        encoder: ObjectId,
        encoder_data: &mut crate::Data,
    ) -> (ObjectId, Box<crate::Data>);

    fn command_encoder_clear_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        texture: &Texture,
        subresource_range: &ImageSubresourceRange,
    );
    fn command_encoder_clear_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );

    fn command_encoder_insert_debug_marker(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        label: &str,
    );
    fn command_encoder_push_debug_group(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        label: &str,
    );
    fn command_encoder_pop_debug_group(&self, encoder: &ObjectId, encoder_data: &crate::Data);

    fn command_encoder_write_timestamp(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn command_encoder_resolve_query_set(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        first_query: u32,
        query_count: u32,
        destination: &ObjectId,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
    );

    fn render_bundle_encoder_finish(
        &self,
        encoder: ObjectId,
        encoder_data: Box<crate::Data>,
        desc: &RenderBundleDescriptor,
    ) -> (ObjectId, Box<crate::Data>);
    fn queue_write_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_validate_write_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()>;
    fn queue_create_staging_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>>;
    fn queue_write_staging_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    );
    fn queue_write_texture(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    );
    fn queue_submit<'a>(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        command_buffers: Box<dyn Iterator<Item = ObjectId> + 'a>,
    ) -> (ObjectId, Arc<crate::Data>);
    fn queue_get_timestamp_period(&self, queue: &ObjectId, queue_data: &crate::Data) -> f32;
    fn queue_on_submitted_work_done(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        callback: Box<dyn FnOnce() + Send + 'static>,
    );

    fn device_start_capture(&self, device: &ObjectId, data: &crate::Data);
    fn device_stop_capture(&self, device: &ObjectId, data: &crate::Data);

    fn compute_pass_set_pipeline(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    );
    fn compute_pass_set_bind_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    );
    fn compute_pass_set_push_constants(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        offset: u32,
        data: &[u8],
    );
    fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        label: &str,
    );
    fn compute_pass_push_debug_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        group_label: &str,
    );
    fn compute_pass_pop_debug_group(&self, pass: &mut ObjectId, pass_data: &mut crate::Data);
    fn compute_pass_write_timestamp(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    );
    fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        z: u32,
    );
    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    );
    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        slot: u32,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_bundle_encoder_draw(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        command_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );

    fn render_pass_set_pipeline(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    );
    fn render_pass_set_bind_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_index_buffer(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        slot: u32,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn render_pass_set_push_constants(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    );
    fn render_pass_draw(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    );
    fn render_pass_draw_indexed(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    );
    fn render_pass_draw_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    );
    fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        command_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn render_pass_set_blend_constant(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        color: Color,
    );
    fn render_pass_set_scissor_rect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn render_pass_set_viewport(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    fn render_pass_set_stencil_reference(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        reference: u32,
    );
    fn render_pass_insert_debug_marker(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        label: &str,
    );
    fn render_pass_push_debug_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        group_label: &str,
    );
    fn render_pass_pop_debug_group(&self, pass: &mut ObjectId, pass_data: &mut crate::Data);
    fn render_pass_write_timestamp(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    );
    fn render_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    );
    fn render_pass_execute_bundles<'a>(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        render_bundles: Box<dyn Iterator<Item = &'a ObjectId> + 'a>,
    );
}

// Blanket impl of DynContext for all types which implement Context.
impl<T> DynContext for T
where
    T: Context + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(ObjectId, Box<crate::Data>), crate::CreateSurfaceError> {
        let (surface, data) =
            Context::instance_create_surface(self, display_handle, window_handle)?;
        Ok((surface.into(), Box::new(data) as _))
    }

    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
    ) -> Pin<Box<dyn Future<Output = Option<(ObjectId, Box<crate::Data>)>> + Send>> {
        let future: T::RequestAdapterFuture = Context::instance_request_adapter(self, options);
        Box::pin(async move {
            let result: Option<(T::AdapterId, T::AdapterData)> = future.await;
            result.map(|(adapter, data)| (adapter.into(), Box::new(data) as _))
        })
    }

    fn adapter_request_device(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Pin<Box<dyn Future<Output = Result<DeviceRequest, RequestDeviceError>> + Send>> {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        let future = Context::adapter_request_device(self, &adapter, adapter_data, desc, trace_dir);

        Box::pin(async move {
            let (device_id, device_data, queue_id, queue_data) = future.await?;
            Ok(DeviceRequest {
                device_id: device_id.into(),
                device_data: Box::new(device_data) as _,
                queue_id: queue_id.into(),
                queue_data: Box::new(queue_data) as _,
            })
        })
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        Context::instance_poll_all_devices(self, force_wait)
    }

    fn adapter_is_surface_supported(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        surface: &ObjectId,
        surface_data: &crate::Data,
    ) -> bool {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        let surface = <T::SurfaceId>::from(*surface);
        let surface_data = downcast_ref(surface_data);
        Context::adapter_is_surface_supported(self, &adapter, adapter_data, &surface, surface_data)
    }

    fn adapter_features(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> Features {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_features(self, &adapter, adapter_data)
    }

    fn adapter_limits(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> Limits {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_limits(self, &adapter, adapter_data)
    }

    fn adapter_downlevel_capabilities(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> DownlevelCapabilities {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_downlevel_capabilities(self, &adapter, adapter_data)
    }

    fn adapter_get_info(&self, adapter: &ObjectId, adapter_data: &crate::Data) -> AdapterInfo {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_info(self, &adapter, adapter_data)
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
        format: TextureFormat,
    ) -> TextureFormatFeatures {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_texture_format_features(self, &adapter, adapter_data, format)
    }
    fn adapter_get_presentation_timestamp(
        &self,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> wgt::PresentationTimestamp {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_get_presentation_timestamp(self, &adapter, adapter_data)
    }

    fn surface_get_capabilities(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
        adapter: &ObjectId,
        adapter_data: &crate::Data,
    ) -> wgt::SurfaceCapabilities {
        let surface = <T::SurfaceId>::from(*surface);
        let surface_data = downcast_ref(surface_data);
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::surface_get_capabilities(self, &surface, surface_data, &adapter, adapter_data)
    }

    fn surface_configure(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
        device: &ObjectId,
        device_data: &crate::Data,
        config: &crate::SurfaceConfiguration,
    ) {
        let surface = <T::SurfaceId>::from(*surface);
        let surface_data = downcast_ref(surface_data);
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::surface_configure(self, &surface, surface_data, &device, device_data, config)
    }

    fn surface_get_current_texture(
        &self,
        surface: &ObjectId,
        surface_data: &crate::Data,
    ) -> (
        Option<ObjectId>,
        Option<Box<crate::Data>>,
        SurfaceStatus,
        Box<dyn Any + Send + Sync>,
    ) {
        let surface = <T::SurfaceId>::from(*surface);
        let surface_data = downcast_ref(surface_data);
        let (texture, texture_data, status, detail) =
            Context::surface_get_current_texture(self, &surface, surface_data);
        let detail = Box::new(detail) as Box<dyn Any + Send + Sync>;
        (
            texture.map(Into::into),
            texture_data.map(|b| Box::new(b) as _),
            status,
            detail,
        )
    }

    fn surface_present(&self, texture: &ObjectId, detail: &(dyn Any + Send + Sync)) {
        let texture = <T::TextureId>::from(*texture);
        Context::surface_present(self, &texture, detail.downcast_ref().unwrap())
    }

    fn surface_texture_discard(&self, texture: &ObjectId, detail: &(dyn Any + Send + Sync)) {
        let texture = <T::TextureId>::from(*texture);
        Context::surface_texture_discard(self, &texture, detail.downcast_ref().unwrap())
    }

    fn device_features(&self, device: &ObjectId, device_data: &crate::Data) -> Features {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_features(self, &device, device_data)
    }

    fn device_limits(&self, device: &ObjectId, device_data: &crate::Data) -> Limits {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_limits(self, &device, device_data)
    }

    fn device_downlevel_properties(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
    ) -> DownlevelCapabilities {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_downlevel_properties(self, &device, device_data)
    }

    fn device_create_shader_module(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (shader_module, data) = Context::device_create_shader_module(
            self,
            &device,
            device_data,
            desc,
            shader_bound_checks,
        );
        (shader_module.into(), Box::new(data) as _)
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (shader_module, data) =
            unsafe { Context::device_create_shader_module_spirv(self, &device, device_data, desc) };
        (shader_module.into(), Box::new(data) as _)
    }

    fn device_create_bind_group_layout(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BindGroupLayoutDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (bind_group_layout, data) =
            Context::device_create_bind_group_layout(self, &device, device_data, desc);
        (bind_group_layout.into(), Box::new(data) as _)
    }

    fn device_create_bind_group(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BindGroupDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (bind_group, data) =
            Context::device_create_bind_group(self, &device, device_data, desc);
        (bind_group.into(), Box::new(data) as _)
    }

    fn device_create_pipeline_layout(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &PipelineLayoutDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (pipeline_layout, data) =
            Context::device_create_pipeline_layout(self, &device, device_data, desc);
        (pipeline_layout.into(), Box::new(data) as _)
    }

    fn device_create_render_pipeline(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &RenderPipelineDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (render_pipeline, data) =
            Context::device_create_render_pipeline(self, &device, device_data, desc);
        (render_pipeline.into(), Box::new(data) as _)
    }

    fn device_create_compute_pipeline(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &ComputePipelineDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (compute_pipeline, data) =
            Context::device_create_compute_pipeline(self, &device, device_data, desc);
        (compute_pipeline.into(), Box::new(data) as _)
    }

    fn device_create_buffer(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &BufferDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (buffer, data) = Context::device_create_buffer(self, &device, device_data, desc);
        (buffer.into(), Box::new(data) as _)
    }

    fn device_create_texture(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &TextureDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (texture, data) = Context::device_create_texture(self, &device, device_data, desc);
        (texture.into(), Box::new(data) as _)
    }

    fn device_create_sampler(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &SamplerDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (sampler, data) = Context::device_create_sampler(self, &device, device_data, desc);
        (sampler.into(), Box::new(data) as _)
    }

    fn device_create_query_set(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &QuerySetDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (query_set, data) = Context::device_create_query_set(self, &device, device_data, desc);
        (query_set.into(), Box::new(data) as _)
    }

    fn device_create_command_encoder(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &CommandEncoderDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (command_encoder, data) =
            Context::device_create_command_encoder(self, &device, device_data, desc);
        (command_encoder.into(), Box::new(data) as _)
    }

    fn device_create_render_bundle_encoder(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        desc: &RenderBundleEncoderDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        let (render_bundle_encoder, data) =
            Context::device_create_render_bundle_encoder(self, &device, device_data, desc);
        (render_bundle_encoder.into(), Box::new(data) as _)
    }

    fn device_drop(&self, device: &ObjectId, device_data: &crate::Data) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_drop(self, &device, device_data)
    }

    fn device_poll(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        maintain: Maintain,
    ) -> bool {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_poll(self, &device, device_data, maintain)
    }

    fn device_on_uncaptured_error(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_on_uncaptured_error(self, &device, device_data, handler)
    }

    fn device_push_error_scope(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
        filter: ErrorFilter,
    ) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_push_error_scope(self, &device, device_data, filter)
    }

    fn device_pop_error_scope(
        &self,
        device: &ObjectId,
        device_data: &crate::Data,
    ) -> Pin<Box<dyn Future<Output = Option<Error>> + Send + 'static>> {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Box::pin(Context::device_pop_error_scope(self, &device, device_data))
    }

    fn buffer_map_async(
        &self,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        mode: MapMode,
        range: Range<BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), BufferAsyncError>) + Send + 'static>,
    ) {
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_map_async(self, &buffer, buffer_data, mode, range, callback)
    }

    fn buffer_get_mapped_range(
        &self,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        sub_range: Range<BufferAddress>,
    ) -> Box<dyn BufferMappedRange> {
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_get_mapped_range(self, &buffer, buffer_data, sub_range)
    }

    fn buffer_unmap(&self, buffer: &ObjectId, buffer_data: &crate::Data) {
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_unmap(self, &buffer, buffer_data)
    }

    fn texture_create_view(
        &self,
        texture: &ObjectId,
        texture_data: &crate::Data,
        desc: &TextureViewDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let texture = <T::TextureId>::from(*texture);
        let texture_data = downcast_ref(texture_data);
        let (texture_view, data) = Context::texture_create_view(self, &texture, texture_data, desc);
        (texture_view.into(), Box::new(data) as _)
    }

    fn surface_drop(&self, surface: &ObjectId, surface_data: &crate::Data) {
        let surface = <T::SurfaceId>::from(*surface);
        let surface_data = downcast_ref(surface_data);
        Context::surface_drop(self, &surface, surface_data)
    }

    fn adapter_drop(&self, adapter: &ObjectId, adapter_data: &crate::Data) {
        let adapter = <T::AdapterId>::from(*adapter);
        let adapter_data = downcast_ref(adapter_data);
        Context::adapter_drop(self, &adapter, adapter_data)
    }

    fn buffer_destroy(&self, buffer: &ObjectId, buffer_data: &crate::Data) {
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_destroy(self, &buffer, buffer_data)
    }

    fn buffer_drop(&self, buffer: &ObjectId, buffer_data: &crate::Data) {
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::buffer_drop(self, &buffer, buffer_data)
    }

    fn texture_destroy(&self, texture: &ObjectId, texture_data: &crate::Data) {
        let texture = <T::TextureId>::from(*texture);
        let texture_data = downcast_ref(texture_data);
        Context::texture_destroy(self, &texture, texture_data)
    }

    fn texture_drop(&self, texture: &ObjectId, texture_data: &crate::Data) {
        let texture = <T::TextureId>::from(*texture);
        let texture_data = downcast_ref(texture_data);
        Context::texture_drop(self, &texture, texture_data)
    }

    fn texture_view_drop(&self, texture_view: &ObjectId, texture_view_data: &crate::Data) {
        let texture_view = <T::TextureViewId>::from(*texture_view);
        let texture_view_data = downcast_ref(texture_view_data);
        Context::texture_view_drop(self, &texture_view, texture_view_data)
    }

    fn sampler_drop(&self, sampler: &ObjectId, sampler_data: &crate::Data) {
        let sampler = <T::SamplerId>::from(*sampler);
        let sampler_data = downcast_ref(sampler_data);
        Context::sampler_drop(self, &sampler, sampler_data)
    }

    fn query_set_drop(&self, query_set: &ObjectId, query_set_data: &crate::Data) {
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::query_set_drop(self, &query_set, query_set_data)
    }

    fn bind_group_drop(&self, bind_group: &ObjectId, bind_group_data: &crate::Data) {
        let bind_group = <T::BindGroupId>::from(*bind_group);
        let bind_group_data = downcast_ref(bind_group_data);
        Context::bind_group_drop(self, &bind_group, bind_group_data)
    }

    fn bind_group_layout_drop(
        &self,
        bind_group_layout: &ObjectId,
        bind_group_layout_data: &crate::Data,
    ) {
        let bind_group_layout = <T::BindGroupLayoutId>::from(*bind_group_layout);
        let bind_group_layout_data = downcast_ref(bind_group_layout_data);
        Context::bind_group_layout_drop(self, &bind_group_layout, bind_group_layout_data)
    }

    fn pipeline_layout_drop(&self, pipeline_layout: &ObjectId, pipeline_layout_data: &crate::Data) {
        let pipeline_layout = <T::PipelineLayoutId>::from(*pipeline_layout);
        let pipeline_layout_data = downcast_ref(pipeline_layout_data);
        Context::pipeline_layout_drop(self, &pipeline_layout, pipeline_layout_data)
    }

    fn shader_module_drop(&self, shader_module: &ObjectId, shader_module_data: &crate::Data) {
        let shader_module = <T::ShaderModuleId>::from(*shader_module);
        let shader_module_data = downcast_ref(shader_module_data);
        Context::shader_module_drop(self, &shader_module, shader_module_data)
    }

    fn command_encoder_drop(&self, command_encoder: &ObjectId, command_encoder_data: &crate::Data) {
        let command_encoder = <T::CommandEncoderId>::from(*command_encoder);
        let command_encoder_data = downcast_ref(command_encoder_data);
        Context::command_encoder_drop(self, &command_encoder, command_encoder_data)
    }

    fn command_buffer_drop(&self, command_buffer: &ObjectId, command_buffer_data: &crate::Data) {
        let command_buffer = <T::CommandBufferId>::from(*command_buffer);
        let command_buffer_data = downcast_ref(command_buffer_data);
        Context::command_buffer_drop(self, &command_buffer, command_buffer_data)
    }

    fn render_bundle_drop(&self, render_bundle: &ObjectId, render_bundle_data: &crate::Data) {
        let render_bundle = <T::RenderBundleId>::from(*render_bundle);
        let render_bundle_data = downcast_ref(render_bundle_data);
        Context::render_bundle_drop(self, &render_bundle, render_bundle_data)
    }

    fn compute_pipeline_drop(&self, pipeline: &ObjectId, pipeline_data: &crate::Data) {
        let pipeline = <T::ComputePipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::compute_pipeline_drop(self, &pipeline, pipeline_data)
    }

    fn render_pipeline_drop(&self, pipeline: &ObjectId, pipeline_data: &crate::Data) {
        let pipeline = <T::RenderPipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_pipeline_drop(self, &pipeline, pipeline_data)
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> (ObjectId, Box<crate::Data>) {
        let pipeline = <T::ComputePipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        let (bind_group_layout, data) =
            Context::compute_pipeline_get_bind_group_layout(self, &pipeline, pipeline_data, index);
        (bind_group_layout.into(), Box::new(data) as _)
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
        index: u32,
    ) -> (ObjectId, Box<crate::Data>) {
        let pipeline = <T::RenderPipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        let (bind_group_layout, data) =
            Context::render_pipeline_get_bind_group_layout(self, &pipeline, pipeline_data, index);
        (bind_group_layout.into(), Box::new(data) as _)
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: &ObjectId,
        source_data: &crate::Data,
        source_offset: BufferAddress,
        destination: &ObjectId,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let source = <T::BufferId>::from(*source);
        let source_data = downcast_ref(source_data);
        let destination = <T::BufferId>::from(*destination);
        let destination_data = downcast_ref(destination_data);
        Context::command_encoder_copy_buffer_to_buffer(
            self,
            &encoder,
            encoder_data,
            &source,
            source_data,
            source_offset,
            &destination,
            destination_data,
            destination_offset,
            copy_size,
        )
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_buffer_to_texture(
            self,
            &encoder,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_texture_to_buffer(
            self,
            &encoder,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_copy_texture_to_texture(
            self,
            &encoder,
            encoder_data,
            source,
            destination,
            copy_size,
        )
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        desc: &ComputePassDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let (compute_pass, data) =
            Context::command_encoder_begin_compute_pass(self, &encoder, encoder_data, desc);
        (compute_pass.into(), Box::new(data) as _)
    }

    fn command_encoder_end_compute_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut(pass_data);
        Context::command_encoder_end_compute_pass(
            self,
            &encoder,
            encoder_data,
            &mut pass,
            pass_data,
        )
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        desc: &RenderPassDescriptor<'_, '_>,
    ) -> (ObjectId, Box<crate::Data>) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let (render_pass, data) =
            Context::command_encoder_begin_render_pass(self, &encoder, encoder_data, desc);
        (render_pass.into(), Box::new(data) as _)
    }

    fn command_encoder_end_render_pass(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut(pass_data);
        Context::command_encoder_end_render_pass(self, &encoder, encoder_data, &mut pass, pass_data)
    }

    fn command_encoder_finish(
        &self,
        encoder: ObjectId,
        encoder_data: &mut crate::Data,
    ) -> (ObjectId, Box<crate::Data>) {
        let (command_buffer, data) =
            Context::command_encoder_finish(self, encoder.into(), downcast_mut(encoder_data));
        (command_buffer.into(), Box::new(data) as _)
    }

    fn command_encoder_clear_texture(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        texture: &Texture,
        subresource_range: &ImageSubresourceRange,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_clear_texture(
            self,
            &encoder,
            encoder_data,
            texture,
            subresource_range,
        )
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_clear_buffer(self, &encoder, encoder_data, buffer, offset, size)
    }

    fn command_encoder_insert_debug_marker(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        label: &str,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_insert_debug_marker(self, &encoder, encoder_data, label)
    }

    fn command_encoder_push_debug_group(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        label: &str,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_push_debug_group(self, &encoder, encoder_data, label)
    }

    fn command_encoder_pop_debug_group(&self, encoder: &ObjectId, encoder_data: &crate::Data) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        Context::command_encoder_pop_debug_group(self, &encoder, encoder_data)
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::command_encoder_write_timestamp(
            self,
            &encoder,
            encoder_data,
            &query_set,
            query_set_data,
            query_index,
        )
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder: &ObjectId,
        encoder_data: &crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        first_query: u32,
        query_count: u32,
        destination: &ObjectId,
        destination_data: &crate::Data,
        destination_offset: BufferAddress,
    ) {
        let encoder = <T::CommandEncoderId>::from(*encoder);
        let encoder_data = downcast_ref(encoder_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        let destination = <T::BufferId>::from(*destination);
        let destination_data = downcast_ref(destination_data);
        Context::command_encoder_resolve_query_set(
            self,
            &encoder,
            encoder_data,
            &query_set,
            query_set_data,
            first_query,
            query_count,
            &destination,
            destination_data,
            destination_offset,
        )
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder: ObjectId,
        encoder_data: Box<crate::Data>,
        desc: &RenderBundleDescriptor,
    ) -> (ObjectId, Box<crate::Data>) {
        let encoder_data = *encoder_data.downcast().unwrap();
        let (render_bundle, data) =
            Context::render_bundle_encoder_finish(self, encoder.into(), encoder_data, desc);
        (render_bundle.into(), Box::new(data) as _)
    }

    fn queue_write_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        data: &[u8],
    ) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_write_buffer(self, &queue, queue_data, &buffer, buffer_data, offset, data)
    }

    fn queue_validate_write_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_validate_write_buffer(
            self,
            &queue,
            queue_data,
            &buffer,
            buffer_data,
            offset,
            size,
        )
    }

    fn queue_create_staging_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        size: BufferSize,
    ) -> Option<Box<dyn QueueWriteBuffer>> {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        Context::queue_create_staging_buffer(self, &queue, queue_data, size)
    }

    fn queue_write_staging_buffer(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        staging_buffer: &dyn QueueWriteBuffer,
    ) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::queue_write_staging_buffer(
            self,
            &queue,
            queue_data,
            &buffer,
            buffer_data,
            offset,
            staging_buffer,
        )
    }

    fn queue_write_texture(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    ) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        Context::queue_write_texture(self, &queue, queue_data, texture, data, data_layout, size)
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    ) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        Context::queue_copy_external_image_to_texture(self, &queue, queue_data, source, dest, size)
    }

    fn queue_submit<'a>(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        command_buffers: Box<dyn Iterator<Item = ObjectId> + 'a>,
    ) -> (ObjectId, Arc<crate::Data>) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        let command_buffers = command_buffers.into_iter().map(<T::CommandBufferId>::from);
        let (submission_index, data) =
            Context::queue_submit(self, &queue, queue_data, command_buffers);
        (submission_index.into(), Arc::new(data) as _)
    }

    fn queue_get_timestamp_period(&self, queue: &ObjectId, queue_data: &crate::Data) -> f32 {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        Context::queue_get_timestamp_period(self, &queue, queue_data)
    }

    fn queue_on_submitted_work_done(
        &self,
        queue: &ObjectId,
        queue_data: &crate::Data,
        callback: Box<dyn FnOnce() + Send + 'static>,
    ) {
        let queue = <T::QueueId>::from(*queue);
        let queue_data = downcast_ref(queue_data);
        Context::queue_on_submitted_work_done(self, &queue, queue_data, callback)
    }

    fn device_start_capture(&self, device: &ObjectId, device_data: &crate::Data) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_start_capture(self, &device, device_data)
    }

    fn device_stop_capture(&self, device: &ObjectId, device_data: &crate::Data) {
        let device = <T::DeviceId>::from(*device);
        let device_data = downcast_ref(device_data);
        Context::device_stop_capture(self, &device, device_data)
    }

    fn compute_pass_set_pipeline(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let pipeline = <T::ComputePipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::compute_pass_set_pipeline(self, &mut pass, pass_data, &pipeline, pipeline_data)
    }

    fn compute_pass_set_bind_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let bind_group = <T::BindGroupId>::from(*bind_group);
        let bind_group_data = downcast_ref(bind_group_data);
        Context::compute_pass_set_bind_group(
            self,
            &mut pass,
            pass_data,
            index,
            &bind_group,
            bind_group_data,
            offsets,
        )
    }

    fn compute_pass_set_push_constants(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        offset: u32,
        data: &[u8],
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_set_push_constants(self, &mut pass, pass_data, offset, data)
    }

    fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        label: &str,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_insert_debug_marker(self, &mut pass, pass_data, label)
    }

    fn compute_pass_push_debug_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        group_label: &str,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_push_debug_group(self, &mut pass, pass_data, group_label)
    }

    fn compute_pass_pop_debug_group(&self, pass: &mut ObjectId, pass_data: &mut crate::Data) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_pop_debug_group(self, &mut pass, pass_data)
    }

    fn compute_pass_write_timestamp(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::compute_pass_write_timestamp(
            self,
            &mut pass,
            pass_data,
            &query_set,
            query_set_data,
            query_index,
        )
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::compute_pass_begin_pipeline_statistics_query(
            self,
            &mut pass,
            pass_data,
            &query_set,
            query_set_data,
            query_index,
        )
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_end_pipeline_statistics_query(self, &mut pass, pass_data)
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        z: u32,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        Context::compute_pass_dispatch_workgroups(self, &mut pass, pass_data, x, y, z)
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let mut pass = <T::ComputePassId>::from(*pass);
        let pass_data = downcast_mut::<T::ComputePassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::compute_pass_dispatch_workgroups_indirect(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let pipeline = <T::RenderPipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_bundle_encoder_set_pipeline(
            self,
            &mut encoder,
            encoder_data,
            &pipeline,
            pipeline_data,
        )
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let bind_group = <T::BindGroupId>::from(*bind_group);
        let bind_group_data = downcast_ref(bind_group_data);
        Context::render_bundle_encoder_set_bind_group(
            self,
            &mut encoder,
            encoder_data,
            index,
            &bind_group,
            bind_group_data,
            offsets,
        )
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_bundle_encoder_set_index_buffer(
            self,
            &mut encoder,
            encoder_data,
            &buffer,
            buffer_data,
            index_format,
            offset,
            size,
        )
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        slot: u32,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_bundle_encoder_set_vertex_buffer(
            self,
            &mut encoder,
            encoder_data,
            slot,
            &buffer,
            buffer_data,
            offset,
            size,
        )
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_set_push_constants(
            self,
            &mut encoder,
            encoder_data,
            stages,
            offset,
            data,
        )
    }

    fn render_bundle_encoder_draw(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_draw(self, &mut encoder, encoder_data, vertices, instances)
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        Context::render_bundle_encoder_draw_indexed(
            self,
            &mut encoder,
            encoder_data,
            indices,
            base_vertex,
            instances,
        )
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_draw_indirect(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_draw_indexed_indirect(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_multi_draw_indirect(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_bundle_encoder_multi_draw_indexed_indirect(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer = <T::BufferId>::from(*count_buffer);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_bundle_encoder_multi_draw_indirect_count(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            &count_buffer,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        encoder: &mut ObjectId,
        encoder_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let mut encoder = <T::RenderBundleEncoderId>::from(*encoder);
        let encoder_data = downcast_mut::<T::RenderBundleEncoderData>(encoder_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer = <T::BufferId>::from(*count_buffer);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_bundle_encoder_multi_draw_indexed_indirect_count(
            self,
            &mut encoder,
            encoder_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            &count_buffer,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_pipeline(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        pipeline: &ObjectId,
        pipeline_data: &crate::Data,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let pipeline = <T::RenderPipelineId>::from(*pipeline);
        let pipeline_data = downcast_ref(pipeline_data);
        Context::render_pass_set_pipeline(self, &mut pass, pass_data, &pipeline, pipeline_data)
    }

    fn render_pass_set_bind_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        index: u32,
        bind_group: &ObjectId,
        bind_group_data: &crate::Data,
        offsets: &[DynamicOffset],
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let bind_group = <T::BindGroupId>::from(*bind_group);
        let bind_group_data = downcast_ref(bind_group_data);
        Context::render_pass_set_bind_group(
            self,
            &mut pass,
            pass_data,
            index,
            &bind_group,
            bind_group_data,
            offsets,
        )
    }

    fn render_pass_set_index_buffer(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_pass_set_index_buffer(
            self,
            &mut pass,
            pass_data,
            &buffer,
            buffer_data,
            index_format,
            offset,
            size,
        )
    }

    fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        slot: u32,
        buffer: &ObjectId,
        buffer_data: &crate::Data,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let buffer = <T::BufferId>::from(*buffer);
        let buffer_data = downcast_ref(buffer_data);
        Context::render_pass_set_vertex_buffer(
            self,
            &mut pass,
            pass_data,
            slot,
            &buffer,
            buffer_data,
            offset,
            size,
        )
    }

    fn render_pass_set_push_constants(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        stages: ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_push_constants(self, &mut pass, pass_data, stages, offset, data)
    }

    fn render_pass_draw(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_draw(self, &mut pass, pass_data, vertices, instances)
    }

    fn render_pass_draw_indexed(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_draw_indexed(
            self,
            &mut pass,
            pass_data,
            indices,
            base_vertex,
            instances,
        )
    }

    fn render_pass_draw_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_draw_indirect(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_draw_indexed_indirect(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
        )
    }

    fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_multi_draw_indirect(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        Context::render_pass_multi_draw_indexed_indirect(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            count,
        )
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer = <T::BufferId>::from(*count_buffer);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_pass_multi_draw_indirect_count(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            &count_buffer,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        indirect_buffer: &ObjectId,
        indirect_buffer_data: &crate::Data,
        indirect_offset: BufferAddress,
        count_buffer: &ObjectId,
        count_buffer_data: &crate::Data,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let indirect_buffer = <T::BufferId>::from(*indirect_buffer);
        let indirect_buffer_data = downcast_ref(indirect_buffer_data);
        let count_buffer = <T::BufferId>::from(*count_buffer);
        let count_buffer_data = downcast_ref(count_buffer_data);
        Context::render_pass_multi_draw_indexed_indirect_count(
            self,
            &mut pass,
            pass_data,
            &indirect_buffer,
            indirect_buffer_data,
            indirect_offset,
            &count_buffer,
            count_buffer_data,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_blend_constant(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        color: Color,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_blend_constant(self, &mut pass, pass_data, color)
    }

    fn render_pass_set_scissor_rect(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_scissor_rect(self, &mut pass, pass_data, x, y, width, height)
    }

    fn render_pass_set_viewport(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_viewport(
            self, &mut pass, pass_data, x, y, width, height, min_depth, max_depth,
        )
    }

    fn render_pass_set_stencil_reference(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        reference: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_set_stencil_reference(self, &mut pass, pass_data, reference)
    }

    fn render_pass_insert_debug_marker(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        label: &str,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_insert_debug_marker(self, &mut pass, pass_data, label)
    }

    fn render_pass_push_debug_group(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        group_label: &str,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_push_debug_group(self, &mut pass, pass_data, group_label)
    }

    fn render_pass_pop_debug_group(&self, pass: &mut ObjectId, pass_data: &mut crate::Data) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_pop_debug_group(self, &mut pass, pass_data)
    }

    fn render_pass_write_timestamp(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::render_pass_write_timestamp(
            self,
            &mut pass,
            pass_data,
            &query_set,
            query_set_data,
            query_index,
        )
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        query_set: &ObjectId,
        query_set_data: &crate::Data,
        query_index: u32,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let query_set = <T::QuerySetId>::from(*query_set);
        let query_set_data = downcast_ref(query_set_data);
        Context::render_pass_begin_pipeline_statistics_query(
            self,
            &mut pass,
            pass_data,
            &query_set,
            query_set_data,
            query_index,
        )
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        Context::render_pass_end_pipeline_statistics_query(self, &mut pass, pass_data)
    }

    fn render_pass_execute_bundles<'a>(
        &self,
        pass: &mut ObjectId,
        pass_data: &mut crate::Data,
        render_bundles: Box<dyn Iterator<Item = &'a ObjectId> + 'a>,
    ) {
        let mut pass = <T::RenderPassId>::from(*pass);
        let pass_data = downcast_mut::<T::RenderPassData>(pass_data);
        let render_bundles = Box::new(
            render_bundles
                .into_iter()
                .map(|id| <T::RenderBundleId>::from(*id)),
        );
        Context::render_pass_execute_bundles(self, &mut pass, pass_data, render_bundles)
    }
}

pub trait QueueWriteBuffer: Send + Sync {
    fn slice(&self) -> &[u8];

    fn slice_mut(&mut self) -> &mut [u8];

    fn as_any(&self) -> &dyn Any;
}

pub trait BufferMappedRange: Debug {
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
