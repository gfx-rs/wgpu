use std::future::Pending;

use crate::{
    context::Unused, Buffer, BufferDescriptor, Error, RequestDeviceError, Surface,
    TextureDescriptor,
};

#[derive(Debug)]
pub(crate) struct Context;

impl crate::Context for Context {
    type AdapterId = Unused;
    type AdapterData = ();
    type DeviceId = Unused;
    type DeviceData = ();
    type QueueId = Unused;
    type QueueData = ();
    type ShaderModuleId = Unused;
    type ShaderModuleData = ();
    type BindGroupLayoutId = Unused;
    type BindGroupLayoutData = ();
    type BindGroupId = Unused;
    type BindGroupData = ();
    type TextureViewId = Unused;
    type TextureViewData = ();
    type SamplerId = Unused;
    type SamplerData = ();
    type BufferId = Unused;
    type BufferData = ();
    type TextureId = Unused;
    type TextureData = ();
    type QuerySetId = Unused;
    type QuerySetData = ();
    type PipelineLayoutId = Unused;
    type PipelineLayoutData = ();
    type RenderPipelineId = Unused;
    type RenderPipelineData = ();
    type ComputePipelineId = Unused;
    type ComputePipelineData = ();
    type CommandEncoderId = Unused;
    type CommandEncoderData = ();
    type ComputePassId = Unused;
    type ComputePassData = ();
    type RenderPassId = Unused;
    type RenderPassData = ();
    type CommandBufferId = Unused;
    type CommandBufferData = ();
    type RenderBundleEncoderId = Unused;
    type RenderBundleEncoderData = ();
    type RenderBundleId = Unused;
    type RenderBundleData = ();
    type SurfaceId = Unused;
    type SurfaceData = ();
    type SurfaceOutputDetail = ();
    type SubmissionIndex = Unused;
    type SubmissionIndexData = ();
    type RequestAdapterFuture = Pending<Option<(Self::AdapterId, Self::AdapterData)>>;
    type RequestDeviceFuture = Pending<
        Result<
            (
                Self::DeviceId,
                Self::DeviceData,
                Self::QueueId,
                Self::QueueData,
            ),
            RequestDeviceError,
        >,
    >;
    type PopErrorScopeFuture = Pending<Option<Error>>;

    fn init(_instance_desc: wgt::InstanceDescriptor) -> Self {
        unimplemented!("no backend enabled")
    }

    unsafe fn instance_create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        _window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError> {
        unimplemented!("no backend enabled")
    }

    fn instance_request_adapter(
        &self,
        _options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture {
        unimplemented!("no backend enabled")
    }

    fn adapter_request_device(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _desc: &crate::DeviceDescriptor<'_>,
        _trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        unimplemented!("no backend enabled")
    }

    fn instance_poll_all_devices(&self, _force_wait: bool) -> bool {
        unimplemented!("no backend enabled")
    }

    fn adapter_is_surface_supported(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> bool {
        unimplemented!("no backend enabled")
    }

    fn adapter_features(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::Features {
        unimplemented!("no backend enabled")
    }

    fn adapter_limits(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::Limits {
        unimplemented!("no backend enabled")
    }

    fn adapter_downlevel_capabilities(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::DownlevelCapabilities {
        unimplemented!("no backend enabled")
    }

    fn adapter_get_info(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::AdapterInfo {
        unimplemented!("no backend enabled")
    }

    fn adapter_get_texture_format_features(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        unimplemented!("no backend enabled")
    }

    fn adapter_get_presentation_timestamp(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        unimplemented!("no backend enabled")
    }

    fn surface_get_capabilities(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        unimplemented!("no backend enabled")
    }

    fn surface_configure(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _config: &crate::SurfaceConfiguration,
    ) {
        unimplemented!("no backend enabled")
    }

    fn surface_get_current_texture(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureId>,
        Option<Self::TextureData>,
        wgt::SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        unimplemented!("no backend enabled")
    }

    fn surface_present(&self, _texture: &Self::TextureId, _detail: &Self::SurfaceOutputDetail) {
        unimplemented!("no backend enabled")
    }

    fn surface_texture_discard(
        &self,
        _texture: &Self::TextureId,
        _detail: &Self::SurfaceOutputDetail,
    ) {
        unimplemented!("no backend enabled")
    }

    fn device_features(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::Features {
        unimplemented!("no backend enabled")
    }

    fn device_limits(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::Limits {
        unimplemented!("no backend enabled")
    }

    fn device_downlevel_properties(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::DownlevelCapabilities {
        unimplemented!("no backend enabled")
    }

    fn device_create_shader_module(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: crate::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        unimplemented!("no backend enabled")
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::ShaderModuleDescriptorSpirV<'_>,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_bind_group_layout(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::BindGroupLayoutDescriptor<'_>,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_bind_group(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::BindGroupDescriptor<'_>,
    ) -> (Self::BindGroupId, Self::BindGroupData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_pipeline_layout(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::PipelineLayoutDescriptor<'_>,
    ) -> (Self::PipelineLayoutId, Self::PipelineLayoutData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_render_pipeline(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::RenderPipelineDescriptor<'_>,
    ) -> (Self::RenderPipelineId, Self::RenderPipelineData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_compute_pipeline(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::ComputePipelineDescriptor<'_>,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_buffer(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::BufferDescriptor<'_>,
    ) -> (Self::BufferId, Self::BufferData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_texture(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::TextureDescriptor<'_>,
    ) -> (Self::TextureId, Self::TextureData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_sampler(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::SamplerDescriptor<'_>,
    ) -> (Self::SamplerId, Self::SamplerData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_query_set(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::QuerySetDescriptor<'_>,
    ) -> (Self::QuerySetId, Self::QuerySetData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_command_encoder(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::CommandEncoderDescriptor<'_>,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData) {
        unimplemented!("no backend enabled")
    }

    fn device_create_render_bundle_encoder(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::RenderBundleEncoderDescriptor<'_>,
    ) -> (Self::RenderBundleEncoderId, Self::RenderBundleEncoderData) {
        unimplemented!("no backend enabled")
    }

    fn device_drop(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        unimplemented!("no backend enabled")
    }

    fn device_set_device_lost_callback(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _device_lost_callback: crate::context::DeviceLostCallback,
    ) {
        unimplemented!("no backend enabled")
    }

    fn device_destroy(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        unimplemented!("no backend enabled")
    }

    fn device_mark_lost(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _message: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn queue_drop(&self, _queue: &Self::QueueId, _queue_data: &Self::QueueData) {
        unimplemented!("no backend enabled")
    }

    fn device_poll(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _maintain: crate::Maintain,
    ) -> bool {
        unimplemented!("no backend enabled")
    }

    fn device_on_uncaptured_error(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _handler: Box<dyn crate::UncapturedErrorHandler>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn device_push_error_scope(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _filter: crate::ErrorFilter,
    ) {
        unimplemented!("no backend enabled")
    }

    fn device_pop_error_scope(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> Self::PopErrorScopeFuture {
        unimplemented!("no backend enabled")
    }

    fn buffer_map_async(
        &self,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _mode: crate::MapMode,
        _range: std::ops::Range<wgt::BufferAddress>,
        _callback: crate::context::BufferMapCallback,
    ) {
        unimplemented!("no backend enabled")
    }

    fn buffer_get_mapped_range(
        &self,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _sub_range: std::ops::Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        unimplemented!("no backend enabled")
    }

    fn buffer_unmap(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        unimplemented!("no backend enabled")
    }

    fn texture_create_view(
        &self,
        _texture: &Self::TextureId,
        _texture_data: &Self::TextureData,
        _desc: &crate::TextureViewDescriptor<'_>,
    ) -> (Self::TextureViewId, Self::TextureViewData) {
        unimplemented!("no backend enabled")
    }

    fn surface_drop(&self, _surface: &Self::SurfaceId, _surface_data: &Self::SurfaceData) {
        unimplemented!("no backend enabled")
    }

    fn adapter_drop(&self, _adapter: &Self::AdapterId, _adapter_data: &Self::AdapterData) {
        unimplemented!("no backend enabled")
    }

    fn buffer_destroy(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        unimplemented!("no backend enabled")
    }

    fn buffer_drop(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        unimplemented!("no backend enabled")
    }

    fn texture_destroy(&self, _texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        unimplemented!("no backend enabled")
    }

    fn texture_drop(&self, _texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        unimplemented!("no backend enabled")
    }

    fn texture_view_drop(
        &self,
        _texture_view: &Self::TextureViewId,
        _texture_view_data: &Self::TextureViewData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn sampler_drop(&self, _sampler: &Self::SamplerId, _sampler_data: &Self::SamplerData) {
        unimplemented!("no backend enabled")
    }

    fn query_set_drop(&self, _query_set: &Self::QuerySetId, _query_set_data: &Self::QuerySetData) {
        unimplemented!("no backend enabled")
    }

    fn bind_group_drop(
        &self,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn bind_group_layout_drop(
        &self,
        _bind_group_layout: &Self::BindGroupLayoutId,
        _bind_group_layout_data: &Self::BindGroupLayoutData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn pipeline_layout_drop(
        &self,
        _pipeline_layout: &Self::PipelineLayoutId,
        _pipeline_layout_data: &Self::PipelineLayoutData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn shader_module_drop(
        &self,
        _shader_module: &Self::ShaderModuleId,
        _shader_module_data: &Self::ShaderModuleData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_drop(
        &self,
        _command_encoder: &Self::CommandEncoderId,
        _command_encoder_data: &Self::CommandEncoderData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_buffer_drop(
        &self,
        _command_buffer: &Self::CommandBufferId,
        _command_buffer_data: &Self::CommandBufferData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_drop(
        &self,
        _render_bundle: &Self::RenderBundleId,
        _render_bundle_data: &Self::RenderBundleData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pipeline_drop(
        &self,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pipeline_drop(
        &self,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
        _index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        unimplemented!("no backend enabled")
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
        _index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: &Self::BufferId,
        _source_data: &Self::BufferData,
        _source_offset: wgt::BufferAddress,
        _destination: &Self::BufferId,
        _destination_data: &Self::BufferData,
        _destination_offset: wgt::BufferAddress,
        _copy_size: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyBuffer<'_>,
        _destination: crate::ImageCopyTexture<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyTexture<'_>,
        _destination: crate::ImageCopyBuffer<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyTexture<'_>,
        _destination: crate::ImageCopyTexture<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_begin_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _desc: &crate::ComputePassDescriptor<'_>,
    ) -> (Self::ComputePassId, Self::ComputePassData) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_end_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_begin_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _desc: &crate::RenderPassDescriptor<'_, '_>,
    ) -> (Self::RenderPassId, Self::RenderPassData) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_end_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_finish(
        &self,
        _encoder: Self::CommandEncoderId,
        _encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_clear_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _texture: &crate::Texture, // TODO: Decompose?
        _subresource_range: &wgt::ImageSubresourceRange,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_clear_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _buffer: &crate::Buffer,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferAddress>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_insert_debug_marker(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_push_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_pop_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_write_timestamp(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn command_encoder_resolve_query_set(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _first_query: u32,
        _query_count: u32,
        _destination: &Self::BufferId,
        _destination_data: &Self::BufferData,
        _destination_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_finish(
        &self,
        _encoder: Self::RenderBundleEncoderId,
        _encoder_data: Self::RenderBundleEncoderData,
        _desc: &crate::RenderBundleDescriptor<'_>,
    ) -> (Self::RenderBundleId, Self::RenderBundleData) {
        unimplemented!("no backend enabled")
    }

    fn queue_write_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _offset: wgt::BufferAddress,
        _data: &[u8],
    ) {
        unimplemented!("no backend enabled")
    }

    fn queue_validate_write_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _offset: wgt::BufferAddress,
        _size: wgt::BufferSize,
    ) -> Option<()> {
        unimplemented!("no backend enabled")
    }

    fn queue_create_staging_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _size: wgt::BufferSize,
    ) -> Option<Box<dyn crate::context::QueueWriteBuffer>> {
        unimplemented!("no backend enabled")
    }

    fn queue_write_staging_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _offset: wgt::BufferAddress,
        _staging_buffer: &dyn crate::context::QueueWriteBuffer,
    ) {
        unimplemented!("no backend enabled")
    }

    fn queue_write_texture(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _texture: crate::ImageCopyTexture<'_>,
        _data: &[u8],
        _data_layout: wgt::ImageDataLayout,
        _size: wgt::Extent3d,
    ) {
        unimplemented!("no backend enabled")
    }

    fn queue_submit<I: Iterator<Item = (Self::CommandBufferId, Self::CommandBufferData)>>(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _command_buffers: I,
    ) -> (Self::SubmissionIndex, Self::SubmissionIndexData) {
        unimplemented!("no backend enabled")
    }

    fn queue_get_timestamp_period(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
    ) -> f32 {
        unimplemented!("no backend enabled")
    }

    fn queue_on_submitted_work_done(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _callback: crate::context::SubmittedWorkDoneCallback,
    ) {
        unimplemented!("no backend enabled")
    }

    fn device_start_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        unimplemented!("no backend enabled")
    }

    fn device_stop_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_set_pipeline(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_set_bind_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _index: u32,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
        _offsets: &[wgt::DynamicOffset],
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _offset: u32,
        _data: &[u8],
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _group_label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_pop_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _x: u32,
        _y: u32,
        _z: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _index: u32,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
        _offsets: &[wgt::DynamicOffset],
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _index_format: wgt::IndexFormat,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferSize>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _slot: u32,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferSize>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_draw(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _vertices: std::ops::Range<u32>,
        _instances: std::ops::Range<u32>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indices: std::ops::Range<u32>,
        _base_vertex: i32,
        _instances: std::ops::Range<u32>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_pipeline(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_bind_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _index: u32,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
        _offsets: &[wgt::DynamicOffset],
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_index_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _index_format: wgt::IndexFormat,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferSize>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_vertex_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _slot: u32,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferSize>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_push_constants(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_draw(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _vertices: std::ops::Range<u32>,
        _instances: std::ops::Range<u32>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_draw_indexed(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indices: std::ops::Range<u32>,
        _base_vertex: i32,
        _instances: std::ops::Range<u32>,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
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
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_blend_constant(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _color: wgt::Color,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_scissor_rect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_viewport(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _x: f32,
        _y: f32,
        _width: f32,
        _height: f32,
        _min_depth: f32,
        _max_depth: f32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_set_stencil_reference(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _reference: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_push_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _group_label: &str,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_pop_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_write_timestamp(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_begin_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_end_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        unimplemented!("no backend enabled")
    }

    fn render_pass_execute_bundles<'a>(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _render_bundles: Box<
            dyn Iterator<Item = (Self::RenderBundleId, &'a Self::RenderBundleData)> + 'a,
        >,
    ) {
        unimplemented!("no backend enabled")
    }
}

impl Context {
    pub unsafe fn from_hal_instance<A: wgc::hal_api::HalApi>(_hal_instance: A::Instance) -> Self {
        unimplemented!("no backend enabled")
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: wgc::hal_api::HalApi>(&self) -> Option<&A::Instance> {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn from_core_instance(_core_instance: wgc::instance::Instance) -> Self {
        unimplemented!("no backend enabled")
    }

    pub fn enumerate_adapters(&self, _backends: wgt::Backends) -> Vec<Unused> {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        _hal_adapter: hal::ExposedAdapter<A>,
    ) -> Unused {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn adapter_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Adapter>) -> R,
        R,
    >(
        &self,
        _adapter: Unused,
        _hal_adapter_callback: F,
    ) -> R {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn create_device_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        _adapter: &Unused,
        _hal_device: hal::OpenDevice<A>,
        _desc: &crate::DeviceDescriptor<'_>,
        _trace_dir: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), crate::RequestDeviceError> {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn create_texture_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        _hal_texture: A::Texture,
        _device: &Device,
        _desc: &TextureDescriptor<'_>,
    ) -> Texture {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn create_buffer_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        _hal_buffer: A::Buffer,
        _device: &Device,
        _desc: &BufferDescriptor<'_>,
    ) -> (Unused, Buffer) {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn device_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        _device: &Device,
        _hal_device_callback: F,
    ) -> R {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn surface_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Surface>) -> R,
        R,
    >(
        &self,
        _surface: &Surface<'_>,
        _hal_surface_callback: F,
    ) -> R {
        unimplemented!("no backend enabled")
    }

    pub unsafe fn texture_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Texture>)>(
        &self,
        _texture: &Texture,
        _hal_texture_callback: F,
    ) {
        unimplemented!("no backend enabled")
    }

    pub fn generate_report(&self) -> wgc::global::GlobalReport {
        unimplemented!("no backend enabled")
    }
}

pub struct Device;

impl Device {
    pub fn id(&self) -> Unused {
        unimplemented!("no backend enabled")
    }
}

pub struct Queue;

impl Queue {
    pub fn id(&self) -> Unused {
        unimplemented!("no backend enabled")
    }
}

pub struct Texture;

impl Texture {
    pub fn id(&self) -> Unused {
        unimplemented!("no backend enabled")
    }
}
