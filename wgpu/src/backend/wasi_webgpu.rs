use crate::{
    AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferDescriptor,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelCapabilities, Features, Limits, MapMode, PipelineLayoutDescriptor,
    RenderBundleEncoderDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
    ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, SurfaceStatus, SurfaceTargetUnsafe,
    TextureDescriptor, TextureViewDescriptor, UncapturedErrorHandler,
};

use std::{future::Future, marker::PhantomData, ops::Range};

wit_bindgen::generate!({
    path: "wit",
    world: "component:webgpu/example",
    exports: {
        world: ExampleTriangle,
    },
});

struct ExampleTriangle;

// TODO: get rid of this.
impl Guest for ExampleTriangle {
    fn start() {}
}

// TODO: get rid of this type.
#[derive(Debug, Clone, Copy)]
pub struct PlaceholderType;
impl From<crate::context::ObjectId> for PlaceholderType {
    fn from(_value: crate::context::ObjectId) -> Self {
        todo!()
    }
}
impl From<PlaceholderType> for crate::context::ObjectId {
    fn from(_value: PlaceholderType) -> Self {
        todo!()
    }
}

// TODO: get rid of this type.
#[derive(Debug, Clone, Copy)]
pub struct PlaceholderFuture<T>(PhantomData<T>);

impl<T> Future for PlaceholderFuture<T> {
    type Output = T;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        todo!()
    }
}

#[derive(Debug)]
pub struct ContextWasiWebgpu(());

impl crate::Context for ContextWasiWebgpu {
    type AdapterId = PlaceholderType;
    type AdapterData = PlaceholderType;
    type DeviceId = PlaceholderType;
    type DeviceData = PlaceholderType;
    type QueueId = PlaceholderType;
    type QueueData = PlaceholderType;
    type ShaderModuleId = PlaceholderType;
    type ShaderModuleData = PlaceholderType;
    type BindGroupLayoutId = PlaceholderType;
    type BindGroupLayoutData = PlaceholderType;
    type BindGroupId = PlaceholderType;
    type BindGroupData = PlaceholderType;
    type TextureViewId = PlaceholderType;
    type TextureViewData = PlaceholderType;
    type SamplerId = PlaceholderType;
    type SamplerData = PlaceholderType;
    type BufferId = PlaceholderType;
    type BufferData = PlaceholderType;
    type TextureId = PlaceholderType;
    type TextureData = PlaceholderType;
    type QuerySetId = PlaceholderType;
    type QuerySetData = PlaceholderType;
    type PipelineLayoutId = PlaceholderType;
    type PipelineLayoutData = PlaceholderType;
    type RenderPipelineId = PlaceholderType;
    type RenderPipelineData = PlaceholderType;
    type ComputePipelineId = PlaceholderType;
    type ComputePipelineData = PlaceholderType;
    type CommandEncoderId = PlaceholderType;
    type CommandEncoderData = PlaceholderType;
    type ComputePassId = PlaceholderType;
    type ComputePassData = PlaceholderType;
    type RenderPassId = PlaceholderType;
    type RenderPassData = PlaceholderType;
    type CommandBufferId = PlaceholderType;
    type CommandBufferData = PlaceholderType;
    type RenderBundleEncoderId = PlaceholderType;
    type RenderBundleEncoderData = PlaceholderType;
    type RenderBundleId = PlaceholderType;
    type RenderBundleData = PlaceholderType;

    type SurfaceId = PlaceholderType;
    type SurfaceData = PlaceholderType;
    type SurfaceOutputDetail = PlaceholderType;
    type SubmissionIndex = PlaceholderType;
    type SubmissionIndexData = PlaceholderFuture<Option<()>>;

    type RequestAdapterFuture = PlaceholderFuture<Option<(PlaceholderType, PlaceholderType)>>;
    type RequestDeviceFuture = PlaceholderFuture<
        Result<
            (
                PlaceholderType,
                PlaceholderType,
                PlaceholderType,
                PlaceholderType,
            ),
            crate::RequestDeviceError,
        >,
    >;
    type PopErrorScopeFuture = PlaceholderFuture<Option<crate::Error>>;

    fn init(_instance_desc: wgt::InstanceDescriptor) -> Self {
        todo!()
    }

    unsafe fn instance_create_surface(
        &self,
        _target: SurfaceTargetUnsafe,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError> {
        todo!()
    }

    fn instance_request_adapter(
        &self,
        _options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture {
        todo!()
    }

    fn adapter_request_device(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _desc: &crate::DeviceDescriptor<'_>,
        _trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        todo!()
    }

    fn instance_poll_all_devices(&self, _force_wait: bool) -> bool {
        todo!()
    }

    fn adapter_is_surface_supported(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> bool {
        todo!()
    }

    fn adapter_features(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Features {
        todo!()
    }

    fn adapter_limits(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Limits {
        todo!()
    }

    fn adapter_downlevel_capabilities(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities {
        todo!()
    }

    fn adapter_get_info(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> AdapterInfo {
        todo!()
    }

    fn adapter_get_texture_format_features(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        _format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        todo!()
    }

    fn adapter_get_presentation_timestamp(
        &self,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        todo!()
    }

    fn surface_get_capabilities(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        _adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        todo!()
    }

    fn surface_configure(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _config: &crate::SurfaceConfiguration,
    ) {
        todo!()
    }

    fn surface_get_current_texture(
        &self,
        _surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureId>,
        Option<Self::TextureData>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        todo!()
    }

    fn surface_present(&self, _texture: &Self::TextureId, _detail: &Self::SurfaceOutputDetail) {
        todo!()
    }

    fn surface_texture_discard(
        &self,
        _texture: &Self::TextureId,
        _detail: &Self::SurfaceOutputDetail,
    ) {
        todo!()
    }

    fn device_features(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> Features {
        todo!()
    }

    fn device_limits(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) -> Limits {
        todo!()
    }

    fn device_downlevel_properties(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> DownlevelCapabilities {
        todo!()
    }

    fn device_create_shader_module(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        todo!()
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        todo!()
    }

    fn device_create_bind_group_layout(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &BindGroupLayoutDescriptor<'_>,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        todo!()
    }

    fn device_create_bind_group(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &BindGroupDescriptor<'_>,
    ) -> (Self::BindGroupId, Self::BindGroupData) {
        todo!()
    }

    fn device_create_pipeline_layout(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &PipelineLayoutDescriptor<'_>,
    ) -> (Self::PipelineLayoutId, Self::PipelineLayoutData) {
        todo!()
    }

    fn device_create_render_pipeline(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &RenderPipelineDescriptor<'_>,
    ) -> (Self::RenderPipelineId, Self::RenderPipelineData) {
        todo!()
    }

    fn device_create_compute_pipeline(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &ComputePipelineDescriptor<'_>,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData) {
        todo!()
    }

    fn device_create_buffer(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &BufferDescriptor<'_>,
    ) -> (Self::BufferId, Self::BufferData) {
        todo!()
    }

    fn device_create_texture(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &TextureDescriptor<'_>,
    ) -> (Self::TextureId, Self::TextureData) {
        todo!()
    }

    fn device_create_sampler(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &SamplerDescriptor<'_>,
    ) -> (Self::SamplerId, Self::SamplerData) {
        todo!()
    }

    fn device_create_query_set(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &crate::QuerySetDescriptor<'_>,
    ) -> (Self::QuerySetId, Self::QuerySetData) {
        todo!()
    }

    fn device_create_command_encoder(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &CommandEncoderDescriptor<'_>,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData) {
        todo!()
    }

    fn device_create_render_bundle_encoder(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> (Self::RenderBundleEncoderId, Self::RenderBundleEncoderData) {
        todo!()
    }

    fn device_drop(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        todo!()
    }

    fn device_set_device_lost_callback(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _device_lost_callback: crate::context::DeviceLostCallback,
    ) {
        todo!()
    }

    fn device_destroy(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        todo!()
    }

    fn device_mark_lost(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _message: &str,
    ) {
        todo!()
    }

    fn queue_drop(&self, _queue: &Self::QueueId, _queue_data: &Self::QueueData) {
        todo!()
    }

    fn device_poll(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _maintain: crate::Maintain,
    ) -> wgt::MaintainResult {
        todo!()
    }

    fn device_on_uncaptured_error(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _handler: Box<dyn UncapturedErrorHandler>,
    ) {
        todo!()
    }

    fn device_push_error_scope(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        _filter: crate::ErrorFilter,
    ) {
        todo!()
    }

    fn device_pop_error_scope(
        &self,
        _device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> Self::PopErrorScopeFuture {
        todo!()
    }

    fn buffer_map_async(
        &self,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _mode: MapMode,
        _range: Range<wgt::BufferAddress>,
        _callback: crate::context::BufferMapCallback,
    ) {
        todo!()
    }

    fn buffer_get_mapped_range(
        &self,
        _buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        _sub_range: Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        todo!()
    }

    fn buffer_unmap(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        todo!()
    }

    fn texture_create_view(
        &self,
        _texture: &Self::TextureId,
        _texture_data: &Self::TextureData,
        _desc: &TextureViewDescriptor<'_>,
    ) -> (Self::TextureViewId, Self::TextureViewData) {
        todo!()
    }

    fn surface_drop(&self, _surface: &Self::SurfaceId, _surface_data: &Self::SurfaceData) {
        todo!()
    }

    fn adapter_drop(&self, _adapter: &Self::AdapterId, _adapter_data: &Self::AdapterData) {
        todo!()
    }

    fn buffer_destroy(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        todo!()
    }

    fn buffer_drop(&self, _buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        todo!()
    }

    fn texture_destroy(&self, _texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        todo!()
    }

    fn texture_drop(&self, _texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        todo!()
    }

    fn texture_view_drop(
        &self,
        _texture_view: &Self::TextureViewId,
        _texture_view_data: &Self::TextureViewData,
    ) {
        todo!()
    }

    fn sampler_drop(&self, _sampler: &Self::SamplerId, _sampler_data: &Self::SamplerData) {
        todo!()
    }

    fn query_set_drop(&self, _query_set: &Self::QuerySetId, _query_set_data: &Self::QuerySetData) {
        todo!()
    }

    fn bind_group_drop(
        &self,
        _bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
    ) {
        todo!()
    }

    fn bind_group_layout_drop(
        &self,
        _bind_group_layout: &Self::BindGroupLayoutId,
        _bind_group_layout_data: &Self::BindGroupLayoutData,
    ) {
        todo!()
    }

    fn pipeline_layout_drop(
        &self,
        _pipeline_layout: &Self::PipelineLayoutId,
        _pipeline_layout_data: &Self::PipelineLayoutData,
    ) {
        todo!()
    }

    fn shader_module_drop(
        &self,
        _shader_module: &Self::ShaderModuleId,
        _shader_module_data: &Self::ShaderModuleData,
    ) {
        todo!()
    }

    fn command_encoder_drop(
        &self,
        _command_encoder: &Self::CommandEncoderId,
        _command_encoder_data: &Self::CommandEncoderData,
    ) {
        todo!()
    }

    fn command_buffer_drop(
        &self,
        _command_buffer: &Self::CommandBufferId,
        _command_buffer_data: &Self::CommandBufferData,
    ) {
        todo!()
    }

    fn render_bundle_drop(
        &self,
        _render_bundle: &Self::RenderBundleId,
        _render_bundle_data: &Self::RenderBundleData,
    ) {
        todo!()
    }

    fn compute_pipeline_drop(
        &self,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        todo!()
    }

    fn render_pipeline_drop(
        &self,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        todo!()
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
        _index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        todo!()
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
        _index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        todo!()
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
        todo!()
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyBuffer<'_>,
        _destination: crate::ImageCopyTexture<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        todo!()
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyTexture<'_>,
        _destination: crate::ImageCopyBuffer<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        todo!()
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _source: crate::ImageCopyTexture<'_>,
        _destination: crate::ImageCopyTexture<'_>,
        _copy_size: wgt::Extent3d,
    ) {
        todo!()
    }

    fn command_encoder_begin_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _desc: &ComputePassDescriptor<'_>,
    ) -> (Self::ComputePassId, Self::ComputePassData) {
        todo!()
    }

    fn command_encoder_end_compute_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        todo!()
    }

    fn command_encoder_begin_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _desc: &crate::RenderPassDescriptor<'_, '_>,
    ) -> (Self::RenderPassId, Self::RenderPassData) {
        todo!()
    }

    fn command_encoder_end_render_pass(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        todo!()
    }

    fn command_encoder_finish(
        &self,
        _encoder: Self::CommandEncoderId,
        _encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData) {
        todo!()
    }

    fn command_encoder_clear_texture(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _texture: &crate::Texture, // TODO: Decompose?
        _subresource_range: &wgt::ImageSubresourceRange,
    ) {
        todo!()
    }

    fn command_encoder_clear_buffer(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _buffer: &crate::Buffer,
        _offset: wgt::BufferAddress,
        _size: Option<wgt::BufferAddress>,
    ) {
        todo!()
    }

    fn command_encoder_insert_debug_marker(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        todo!()
    }

    fn command_encoder_push_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _label: &str,
    ) {
        todo!()
    }

    fn command_encoder_pop_debug_group(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
    ) {
        todo!()
    }

    fn command_encoder_write_timestamp(
        &self,
        _encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        todo!()
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
        todo!()
    }

    fn render_bundle_encoder_finish(
        &self,
        _encoder: Self::RenderBundleEncoderId,
        _encoder_data: Self::RenderBundleEncoderData,
        _desc: &crate::RenderBundleDescriptor<'_>,
    ) -> (Self::RenderBundleId, Self::RenderBundleData) {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn queue_create_staging_buffer(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _size: wgt::BufferSize,
    ) -> Option<Box<dyn crate::context::QueueWriteBuffer>> {
        todo!()
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
        todo!()
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
        todo!()
    }

    // TODO: do we need this?
    fn queue_copy_external_image_to_texture(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _source: &wgt::ImageCopyExternalImage,
        _dest: crate::ImageCopyTextureTagged<'_>,
        _size: wgt::Extent3d,
    ) {
        todo!()
    }

    fn queue_submit<I: Iterator<Item = (Self::CommandBufferId, Self::CommandBufferData)>>(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _command_buffers: I,
    ) -> (Self::SubmissionIndex, Self::SubmissionIndexData) {
        todo!()
    }

    fn queue_get_timestamp_period(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
    ) -> f32 {
        todo!()
    }

    fn queue_on_submitted_work_done(
        &self,
        _queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        _callback: crate::context::SubmittedWorkDoneCallback,
    ) {
        todo!()
    }

    fn device_start_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        todo!()
    }

    fn device_stop_capture(&self, _device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        todo!()
    }

    fn compute_pass_set_pipeline(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        todo!()
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
        todo!()
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _offset: u32,
        _data: &[u8],
    ) {
        todo!()
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _label: &str,
    ) {
        todo!()
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _group_label: &str,
    ) {
        todo!()
    }

    fn compute_pass_pop_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        todo!()
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        todo!()
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        todo!()
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
    ) {
        todo!()
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _x: u32,
        _y: u32,
        _z: u32,
    ) {
        todo!()
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        _pass: &mut Self::ComputePassId,
        _pass_data: &mut Self::ComputePassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        todo!()
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        todo!()
    }

    fn render_bundle_encoder_draw(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _vertices: Range<u32>,
        _instances: Range<u32>,
    ) {
        todo!()
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indices: Range<u32>,
        _base_vertex: i32,
        _instances: Range<u32>,
    ) {
        todo!()
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        todo!()
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn render_pass_set_pipeline(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn render_pass_set_push_constants(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _stages: wgt::ShaderStages,
        _offset: u32,
        _data: &[u8],
    ) {
        todo!()
    }

    fn render_pass_draw(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _vertices: Range<u32>,
        _instances: Range<u32>,
    ) {
        todo!()
    }

    fn render_pass_draw_indexed(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indices: Range<u32>,
        _base_vertex: i32,
        _instances: Range<u32>,
    ) {
        todo!()
    }

    fn render_pass_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        todo!()
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
    ) {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn render_pass_set_blend_constant(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _color: wgt::Color,
    ) {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn render_pass_set_stencil_reference(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _reference: u32,
    ) {
        todo!()
    }

    fn render_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _label: &str,
    ) {
        todo!()
    }

    fn render_pass_push_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _group_label: &str,
    ) {
        todo!()
    }

    fn render_pass_pop_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        todo!()
    }

    fn render_pass_write_timestamp(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        todo!()
    }

    fn render_pass_begin_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_index: u32,
    ) {
        todo!()
    }

    fn render_pass_end_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        todo!()
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        _query_index: u32,
    ) {
        todo!()
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
    ) {
        todo!()
    }

    fn render_pass_execute_bundles(
        &self,
        _pass: &mut Self::RenderPassId,
        _pass_data: &mut Self::RenderPassData,
        _render_bundles: &mut dyn Iterator<Item = (Self::RenderBundleId, &Self::RenderBundleData)>,
    ) {
        todo!()
    }
}
