use alloc::sync::Arc;
use core::cell::RefCell;
use core::fmt;
use wgpu_core::binding_model::{BindGroup, BindGroupLayout, PipelineLayout};
use wgpu_core::command::{
    CommandBuffer, CommandEncoder, ComputePass, RenderBundle, RenderBundleEncoder, RenderPass,
};
use wgpu_core::device::queue::Queue;
use wgpu_core::device::Device;
use wgpu_core::instance::{Adapter, Instance};
use wgpu_core::pipeline::{ComputePipeline, RenderPipeline, ShaderModule};
use wgpu_core::resource::{Buffer, ExternalTexture, QuerySet, Sampler, Texture, TextureView};
use wgpu_core_remote_types::id::{
    BindGroupId, ComputePassEncoderId, ExternalTextureId, RenderBundleEncoderId, RenderBundleId,
    RenderPassEncoderId, SamplerId, ShaderModuleId, TextureViewId,
};

use crate::hub::Hub;
use crate::id::{
    AdapterId, BindGroupLayoutId, BufferId, CommandBufferId, CommandEncoderId, ComputePipelineId,
    DeviceId, PipelineLayoutId, QuerySetId, QueueId, RenderPipelineId, TextureId,
};

mod bundle;
mod command_encoder;
mod compute_pass;
mod device;
mod instance;
mod queue;
mod render_pass;

/// Wrapper around [`Instance`] that uses [`Hub`] to store all resources created by the instance behind an [`Id`].
///
/// All resource methods are implemented on [`Global`] and accept types from [`wgpu_core_remote_types`]
/// (which use [`Id`]s instead of concrete resources) and maps them into [`wgpu_core`] types
/// (the process which also includes resolving the IDs).
///
/// [`Id`]: crate::id::Id
pub struct Global {
    pub(crate) hub: RefCell<Hub>,
    // the instance must be dropped last
    pub instance: Arc<Instance>,
}

impl Global {
    pub fn new(
        name: &str,
        instance_desc: wgt::InstanceDescriptor,
        telemetry: Option<hal::Telemetry>,
    ) -> Self {
        Self {
            instance: Instance::new(name, instance_desc, telemetry),
            hub: RefCell::new(Hub::new()),
        }
    }

    pub fn from_instance(instance: Arc<Instance>) -> Self {
        Self {
            instance,
            hub: RefCell::new(Hub::new()),
        }
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

// methods to import and resolve resources in the global hub
impl Global {
    /// Import [`Arc<Adapter>`] into the global hub,
    /// returning an [`AdapterId`] under which the adapter is stored.
    pub fn import_adapter(&self, adapter: Arc<Adapter>, id_in: AdapterId) -> AdapterId {
        let mut hub = self.hub.borrow_mut();
        hub.adapters.assign(id_in, adapter)
    }

    /// Resolve an [`AdapterId`] to the corresponding [`Arc<Adapter>`] in the global hub.
    pub fn resolve_adapter_id(&self, adapter_id: AdapterId) -> Arc<Adapter> {
        self.hub.borrow().adapters.get(adapter_id)
    }

    /// Import [`Arc<Device>`] into the global hub,
    /// returning a [`DeviceId`] under which the device is stored.
    pub fn import_device(&self, device: Arc<Device>, id_in: DeviceId) -> DeviceId {
        let mut hub = self.hub.borrow_mut();
        hub.devices.assign(id_in, device)
    }

    /// Resolve a [`DeviceId`] to the corresponding [`Arc<Device>`] in the global hub.
    pub fn resolve_device_id(&self, device_id: DeviceId) -> Arc<Device> {
        self.hub.borrow().devices.get(device_id)
    }

    /// Import [`Arc<Queue>`] into the global hub,
    /// returning a [`QueueId`] under which the queue is stored.
    pub fn import_queue(&self, queue: Arc<Queue>, id_in: QueueId) -> QueueId {
        let mut hub = self.hub.borrow_mut();
        hub.queues.assign(id_in, queue)
    }

    /// Resolve a [`QueueId`] to the corresponding [`Arc<Queue>`] in the global hub.
    pub fn resolve_queue_id(&self, queue_id: QueueId) -> Arc<Queue> {
        self.hub.borrow().queues.get(queue_id)
    }

    /// Import [`Arc<PipelineLayout>`] into the global hub,
    /// returning a [`PipelineLayoutId`] under which the pipeline layout is stored.
    pub fn import_pipeline_layout(
        &self,
        pipeline_layout: Arc<PipelineLayout>,
        id_in: PipelineLayoutId,
    ) -> PipelineLayoutId {
        let mut hub = self.hub.borrow_mut();
        hub.pipeline_layouts.assign(id_in, pipeline_layout)
    }

    /// Resolve a [`PipelineLayoutId`] to the corresponding [`Arc<PipelineLayout>`] in the global hub.
    pub fn resolve_pipeline_layout_id(
        &self,
        pipeline_layout_id: PipelineLayoutId,
    ) -> Arc<PipelineLayout> {
        self.hub.borrow().pipeline_layouts.get(pipeline_layout_id)
    }

    /// Import [`Arc<ShaderModule>`] into the global hub,
    /// returning a [`ShaderModuleId`] under which the shader module is stored.
    pub fn import_shader_module(
        &self,
        shader_module: Arc<ShaderModule>,
        id_in: ShaderModuleId,
    ) -> ShaderModuleId {
        let mut hub = self.hub.borrow_mut();
        hub.shader_modules.assign(id_in, shader_module)
    }

    /// Resolve a [`ShaderModuleId`] to the corresponding [`Arc<ShaderModule>`] in the global hub.
    pub fn resolve_shader_module_id(&self, shader_module_id: ShaderModuleId) -> Arc<ShaderModule> {
        self.hub.borrow().shader_modules.get(shader_module_id)
    }

    /// Import [`Arc<BindGroupLayout>`] into the global hub,
    /// returning a [`BindGroupLayoutId`] under which the bind group layout is stored.
    pub fn import_bind_group_layout(
        &self,
        bind_group_layout: Arc<BindGroupLayout>,
        id_in: BindGroupLayoutId,
    ) -> BindGroupLayoutId {
        let mut hub = self.hub.borrow_mut();
        hub.bind_group_layouts.assign(id_in, bind_group_layout)
    }

    /// Resolve a [`BindGroupLayoutId`] to the corresponding [`Arc<BindGroupLayout>`] in the global hub.
    pub fn resolve_bind_group_layout_id(
        &self,
        bind_group_layout_id: BindGroupLayoutId,
    ) -> Arc<BindGroupLayout> {
        self.hub
            .borrow()
            .bind_group_layouts
            .get(bind_group_layout_id)
    }

    /// Import [`Arc<BindGroup>`] into the global hub,
    /// returning a [`BindGroupId`] under which the bind group is stored.
    pub fn import_bind_group(&self, bind_group: Arc<BindGroup>, id_in: BindGroupId) -> BindGroupId {
        let mut hub = self.hub.borrow_mut();
        hub.bind_groups.assign(id_in, bind_group)
    }

    /// Resolve a [`BindGroupId`] to the corresponding [`Arc<BindGroup>`] in the global hub.
    pub fn resolve_bind_group_id(&self, bind_group_id: BindGroupId) -> Arc<BindGroup> {
        self.hub.borrow().bind_groups.get(bind_group_id)
    }

    /// Import [`Arc<CommandEncoder>`] into the global hub,
    /// returning a [`CommandEncoderId`] under which the command encoder is stored.
    pub fn import_command_encoder(
        &self,
        command_encoder: Arc<CommandEncoder>,
        id_in: CommandEncoderId,
    ) -> CommandEncoderId {
        let mut hub = self.hub.borrow_mut();
        hub.command_encoders.assign(id_in, command_encoder)
    }

    /// Resolve a [`CommandEncoderId`] to the corresponding [`Arc<CommandEncoder>`] in the global hub.
    pub fn resolve_command_encoder_id(
        &self,
        command_encoder_id: CommandEncoderId,
    ) -> Arc<CommandEncoder> {
        self.hub.borrow().command_encoders.get(command_encoder_id)
    }

    /// Import [`Arc<CommandBuffer>`] into the global hub,
    /// returning a [`CommandBufferId`] under which the command buffer is stored.
    pub fn import_command_buffer(
        &self,
        command_buffer: Arc<CommandBuffer>,
        id_in: CommandBufferId,
    ) -> CommandBufferId {
        let mut hub = self.hub.borrow_mut();
        hub.command_buffers.assign(id_in, command_buffer)
    }

    /// Resolve a [`CommandBufferId`] to the corresponding [`Arc<CommandBuffer>`] in the global hub.
    pub fn resolve_command_buffer_id(
        &self,
        command_buffer_id: CommandBufferId,
    ) -> Arc<CommandBuffer> {
        self.hub.borrow().command_buffers.get(command_buffer_id)
    }

    /// Import [`Arc<RenderBundle>`] into the global hub,
    /// returning a [`RenderBundleId`] under which the render bundle is stored.
    pub fn import_render_bundle(
        &self,
        render_bundle: Arc<RenderBundle>,
        id_in: RenderBundleId,
    ) -> RenderBundleId {
        let mut hub = self.hub.borrow_mut();
        hub.render_bundles.assign(id_in, render_bundle)
    }

    /// Resolve a [`RenderBundleId`] to the corresponding [`Arc<RenderBundle>`] in the global hub.
    pub fn resolve_render_bundle_id(&self, render_bundle_id: RenderBundleId) -> Arc<RenderBundle> {
        self.hub.borrow().render_bundles.get(render_bundle_id)
    }

    /// Import [`Arc<RenderPipeline>`] into the global hub,
    /// returning a [`RenderPipelineId`] under which the render pipeline is stored.
    pub fn import_render_pipeline(
        &self,
        render_pipeline: Arc<RenderPipeline>,
        id_in: RenderPipelineId,
    ) -> RenderPipelineId {
        let mut hub = self.hub.borrow_mut();
        hub.render_pipelines.assign(id_in, render_pipeline)
    }

    /// Resolve a [`RenderPipelineId`] to the corresponding [`Arc<RenderPipeline>`] in the global hub.
    pub fn resolve_render_pipeline_id(
        &self,
        render_pipeline_id: RenderPipelineId,
    ) -> Arc<RenderPipeline> {
        self.hub.borrow().render_pipelines.get(render_pipeline_id)
    }

    /// Import [`Arc<ComputePipeline>`] into the global hub,
    /// returning a [`ComputePipelineId`] under which the compute pipeline is stored.
    pub fn import_compute_pipeline(
        &self,
        compute_pipeline: Arc<ComputePipeline>,
        id_in: ComputePipelineId,
    ) -> ComputePipelineId {
        let mut hub = self.hub.borrow_mut();
        hub.compute_pipelines.assign(id_in, compute_pipeline)
    }

    /// Resolve a [`ComputePipelineId`] to the corresponding [`Arc<ComputePipeline>`] in the global hub.
    pub fn resolve_compute_pipeline_id(
        &self,
        compute_pipeline_id: ComputePipelineId,
    ) -> Arc<ComputePipeline> {
        self.hub.borrow().compute_pipelines.get(compute_pipeline_id)
    }

    /// Import [`Arc<QuerySet>`] into the global hub,
    /// returning a [`QuerySetId`] under which the query set is stored.
    pub fn import_query_set(&self, query_set: Arc<QuerySet>, id_in: QuerySetId) -> QuerySetId {
        let mut hub = self.hub.borrow_mut();
        hub.query_sets.assign(id_in, query_set)
    }

    /// Resolve a [`QuerySetId`] to the corresponding [`Arc<QuerySet>`] in the global hub.
    pub fn resolve_query_set_id(&self, query_set_id: QuerySetId) -> Arc<QuerySet> {
        self.hub.borrow().query_sets.get(query_set_id)
    }

    /// Import [`Arc<Buffer>`] into the global hub,
    /// returning a [`BufferId`] under which the buffer is stored.
    pub fn import_buffer(&self, buffer: Arc<Buffer>, id_in: BufferId) -> BufferId {
        let mut hub = self.hub.borrow_mut();
        hub.buffers.assign(id_in, buffer)
    }

    /// Resolve a [`BufferId`] to the corresponding [`Arc<Buffer>`] in the global hub.
    pub fn resolve_buffer_id(&self, buffer_id: BufferId) -> Arc<Buffer> {
        self.hub.borrow().buffers.get(buffer_id)
    }

    /// Import [`Arc<Texture>`] into the global hub,
    /// returning a [`TextureId`] under which the texture is stored.
    pub fn import_texture(&self, texture: Arc<Texture>, id_in: TextureId) -> TextureId {
        let mut hub = self.hub.borrow_mut();
        hub.textures.assign(id_in, texture)
    }

    /// Resolve a [`TextureId`] to the corresponding [`Arc<Texture>`] in the global hub.
    pub fn resolve_texture_id(&self, texture_id: TextureId) -> Arc<Texture> {
        self.hub.borrow().textures.get(texture_id)
    }

    /// Import [`Arc<TextureView>`] into the global hub,
    /// returning a [`TextureViewId`] under which the texture view is stored.
    pub fn import_texture_view(
        &self,
        texture_view: Arc<TextureView>,
        id_in: TextureViewId,
    ) -> TextureViewId {
        let mut hub = self.hub.borrow_mut();
        hub.texture_views.assign(id_in, texture_view)
    }

    /// Resolve a [`TextureViewId`] to the corresponding [`Arc<TextureView>`] in the global hub.
    pub fn resolve_texture_view_id(&self, texture_view_id: TextureViewId) -> Arc<TextureView> {
        self.hub.borrow().texture_views.get(texture_view_id)
    }

    /// Import [`Arc<ExternalTexture>`] into the global hub,
    /// returning a [`ExternalTextureId`] under which the external texture is stored.
    pub fn import_external_texture(
        &self,
        external_texture: Arc<ExternalTexture>,
        id_in: ExternalTextureId,
    ) -> ExternalTextureId {
        let mut hub = self.hub.borrow_mut();
        hub.external_textures.assign(id_in, external_texture)
    }

    /// Resolve a [`ExternalTextureId`] to the corresponding [`Arc<ExternalTexture>`] in the global hub.
    pub fn resolve_external_texture_id(
        &self,
        external_texture_id: ExternalTextureId,
    ) -> Arc<ExternalTexture> {
        self.hub.borrow().external_textures.get(external_texture_id)
    }

    /// Import [`Arc<Sampler>`] into the global hub,
    /// returning a [`SamplerId`] under which the sampler is stored.
    pub fn import_sampler(&self, sampler: Arc<Sampler>, id_in: SamplerId) -> SamplerId {
        let mut hub = self.hub.borrow_mut();
        hub.samplers.assign(id_in, sampler)
    }

    /// Resolve a [`SamplerId`] to the corresponding [`Arc<Sampler>`] in the global hub.
    pub fn resolve_sampler_id(&self, sampler_id: SamplerId) -> Arc<Sampler> {
        self.hub.borrow().samplers.get(sampler_id)
    }

    /// Import [`RenderPass`] into the global hub,
    /// returning a [`RenderPassEncoderId`] under which the render pass is stored.
    pub fn import_render_pass(
        &self,
        render_pass: RenderPass,
        id_in: RenderPassEncoderId,
    ) -> RenderPassEncoderId {
        let mut hub = self.hub.borrow_mut();
        hub.render_passes.assign(id_in, render_pass)
    }

    /// Import [`ComputePass`] into the global hub,
    /// returning a [`ComputePassEncoderId`] under which the compute pass is stored.
    pub fn import_compute_pass(
        &self,
        compute_pass: ComputePass,
        id_in: ComputePassEncoderId,
    ) -> ComputePassEncoderId {
        let mut hub = self.hub.borrow_mut();
        hub.compute_passes.assign(id_in, compute_pass)
    }

    /// Import [`RenderBundleEncoder`] into the global hub,
    /// returning a [`RenderBundleEncoderId`] under which the render bundle encoder is stored.
    pub fn import_render_bundle_encoder(
        &self,
        render_bundle_encoder: RenderBundleEncoder,
        id_in: RenderBundleEncoderId,
    ) -> RenderBundleEncoderId {
        let mut hub = self.hub.borrow_mut();
        hub.render_bundle_encoders
            .assign(id_in, render_bundle_encoder)
    }
}

impl fmt::Debug for Global {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Global").finish()
    }
}
