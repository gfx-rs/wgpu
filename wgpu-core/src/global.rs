use alloc::{borrow::ToOwned as _, sync::Arc};
use core::fmt;

use crate::{
    binding_model::{BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, CommandEncoder},
    device::{queue::Queue, Device},
    hub::{Hub, HubReport},
    id::{
        AdapterId, BindGroupLayoutId, CommandBufferId, CommandEncoderId, ComputePipelineId,
        DeviceId, PipelineLayoutId, QuerySetId, QueueId, RenderPipelineId, TextureId,
    },
    instance::{Adapter, Instance, Surface},
    pipeline::{ComputePipeline, RenderPipeline},
    registry::{Registry, RegistryReport},
    resource::{QuerySet, Texture},
    resource_log,
};

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalReport {
    pub surfaces: RegistryReport,
    pub hub: HubReport,
}

impl GlobalReport {
    pub fn surfaces(&self) -> &RegistryReport {
        &self.surfaces
    }
    pub fn hub_report(&self) -> &HubReport {
        &self.hub
    }
}

pub struct Global {
    pub(crate) surfaces: Registry<Arc<Surface>>,
    pub(crate) hub: Hub,
    // the instance must be dropped last
    pub instance: Instance,
}

impl Global {
    pub fn new(
        name: &str,
        instance_desc: wgt::InstanceDescriptor,
        telemetry: Option<hal::Telemetry>,
    ) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc, telemetry),
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: hal::Api>(name: &str, hal_instance: A::Instance) -> Self {
        profiling::scope!("Global::new");

        Self {
            instance: Instance::from_hal_instance::<A>(name.to_owned(), hal_instance),
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: hal::Api>(&self) -> Option<&A::Instance> {
        unsafe { self.instance.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw handles obtained from the Instance must not be manually destroyed
    pub unsafe fn from_instance(instance: Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance,
            surfaces: Registry::new(),
            hub: Hub::new(),
        }
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            hub: self.hub.generate_report(),
        }
    }
}

// methods to import and resolve resources in the global hub
impl Global {
    /// Import [`Arc<Adapter>`] into the global hub,
    /// returning an [`AdapterId`] under which the adapter is stored.
    pub fn import_adapter(&self, adapter: Arc<Adapter>, id_in: Option<AdapterId>) -> AdapterId {
        let fid = self.hub.adapters.prepare(id_in);
        fid.assign(adapter)
    }

    /// Resolve an [`AdapterId`] to the corresponding [`Arc<Adapter>`] in the global hub.
    pub fn resolve_adapter_id(&self, adapter_id: AdapterId) -> Arc<Adapter> {
        self.hub.adapters.get(adapter_id)
    }

    /// Import [`Arc<Device>`] into the global hub,
    /// returning a [`DeviceId`] under which the device is stored.
    pub fn import_device(&self, device: Arc<Device>, id_in: Option<DeviceId>) -> DeviceId {
        let fid = self.hub.devices.prepare(id_in);
        fid.assign(device)
    }

    /// Resolve a [`DeviceId`] to the corresponding [`Arc<Device>`] in the global hub.
    pub fn resolve_device_id(&self, device_id: DeviceId) -> Arc<Device> {
        self.hub.devices.get(device_id)
    }

    /// Import [`Arc<Queue>`] into the global hub,
    /// returning a [`QueueId`] under which the queue is stored.
    pub fn import_queue(&self, queue: Arc<Queue>, id_in: Option<QueueId>) -> QueueId {
        let fid = self.hub.queues.prepare(id_in);
        fid.assign(queue)
    }

    /// Resolve a [`QueueId`] to the corresponding [`Arc<Queue>`] in the global hub.
    pub fn resolve_queue_id(&self, queue_id: QueueId) -> Arc<Queue> {
        self.hub.queues.get(queue_id)
    }

    /// Import [`Arc<PipelineLayout>`] into the global hub,
    /// returning a [`PipelineLayoutId`] under which the pipeline layout is stored.
    pub fn import_pipeline_layout(
        &self,
        pipeline_layout: Arc<PipelineLayout>,
        id_in: Option<PipelineLayoutId>,
    ) -> PipelineLayoutId {
        let fid = self.hub.pipeline_layouts.prepare(id_in);
        fid.assign(pipeline_layout)
    }

    /// Resolve a [`PipelineLayoutId`] to the corresponding [`Arc<PipelineLayout>`] in the global hub.
    pub fn resolve_pipeline_layout_id(
        &self,
        pipeline_layout_id: PipelineLayoutId,
    ) -> Arc<PipelineLayout> {
        self.hub.pipeline_layouts.get(pipeline_layout_id)
    }

    /// Import [`Arc<BindGroupLayout>`] into the global hub,
    /// returning a [`BindGroupLayoutId`] under which the bind group layout is stored.
    pub fn import_bind_group_layout(
        &self,
        bind_group_layout: Arc<BindGroupLayout>,
        id_in: Option<BindGroupLayoutId>,
    ) -> BindGroupLayoutId {
        let fid = self.hub.bind_group_layouts.prepare(id_in);
        fid.assign(bind_group_layout)
    }

    /// Resolve a [`BindGroupLayoutId`] to the corresponding [`Arc<BindGroupLayout>`] in the global hub.
    pub fn resolve_bind_group_layout_id(
        &self,
        bind_group_layout_id: BindGroupLayoutId,
    ) -> Arc<BindGroupLayout> {
        self.hub.bind_group_layouts.get(bind_group_layout_id)
    }

    /// Import [`Arc<CommandEncoder>`] into the global hub,
    /// returning a [`CommandEncoderId`] under which the command encoder is stored.
    pub fn import_command_encoder(
        &self,
        command_encoder: Arc<CommandEncoder>,
        id_in: Option<CommandEncoderId>,
    ) -> CommandEncoderId {
        let fid = self.hub.command_encoders.prepare(id_in);
        fid.assign(command_encoder)
    }

    /// Resolve a [`CommandEncoderId`] to the corresponding [`Arc<CommandEncoder>`] in the global hub.
    pub fn resolve_command_encoder_id(
        &self,
        command_encoder_id: CommandEncoderId,
    ) -> Arc<CommandEncoder> {
        self.hub.command_encoders.get(command_encoder_id)
    }

    /// Import [`Arc<CommandBuffer>`] into the global hub,
    /// returning a [`CommandBufferId`] under which the command buffer is stored.
    pub fn import_command_buffer(
        &self,
        command_buffer: Arc<CommandBuffer>,
        id_in: Option<CommandBufferId>,
    ) -> CommandBufferId {
        let fid = self.hub.command_buffers.prepare(id_in);
        fid.assign(command_buffer)
    }

    /// Resolve a [`CommandBufferId`] to the corresponding [`Arc<CommandBuffer>`] in the global hub.
    pub fn resolve_command_buffer_id(
        &self,
        command_buffer_id: CommandBufferId,
    ) -> Arc<CommandBuffer> {
        self.hub.command_buffers.get(command_buffer_id)
    }

    /// Import [`Arc<RenderPipeline>`] into the global hub,
    /// returning a [`RenderPipelineId`] under which the render pipeline is stored.
    pub fn import_render_pipeline(
        &self,
        render_pipeline: Arc<RenderPipeline>,
        id_in: Option<RenderPipelineId>,
    ) -> RenderPipelineId {
        let fid = self.hub.render_pipelines.prepare(id_in);
        fid.assign(render_pipeline)
    }

    /// Resolve a [`RenderPipelineId`] to the corresponding [`Arc<RenderPipeline>`] in the global hub.
    pub fn resolve_render_pipeline_id(
        &self,
        render_pipeline_id: RenderPipelineId,
    ) -> Arc<RenderPipeline> {
        self.hub.render_pipelines.get(render_pipeline_id)
    }

    /// Import [`Arc<ComputePipeline>`] into the global hub,
    /// returning a [`ComputePipelineId`] under which the compute pipeline is stored.
    pub fn import_compute_pipeline(
        &self,
        compute_pipeline: Arc<ComputePipeline>,
        id_in: Option<ComputePipelineId>,
    ) -> ComputePipelineId {
        let fid = self.hub.compute_pipelines.prepare(id_in);
        fid.assign(compute_pipeline)
    }

    /// Resolve a [`ComputePipelineId`] to the corresponding [`Arc<ComputePipeline>`] in the global hub.
    pub fn resolve_compute_pipeline_id(
        &self,
        compute_pipeline_id: ComputePipelineId,
    ) -> Arc<ComputePipeline> {
        self.hub.compute_pipelines.get(compute_pipeline_id)
    }

    /// Import [`Arc<QuerySet>`] into the global hub,
    /// returning a [`QuerySetId`] under which the query set is stored.
    pub fn import_query_set(
        &self,
        query_set: Arc<QuerySet>,
        id_in: Option<QuerySetId>,
    ) -> QuerySetId {
        let fid = self.hub.query_sets.prepare(id_in);
        fid.assign(query_set)
    }

    /// Resolve a [`QuerySetId`] to the corresponding [`Arc<QuerySet>`] in the global hub.
    pub fn resolve_query_set_id(&self, query_set_id: QuerySetId) -> Arc<QuerySet> {
        self.hub.query_sets.get(query_set_id)
    }

    /// Import [`Arc<Texture>`] into the global hub,
    /// returning a [`TextureId`] under which the texture is stored.
    pub fn import_texture(&self, texture: Arc<Texture>, id_in: Option<TextureId>) -> TextureId {
        let fid = self.hub.textures.prepare(id_in);
        fid.assign(texture)
    }

    /// Resolve a [`TextureId`] to the corresponding [`Arc<Texture>`] in the global hub.
    pub fn resolve_texture_id(&self, texture_id: TextureId) -> Arc<Texture> {
        self.hub.textures.get(texture_id)
    }
}

impl fmt::Debug for Global {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Global").finish()
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        resource_log!("Global::drop");
    }
}

#[cfg(send_sync)]
fn _test_send_sync(global: &Global) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
