use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};
use core::ptr::NonNull;
use wgpu_core_remote_types::encoders::RenderBundleDescriptor;

use wgpu_core::{
    binding_model::{self},
    command,
    device::{DeviceLostClosure, WaitIdleError},
    error::EmptyErrorScopeStack,
    pipeline::{
        self, ProgrammableStageDescriptor, RenderPipelineVertexProcessor,
        ResolvedGeneralRenderPipelineDescriptor,
    },
    resource::{
        self, BufferAccessError, BufferAccessResult, BufferMapOperation, CreateBufferError,
    },
    Label, LabelHelpers, SubmissionIndex,
};

use crate::{
    global::Global,
    hub::Hub,
    id::{self, DeviceId, QueueId},
    registry::Registry,
};

use wgt::{error::WebGpuError, BufferAddress};

pub use wgpu_core_remote_types::binding_model::*;

pub type ComputePipelineDescriptor<'a> = pipeline::ComputePipelineDescriptor<
    'a,
    id::PipelineLayoutId,
    id::ShaderModuleId,
    id::PipelineCacheId,
>;

pub type RenderPipelineDescriptor<'a> = pipeline::RenderPipelineDescriptor<
    'a,
    id::PipelineLayoutId,
    id::ShaderModuleId,
    id::PipelineCacheId,
>;

impl Global {
    pub fn device_features(&self, device_id: DeviceId) -> wgt::Features {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);
        *device.features()
    }

    pub fn device_limits(&self, device_id: DeviceId) -> wgt::Limits {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);
        device.limits().clone()
    }

    pub fn device_adapter_info(&self, device_id: DeviceId) -> wgt::AdapterInfo {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);
        device.adapter_info()
    }

    pub fn device_downlevel_properties(&self, device_id: DeviceId) -> wgt::DownlevelCapabilities {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);
        device.downlevel().clone()
    }

    pub fn device_create_buffer(
        &self,
        device_id: DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: id::BufferId,
    ) -> (id::BufferId, Option<CreateBufferError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            buffers, devices, ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (buffer, error) = device.create_buffer(desc);

        let id = buffers.assign(id_in, buffer);

        (id, error)
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// Ensure that future attempts to use `id_in` as a buffer ID will propagate
    /// the error, following the WebGPU ["contagious invalidity"] style.
    ///
    /// Firefox uses this function to comply strictly with the WebGPU spec,
    /// which requires [`GPUBufferDescriptor`] validation to be generated on the
    /// Device timeline and leave the newly created [`GPUBuffer`] invalid.
    ///
    /// Ideally, we would simply let [`Device::create_buffer`] take care of all
    /// of this, but some errors must be detected before we can even construct a
    /// [`wgpu_types::BufferDescriptor`] to give it. For example, the WebGPU API
    /// allows a `GPUBufferDescriptor`'s [`usage`] property to be any WebIDL
    /// `unsigned long` value, but we can't construct a
    /// [`wgpu_types::BufferUsages`] value from values with unassigned bits
    /// set. This means we must validate `usage` before we can call
    /// `Device::create_buffer`.
    ///
    /// When that validation fails, we must arrange for the buffer id to be
    /// considered invalid. This method provides the means to do so.
    ///
    /// ["contagious invalidity"]: https://www.w3.org/TR/webgpu/#invalidity
    /// [`GPUBufferDescriptor`]: https://www.w3.org/TR/webgpu/#dictdef-gpubufferdescriptor
    /// [`GPUBuffer`]: https://www.w3.org/TR/webgpu/#gpubuffer
    /// [`wgpu_types::BufferDescriptor`]: wgt::BufferDescriptor
    /// [`Device::create_buffer`]: wgpu_core::device::Device::create_buffer
    /// [`usage`]: https://www.w3.org/TR/webgpu/#dom-gputexturedescriptor-usage
    /// [`wgpu_types::BufferUsages`]: wgt::BufferUsages
    pub fn create_buffer_error(
        &self,
        device_id: DeviceId,
        id_in: id::BufferId,
        desc: &resource::BufferDescriptor,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            buffers, devices, ..
        } = &mut *hub;
        let device = devices.get(device_id);
        buffers.assign(id_in, resource::Buffer::invalid(device, desc));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_render_bundle_error(
        &self,
        device_id: DeviceId,
        id_in: id::RenderBundleId,
        desc: &command::RenderBundleDescriptor,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundles,
            devices,
            ..
        } = &mut *hub;
        let device = devices.get(device_id);
        render_bundles.assign(id_in, command::RenderBundle::invalid(device, desc));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_texture_error(
        &self,
        device_id: DeviceId,
        id_in: id::TextureId,
        desc: &resource::TextureDescriptor,
    ) -> id::TextureId {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            textures, devices, ..
        } = &mut *hub;
        let device = devices.get(device_id);
        let texture = device.create_texture_error(desc);
        textures.assign(id_in, texture)
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_external_texture_error(
        &self,
        device_id: DeviceId,
        id_in: id::ExternalTextureId,
        desc: &resource::ExternalTextureDescriptor,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            external_textures,
            devices,
            ..
        } = &mut *hub;
        let device = devices.get(device_id);
        external_textures.assign(id_in, resource::ExternalTexture::invalid(device, desc));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// In JavaScript environments, it is possible to call `GPUDevice.createBindGroupLayout` with
    /// entries that are invalid. Because our Rust's types for bind group layouts prevent even
    /// calling [`Self::device_create_bind_group`], we let standards-compliant environments
    /// register an invalid bind group layout so this crate's API can still be consistently used.
    ///
    /// See [`Self::create_buffer_error`] for additional context and explanation.
    pub fn create_bind_group_layout_error(
        &self,
        device_id: DeviceId,
        id_in: id::BindGroupLayoutId,
        label: Option<Cow<'_, str>>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            bind_group_layouts,
            devices,
            ..
        } = &mut *hub;
        let device = devices.get(device_id);
        bind_group_layouts.assign(
            id_in,
            binding_model::BindGroupLayout::invalid(&device, label.to_string()),
        );
    }

    pub fn buffer_destroy(&self, buffer_id: id::BufferId) {
        let hub = self.hub.borrow();

        let buffer = hub.buffers.get(buffer_id);

        buffer.destroy();
    }

    pub fn buffer_drop(&self, buffer_id: id::BufferId) {
        let mut hub = self.hub.borrow_mut();

        let _buffer = hub.buffers.remove(buffer_id);
    }

    pub fn device_create_texture(
        &self,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: id::TextureId,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        let mut hub = self.hub.borrow_mut();

        let Hub {
            textures, devices, ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (texture, error) = device.create_texture(desc);

        let id = textures.assign(id_in, texture);

        (id, error)
    }

    pub fn device_validate_texture_descriptor(
        &self,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
    ) -> Option<resource::CreateTextureError> {
        let hub = self.hub.borrow();
        hub.devices
            .get(device_id)
            .validate_texture_descriptor(desc)
            .err()
    }

    /// # Safety
    ///
    /// - `hal_texture` must be created from `device_id` corresponding raw handle.
    /// - `hal_texture` must be created respecting `desc`
    /// - `hal_texture` must be initialized
    /// - The `initial_state` must match the actual driver-side state of
    ///   the wrapped resource at the moment of wrap.
    pub unsafe fn create_texture_from_hal(
        &self,
        hal_texture: Box<dyn hal::DynTexture>,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        initial_state: wgt::TextureUses,
        id_in: id::TextureId,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            textures, devices, ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (texture, error) =
            unsafe { device.create_texture_from_hal(hal_texture, desc, initial_state) };

        let id = textures.assign(id_in, texture);
        (id, error)
    }

    /// # Safety
    ///
    /// - `hal_buffer` must be created from `device_id` corresponding raw handle.
    /// - `hal_buffer` must be created respecting `desc`
    /// - `hal_buffer` must be initialized
    /// - `hal_buffer` must not have zero size.
    pub unsafe fn create_buffer_from_hal<A: hal::Api>(
        &self,
        hal_buffer: A::Buffer,
        device_id: DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: id::BufferId,
    ) -> (id::BufferId, Option<CreateBufferError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            buffers, devices, ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (buffer, err) = unsafe { device.create_buffer_from_hal(Box::new(hal_buffer), desc) };

        let id = buffers.assign(id_in, buffer);

        (id, err)
    }

    pub fn texture_destroy(&self, texture_id: id::TextureId) {
        let hub = self.hub.borrow();

        let texture = hub.textures.get(texture_id);

        texture.destroy();
    }

    pub fn texture_drop(&self, texture_id: id::TextureId) {
        let mut hub = self.hub.borrow_mut();

        hub.textures.remove(texture_id);
    }

    pub fn texture_create_view(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: id::TextureViewId,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            textures,
            texture_views,
            ..
        } = &mut *hub;

        let texture = textures.get(texture_id);

        let (view, error) = texture.create_view(desc);

        let id = texture_views.assign(id_in, view);

        (id, error)
    }

    pub fn texture_view_drop(&self, texture_view_id: id::TextureViewId) {
        let mut hub = self.hub.borrow_mut();

        let _view = hub.texture_views.remove(texture_view_id);
    }

    pub fn device_create_external_texture(
        &self,
        device_id: DeviceId,
        desc: &resource::ExternalTextureDescriptor,
        planes: &[id::TextureViewId],
        id_in: id::ExternalTextureId,
    ) -> (
        id::ExternalTextureId,
        Option<resource::CreateExternalTextureError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            external_textures,
            devices,
            texture_views,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let planes = planes
            .iter()
            .map(|plane_id| texture_views.get(*plane_id))
            .collect::<Vec<_>>();

        let (external_texture, error) = device.create_external_texture(desc, &planes);

        let id = external_textures.assign(id_in, external_texture);

        (id, error)
    }

    pub fn external_texture_destroy(&self, external_texture_id: id::ExternalTextureId) {
        let hub = self.hub.borrow();

        let external_texture = hub.external_textures.get(external_texture_id);

        external_texture.destroy();
    }

    pub fn external_texture_drop(&self, external_texture_id: id::ExternalTextureId) {
        let mut hub = self.hub.borrow_mut();

        let _external_texture = hub.external_textures.remove(external_texture_id);
    }

    pub fn device_create_sampler(
        &self,
        device_id: DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: id::SamplerId,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            samplers, devices, ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (sampler, error) = device.create_sampler(desc);

        let id = samplers.assign(id_in, sampler);

        (id, error)
    }

    pub fn sampler_drop(&self, sampler_id: id::SamplerId) {
        let mut hub = self.hub.borrow_mut();

        let _sampler = hub.samplers.remove(sampler_id);
    }

    pub fn device_create_bind_group_layout(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: id::BindGroupLayoutId,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            bind_group_layouts,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (bgl, error) = device.create_bind_group_layout(desc);

        let id = bind_group_layouts.assign(id_in, bgl);

        (id, error)
    }

    pub fn bind_group_layout_drop(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        let mut hub = self.hub.borrow_mut();

        let _layout = hub.bind_group_layouts.remove(bind_group_layout_id);
    }

    pub fn device_create_pipeline_layout(
        &self,
        device_id: DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor<id::BindGroupLayoutId>,
        id_in: id::PipelineLayoutId,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            pipeline_layouts,
            devices,
            bind_group_layouts,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let bind_group_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl_id| bgl_id.map(|bgl_id| bind_group_layouts.get(bgl_id)))
            .collect::<Vec<_>>();

        let desc = binding_model::PipelineLayoutDescriptor {
            label: desc.label.clone(),
            bind_group_layouts: Cow::Owned(bind_group_layouts),
            immediate_size: desc.immediate_size,
        };

        let (layout, error) = device.create_pipeline_layout(&desc);
        let id = pipeline_layouts.assign(id_in, layout);
        (id, error)
    }

    pub fn pipeline_layout_drop(&self, pipeline_layout_id: id::PipelineLayoutId) {
        let mut hub = self.hub.borrow_mut();

        let _layout = hub.pipeline_layouts.remove(pipeline_layout_id);
    }

    pub fn device_create_bind_group(
        &self,
        device_id: DeviceId,
        desc: &BindGroupDescriptor,
        id_in: id::BindGroupId,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            bind_groups,
            devices,
            bind_group_layouts,
            buffers,
            samplers,
            texture_views,
            external_textures,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let layout = bind_group_layouts.get(desc.layout);

        fn resolve_entry<'a>(
            e: &'a BindGroupEntry,
            buffers: &mut Registry<Arc<resource::Buffer>>,
            samplers: &mut Registry<Arc<resource::Sampler>>,
            texture_views: &mut Registry<Arc<resource::TextureView>>,
            external_textures: &mut Registry<Arc<resource::ExternalTexture>>,
        ) -> binding_model::BindGroupEntry<'a> {
            let resolve_buffer = |bb: &BufferBinding| {
                let buffer = buffers.get(bb.buffer);
                binding_model::BufferBinding {
                    buffer,
                    offset: bb.offset,
                    size: bb.size,
                }
            };
            let resolve_sampler = |id: &id::SamplerId| samplers.get(*id);
            let resolve_view = |id: &id::TextureViewId| texture_views.get(*id);
            let resolve_external_texture = |id: &id::ExternalTextureId| external_textures.get(*id);
            let resource = match e.resource {
                BindingResource::Buffer(ref buffer) => {
                    binding_model::BindingResource::Buffer(resolve_buffer(buffer))
                }
                BindingResource::Sampler(ref sampler) => {
                    binding_model::BindingResource::Sampler(resolve_sampler(sampler))
                }
                BindingResource::TextureView(ref view) => {
                    binding_model::BindingResource::TextureView(resolve_view(view))
                }
                BindingResource::ExternalTexture(ref et) => {
                    binding_model::BindingResource::ExternalTexture(resolve_external_texture(et))
                }
            };
            binding_model::BindGroupEntry {
                binding: e.binding,
                resource,
            }
        }

        let entries = desc
            .entries
            .iter()
            .map(|e| resolve_entry(e, buffers, samplers, texture_views, external_textures))
            .collect::<Vec<_>>();
        let entries = Cow::Owned(entries);

        let desc = binding_model::BindGroupDescriptor {
            label: desc.label.clone(),
            layout,
            entries,
        };

        let (bind_group, error) = device.create_bind_group(&desc);

        let id = bind_groups.assign(id_in, bind_group);
        (id, error)
    }

    pub fn bind_group_drop(&self, bind_group_id: id::BindGroupId) {
        let mut hub = self.hub.borrow_mut();

        let _bind_group = hub.bind_groups.remove(bind_group_id);
    }

    /// Create a shader module with the given `source`.
    ///
    /// <div class="warning">
    // NOTE: Keep this in sync with `naga::front::wgsl::parse_str`!
    // NOTE: Keep this in sync with `wgpu::Device::create_shader_module`!
    ///
    /// This function may consume a lot of stack space. Compiler-enforced limits for parsing
    /// recursion exist; if shader compilation runs into them, it will return an error gracefully.
    /// However, on some build profiles and platforms, the default stack size for a thread may be
    /// exceeded before this limit is reached during parsing. Callers should ensure that there is
    /// enough stack space for this, particularly if calls to this method are exposed to user
    /// input.
    ///
    /// </div>
    pub fn device_create_shader_module(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: pipeline::ShaderModuleSource,
        id_in: id::ShaderModuleId,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            shader_modules,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (shader, error) = device.create_shader_module(desc, source);

        let id = shader_modules.assign(id_in, shader);

        (id, error)
    }

    /// # Safety
    ///
    /// This function passes source code or binary to the backend as-is and can potentially result in a
    /// driver crash.
    pub unsafe fn device_create_shader_module_passthrough(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ShaderModuleDescriptorPassthrough<'_>,
        id_in: id::ShaderModuleId,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            shader_modules,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (shader, error) = unsafe { device.create_shader_module_passthrough(desc) };

        let id = shader_modules.assign(id_in, shader);

        (id, error)
    }

    pub fn shader_module_drop(&self, shader_module_id: id::ShaderModuleId) {
        let mut hub = self.hub.borrow_mut();

        let _shader_module = hub.shader_modules.remove(shader_module_id);
    }

    pub fn device_create_command_encoder(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: id::CommandEncoderId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            command_encoders,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let cmd_enc = device.create_command_encoder(desc);

        command_encoders.assign(id_in, cmd_enc);
    }

    pub fn command_encoder_drop(&self, command_encoder_id: id::CommandEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let _cmd_enc = hub.command_encoders.remove(command_encoder_id);
    }

    pub fn command_buffer_drop(&self, command_buffer_id: id::CommandBufferId) {
        let mut hub = self.hub.borrow_mut();
        let _cmd_buf = hub.command_buffers.remove(command_buffer_id);
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
        id_in: id::RenderBundleEncoderId,
    ) -> (
        id::RenderBundleEncoderId,
        Option<command::CreateRenderBundleError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);
        let (render_bundle_encoder, error) = device.create_render_bundle_encoder(desc);

        let id = render_bundle_encoders.assign(id_in, *render_bundle_encoder);

        (id, error)
    }

    pub fn render_bundle_encoder_finish(
        &self,
        render_bundle_encoder_id: id::RenderBundleEncoderId,
        desc: RenderBundleDescriptor,
        id_in: id::RenderBundleId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            render_bundles,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(render_bundle_encoder_id);

        let RenderBundleDescriptor { label } = desc;
        let desc = wgt::RenderBundleDescriptor { label };

        let render_bundle = bundle_encoder.finish(&desc);

        render_bundles.assign(id_in, render_bundle);
    }

    pub fn render_bundle_encoder_drop(&self, render_bundle_encoder_id: id::RenderBundleEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let _bundle_encoder = hub.render_bundle_encoders.remove(render_bundle_encoder_id);
    }

    pub fn render_bundle_drop(&self, render_bundle_id: id::RenderBundleId) {
        let mut hub = self.hub.borrow_mut();
        let _bundle = hub.render_bundles.remove(render_bundle_id);
    }

    pub fn device_create_query_set(
        &self,
        device_id: DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: id::QuerySetId,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            query_sets,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (query_set, error) = device.create_query_set(desc);

        let id = query_sets.assign(id_in, query_set);

        (id, error)
    }

    pub fn query_set_destroy(&self, query_set_id: id::QuerySetId) {
        let hub = self.hub.borrow();

        let query_set = hub.query_sets.get(query_set_id);

        query_set.destroy();
    }

    pub fn query_set_drop(&self, query_set_id: id::QuerySetId) {
        let mut hub = self.hub.borrow_mut();

        let _query_set = hub.query_sets.remove(query_set_id);
    }

    pub fn device_create_render_pipeline(
        &self,
        device_id: DeviceId,
        desc: &RenderPipelineDescriptor,
        id_in: id::RenderPipelineId,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_pipelines,
            devices,
            shader_modules,
            pipeline_layouts,
            pipeline_caches,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let layout = desc.layout.map(|layout| pipeline_layouts.get(layout));

        let cache = desc.cache.map(|cache| pipeline_caches.get(cache));

        let vertex = {
            let module = shader_modules.get(desc.vertex.stage.module);
            let stage = ProgrammableStageDescriptor {
                module,
                entry_point: desc.vertex.stage.entry_point.clone(),
                constants: desc.vertex.stage.constants.clone(),
                zero_initialize_workgroup_memory: desc
                    .vertex
                    .stage
                    .zero_initialize_workgroup_memory,
            };
            RenderPipelineVertexProcessor::Vertex(pipeline::VertexState {
                stage,
                buffers: desc.vertex.buffers.clone(),
            })
        };

        let fragment = if let Some(ref state) = desc.fragment {
            let module = shader_modules.get(state.stage.module);

            let stage = ProgrammableStageDescriptor {
                module,
                entry_point: state.stage.entry_point.clone(),
                constants: state.stage.constants.clone(),
                zero_initialize_workgroup_memory: state.stage.zero_initialize_workgroup_memory,
            };
            Some(pipeline::FragmentState {
                stage,
                targets: state.targets.clone(),
            })
        } else {
            None
        };

        let desc = ResolvedGeneralRenderPipelineDescriptor {
            label: desc.label.clone(),
            layout,
            vertex,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment,
            multiview_mask: desc.multiview_mask,
            cache,
        };

        let (pipeline, error) = device.create_render_pipeline(desc);

        let id = render_pipelines.assign(id_in, pipeline);

        (id, error)
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: id::BindGroupLayoutId,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            bind_group_layouts,
            render_pipelines,
            ..
        } = &mut *hub;

        let pipeline = render_pipelines.get(pipeline_id);

        let (bgl, error) = pipeline.get_bind_group_layout(index);

        let id = bind_group_layouts.assign(id_in, bgl);

        (id, error)
    }

    pub fn render_pipeline_drop(&self, render_pipeline_id: id::RenderPipelineId) {
        let mut hub = self.hub.borrow_mut();

        let _pipeline = hub.render_pipelines.remove(render_pipeline_id);
    }

    pub fn device_create_compute_pipeline(
        &self,
        device_id: DeviceId,
        desc: &ComputePipelineDescriptor,
        id_in: id::ComputePipelineId,
    ) -> (
        id::ComputePipelineId,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_pipelines,
            devices,
            shader_modules,
            pipeline_layouts,
            pipeline_caches,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let layout = desc.layout.map(|layout| pipeline_layouts.get(layout));

        let cache = desc.cache.map(|cache| pipeline_caches.get(cache));

        let module = shader_modules.get(desc.stage.module);

        let stage = ProgrammableStageDescriptor {
            module,
            entry_point: desc.stage.entry_point.clone(),
            constants: desc.stage.constants.clone(),
            zero_initialize_workgroup_memory: desc.stage.zero_initialize_workgroup_memory,
        };

        let desc = pipeline::ComputePipelineDescriptor {
            label: desc.label.clone(),
            layout,
            stage,
            cache,
        };

        let (pipeline, error) = device.create_compute_pipeline(desc);

        let id = compute_pipelines.assign(id_in, pipeline);

        (id, error)
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: id::BindGroupLayoutId,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            bind_group_layouts,
            compute_pipelines,
            ..
        } = &mut *hub;

        let pipeline = compute_pipelines.get(pipeline_id);

        let (bgl, error) = pipeline.get_bind_group_layout(index);

        let id = bind_group_layouts.assign(id_in, bgl);

        (id, error)
    }

    pub fn compute_pipeline_drop(&self, compute_pipeline_id: id::ComputePipelineId) {
        let mut hub = self.hub.borrow_mut();
        let _pipeline = hub.compute_pipelines.remove(compute_pipeline_id);
    }

    /// # Safety
    /// The `data` argument of `desc` must have been returned by
    /// [Self::pipeline_cache_get_data] for the same adapter
    pub unsafe fn device_create_pipeline_cache(
        &self,
        device_id: DeviceId,
        desc: &pipeline::PipelineCacheDescriptor<'_>,
        id_in: id::PipelineCacheId,
    ) -> (
        id::PipelineCacheId,
        Option<pipeline::CreatePipelineCacheError>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            pipeline_caches,
            devices,
            ..
        } = &mut *hub;

        let device = devices.get(device_id);

        let (cache, error) = unsafe { device.create_pipeline_cache(desc) };

        let id = pipeline_caches.assign(id_in, cache);

        (id, error)
    }

    pub fn pipeline_cache_drop(&self, pipeline_cache_id: id::PipelineCacheId) {
        let mut hub = self.hub.borrow_mut();

        let _cache = hub.pipeline_caches.remove(pipeline_cache_id);
    }

    /// Check `device_id` for freeable resources and completed buffer mappings.
    ///
    /// Return `queue_empty` indicating whether there are more queue submissions still in flight.
    pub fn device_poll(
        &self,
        device_id: DeviceId,
        poll_type: wgt::PollType<SubmissionIndex>,
    ) -> Result<wgt::PollStatus, WaitIdleError> {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.poll(poll_type)
    }

    /// Poll all devices on all backends.
    ///
    /// This is the implementation of `wgpu::Instance::poll_all`.
    ///
    /// Return `all_queue_empty` indicating whether there are more queue
    /// submissions still in flight.
    pub fn poll_all_devices(&self, force_wait: bool) -> Result<bool, WaitIdleError> {
        self.instance.poll_all_devices(force_wait)
    }

    /// # Safety
    ///
    /// - See [wgpu::Device::start_graphics_debugger_capture][api] for details the safety.
    ///
    /// [api]: ../../wgpu/struct.Device.html#method.start_graphics_debugger_capture
    pub unsafe fn device_start_graphics_debugger_capture(&self, device_id: DeviceId) {
        let hub = self.hub.borrow();
        unsafe {
            hub.devices.get(device_id).start_graphics_debugger_capture();
        }
    }

    /// # Safety
    ///
    /// - See [wgpu::Device::stop_graphics_debugger_capture][api] for details the safety.
    ///
    /// [api]: ../../wgpu/struct.Device.html#method.stop_graphics_debugger_capture
    pub unsafe fn device_stop_graphics_debugger_capture(&self, device_id: DeviceId) {
        let hub = self.hub.borrow();
        unsafe {
            hub.devices.get(device_id).stop_graphics_debugger_capture();
        }
    }

    pub fn pipeline_cache_get_data(&self, id: id::PipelineCacheId) -> Option<Vec<u8>> {
        let hub = self.hub.borrow();

        hub.pipeline_caches.get(id).get_data()
    }

    pub fn device_drop(&self, device_id: DeviceId) {
        let mut hub = self.hub.borrow_mut();
        hub.devices.remove(device_id);
    }

    /// `device_lost_closure` might never be called.
    pub fn device_set_device_lost_closure(
        &self,
        device_id: DeviceId,
        device_lost_closure: DeviceLostClosure,
    ) {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.set_device_lost_closure(device_lost_closure);
    }

    pub fn device_destroy(&self, device_id: DeviceId) {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.destroy();
    }

    pub fn device_get_internal_counters(&self, device_id: DeviceId) -> wgt::InternalCounters {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.get_internal_counters()
    }

    pub fn device_generate_allocator_report(
        &self,
        device_id: DeviceId,
    ) -> Option<wgt::AllocatorReport> {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.generate_allocator_report()
    }

    pub fn queue_drop(&self, queue_id: QueueId) {
        let mut hub = self.hub.borrow_mut();
        hub.queues.remove(queue_id);
    }

    /// `op.callback` is always called, even in case of errors.
    pub fn buffer_map_async(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
        op: BufferMapOperation,
    ) -> Result<SubmissionIndex, BufferAccessError> {
        let hub = self.hub.borrow();

        let buffer = hub.buffers.get(buffer_id);

        buffer.map_async(offset, size, op)
    }

    pub fn buffer_get_mapped_range(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(NonNull<u8>, u64), BufferAccessError> {
        let hub = self.hub.borrow();

        let buffer = hub.buffers.get(buffer_id);

        buffer.get_mapped_range(offset, size)
    }

    pub fn buffer_unmap(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        let hub = self.hub.borrow();

        let buffer = hub.buffers.get(buffer_id);

        buffer.unmap()
    }

    pub fn device_on_uncaptured_error(
        &self,
        device_id: DeviceId,
        handler: Arc<dyn wgt::error::UncapturedErrorHandler>,
    ) {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.on_uncaptured_error(handler);
    }

    pub fn device_push_error_scope(
        &self,
        device_id: DeviceId,
        error_scope: wgt::error::ErrorFilter,
    ) {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.push_error_scope(error_scope)
    }

    pub fn device_pop_error_scope(
        &self,
        device_id: DeviceId,
    ) -> Result<Option<wgt::error::Error>, EmptyErrorScopeStack> {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.pop_error_scope()
    }

    pub fn device_handle_error(
        &self,
        device_id: DeviceId,
        source: impl WebGpuError + Send + Sync + 'static,
        label: Option<&str>,
        fn_ident: &'static str,
    ) {
        let hub = self.hub.borrow();
        let device = hub.devices.get(device_id);

        device.handle_error(source, label, fn_ident);
    }
}
