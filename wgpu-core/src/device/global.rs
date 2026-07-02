use alloc::{borrow::Cow, boxed::Box, sync::Arc, vec::Vec};
use core::{ptr::NonNull, sync::atomic::Ordering};

#[cfg(feature = "trace")]
use crate::device::trace::{self, IntoTrace};
use crate::{
    api_log,
    binding_model::{
        self, BindGroupEntry, BindingResource, BufferBinding, ResolvedBindGroupDescriptor,
        ResolvedBindGroupEntry, ResolvedBindingResource, ResolvedBufferBinding,
    },
    command::{self, CommandEncoder},
    conv,
    device::{life::WaitIdleError, DeviceError, DeviceLostClosure},
    global::Global,
    id::{self, AdapterId, DeviceId, QueueId, SurfaceId},
    instance::{self, Adapter, Surface},
    pipeline::{
        self, RenderPipelineVertexProcessor, ResolvedComputePipelineDescriptor,
        ResolvedFragmentState, ResolvedGeneralRenderPipelineDescriptor, ResolvedMeshState,
        ResolvedProgrammableStageDescriptor, ResolvedTaskState, ResolvedVertexState,
    },
    present,
    resource::{
        self, BufferAccessError, BufferAccessResult, BufferMapOperation, CreateBufferError,
        Fallible,
    },
    storage::Storage,
    Label, LabelHelpers,
};

use wgt::{BufferAddress, TextureFormat};

use super::{surface_config, UserClosures};

impl Global {
    pub fn adapter_is_surface_supported(
        &self,
        adapter_id: AdapterId,
        surface_id: SurfaceId,
    ) -> bool {
        let surface = self.surfaces.get(surface_id);
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.is_surface_supported(&surface)
    }

    pub fn surface_get_capabilities(
        &self,
        surface_id: SurfaceId,
        adapter_id: AdapterId,
    ) -> Result<wgt::SurfaceCapabilities, instance::GetSurfaceSupportError> {
        profiling::scope!("Surface::get_capabilities");
        self.fetch_adapter_and_surface::<_, _>(surface_id, adapter_id, |adapter, surface| {
            let mut hal_caps = surface.get_capabilities(adapter)?;

            hal_caps.formats.sort_by_key(|fc| !fc.format.is_srgb());

            let usages = conv::map_texture_usage_from_hal(hal_caps.usage);

            // `SurfaceCapabilities::formats` lists only the formats a
            // color-space-unaware application can configure via
            // `SurfaceColorSpace::Auto`, i.e. those for which `Auto` resolves to a
            // concrete color space. (The full `format_capabilities` still reports
            // every color space, including HDR ones, for explicit opt-in.)
            Ok(wgt::SurfaceCapabilities {
                formats: hal_caps
                    .formats
                    .iter()
                    .filter(|fc| {
                        surface_config::resolve_auto_color_space(fc.format, fc.color_spaces)
                            .is_some()
                    })
                    .map(|fc| fc.format)
                    .collect(),
                format_capabilities: hal_caps.formats,
                present_modes: hal_caps.present_modes,
                alpha_modes: hal_caps.composite_alpha_modes,
                usages,
            })
        })
    }

    /// Returns the HDR and luminance characteristics of the display backing
    /// `surface_id` on `adapter_id`.
    ///
    /// Reports the raw display state, independent of the surface's configured
    /// color space; see [`wgt::DisplayHdrInfo`] for per-field platform coverage.
    /// Returns [`wgt::DisplayHdrInfo::default`] (all fields `None`) when nothing
    /// is known: the surface is not on `adapter_id`'s backend, the backend has
    /// no display-query path, or the Metal backend is queried off the main
    /// thread.
    pub fn surface_display_hdr_info(
        &self,
        surface_id: SurfaceId,
        adapter_id: AdapterId,
    ) -> wgt::DisplayHdrInfo {
        profiling::scope!("Surface::display_hdr_info");
        self.fetch_adapter_and_surface(surface_id, adapter_id, |adapter, surface| {
            surface.display_hdr_info(adapter)
        })
    }

    fn fetch_adapter_and_surface<F: FnOnce(&Adapter, &Surface) -> B, B>(
        &self,
        surface_id: SurfaceId,
        adapter_id: AdapterId,
        get_supported_callback: F,
    ) -> B {
        let surface = self.surfaces.get(surface_id);
        let adapter = self.hub.adapters.get(adapter_id);
        get_supported_callback(&adapter, &surface)
    }

    pub fn device_features(&self, device_id: DeviceId) -> wgt::Features {
        let device = self.hub.devices.get(device_id);
        device.features
    }

    pub fn device_limits(&self, device_id: DeviceId) -> wgt::Limits {
        let device = self.hub.devices.get(device_id);
        device.limits.clone()
    }

    pub fn device_adapter_info(&self, device_id: DeviceId) -> wgt::AdapterInfo {
        let device = self.hub.devices.get(device_id);
        device.adapter.get_info()
    }

    pub fn device_downlevel_properties(&self, device_id: DeviceId) -> wgt::DownlevelCapabilities {
        let device = self.hub.devices.get(device_id);
        device.downlevel.clone()
    }

    pub fn device_create_buffer(
        &self,
        device_id: DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Option<id::BufferId>,
    ) -> (id::BufferId, Option<CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = &self.hub;
        let fid = hub.buffers.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (buffer, error) = device.create_buffer(desc);

        let id = fid.assign(buffer);

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
    /// Ideally, we would simply let [`device_create_buffer`] take care of all
    /// of this, but some errors must be detected before we can even construct a
    /// [`wgpu_types::BufferDescriptor`] to give it. For example, the WebGPU API
    /// allows a `GPUBufferDescriptor`'s [`usage`] property to be any WebIDL
    /// `unsigned long` value, but we can't construct a
    /// [`wgpu_types::BufferUsages`] value from values with unassigned bits
    /// set. This means we must validate `usage` before we can call
    /// `device_create_buffer`.
    ///
    /// When that validation fails, we must arrange for the buffer id to be
    /// considered invalid. This method provides the means to do so.
    ///
    /// ["contagious invalidity"]: https://www.w3.org/TR/webgpu/#invalidity
    /// [`GPUBufferDescriptor`]: https://www.w3.org/TR/webgpu/#dictdef-gpubufferdescriptor
    /// [`GPUBuffer`]: https://www.w3.org/TR/webgpu/#gpubuffer
    /// [`wgpu_types::BufferDescriptor`]: wgt::BufferDescriptor
    /// [`device_create_buffer`]: Global::device_create_buffer
    /// [`usage`]: https://www.w3.org/TR/webgpu/#dom-gputexturedescriptor-usage
    /// [`wgpu_types::BufferUsages`]: wgt::BufferUsages
    pub fn create_buffer_error(
        &self,
        device_id: DeviceId,
        id_in: Option<id::BufferId>,
        desc: &resource::BufferDescriptor,
    ) {
        let fid = self.hub.buffers.prepare(id_in);
        let device = self.hub.devices.get(device_id);
        fid.assign(resource::Buffer::invalid(device, desc));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_render_bundle_error(
        &self,
        device_id: DeviceId,
        id_in: Option<id::RenderBundleId>,
        desc: &command::RenderBundleDescriptor,
    ) {
        let device = self.hub.devices.get(device_id);
        let fid = self.hub.render_bundles.prepare(id_in);
        fid.assign(command::RenderBundle::invalid(device, desc));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_texture_error(
        &self,
        device_id: DeviceId,
        id_in: Option<id::TextureId>,
        desc: &resource::TextureDescriptor,
    ) -> id::TextureId {
        let fid = self.hub.textures.prepare(id_in);
        let device = self.hub.devices.get(device_id);
        let texture = device.create_texture_error(desc);
        fid.assign(texture)
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See [`Self::create_buffer_error`] for more context and explanation.
    pub fn create_external_texture_error(
        &self,
        device_id: DeviceId,
        id_in: Option<id::ExternalTextureId>,
        desc: &resource::ExternalTextureDescriptor,
    ) {
        let fid = self.hub.external_textures.prepare(id_in);
        let device = self.hub.devices.get(device_id);
        fid.assign(resource::ExternalTexture::invalid(device, desc));
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
        id_in: Option<id::BindGroupLayoutId>,
        label: Option<Cow<'_, str>>,
    ) {
        let fid = self.hub.bind_group_layouts.prepare(id_in);
        let device = self.hub.devices.get(device_id);
        fid.assign(binding_model::BindGroupLayout::invalid(
            &device,
            label.to_string(),
        ));
    }

    pub fn buffer_destroy(&self, buffer_id: id::BufferId) {
        profiling::scope!("Buffer::destroy");
        api_log!("Buffer::destroy {buffer_id:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id);

        buffer.destroy();
    }

    pub fn buffer_drop(&self, buffer_id: id::BufferId) {
        profiling::scope!("Buffer::drop");
        api_log!("Buffer::drop {buffer_id:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.remove(buffer_id);

        let _ = buffer.unmap();
    }

    pub fn device_create_texture(
        &self,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Option<id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = &self.hub;

        let fid = hub.textures.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (texture, error) = device.create_texture(desc);

        let id = fid.assign(texture);

        (id, error)
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
        id_in: Option<id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture_from_hal");

        let hub = &self.hub;

        let fid = hub.textures.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let error = 'error: {
            let texture = match device.create_texture_from_hal(hal_texture, desc, initial_state) {
                Ok(texture) => texture,
                Err(error) => break 'error error,
            };

            // NB: Any change done through the raw texture handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(
                    texture.to_trace(),
                    desc.clone(),
                ));
            }

            let id = fid.assign(texture);
            api_log!("Device::create_texture({desc:?}) -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Arc::new(resource::Texture::invalid(&device, desc)));
        (id, Some(error))
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
        id_in: Option<id::BufferId>,
    ) -> (id::BufferId, Option<CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = &self.hub;
        let fid = hub.buffers.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (buffer, err) = unsafe { device.create_buffer_from_hal(Box::new(hal_buffer), desc) };

        let id = fid.assign(buffer);

        (id, err)
    }

    pub fn texture_destroy(&self, texture_id: id::TextureId) {
        profiling::scope!("Texture::destroy");
        api_log!("Texture::destroy {texture_id:?}");

        let hub = &self.hub;

        let texture = hub.textures.get(texture_id);

        #[cfg(feature = "trace")]
        if let Some(trace) = texture.device.trace.lock().as_mut() {
            trace.add(trace::Action::DestroyTexture(texture.to_trace()));
        }

        texture.destroy();
    }

    pub fn texture_drop(&self, texture_id: id::TextureId) {
        profiling::scope!("Texture::drop");
        api_log!("Texture::drop {texture_id:?}");

        let hub = &self.hub;

        hub.textures.remove(texture_id);
    }

    pub fn texture_create_view(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: Option<id::TextureViewId>,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        profiling::scope!("Texture::create_view");

        let hub = &self.hub;

        let fid = hub.texture_views.prepare(id_in);

        let texture = hub.textures.get(texture_id);
        let device = &texture.device;

        let (view, error) = device.create_texture_view(&texture, desc);

        let id = fid.assign(view);

        (id, error)
    }

    pub fn texture_view_drop(&self, texture_view_id: id::TextureViewId) {
        let hub = &self.hub;

        let _view = hub.texture_views.remove(texture_view_id);
    }

    pub fn device_create_external_texture(
        &self,
        device_id: DeviceId,
        desc: &resource::ExternalTextureDescriptor,
        planes: &[id::TextureViewId],
        id_in: Option<id::ExternalTextureId>,
    ) -> (
        id::ExternalTextureId,
        Option<resource::CreateExternalTextureError>,
    ) {
        profiling::scope!("Device::create_external_texture");

        let hub = &self.hub;

        let fid = hub.external_textures.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let planes = planes
            .iter()
            .map(|plane_id| self.hub.texture_views.get(*plane_id))
            .collect::<Vec<_>>();

        let (external_texture, error) = device.create_external_texture(desc, &planes);

        let id = fid.assign(external_texture);

        (id, error)
    }

    pub fn external_texture_destroy(&self, external_texture_id: id::ExternalTextureId) {
        profiling::scope!("ExternalTexture::destroy");
        api_log!("ExternalTexture::destroy {external_texture_id:?}");

        let hub = &self.hub;

        let external_texture = hub.external_textures.get(external_texture_id);

        external_texture.destroy();
    }

    pub fn external_texture_drop(&self, external_texture_id: id::ExternalTextureId) {
        profiling::scope!("ExternalTexture::drop");
        api_log!("ExternalTexture::drop {external_texture_id:?}");

        let hub = &self.hub;

        let _external_texture = hub.external_textures.remove(external_texture_id);
    }

    pub fn device_create_sampler(
        &self,
        device_id: DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Option<id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        let hub = &self.hub;
        let fid = hub.samplers.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (sampler, error) = device.create_sampler(desc);

        let id = fid.assign(sampler);

        (id, error)
    }

    pub fn sampler_drop(&self, sampler_id: id::SamplerId) {
        let hub = &self.hub;

        let _sampler = hub.samplers.remove(sampler_id);
    }

    pub fn device_create_bind_group_layout(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        profiling::scope!("Device::create_bind_group_layout");

        let hub = &self.hub;
        let fid = hub.bind_group_layouts.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (bgl, error) = device.create_bind_group_layout(desc);

        let id = fid.assign(bgl);

        api_log!("Device::create_bind_group_layout -> {id:?}");

        (id, error)
    }

    pub fn bind_group_layout_drop(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("BindGroupLayout::drop");
        api_log!("BindGroupLayout::drop {bind_group_layout_id:?}");

        let hub = &self.hub;

        let _layout = hub.bind_group_layouts.remove(bind_group_layout_id);
    }

    pub fn device_create_pipeline_layout(
        &self,
        device_id: DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Option<id::PipelineLayoutId>,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        profiling::scope!("Device::create_pipeline_layout");

        let hub = &self.hub;
        let fid = hub.pipeline_layouts.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let bind_group_layouts = {
            let bind_group_layouts_guard = hub.bind_group_layouts.read();
            desc.bind_group_layouts
                .iter()
                .map(|bgl_id| bgl_id.map(|bgl_id| bind_group_layouts_guard.get(bgl_id)))
                .collect::<Vec<_>>()
        };

        let desc = binding_model::ResolvedPipelineLayoutDescriptor {
            label: desc.label.clone(),
            bind_group_layouts: Cow::Owned(bind_group_layouts),
            immediate_size: desc.immediate_size,
        };

        let (layout, error) = device.create_pipeline_layout(&desc);
        let id = fid.assign(layout);
        (id, error)
    }

    pub fn pipeline_layout_drop(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");
        api_log!("PipelineLayout::drop {pipeline_layout_id:?}");

        let hub = &self.hub;

        let _layout = hub.pipeline_layouts.remove(pipeline_layout_id);
    }

    pub fn device_create_bind_group(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Option<id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("Device::create_bind_group");

        let hub = &self.hub;
        let fid = hub.bind_groups.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            if let Err(e) = device.check_is_valid() {
                break 'error e.into();
            }

            let layout = hub.bind_group_layouts.get(desc.layout);

            fn resolve_entry<'a>(
                e: &BindGroupEntry<'a>,
                buffer_storage: &Storage<Arc<resource::Buffer>>,
                sampler_storage: &Storage<Arc<resource::Sampler>>,
                texture_view_storage: &Storage<Arc<resource::TextureView>>,
                tlas_storage: &Storage<Arc<resource::Tlas>>,
                external_texture_storage: &Storage<Arc<resource::ExternalTexture>>,
            ) -> Result<ResolvedBindGroupEntry<'a>, binding_model::CreateBindGroupError>
            {
                let resolve_buffer =
                    |bb: &BufferBinding| -> Result<_, binding_model::CreateBindGroupError> {
                        let buffer = buffer_storage.get(bb.buffer);
                        buffer.check_is_valid()?;
                        Ok(ResolvedBufferBinding {
                            buffer,
                            offset: bb.offset,
                            size: bb.size,
                        })
                    };
                let resolve_sampler = |id: &id::SamplerId| sampler_storage.get(*id);
                let resolve_view = |id: &id::TextureViewId| texture_view_storage.get(*id);
                let resolve_tlas = |id: &id::TlasId| tlas_storage.get(*id);
                let resolve_external_texture =
                    |id: &id::ExternalTextureId| external_texture_storage.get(*id);
                let resource = match e.resource {
                    BindingResource::Buffer(ref buffer) => {
                        ResolvedBindingResource::Buffer(resolve_buffer(buffer)?)
                    }
                    BindingResource::BufferArray(ref buffers) => {
                        let buffers = buffers
                            .iter()
                            .map(resolve_buffer)
                            .collect::<Result<Vec<_>, _>>()?;
                        ResolvedBindingResource::BufferArray(Cow::Owned(buffers))
                    }
                    BindingResource::Sampler(ref sampler) => {
                        ResolvedBindingResource::Sampler(resolve_sampler(sampler))
                    }
                    BindingResource::SamplerArray(ref samplers) => {
                        let samplers = samplers.iter().map(resolve_sampler).collect::<Vec<_>>();
                        ResolvedBindingResource::SamplerArray(Cow::Owned(samplers))
                    }
                    BindingResource::TextureView(ref view) => {
                        ResolvedBindingResource::TextureView(resolve_view(view))
                    }
                    BindingResource::TextureViewArray(ref views) => {
                        let views = views.iter().map(resolve_view).collect::<Vec<_>>();
                        ResolvedBindingResource::TextureViewArray(Cow::Owned(views))
                    }
                    BindingResource::AccelerationStructure(ref tlas) => {
                        ResolvedBindingResource::AccelerationStructure(resolve_tlas(tlas))
                    }
                    BindingResource::AccelerationStructureArray(ref tlas_array) => {
                        let tlas_array = tlas_array.iter().map(resolve_tlas).collect::<Vec<_>>();
                        ResolvedBindingResource::AccelerationStructureArray(Cow::Owned(tlas_array))
                    }
                    BindingResource::ExternalTexture(ref et) => {
                        ResolvedBindingResource::ExternalTexture(resolve_external_texture(et))
                    }
                };
                Ok(ResolvedBindGroupEntry {
                    binding: e.binding,
                    resource,
                })
            }

            let entries = {
                let buffer_guard = hub.buffers.read();
                let texture_view_guard = hub.texture_views.read();
                let sampler_guard = hub.samplers.read();
                let tlas_guard = hub.tlas_s.read();
                let external_texture_guard = hub.external_textures.read();
                desc.entries
                    .iter()
                    .map(|e| {
                        resolve_entry(
                            e,
                            &buffer_guard,
                            &sampler_guard,
                            &texture_view_guard,
                            &tlas_guard,
                            &external_texture_guard,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let entries = match entries {
                Ok(entries) => Cow::Owned(entries),
                Err(e) => break 'error e,
            };

            let desc = ResolvedBindGroupDescriptor {
                label: desc.label.clone(),
                layout,
                entries,
            };
            #[cfg(feature = "trace")]
            let trace_desc = (&desc).to_trace();

            let bind_group = match device.create_bind_group(desc) {
                Ok(bind_group) => bind_group,
                Err(e) => break 'error e,
            };

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroup(
                    bind_group.to_trace(),
                    trace_desc,
                ));
            }

            let id = fid.assign(Fallible::Valid(bind_group));

            api_log!("Device::create_bind_group -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn bind_group_drop(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("BindGroup::drop");
        api_log!("BindGroup::drop {bind_group_id:?}");

        let hub = &self.hub;

        let _bind_group = hub.bind_groups.remove(bind_group_id);

        #[cfg(feature = "trace")]
        if let Ok(bind_group) = _bind_group.get() {
            if let Some(t) = bind_group.device.trace.lock().as_mut() {
                t.add(trace::Action::DropBindGroup(bind_group.to_trace()));
            }
        }
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
        id_in: Option<id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = &self.hub;
        let fid = hub.shader_modules.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (shader, error) = device.create_shader_module(desc, source);

        let id = fid.assign(shader);

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
        id_in: Option<id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        let hub = &self.hub;
        let fid = hub.shader_modules.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (shader, error) = unsafe { device.create_shader_module_passthrough(desc) };

        let id = fid.assign(shader);

        (id, error)
    }

    pub fn shader_module_drop(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");
        api_log!("ShaderModule::drop {shader_module_id:?}");

        let hub = &self.hub;

        let _shader_module = hub.shader_modules.remove(shader_module_id);
    }

    pub fn device_create_command_encoder(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Option<id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = &self.hub;
        let fid = hub.command_encoders.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let error = 'error: {
            let cmd_enc = match device.create_command_encoder(&desc.label) {
                Ok(cmd_enc) => cmd_enc,
                Err(e) => break 'error e,
            };

            let id = fid.assign(cmd_enc);
            api_log!("Device::create_command_encoder -> {id:?}");
            return (id, None);
        };

        let id = fid.assign(Arc::new(CommandEncoder::new_invalid(
            &device,
            &desc.label,
            error.clone().into(),
        )));
        (id, Some(error))
    }

    pub fn command_encoder_drop(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");
        api_log!("CommandEncoder::drop {command_encoder_id:?}");
        let _cmd_enc = self.hub.command_encoders.remove(command_encoder_id);
    }

    pub fn command_buffer_drop(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        api_log!("CommandBuffer::drop {command_buffer_id:?}");
        let _cmd_buf = self.hub.command_buffers.remove(command_buffer_id);
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
    ) -> (
        Box<command::RenderBundleEncoder>,
        Option<command::CreateRenderBundleError>,
    ) {
        profiling::scope!("Device::create_render_bundle_encoder");
        api_log!("Device::device_create_render_bundle_encoder");
        let device = self.hub.devices.get(device_id);
        let (encoder, error) =
            match command::RenderBundleEncoder::new(desc, Some(&device), device_id) {
                Ok(encoder) => (encoder, None),
                Err(e) => (command::RenderBundleEncoder::dummy(device_id), Some(e)),
            };
        (Box::new(encoder), error)
    }

    pub fn device_create_render_bundle_encoder_with_id(
        &self,
        device_id: DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
        id_in: Option<id::RenderBundleEncoderId>,
    ) -> (
        id::RenderBundleEncoderId,
        Option<command::CreateRenderBundleError>,
    ) {
        let fid = self.hub.render_bundle_encoders.prepare(id_in);

        let (render_bundle_encoder, error) =
            self.device_create_render_bundle_encoder(device_id, desc);

        // no lock rank here because only one thread should be using compute pass
        // and it's only used by id variants of compute pass methods on global
        // so no deadlock (or concurrent lock) should happen in practise
        let id = fid.assign(Arc::new(parking_lot::Mutex::new(*render_bundle_encoder)));

        (id, error)
    }

    pub fn render_bundle_encoder_finish(
        &self,
        bundle_encoder: &mut command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Option<id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("RenderBundleEncoder::finish");

        let hub = &self.hub;

        let fid = hub.render_bundles.prepare(id_in);

        let device = self.hub.devices.get(bundle_encoder.parent());

        let (render_bundle, error) = bundle_encoder.finish(desc, &device, hub);

        let id = fid.assign(render_bundle);

        (id, error)
    }

    pub fn render_bundle_encoder_finish_with_id(
        &self,
        render_bundle_encoder_id: id::RenderBundleEncoderId,
        desc: &command::RenderBundleDescriptor,
        id_in: Option<id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        let bundle_encoder = self
            .hub
            .render_bundle_encoders
            .get(render_bundle_encoder_id);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        let (id, error) = self.render_bundle_encoder_finish(&mut bundle_encoder, desc, id_in);

        (id, error)
    }

    pub fn render_bundle_encoder_drop(&self, render_bundle_encoder_id: id::RenderBundleEncoderId) {
        let hub = &self.hub;

        let _bundle_encoder = hub.render_bundle_encoders.remove(render_bundle_encoder_id);
    }

    pub fn render_bundle_drop(&self, render_bundle_id: id::RenderBundleId) {
        let hub = &self.hub;

        let _bundle = hub.render_bundles.remove(render_bundle_id);
    }

    pub fn device_create_query_set(
        &self,
        device_id: DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: Option<id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("Device::create_query_set");

        let hub = &self.hub;
        let fid = hub.query_sets.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (query_set, error) = device.create_query_set(desc);

        let id = fid.assign(query_set);

        (id, error)
    }

    pub fn query_set_destroy(&self, query_set_id: id::QuerySetId) {
        let hub = &self.hub;

        let query_set = hub.query_sets.get(query_set_id);

        query_set.destroy();
    }

    pub fn query_set_drop(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");
        api_log!("QuerySet::drop {query_set_id:?}");

        let hub = &self.hub;

        let _query_set = hub.query_sets.remove(query_set_id);
    }

    pub fn device_create_render_pipeline(
        &self,
        device_id: DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Option<id::RenderPipelineId>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("Device::create_render_pipeline");

        let hub = &self.hub;

        let fid = hub.render_pipelines.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        self.device_create_general_render_pipeline(desc.clone().into(), device, fid)
    }

    pub fn device_create_mesh_pipeline(
        &self,
        device_id: DeviceId,
        desc: &pipeline::MeshPipelineDescriptor,
        id_in: Option<id::RenderPipelineId>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        let hub = &self.hub;

        let fid = hub.render_pipelines.prepare(id_in);

        let device = self.hub.devices.get(device_id);
        self.device_create_general_render_pipeline(desc.clone().into(), device, fid)
    }

    fn device_create_general_render_pipeline(
        &self,
        desc: pipeline::GeneralRenderPipelineDescriptor,
        device: Arc<crate::device::resource::Device>,
        fid: crate::registry::FutureId<Arc<pipeline::RenderPipeline>>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("Device::create_general_render_pipeline");

        let hub = &self.hub;

        // eventually there will be no error handling here only id to object mapping
        let error = 'error: {
            // until then we also need this
            if let Err(e) = device.check_is_valid() {
                break 'error e.into();
            }

            let layout = desc.layout.map(|layout| hub.pipeline_layouts.get(layout));

            let cache = desc.cache.map(|cache| hub.pipeline_caches.get(cache));

            let vertex = match desc.vertex {
                RenderPipelineVertexProcessor::Vertex(ref vertex) => {
                    let module = hub.shader_modules.get(vertex.stage.module);
                    let stage = ResolvedProgrammableStageDescriptor {
                        module,
                        entry_point: vertex.stage.entry_point.clone(),
                        constants: vertex.stage.constants.clone(),
                        zero_initialize_workgroup_memory: vertex
                            .stage
                            .zero_initialize_workgroup_memory,
                    };
                    RenderPipelineVertexProcessor::Vertex(ResolvedVertexState {
                        stage,
                        buffers: vertex.buffers.clone(),
                    })
                }
                RenderPipelineVertexProcessor::Mesh(ref task, ref mesh) => {
                    let task_module = if let Some(task) = task {
                        let module = hub.shader_modules.get(task.stage.module);

                        let state = ResolvedProgrammableStageDescriptor {
                            module,
                            entry_point: task.stage.entry_point.clone(),
                            constants: task.stage.constants.clone(),
                            zero_initialize_workgroup_memory: task
                                .stage
                                .zero_initialize_workgroup_memory,
                        };
                        Some(ResolvedTaskState { stage: state })
                    } else {
                        None
                    };
                    let mesh_module = hub.shader_modules.get(mesh.stage.module);
                    let mesh_stage = ResolvedProgrammableStageDescriptor {
                        module: mesh_module,
                        entry_point: mesh.stage.entry_point.clone(),
                        constants: mesh.stage.constants.clone(),
                        zero_initialize_workgroup_memory: mesh
                            .stage
                            .zero_initialize_workgroup_memory,
                    };
                    RenderPipelineVertexProcessor::Mesh(
                        task_module,
                        ResolvedMeshState { stage: mesh_stage },
                    )
                }
            };

            let fragment = if let Some(ref state) = desc.fragment {
                let module = hub.shader_modules.get(state.stage.module);

                let stage = ResolvedProgrammableStageDescriptor {
                    module,
                    entry_point: state.stage.entry_point.clone(),
                    constants: state.stage.constants.clone(),
                    zero_initialize_workgroup_memory: state.stage.zero_initialize_workgroup_memory,
                };
                Some(ResolvedFragmentState {
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

            let id = fid.assign(pipeline);
            api_log!("Device::create_render_pipeline -> {id:?}");

            return (id, error);
        };

        let id = fid.assign(pipeline::RenderPipeline::invalid(
            device.clone(),
            desc.label.to_string(),
        ));

        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = &self.hub;

        let fid = hub.bind_group_layouts.prepare(id_in);

        let pipeline = hub.render_pipelines.get(pipeline_id);

        let (bgl, error) = pipeline.get_bind_group_layout(index);

        let id = fid.assign(bgl);

        (id, error)
    }

    pub fn render_pipeline_drop(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("RenderPipeline::drop");
        api_log!("RenderPipeline::drop {render_pipeline_id:?}");

        let hub = &self.hub;

        let _pipeline = hub.render_pipelines.remove(render_pipeline_id);
    }

    pub fn device_create_compute_pipeline(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: Option<id::ComputePipelineId>,
    ) -> (
        id::ComputePipelineId,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        profiling::scope!("Device::create_compute_pipeline");

        let hub = &self.hub;

        let fid = hub.compute_pipelines.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        // eventually there will be no error handling here only id to object mapping
        let error = 'error: {
            // until then we also need this
            if let Err(e) = device.check_is_valid() {
                break 'error e.into();
            }

            let layout = desc.layout.map(|layout| hub.pipeline_layouts.get(layout));

            let cache = desc.cache.map(|cache| hub.pipeline_caches.get(cache));

            let module = hub.shader_modules.get(desc.stage.module);

            let stage = ResolvedProgrammableStageDescriptor {
                module,
                entry_point: desc.stage.entry_point.clone(),
                constants: desc.stage.constants.clone(),
                zero_initialize_workgroup_memory: desc.stage.zero_initialize_workgroup_memory,
            };

            let desc = ResolvedComputePipelineDescriptor {
                label: desc.label.clone(),
                layout,
                stage,
                cache,
            };

            let (pipeline, error) = device.create_compute_pipeline(desc);

            let id = fid.assign(pipeline);
            api_log!("Device::create_compute_pipeline -> {id:?}");

            return (id, error);
        };

        let id = fid.assign(pipeline::ComputePipeline::invalid(
            device,
            desc.label.to_string(),
        ));

        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = &self.hub;

        let fid = hub.bind_group_layouts.prepare(id_in);

        let pipeline = hub.compute_pipelines.get(pipeline_id);

        let (bgl, error) = pipeline.get_bind_group_layout(index);

        let id = fid.assign(bgl);

        (id, error)
    }

    pub fn compute_pipeline_drop(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("ComputePipeline::drop");
        api_log!("ComputePipeline::drop {compute_pipeline_id:?}");

        let hub = &self.hub;

        let _pipeline = hub.compute_pipelines.remove(compute_pipeline_id);
    }

    /// # Safety
    /// The `data` argument of `desc` must have been returned by
    /// [Self::pipeline_cache_get_data] for the same adapter
    pub unsafe fn device_create_pipeline_cache(
        &self,
        device_id: DeviceId,
        desc: &pipeline::PipelineCacheDescriptor<'_>,
        id_in: Option<id::PipelineCacheId>,
    ) -> (
        id::PipelineCacheId,
        Option<pipeline::CreatePipelineCacheError>,
    ) {
        profiling::scope!("Device::create_pipeline_cache");

        let hub = &self.hub;

        let fid = hub.pipeline_caches.prepare(id_in);
        let device = self.hub.devices.get(device_id);

        let (cache, error) = unsafe { device.create_pipeline_cache(desc) };

        let id = fid.assign(cache);

        (id, error)
    }

    pub fn pipeline_cache_drop(&self, pipeline_cache_id: id::PipelineCacheId) {
        let hub = &self.hub;

        let _cache = hub.pipeline_caches.remove(pipeline_cache_id);
    }

    pub fn surface_configure(
        &self,
        surface_id: SurfaceId,
        device_id: DeviceId,
        config: &wgt::SurfaceConfiguration<Vec<TextureFormat>>,
    ) -> Option<present::ConfigureSurfaceError> {
        let device = self.hub.devices.get(device_id);
        let surface = self.surfaces.get(surface_id);

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(trace::Action::ConfigureSurface(
                surface.to_trace(),
                config.clone(),
            ));
        }

        device.configure_surface(&surface, config)
    }

    /// Check `device_id` for freeable resources and completed buffer mappings.
    ///
    /// Return `queue_empty` indicating whether there are more queue submissions still in flight.
    pub fn device_poll(
        &self,
        device_id: DeviceId,
        poll_type: wgt::PollType<crate::SubmissionIndex>,
    ) -> Result<wgt::PollStatus, WaitIdleError> {
        api_log!("Device::poll {poll_type:?}");

        let device = self.hub.devices.get(device_id);

        let (closures, result) = device.poll_and_return_closures(poll_type);

        closures.fire();

        result
    }

    /// Poll all devices belonging to the specified backend.
    ///
    /// If `force_wait` is true, block until all buffer mappings are done.
    ///
    /// Return `all_queue_empty` indicating whether there are more queue
    /// submissions still in flight.
    fn poll_all_devices_of_api(
        &self,
        force_wait: bool,
        closure_list: &mut UserClosures,
    ) -> Result<bool, WaitIdleError> {
        profiling::scope!("poll_device");

        let hub = &self.hub;
        let mut all_queue_empty = true;
        {
            let device_guard = hub.devices.read();

            for (_id, device) in device_guard.iter() {
                let poll_type = if force_wait {
                    // TODO(#8286): Should expose timeout to poll_all.
                    wgt::PollType::wait_indefinitely()
                } else {
                    wgt::PollType::Poll
                };

                let (closures, result) = device.poll_and_return_closures(poll_type);

                let is_queue_empty = matches!(result, Ok(wgt::PollStatus::QueueEmpty));

                all_queue_empty &= is_queue_empty;

                closure_list.extend(closures);
            }
        }

        Ok(all_queue_empty)
    }

    /// Poll all devices on all backends.
    ///
    /// This is the implementation of `wgpu::Instance::poll_all`.
    ///
    /// Return `all_queue_empty` indicating whether there are more queue
    /// submissions still in flight.
    pub fn poll_all_devices(&self, force_wait: bool) -> Result<bool, WaitIdleError> {
        api_log!("poll_all_devices");
        let mut closures = UserClosures::default();
        let all_queue_empty = self.poll_all_devices_of_api(force_wait, &mut closures)?;

        closures.fire();

        Ok(all_queue_empty)
    }

    /// # Safety
    ///
    /// - See [wgpu::Device::start_graphics_debugger_capture][api] for details the safety.
    ///
    /// [api]: ../../wgpu/struct.Device.html#method.start_graphics_debugger_capture
    pub unsafe fn device_start_graphics_debugger_capture(&self, device_id: DeviceId) {
        unsafe {
            self.hub
                .devices
                .get(device_id)
                .start_graphics_debugger_capture();
        }
    }

    /// # Safety
    ///
    /// - See [wgpu::Device::stop_graphics_debugger_capture][api] for details the safety.
    ///
    /// [api]: ../../wgpu/struct.Device.html#method.stop_graphics_debugger_capture
    pub unsafe fn device_stop_graphics_debugger_capture(&self, device_id: DeviceId) {
        unsafe {
            self.hub
                .devices
                .get(device_id)
                .stop_graphics_debugger_capture();
        }
    }

    pub fn pipeline_cache_get_data(&self, id: id::PipelineCacheId) -> Option<Vec<u8>> {
        let hub = &self.hub;

        hub.pipeline_caches.get(id).get_data()
    }

    pub fn device_drop(&self, device_id: DeviceId) {
        profiling::scope!("Device::drop");
        api_log!("Device::drop {device_id:?}");

        self.hub.devices.remove(device_id);
    }

    /// `device_lost_closure` might never be called.
    pub fn device_set_device_lost_closure(
        &self,
        device_id: DeviceId,
        device_lost_closure: DeviceLostClosure,
    ) {
        let device = self.hub.devices.get(device_id);

        device
            .device_lost_closure
            .lock()
            .replace(device_lost_closure);
    }

    pub fn device_destroy(&self, device_id: DeviceId) {
        api_log!("Device::destroy {device_id:?}");

        let device = self.hub.devices.get(device_id);

        // Follow the steps at
        // https://gpuweb.github.io/gpuweb/#dom-gpudevice-destroy.
        // It's legal to call destroy multiple times, but if the device
        // is already invalid, there's nothing more to do. There's also
        // no need to return an error.
        if !device.is_valid() {
            return;
        }

        // The last part of destroy is to lose the device. The spec says
        // delay that until all "currently-enqueued operations on any
        // queue on this device are completed." This is accomplished by
        // setting valid to false, and then relying upon maintain to
        // check for empty queues and a DeviceLostClosure. At that time,
        // the DeviceLostClosure will be called with "destroyed" as the
        // reason.
        device.valid.store(false, Ordering::Release);
    }

    pub fn device_get_internal_counters(&self, device_id: DeviceId) -> wgt::InternalCounters {
        let device = self.hub.devices.get(device_id);
        wgt::InternalCounters {
            hal: device.get_hal_counters(),
            core: wgt::CoreCounters {},
        }
    }

    pub fn device_generate_allocator_report(
        &self,
        device_id: DeviceId,
    ) -> Option<wgt::AllocatorReport> {
        let device = self.hub.devices.get(device_id);
        device.generate_allocator_report()
    }

    #[cfg(feature = "trace")]
    pub fn device_take_trace(
        &self,
        device_id: DeviceId,
    ) -> Option<Box<dyn trace::Trace + Send + Sync + 'static>> {
        let device = self.hub.devices.get(device_id);
        device.take_trace()
    }

    pub fn queue_drop(&self, queue_id: QueueId) {
        profiling::scope!("Queue::drop");
        api_log!("Queue::drop {queue_id:?}");

        self.hub.queues.remove(queue_id);
    }

    /// `op.callback` is always called, even in case of errors.
    pub fn buffer_map_async(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
        op: BufferMapOperation,
    ) -> Result<crate::SubmissionIndex, BufferAccessError> {
        profiling::scope!("Buffer::map_async");
        api_log!("Buffer::map_async {buffer_id:?} offset {offset:?} size {size:?} op: {op:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id);

        buffer.map_async(offset, size, op)
    }

    pub fn buffer_get_mapped_range(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(NonNull<u8>, u64), BufferAccessError> {
        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id);

        buffer.get_mapped_range(offset, size)
    }

    pub fn buffer_unmap(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        profiling::scope!("unmap", "Buffer");
        api_log!("Buffer::unmap {buffer_id:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id);

        buffer.unmap()
    }
}
