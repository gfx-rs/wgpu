#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    api_log,
    binding_model::{
        self, BindGroupEntry, BindingResource, BufferBinding, ResolvedBindGroupDescriptor,
        ResolvedBindGroupEntry, ResolvedBindingResource, ResolvedBufferBinding,
    },
    command::{self, CommandBuffer},
    conv,
    device::{bgl, life::WaitIdleError, DeviceError, DeviceLostClosure, DeviceLostReason},
    global::Global,
    hal_api::HalApi,
    id::{self, AdapterId, DeviceId, QueueId, SurfaceId},
    instance::{self, Adapter, Surface},
    pipeline::{
        self, ResolvedComputePipelineDescriptor, ResolvedFragmentState,
        ResolvedProgrammableStageDescriptor, ResolvedRenderPipelineDescriptor, ResolvedVertexState,
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

use std::{
    borrow::Cow,
    ptr::NonNull,
    sync::{atomic::Ordering, Arc},
};

use super::{ImplicitPipelineIds, UserClosures};

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

            hal_caps.formats.sort_by_key(|f| !f.is_srgb());

            let usages = conv::map_texture_usage_from_hal(hal_caps.usage);

            Ok(wgt::SurfaceCapabilities {
                formats: hal_caps.formats,
                present_modes: hal_caps.present_modes,
                alpha_modes: hal_caps.composite_alpha_modes,
                usages,
            })
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let mut desc = desc.clone();
                let mapped_at_creation = std::mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsages::COPY_DST;
                }
                trace.add(trace::Action::CreateBuffer(fid.id(), desc));
            }

            let buffer = match device.create_buffer(desc) {
                Ok(buffer) => buffer,
                Err(e) => {
                    break 'error e;
                }
            };

            let id = fid.assign(Fallible::Valid(buffer));

            api_log!(
                "Device::create_buffer({:?}{}) -> {id:?}",
                desc.label.as_deref().unwrap_or(""),
                if desc.mapped_at_creation {
                    ", mapped_at_creation"
                } else {
                    ""
                }
            );

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
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
        id_in: Option<id::BufferId>,
        desc: &resource::BufferDescriptor,
    ) {
        let fid = self.hub.buffers.prepare(id_in);
        fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
    }

    pub fn create_render_bundle_error(
        &self,
        id_in: Option<id::RenderBundleId>,
        desc: &command::RenderBundleDescriptor,
    ) {
        let fid = self.hub.render_bundles.prepare(id_in);
        fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See `create_buffer_error` for more context and explanation.
    pub fn create_texture_error(
        &self,
        id_in: Option<id::TextureId>,
        desc: &resource::TextureDescriptor,
    ) {
        let fid = self.hub.textures.prepare(id_in);
        fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
    }

    #[cfg(feature = "replay")]
    pub fn device_set_buffer_data(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &[u8],
    ) -> BufferAccessResult {
        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id).get()?;

        let device = &buffer.device;

        device.check_is_valid()?;
        buffer.check_usage(wgt::BufferUsages::MAP_WRITE)?;

        let last_submission = device
            .lock_life()
            .get_buffer_latest_submission_index(&buffer);

        if let Some(last_submission) = last_submission {
            device.wait_for_submit(last_submission)?;
        }

        let snatch_guard = device.snatchable_lock.read();
        let raw_buf = buffer.try_raw(&snatch_guard)?;

        let mapping = unsafe {
            device
                .raw()
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
        }
        .map_err(|e| device.handle_hal_error(e))?;

        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len()) };

        if !mapping.is_coherent {
            #[allow(clippy::single_range_in_vec_init)]
            unsafe {
                device
                    .raw()
                    .flush_mapped_ranges(raw_buf, &[offset..offset + data.len() as u64])
            };
        }

        unsafe { device.raw().unmap_buffer(raw_buf) };

        Ok(())
    }

    pub fn buffer_destroy(&self, buffer_id: id::BufferId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Buffer::destroy");
        api_log!("Buffer::destroy {buffer_id:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id).get()?;

        #[cfg(feature = "trace")]
        if let Some(trace) = buffer.device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeBuffer(buffer_id));
        }

        let _ = buffer.unmap(
            #[cfg(feature = "trace")]
            buffer_id,
        );

        buffer.destroy()
    }

    pub fn buffer_drop(&self, buffer_id: id::BufferId) {
        profiling::scope!("Buffer::drop");
        api_log!("Buffer::drop {buffer_id:?}");

        let hub = &self.hub;

        let buffer = match hub.buffers.remove(buffer_id).get() {
            Ok(buffer) => buffer,
            Err(_) => {
                return;
            }
        };

        #[cfg(feature = "trace")]
        if let Some(t) = buffer.device.trace.lock().as_mut() {
            t.add(trace::Action::DestroyBuffer(buffer_id));
        }

        let _ = buffer.unmap(
            #[cfg(feature = "trace")]
            buffer_id,
        );
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let texture = match device.create_texture(desc) {
                Ok(texture) => texture,
                Err(error) => break 'error error,
            };

            let id = fid.assign(Fallible::Valid(texture));
            api_log!("Device::create_texture({desc:?}) -> {id:?}");

            return (id, None);
        };

        log::error!("Device::create_texture error: {error}");

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    /// # Safety
    ///
    /// - `hal_texture` must be created from `device_id` corresponding raw handle.
    /// - `hal_texture` must be created respecting `desc`
    /// - `hal_texture` must be initialized
    pub unsafe fn create_texture_from_hal(
        &self,
        hal_texture: Box<dyn hal::DynTexture>,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Option<id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture_from_hal");

        let hub = &self.hub;

        let fid = hub.textures.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            // NB: Any change done through the raw texture handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let texture = match device.create_texture_from_hal(hal_texture, desc) {
                Ok(texture) => texture,
                Err(error) => break 'error error,
            };

            let id = fid.assign(Fallible::Valid(texture));
            api_log!("Device::create_texture({desc:?}) -> {id:?}");

            return (id, None);
        };

        log::error!("Device::create_texture error: {error}");

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    /// # Safety
    ///
    /// - `hal_buffer` must be created from `device_id` corresponding raw handle.
    /// - `hal_buffer` must be created respecting `desc`
    /// - `hal_buffer` must be initialized
    pub unsafe fn create_buffer_from_hal<A: HalApi>(
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

        // NB: Any change done through the raw buffer handle will not be
        // recorded in the replay
        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            trace.add(trace::Action::CreateBuffer(fid.id(), desc.clone()));
        }

        let (buffer, err) = device.create_buffer_from_hal(Box::new(hal_buffer), desc);

        let id = fid.assign(buffer);
        api_log!("Device::create_buffer -> {id:?}");

        (id, err)
    }

    pub fn texture_destroy(&self, texture_id: id::TextureId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Texture::destroy");
        api_log!("Texture::destroy {texture_id:?}");

        let hub = &self.hub;

        let texture = hub.textures.get(texture_id).get()?;

        #[cfg(feature = "trace")]
        if let Some(trace) = texture.device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeTexture(texture_id));
        }

        texture.destroy()
    }

    pub fn texture_drop(&self, texture_id: id::TextureId) {
        profiling::scope!("Texture::drop");
        api_log!("Texture::drop {texture_id:?}");

        let hub = &self.hub;

        let _texture = hub.textures.remove(texture_id);
        #[cfg(feature = "trace")]
        if let Ok(texture) = _texture.get() {
            if let Some(t) = texture.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyTexture(texture_id));
            }
        }
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

        let error = 'error: {
            let texture = match hub.textures.get(texture_id).get() {
                Ok(texture) => texture,
                Err(e) => break 'error e.into(),
            };
            let device = &texture.device;

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTextureView {
                    id: fid.id(),
                    parent_id: texture_id,
                    desc: desc.clone(),
                });
            }

            let view = match device.create_texture_view(&texture, desc) {
                Ok(view) => view,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(view));

            api_log!("Texture::create_view({texture_id:?}) -> {id:?}");

            return (id, None);
        };

        log::error!("Texture::create_view({texture_id:?}) error: {error}");
        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn texture_view_drop(
        &self,
        texture_view_id: id::TextureViewId,
    ) -> Result<(), resource::TextureViewDestroyError> {
        profiling::scope!("TextureView::drop");
        api_log!("TextureView::drop {texture_view_id:?}");

        let hub = &self.hub;

        let _view = hub.texture_views.remove(texture_view_id);

        #[cfg(feature = "trace")]
        if let Ok(view) = _view.get() {
            if let Some(t) = view.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyTextureView(texture_view_id));
            }
        }
        Ok(())
    }

    pub fn device_create_sampler(
        &self,
        device_id: DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Option<id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        profiling::scope!("Device::create_sampler");

        let hub = &self.hub;
        let fid = hub.samplers.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateSampler(fid.id(), desc.clone()));
            }

            let sampler = match device.create_sampler(desc) {
                Ok(sampler) => sampler,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(sampler));
            api_log!("Device::create_sampler -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn sampler_drop(&self, sampler_id: id::SamplerId) {
        profiling::scope!("Sampler::drop");
        api_log!("Sampler::drop {sampler_id:?}");

        let hub = &self.hub;

        let _sampler = hub.samplers.remove(sampler_id);

        #[cfg(feature = "trace")]
        if let Ok(sampler) = _sampler.get() {
            if let Some(t) = sampler.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroySampler(sampler_id));
            }
        }
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroupLayout(fid.id(), desc.clone()));
            }

            // this check can't go in the body of `create_bind_group_layout` since the closure might not get called
            if let Err(e) = device.check_is_valid() {
                break 'error e.into();
            }

            let entry_map = match bgl::EntryMap::from_entries(&device.limits, &desc.entries) {
                Ok(map) => map,
                Err(e) => break 'error e,
            };

            let bgl_result = device.bgl_pool.get_or_init(entry_map, |entry_map| {
                let bgl =
                    device.create_bind_group_layout(&desc.label, entry_map, bgl::Origin::Pool)?;
                bgl.exclusive_pipeline
                    .set(binding_model::ExclusivePipeline::None)
                    .unwrap();
                Ok(bgl)
            });

            let layout = match bgl_result {
                Ok(layout) => layout,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(layout.clone()));

            api_log!("Device::create_bind_group_layout -> {id:?}");
            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn bind_group_layout_drop(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("BindGroupLayout::drop");
        api_log!("BindGroupLayout::drop {bind_group_layout_id:?}");

        let hub = &self.hub;

        let _layout = hub.bind_group_layouts.remove(bind_group_layout_id);

        #[cfg(feature = "trace")]
        if let Ok(layout) = _layout.get() {
            if let Some(t) = layout.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyBindGroupLayout(bind_group_layout_id));
            }
        }
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreatePipelineLayout(fid.id(), desc.clone()));
            }

            let bind_group_layouts = {
                let bind_group_layouts_guard = hub.bind_group_layouts.read();
                desc.bind_group_layouts
                    .iter()
                    .map(|bgl_id| bind_group_layouts_guard.get(*bgl_id).get())
                    .collect::<Result<Vec<_>, _>>()
            };

            let bind_group_layouts = match bind_group_layouts {
                Ok(bind_group_layouts) => bind_group_layouts,
                Err(e) => break 'error e.into(),
            };

            let desc = binding_model::ResolvedPipelineLayoutDescriptor {
                label: desc.label.clone(),
                bind_group_layouts: Cow::Owned(bind_group_layouts),
                push_constant_ranges: desc.push_constant_ranges.clone(),
            };

            let layout = match device.create_pipeline_layout(&desc) {
                Ok(layout) => layout,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(layout));
            api_log!("Device::create_pipeline_layout -> {id:?}");
            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn pipeline_layout_drop(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");
        api_log!("PipelineLayout::drop {pipeline_layout_id:?}");

        let hub = &self.hub;

        let _layout = hub.pipeline_layouts.remove(pipeline_layout_id);

        #[cfg(feature = "trace")]
        if let Ok(layout) = _layout.get() {
            if let Some(t) = layout.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyPipelineLayout(pipeline_layout_id));
            }
        }
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

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroup(fid.id(), desc.clone()));
            }

            let layout = match hub.bind_group_layouts.get(desc.layout).get() {
                Ok(layout) => layout,
                Err(e) => break 'error e.into(),
            };

            fn resolve_entry<'a>(
                e: &BindGroupEntry<'a>,
                buffer_storage: &Storage<Fallible<resource::Buffer>>,
                sampler_storage: &Storage<Fallible<resource::Sampler>>,
                texture_view_storage: &Storage<Fallible<resource::TextureView>>,
            ) -> Result<ResolvedBindGroupEntry<'a>, binding_model::CreateBindGroupError>
            {
                let resolve_buffer = |bb: &BufferBinding| {
                    buffer_storage
                        .get(bb.buffer_id)
                        .get()
                        .map(|buffer| ResolvedBufferBinding {
                            buffer,
                            offset: bb.offset,
                            size: bb.size,
                        })
                        .map_err(binding_model::CreateBindGroupError::from)
                };
                let resolve_sampler = |id: &id::SamplerId| {
                    sampler_storage
                        .get(*id)
                        .get()
                        .map_err(binding_model::CreateBindGroupError::from)
                };
                let resolve_view = |id: &id::TextureViewId| {
                    texture_view_storage
                        .get(*id)
                        .get()
                        .map_err(binding_model::CreateBindGroupError::from)
                };
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
                        ResolvedBindingResource::Sampler(resolve_sampler(sampler)?)
                    }
                    BindingResource::SamplerArray(ref samplers) => {
                        let samplers = samplers
                            .iter()
                            .map(resolve_sampler)
                            .collect::<Result<Vec<_>, _>>()?;
                        ResolvedBindingResource::SamplerArray(Cow::Owned(samplers))
                    }
                    BindingResource::TextureView(ref view) => {
                        ResolvedBindingResource::TextureView(resolve_view(view)?)
                    }
                    BindingResource::TextureViewArray(ref views) => {
                        let views = views
                            .iter()
                            .map(resolve_view)
                            .collect::<Result<Vec<_>, _>>()?;
                        ResolvedBindingResource::TextureViewArray(Cow::Owned(views))
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
                desc.entries
                    .iter()
                    .map(|e| resolve_entry(e, &buffer_guard, &sampler_guard, &texture_view_guard))
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

            let bind_group = match device.create_bind_group(desc) {
                Ok(bind_group) => bind_group,
                Err(e) => break 'error e,
            };

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
        if let Ok(_bind_group) = _bind_group.get() {
            if let Some(t) = _bind_group.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyBindGroup(bind_group_id));
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let data = match source {
                    #[cfg(feature = "wgsl")]
                    pipeline::ShaderModuleSource::Wgsl(ref code) => {
                        trace.make_binary("wgsl", code.as_bytes())
                    }
                    #[cfg(feature = "glsl")]
                    pipeline::ShaderModuleSource::Glsl(ref code, _) => {
                        trace.make_binary("glsl", code.as_bytes())
                    }
                    #[cfg(feature = "spirv")]
                    pipeline::ShaderModuleSource::SpirV(ref code, _) => {
                        trace.make_binary("spirv", bytemuck::cast_slice::<u32, u8>(code))
                    }
                    pipeline::ShaderModuleSource::Naga(ref module) => {
                        let string =
                            ron::ser::to_string_pretty(module, ron::ser::PrettyConfig::default())
                                .unwrap();
                        trace.make_binary("ron", string.as_bytes())
                    }
                    pipeline::ShaderModuleSource::Dummy(_) => {
                        panic!("found `ShaderModuleSource::Dummy`")
                    }
                };
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader = match device.create_shader_module(desc, source) {
                Ok(shader) => shader,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(shader));
            api_log!("Device::create_shader_module -> {id:?}");
            return (id, None);
        };

        log::error!("Device::create_shader_module error: {error}");

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    // Unsafe-ness of internal calls has little to do with unsafe-ness of this.
    #[allow(unused_unsafe)]
    /// # Safety
    ///
    /// This function passes SPIR-V binary to the backend as-is and can potentially result in a
    /// driver crash.
    pub unsafe fn device_create_shader_module_spirv(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: Cow<[u32]>,
        id_in: Option<id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = &self.hub;
        let fid = hub.shader_modules.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let data = trace.make_binary("spv", unsafe {
                    std::slice::from_raw_parts(source.as_ptr().cast::<u8>(), source.len() * 4)
                });
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader = match unsafe { device.create_shader_module_spirv(desc, &source) } {
                Ok(shader) => shader,
                Err(e) => break 'error e,
            };
            let id = fid.assign(Fallible::Valid(shader));
            api_log!("Device::create_shader_module_spirv -> {id:?}");
            return (id, None);
        };

        log::error!("Device::create_shader_module_spirv error: {error}");

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn shader_module_drop(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");
        api_log!("ShaderModule::drop {shader_module_id:?}");

        let hub = &self.hub;

        let _shader_module = hub.shader_modules.remove(shader_module_id);

        #[cfg(feature = "trace")]
        if let Ok(shader_module) = _shader_module.get() {
            if let Some(t) = shader_module.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyShaderModule(shader_module_id));
            }
        }
    }

    pub fn device_create_command_encoder(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Option<id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = &self.hub;
        let fid = hub
            .command_buffers
            .prepare(id_in.map(|id| id.into_command_buffer_id()));

        let device = self.hub.devices.get(device_id);

        let error = 'error: {
            let command_buffer = match device.create_command_encoder(&desc.label) {
                Ok(command_buffer) => command_buffer,
                Err(e) => break 'error e,
            };

            let id = fid.assign(command_buffer);
            api_log!("Device::create_command_encoder -> {id:?}");
            return (id.into_command_encoder_id(), None);
        };

        let id = fid.assign(Arc::new(CommandBuffer::new_invalid(&device, &desc.label)));
        (id.into_command_encoder_id(), Some(error))
    }

    pub fn command_encoder_drop(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");
        api_log!("CommandEncoder::drop {command_encoder_id:?}");

        let hub = &self.hub;

        let _cmd_buf = hub
            .command_buffers
            .remove(command_encoder_id.into_command_buffer_id());
    }

    pub fn command_buffer_drop(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        api_log!("CommandBuffer::drop {command_buffer_id:?}");
        self.command_encoder_drop(command_buffer_id.into_command_encoder_id())
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
    ) -> (
        *mut command::RenderBundleEncoder,
        Option<command::CreateRenderBundleError>,
    ) {
        profiling::scope!("Device::create_render_bundle_encoder");
        api_log!("Device::device_create_render_bundle_encoder");
        let (encoder, error) = match command::RenderBundleEncoder::new(desc, device_id, None) {
            Ok(encoder) => (encoder, None),
            Err(e) => (command::RenderBundleEncoder::dummy(device_id), Some(e)),
        };
        (Box::into_raw(Box::new(encoder)), error)
    }

    pub fn render_bundle_encoder_finish(
        &self,
        bundle_encoder: command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Option<id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("RenderBundleEncoder::finish");

        let hub = &self.hub;

        let fid = hub.render_bundles.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(bundle_encoder.parent());

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateRenderBundle {
                    id: fid.id(),
                    desc: trace::new_render_bundle_encoder_descriptor(
                        desc.label.clone(),
                        &bundle_encoder.context,
                        bundle_encoder.is_depth_read_only,
                        bundle_encoder.is_stencil_read_only,
                    ),
                    base: bundle_encoder.to_base_pass(),
                });
            }

            let render_bundle = match bundle_encoder.finish(desc, &device, hub) {
                Ok(bundle) => bundle,
                Err(e) => break 'error e,
            };

            let id = fid.assign(Fallible::Valid(render_bundle));
            api_log!("RenderBundleEncoder::finish -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn render_bundle_drop(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("RenderBundle::drop");
        api_log!("RenderBundle::drop {render_bundle_id:?}");

        let hub = &self.hub;

        let _bundle = hub.render_bundles.remove(render_bundle_id);

        #[cfg(feature = "trace")]
        if let Ok(bundle) = _bundle.get() {
            if let Some(t) = bundle.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyRenderBundle(render_bundle_id));
            }
        }
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

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateQuerySet {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let query_set = match device.create_query_set(desc) {
                Ok(query_set) => query_set,
                Err(err) => break 'error err,
            };

            let id = fid.assign(Fallible::Valid(query_set));
            api_log!("Device::create_query_set -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));
        (id, Some(error))
    }

    pub fn query_set_drop(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");
        api_log!("QuerySet::drop {query_set_id:?}");

        let hub = &self.hub;

        let _query_set = hub.query_sets.remove(query_set_id);

        #[cfg(feature = "trace")]
        if let Ok(query_set) = _query_set.get() {
            if let Some(trace) = query_set.device.trace.lock().as_mut() {
                trace.add(trace::Action::DestroyQuerySet(query_set_id));
            }
        }
    }

    pub fn device_create_render_pipeline(
        &self,
        device_id: DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Option<id::RenderPipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<'_>>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("Device::create_render_pipeline");

        let hub = &self.hub;

        let missing_implicit_pipeline_ids =
            desc.layout.is_none() && id_in.is_some() && implicit_pipeline_ids.is_none();

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

        let error = 'error: {
            if missing_implicit_pipeline_ids {
                // TODO: categorize this error as API misuse
                break 'error pipeline::ImplicitLayoutError::MissingImplicitPipelineIds.into();
            }

            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let layout = desc
                .layout
                .map(|layout| hub.pipeline_layouts.get(layout).get())
                .transpose();
            let layout = match layout {
                Ok(layout) => layout,
                Err(e) => break 'error e.into(),
            };

            let cache = desc
                .cache
                .map(|cache| hub.pipeline_caches.get(cache).get())
                .transpose();
            let cache = match cache {
                Ok(cache) => cache,
                Err(e) => break 'error e.into(),
            };

            let vertex = {
                let module = hub
                    .shader_modules
                    .get(desc.vertex.stage.module)
                    .get()
                    .map_err(|e| pipeline::CreateRenderPipelineError::Stage {
                        stage: wgt::ShaderStages::VERTEX,
                        error: e.into(),
                    });
                let module = match module {
                    Ok(module) => module,
                    Err(e) => break 'error e,
                };
                let stage = ResolvedProgrammableStageDescriptor {
                    module,
                    entry_point: desc.vertex.stage.entry_point.clone(),
                    constants: desc.vertex.stage.constants.clone(),
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .stage
                        .zero_initialize_workgroup_memory,
                };
                ResolvedVertexState {
                    stage,
                    buffers: desc.vertex.buffers.clone(),
                }
            };

            let fragment = if let Some(ref state) = desc.fragment {
                let module = hub
                    .shader_modules
                    .get(state.stage.module)
                    .get()
                    .map_err(|e| pipeline::CreateRenderPipelineError::Stage {
                        stage: wgt::ShaderStages::FRAGMENT,
                        error: e.into(),
                    });
                let module = match module {
                    Ok(module) => module,
                    Err(e) => break 'error e,
                };
                let stage = ResolvedProgrammableStageDescriptor {
                    module,
                    entry_point: state.stage.entry_point.clone(),
                    constants: state.stage.constants.clone(),
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .stage
                        .zero_initialize_workgroup_memory,
                };
                Some(ResolvedFragmentState {
                    stage,
                    targets: state.targets.clone(),
                })
            } else {
                None
            };

            let desc = ResolvedRenderPipelineDescriptor {
                label: desc.label.clone(),
                layout,
                vertex,
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil.clone(),
                multisample: desc.multisample,
                fragment,
                multiview: desc.multiview,
                cache,
            };

            let pipeline = match device.create_render_pipeline(desc) {
                Ok(pair) => pair,
                Err(e) => break 'error e,
            };

            if let Some(ids) = implicit_context.as_ref() {
                let group_count = pipeline.layout.bind_group_layouts.len();
                if ids.group_ids.len() < group_count {
                    log::error!(
                        "Not enough bind group IDs ({}) specified for the implicit layout ({})",
                        ids.group_ids.len(),
                        group_count
                    );
                    // TODO: categorize this error as API misuse
                    break 'error pipeline::ImplicitLayoutError::MissingIds(group_count as _)
                        .into();
                }

                let mut pipeline_layout_guard = hub.pipeline_layouts.write();
                let mut bgl_guard = hub.bind_group_layouts.write();
                pipeline_layout_guard.insert(ids.root_id, Fallible::Valid(pipeline.layout.clone()));
                let mut group_ids = ids.group_ids.iter();
                // NOTE: If the first iterator is longer than the second, the `.zip()` impl will still advance the
                // the first iterator before realizing that the second iterator has finished.
                // The `pipeline.layout.bind_group_layouts` iterator will always be shorter than `ids.group_ids`,
                // so using it as the first iterator for `.zip()` will work properly.
                for (bgl, bgl_id) in pipeline
                    .layout
                    .bind_group_layouts
                    .iter()
                    .zip(&mut group_ids)
                {
                    bgl_guard.insert(*bgl_id, Fallible::Valid(bgl.clone()));
                }
                for bgl_id in group_ids {
                    bgl_guard.insert(*bgl_id, Fallible::Invalid(Arc::new(String::new())));
                }
            }

            let id = fid.assign(Fallible::Valid(pipeline));
            api_log!("Device::create_render_pipeline -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layouts.
        if let Some(ids) = implicit_context {
            let mut pipeline_layout_guard = hub.pipeline_layouts.write();
            let mut bgl_guard = hub.bind_group_layouts.write();
            pipeline_layout_guard.insert(ids.root_id, Fallible::Invalid(Arc::new(String::new())));
            for bgl_id in ids.group_ids {
                bgl_guard.insert(bgl_id, Fallible::Invalid(Arc::new(String::new())));
            }
        }

        log::error!("Device::create_render_pipeline error: {error}");

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

        let error = 'error: {
            let pipeline = match hub.render_pipelines.get(pipeline_id).get() {
                Ok(pipeline) => pipeline,
                Err(e) => break 'error e.into(),
            };
            let id = match pipeline.layout.bind_group_layouts.get(index as usize) {
                Some(bg) => fid.assign(Fallible::Valid(bg.clone())),
                None => {
                    break 'error binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index)
                }
            };
            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(String::new())));
        (id, Some(error))
    }

    pub fn render_pipeline_drop(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("RenderPipeline::drop");
        api_log!("RenderPipeline::drop {render_pipeline_id:?}");

        let hub = &self.hub;

        let _pipeline = hub.render_pipelines.remove(render_pipeline_id);

        #[cfg(feature = "trace")]
        if let Ok(pipeline) = _pipeline.get() {
            if let Some(t) = pipeline.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyRenderPipeline(render_pipeline_id));
            }
        }
    }

    pub fn device_create_compute_pipeline(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: Option<id::ComputePipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<'_>>,
    ) -> (
        id::ComputePipelineId,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        profiling::scope!("Device::create_compute_pipeline");

        let hub = &self.hub;

        let missing_implicit_pipeline_ids =
            desc.layout.is_none() && id_in.is_some() && implicit_pipeline_ids.is_none();

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

        let error = 'error: {
            if missing_implicit_pipeline_ids {
                // TODO: categorize this error as API misuse
                break 'error pipeline::ImplicitLayoutError::MissingImplicitPipelineIds.into();
            }

            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateComputePipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let layout = desc
                .layout
                .map(|layout| hub.pipeline_layouts.get(layout).get())
                .transpose();
            let layout = match layout {
                Ok(layout) => layout,
                Err(e) => break 'error e.into(),
            };

            let cache = desc
                .cache
                .map(|cache| hub.pipeline_caches.get(cache).get())
                .transpose();
            let cache = match cache {
                Ok(cache) => cache,
                Err(e) => break 'error e.into(),
            };

            let module = hub.shader_modules.get(desc.stage.module).get();
            let module = match module {
                Ok(module) => module,
                Err(e) => break 'error e.into(),
            };
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

            let pipeline = match device.create_compute_pipeline(desc) {
                Ok(pair) => pair,
                Err(e) => break 'error e,
            };

            if let Some(ids) = implicit_context.as_ref() {
                let group_count = pipeline.layout.bind_group_layouts.len();
                if ids.group_ids.len() < group_count {
                    log::error!(
                        "Not enough bind group IDs ({}) specified for the implicit layout ({})",
                        ids.group_ids.len(),
                        group_count
                    );
                    // TODO: categorize this error as API misuse
                    break 'error pipeline::ImplicitLayoutError::MissingIds(group_count as _)
                        .into();
                }

                let mut pipeline_layout_guard = hub.pipeline_layouts.write();
                let mut bgl_guard = hub.bind_group_layouts.write();
                pipeline_layout_guard.insert(ids.root_id, Fallible::Valid(pipeline.layout.clone()));
                let mut group_ids = ids.group_ids.iter();
                // NOTE: If the first iterator is longer than the second, the `.zip()` impl will still advance the
                // the first iterator before realizing that the second iterator has finished.
                // The `pipeline.layout.bind_group_layouts` iterator will always be shorter than `ids.group_ids`,
                // so using it as the first iterator for `.zip()` will work properly.
                for (bgl, bgl_id) in pipeline
                    .layout
                    .bind_group_layouts
                    .iter()
                    .zip(&mut group_ids)
                {
                    bgl_guard.insert(*bgl_id, Fallible::Valid(bgl.clone()));
                }
                for bgl_id in group_ids {
                    bgl_guard.insert(*bgl_id, Fallible::Invalid(Arc::new(String::new())));
                }
            }

            let id = fid.assign(Fallible::Valid(pipeline));
            api_log!("Device::create_compute_pipeline -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layouts.
        if let Some(ids) = implicit_context {
            let mut pipeline_layout_guard = hub.pipeline_layouts.write();
            let mut bgl_guard = hub.bind_group_layouts.write();
            pipeline_layout_guard.insert(ids.root_id, Fallible::Invalid(Arc::new(String::new())));
            for bgl_id in ids.group_ids {
                bgl_guard.insert(bgl_id, Fallible::Invalid(Arc::new(String::new())));
            }
        }

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

        let error = 'error: {
            let pipeline = match hub.compute_pipelines.get(pipeline_id).get() {
                Ok(pipeline) => pipeline,
                Err(e) => break 'error e.into(),
            };

            let id = match pipeline.layout.bind_group_layouts.get(index as usize) {
                Some(bg) => fid.assign(Fallible::Valid(bg.clone())),
                None => {
                    break 'error binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index)
                }
            };

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(String::new())));
        (id, Some(error))
    }

    pub fn compute_pipeline_drop(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("ComputePipeline::drop");
        api_log!("ComputePipeline::drop {compute_pipeline_id:?}");

        let hub = &self.hub;

        let _pipeline = hub.compute_pipelines.remove(compute_pipeline_id);

        #[cfg(feature = "trace")]
        if let Ok(pipeline) = _pipeline.get() {
            if let Some(t) = pipeline.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyComputePipeline(compute_pipeline_id));
            }
        }
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
        let error: pipeline::CreatePipelineCacheError = 'error: {
            let device = self.hub.devices.get(device_id);

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreatePipelineCache {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let cache = unsafe { device.create_pipeline_cache(desc) };
            match cache {
                Ok(cache) => {
                    let id = fid.assign(Fallible::Valid(cache));
                    api_log!("Device::create_pipeline_cache -> {id:?}");
                    return (id, None);
                }
                Err(e) => break 'error e,
            }
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(desc.label.to_string())));

        (id, Some(error))
    }

    pub fn pipeline_cache_drop(&self, pipeline_cache_id: id::PipelineCacheId) {
        profiling::scope!("PipelineCache::drop");
        api_log!("PipelineCache::drop {pipeline_cache_id:?}");

        let hub = &self.hub;

        let _cache = hub.pipeline_caches.remove(pipeline_cache_id);

        #[cfg(feature = "trace")]
        if let Ok(cache) = _cache.get() {
            if let Some(t) = cache.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyPipelineCache(pipeline_cache_id));
            }
        }
    }

    pub fn surface_configure(
        &self,
        surface_id: SurfaceId,
        device_id: DeviceId,
        config: &wgt::SurfaceConfiguration<Vec<TextureFormat>>,
    ) -> Option<present::ConfigureSurfaceError> {
        use present::ConfigureSurfaceError as E;
        profiling::scope!("surface_configure");

        fn validate_surface_configuration(
            config: &mut hal::SurfaceConfiguration,
            caps: &hal::SurfaceCapabilities,
            max_texture_dimension_2d: u32,
        ) -> Result<(), E> {
            let width = config.extent.width;
            let height = config.extent.height;

            if width > max_texture_dimension_2d || height > max_texture_dimension_2d {
                return Err(E::TooLarge {
                    width,
                    height,
                    max_texture_dimension_2d,
                });
            }

            if !caps.present_modes.contains(&config.present_mode) {
                // Automatic present mode checks.
                //
                // The "Automatic" modes are never supported by the backends.
                let fallbacks = match config.present_mode {
                    wgt::PresentMode::AutoVsync => {
                        &[wgt::PresentMode::FifoRelaxed, wgt::PresentMode::Fifo][..]
                    }
                    // Always end in FIFO to make sure it's always supported
                    wgt::PresentMode::AutoNoVsync => &[
                        wgt::PresentMode::Immediate,
                        wgt::PresentMode::Mailbox,
                        wgt::PresentMode::Fifo,
                    ][..],
                    _ => {
                        return Err(E::UnsupportedPresentMode {
                            requested: config.present_mode,
                            available: caps.present_modes.clone(),
                        });
                    }
                };

                let new_mode = fallbacks
                    .iter()
                    .copied()
                    .find(|fallback| caps.present_modes.contains(fallback))
                    .unwrap_or_else(|| {
                        unreachable!(
                            "Fallback system failed to choose present mode. \
                            This is a bug. Mode: {:?}, Options: {:?}",
                            config.present_mode, &caps.present_modes
                        );
                    });

                api_log!(
                    "Automatically choosing presentation mode by rule {:?}. Chose {new_mode:?}",
                    config.present_mode
                );
                config.present_mode = new_mode;
            }
            if !caps.formats.contains(&config.format) {
                return Err(E::UnsupportedFormat {
                    requested: config.format,
                    available: caps.formats.clone(),
                });
            }
            if !caps
                .composite_alpha_modes
                .contains(&config.composite_alpha_mode)
            {
                let new_alpha_mode = 'alpha: {
                    // Automatic alpha mode checks.
                    let fallbacks = match config.composite_alpha_mode {
                        wgt::CompositeAlphaMode::Auto => &[
                            wgt::CompositeAlphaMode::Opaque,
                            wgt::CompositeAlphaMode::Inherit,
                        ][..],
                        _ => {
                            return Err(E::UnsupportedAlphaMode {
                                requested: config.composite_alpha_mode,
                                available: caps.composite_alpha_modes.clone(),
                            });
                        }
                    };

                    for &fallback in fallbacks {
                        if caps.composite_alpha_modes.contains(&fallback) {
                            break 'alpha fallback;
                        }
                    }

                    unreachable!(
                        "Fallback system failed to choose alpha mode. This is a bug. \
                                  AlphaMode: {:?}, Options: {:?}",
                        config.composite_alpha_mode, &caps.composite_alpha_modes
                    );
                };

                api_log!(
                    "Automatically choosing alpha mode by rule {:?}. Chose {new_alpha_mode:?}",
                    config.composite_alpha_mode
                );
                config.composite_alpha_mode = new_alpha_mode;
            }
            if !caps.usage.contains(config.usage) {
                return Err(E::UnsupportedUsage {
                    requested: config.usage,
                    available: caps.usage,
                });
            }
            if width == 0 || height == 0 {
                return Err(E::ZeroArea);
            }
            Ok(())
        }

        log::debug!("configuring surface with {:?}", config);

        let error = 'error: {
            // User callbacks must not be called while we are holding locks.
            let user_callbacks;
            {
                let device = self.hub.devices.get(device_id);

                #[cfg(feature = "trace")]
                if let Some(ref mut trace) = *device.trace.lock() {
                    trace.add(trace::Action::ConfigureSurface(surface_id, config.clone()));
                }

                if let Err(e) = device.check_is_valid() {
                    break 'error e.into();
                }

                let surface = self.surfaces.get(surface_id);

                let caps = match surface.get_capabilities(&device.adapter) {
                    Ok(caps) => caps,
                    Err(_) => break 'error E::UnsupportedQueueFamily,
                };

                let mut hal_view_formats = vec![];
                for format in config.view_formats.iter() {
                    if *format == config.format {
                        continue;
                    }
                    if !caps.formats.contains(&config.format) {
                        break 'error E::UnsupportedFormat {
                            requested: config.format,
                            available: caps.formats,
                        };
                    }
                    if config.format.remove_srgb_suffix() != format.remove_srgb_suffix() {
                        break 'error E::InvalidViewFormat(*format, config.format);
                    }
                    hal_view_formats.push(*format);
                }

                if !hal_view_formats.is_empty() {
                    if let Err(missing_flag) =
                        device.require_downlevel_flags(wgt::DownlevelFlags::SURFACE_VIEW_FORMATS)
                    {
                        break 'error E::MissingDownlevelFlags(missing_flag);
                    }
                }

                let maximum_frame_latency = config.desired_maximum_frame_latency.clamp(
                    *caps.maximum_frame_latency.start(),
                    *caps.maximum_frame_latency.end(),
                );
                let mut hal_config = hal::SurfaceConfiguration {
                    maximum_frame_latency,
                    present_mode: config.present_mode,
                    composite_alpha_mode: config.alpha_mode,
                    format: config.format,
                    extent: wgt::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    usage: conv::map_texture_usage(config.usage, hal::FormatAspects::COLOR),
                    view_formats: hal_view_formats,
                };

                if let Err(error) = validate_surface_configuration(
                    &mut hal_config,
                    &caps,
                    device.limits.max_texture_dimension_2d,
                ) {
                    break 'error error;
                }

                // Wait for all work to finish before configuring the surface.
                let snatch_guard = device.snatchable_lock.read();
                let fence = device.fence.read();
                match device.maintain(fence, wgt::Maintain::Wait, snatch_guard) {
                    Ok((closures, _)) => {
                        user_callbacks = closures;
                    }
                    Err(e) => {
                        break 'error e.into();
                    }
                }

                // All textures must be destroyed before the surface can be re-configured.
                if let Some(present) = surface.presentation.lock().take() {
                    if present.acquired_texture.is_some() {
                        break 'error E::PreviousOutputExists;
                    }
                }

                // TODO: Texture views may still be alive that point to the texture.
                // this will allow the user to render to the surface texture, long after
                // it has been removed.
                //
                // https://github.com/gfx-rs/wgpu/issues/4105

                let surface_raw = surface.raw(device.backend()).unwrap();
                match unsafe { surface_raw.configure(device.raw(), &hal_config) } {
                    Ok(()) => (),
                    Err(error) => {
                        break 'error match error {
                            hal::SurfaceError::Outdated | hal::SurfaceError::Lost => {
                                E::InvalidSurface
                            }
                            hal::SurfaceError::Device(error) => {
                                E::Device(device.handle_hal_error(error))
                            }
                            hal::SurfaceError::Other(message) => {
                                log::error!("surface configuration failed: {}", message);
                                E::InvalidSurface
                            }
                        }
                    }
                }

                let mut presentation = surface.presentation.lock();
                *presentation = Some(present::Presentation {
                    device,
                    config: config.clone(),
                    acquired_texture: None,
                });
            }

            user_callbacks.fire();
            return None;
        };

        Some(error)
    }

    /// Check `device_id` for freeable resources and completed buffer mappings.
    ///
    /// Return `queue_empty` indicating whether there are more queue submissions still in flight.
    pub fn device_poll(
        &self,
        device_id: DeviceId,
        maintain: wgt::Maintain<crate::SubmissionIndex>,
    ) -> Result<bool, WaitIdleError> {
        api_log!("Device::poll {maintain:?}");

        let device = self.hub.devices.get(device_id);

        let DevicePoll {
            closures,
            queue_empty,
        } = Self::poll_single_device(&device, maintain)?;

        closures.fire();

        Ok(queue_empty)
    }

    fn poll_single_device(
        device: &crate::device::Device,
        maintain: wgt::Maintain<crate::SubmissionIndex>,
    ) -> Result<DevicePoll, WaitIdleError> {
        let snatch_guard = device.snatchable_lock.read();
        let fence = device.fence.read();
        let (closures, queue_empty) = device.maintain(fence, maintain, snatch_guard)?;

        // Some deferred destroys are scheduled in maintain so run this right after
        // to avoid holding on to them until the next device poll.
        device.deferred_resource_destruction();

        Ok(DevicePoll {
            closures,
            queue_empty,
        })
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
        closures: &mut UserClosures,
    ) -> Result<bool, WaitIdleError> {
        profiling::scope!("poll_device");

        let hub = &self.hub;
        let mut all_queue_empty = true;
        {
            let device_guard = hub.devices.read();

            for (_id, device) in device_guard.iter() {
                let maintain = if force_wait {
                    wgt::Maintain::Wait
                } else {
                    wgt::Maintain::Poll
                };

                let DevicePoll {
                    closures: cbs,
                    queue_empty,
                } = Self::poll_single_device(device, maintain)?;

                all_queue_empty &= queue_empty;

                closures.extend(cbs);
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

    pub fn device_start_capture(&self, device_id: DeviceId) {
        api_log!("Device::start_capture");

        let device = self.hub.devices.get(device_id);

        if !device.is_valid() {
            return;
        }
        unsafe { device.raw().start_capture() };
    }

    pub fn device_stop_capture(&self, device_id: DeviceId) {
        api_log!("Device::stop_capture");

        let device = self.hub.devices.get(device_id);

        if !device.is_valid() {
            return;
        }
        unsafe { device.raw().stop_capture() };
    }

    pub fn pipeline_cache_get_data(&self, id: id::PipelineCacheId) -> Option<Vec<u8>> {
        use crate::pipeline_cache;
        api_log!("PipelineCache::get_data");
        let hub = &self.hub;

        if let Ok(cache) = hub.pipeline_caches.get(id).get() {
            // TODO: Is this check needed?
            if !cache.device.is_valid() {
                return None;
            }
            let mut vec = unsafe { cache.device.raw().pipeline_cache_get_data(cache.raw()) }?;
            let validation_key = cache.device.raw().pipeline_cache_validation_key()?;

            let mut header_contents = [0; pipeline_cache::HEADER_LENGTH];
            pipeline_cache::add_cache_header(
                &mut header_contents,
                &vec,
                &cache.device.adapter.raw.info,
                validation_key,
            );

            let deleted = vec.splice(..0, header_contents).collect::<Vec<_>>();
            debug_assert!(deleted.is_empty());

            return Some(vec);
        }
        None
    }

    pub fn device_drop(&self, device_id: DeviceId) {
        profiling::scope!("Device::drop");
        api_log!("Device::drop {device_id:?}");

        let device = self.hub.devices.remove(device_id);
        let device_lost_closure = device.lock_life().device_lost_closure.take();
        if let Some(closure) = device_lost_closure {
            closure.call(DeviceLostReason::Dropped, String::from("Device dropped."));
        }

        // The things `Device::prepare_to_die` takes care are mostly
        // unnecessary here. We know our queue is empty, so we don't
        // need to wait for submissions or triage them. We know we were
        // just polled, so `life_tracker.free_resources` is empty.
        debug_assert!(device.lock_life().queue_empty());
        device.pending_writes.lock().deactivate();

        drop(device);
    }

    // This closure will be called exactly once during "lose the device",
    // or when it is replaced.
    pub fn device_set_device_lost_closure(
        &self,
        device_id: DeviceId,
        device_lost_closure: DeviceLostClosure,
    ) {
        let device = self.hub.devices.get(device_id);

        let mut life_tracker = device.lock_life();
        if let Some(existing_closure) = life_tracker.device_lost_closure.take() {
            // It's important to not hold the lock while calling the closure.
            drop(life_tracker);
            existing_closure.call(DeviceLostReason::ReplacedCallback, "".to_string());
            life_tracker = device.lock_life();
        }
        life_tracker.device_lost_closure = Some(device_lost_closure);
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

    pub fn queue_drop(&self, queue_id: QueueId) {
        profiling::scope!("Queue::drop");
        api_log!("Queue::drop {queue_id:?}");

        self.hub.queues.remove(queue_id);
    }

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

        let map_result = match hub.buffers.get(buffer_id).get() {
            Ok(buffer) => buffer.map_async(offset, size, op),
            Err(e) => Err((op, e.into())),
        };

        match map_result {
            Ok(submission_index) => Ok(submission_index),
            Err((mut operation, err)) => {
                if let Some(callback) = operation.callback.take() {
                    callback.call(Err(err.clone()));
                }
                log::error!("Buffer::map_async error: {err}");
                Err(err)
            }
        }
    }

    pub fn buffer_get_mapped_range(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(NonNull<u8>, u64), BufferAccessError> {
        profiling::scope!("Buffer::get_mapped_range");
        api_log!("Buffer::get_mapped_range {buffer_id:?} offset {offset:?} size {size:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id).get()?;

        {
            let snatch_guard = buffer.device.snatchable_lock.read();
            buffer.check_destroyed(&snatch_guard)?;
        }

        let range_size = if let Some(size) = size {
            size
        } else if offset > buffer.size {
            0
        } else {
            buffer.size - offset
        };

        if offset % wgt::MAP_ALIGNMENT != 0 {
            return Err(BufferAccessError::UnalignedOffset { offset });
        }
        if range_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(BufferAccessError::UnalignedRangeSize { range_size });
        }
        let map_state = &*buffer.map_state.lock();
        match *map_state {
            resource::BufferMapState::Init { ref staging_buffer } => {
                // offset (u64) can not be < 0, so no need to validate the lower bound
                if offset + range_size > buffer.size {
                    return Err(BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: buffer.size,
                    });
                }
                let ptr = unsafe { staging_buffer.ptr() };
                let ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().offset(offset as isize)) };
                Ok((ptr, range_size))
            }
            resource::BufferMapState::Active {
                ref mapping,
                ref range,
                ..
            } => {
                if offset < range.start {
                    return Err(BufferAccessError::OutOfBoundsUnderrun {
                        index: offset,
                        min: range.start,
                    });
                }
                if offset + range_size > range.end {
                    return Err(BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: range.end,
                    });
                }
                // ptr points to the beginning of the range we mapped in map_async
                // rather than the beginning of the buffer.
                let relative_offset = (offset - range.start) as isize;
                unsafe {
                    Ok((
                        NonNull::new_unchecked(mapping.ptr.as_ptr().offset(relative_offset)),
                        range_size,
                    ))
                }
            }
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                Err(BufferAccessError::NotMapped)
            }
        }
    }
    pub fn buffer_unmap(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        profiling::scope!("unmap", "Buffer");
        api_log!("Buffer::unmap {buffer_id:?}");

        let hub = &self.hub;

        let buffer = hub.buffers.get(buffer_id).get()?;

        let snatch_guard = buffer.device.snatchable_lock.read();
        buffer.check_destroyed(&snatch_guard)?;
        drop(snatch_guard);

        buffer.device.check_is_valid()?;
        buffer.unmap(
            #[cfg(feature = "trace")]
            buffer_id,
        )
    }
}

struct DevicePoll {
    closures: UserClosures,
    queue_empty: bool,
}
