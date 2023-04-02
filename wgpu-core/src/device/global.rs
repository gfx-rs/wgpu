#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model, command, conv,
    device::{life::WaitIdleError, map_buffer, queue, Device, DeviceError, HostMap},
    global::Global,
    hal_api::HalApi,
    id::{self, AdapterId, DeviceId, SurfaceId},
    identity::{GlobalIdentityHandlerFactory, Input},
    init_tracker::TextureInitTracker,
    instance::{self, Adapter, Surface},
    pipeline, present,
    resource::{self, Buffer, BufferAccessResult, BufferMapState, Resource},
    resource::{BufferAccessError, BufferMapOperation, TextureClearMode},
    validation::check_buffer_usage,
    FastHashMap, Label, LabelHelpers as _,
};

use hal::{CommandEncoder as _, Device as _};
use parking_lot::RwLock;
use smallvec::SmallVec;

use wgt::{BufferAddress, TextureFormat};

use std::{
    borrow::Cow,
    iter, mem,
    ops::Range,
    ptr,
    sync::{atomic::Ordering, Arc},
};

use super::{BufferMapPendingClosure, ImplicitPipelineIds, InvalidDevice, UserClosures};

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_is_surface_supported<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        surface_id: SurfaceId,
    ) -> Result<bool, instance::IsSurfaceSupportedError> {
        let hub = A::hub(self);

        let surface_guard = self.surfaces.read();
        let adapter_guard = hub.adapters.read();
        let adapter = adapter_guard
            .get(adapter_id)
            .map_err(|_| instance::IsSurfaceSupportedError::InvalidAdapter)?;
        let surface = surface_guard
            .get(surface_id)
            .map_err(|_| instance::IsSurfaceSupportedError::InvalidSurface)?;
        Ok(adapter.is_surface_supported(surface))
    }

    pub fn surface_get_capabilities<A: HalApi>(
        &self,
        surface_id: SurfaceId,
        adapter_id: AdapterId,
    ) -> Result<wgt::SurfaceCapabilities, instance::GetSurfaceSupportError> {
        profiling::scope!("Surface::get_capabilities");
        self.fetch_adapter_and_surface::<A, _, _>(surface_id, adapter_id, |adapter, surface| {
            let mut hal_caps = surface.get_capabilities(adapter)?;

            hal_caps.formats.sort_by_key(|f| !f.is_srgb());

            Ok(wgt::SurfaceCapabilities {
                formats: hal_caps.formats,
                present_modes: hal_caps.present_modes,
                alpha_modes: hal_caps.composite_alpha_modes,
            })
        })
    }

    fn fetch_adapter_and_surface<
        A: HalApi,
        F: FnOnce(&Adapter<A>, &Surface) -> Result<B, instance::GetSurfaceSupportError>,
        B,
    >(
        &self,
        surface_id: SurfaceId,
        adapter_id: AdapterId,
        get_supported_callback: F,
    ) -> Result<B, instance::GetSurfaceSupportError> {
        let hub = A::hub(self);

        let surface_guard = self.surfaces.read();
        let adapter_guard = hub.adapters.read();
        let adapter = adapter_guard
            .get(adapter_id)
            .map_err(|_| instance::GetSurfaceSupportError::InvalidAdapter)?;
        let surface = surface_guard
            .get(surface_id)
            .map_err(|_| instance::GetSurfaceSupportError::InvalidSurface)?;

        get_supported_callback(adapter, surface)
    }

    pub fn device_features<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::Features, InvalidDevice> {
        let hub = A::hub(self);
        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.features)
    }

    pub fn device_limits<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::Limits, InvalidDevice> {
        let hub = A::hub(self);
        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.limits.clone())
    }

    pub fn device_downlevel_properties<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidDevice> {
        let hub = A::hub(self);
        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.downlevel.clone())
    }

    pub fn device_create_buffer<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Input<G, id::BufferId>,
    ) -> (id::BufferId, Option<resource::CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = A::hub(self);
        let fid = hub.buffers.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let mut desc = desc.clone();
                let mapped_at_creation = mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsages::COPY_DST;
                }
                trace.add(trace::Action::CreateBuffer(fid.id(), desc));
            }

            let buffer = match device.create_buffer(device_id, desc, false) {
                Ok(buffer) => buffer,
                Err(e) => break e,
            };

            let buffer_use = if !desc.mapped_at_creation {
                hal::BufferUses::empty()
            } else if desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // buffer is mappable, so we are just doing that at start
                let map_size = buffer.size;
                let ptr = if map_size == 0 {
                    std::ptr::NonNull::dangling()
                } else {
                    match map_buffer(
                        device.raw.as_ref().unwrap(),
                        &buffer,
                        0,
                        map_size,
                        HostMap::Write,
                    ) {
                        Ok(ptr) => ptr,
                        Err(e) => {
                            device.lock_life().schedule_resource_destruction(
                                queue::TempResource::Buffer(Arc::new(buffer)),
                                !0,
                            );
                            break e.into();
                        }
                    }
                };
                *buffer.map_state.lock() = resource::BufferMapState::Active {
                    ptr,
                    range: 0..map_size,
                    host: HostMap::Write,
                };
                hal::BufferUses::MAP_WRITE
            } else {
                // buffer needs staging area for initialization only
                let stage_desc = wgt::BufferDescriptor {
                    label: Some(Cow::Borrowed(
                        "(wgpu internal) initializing unmappable buffer",
                    )),
                    size: desc.size,
                    usage: wgt::BufferUsages::MAP_WRITE | wgt::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                };
                let stage = match device.create_buffer(device_id, &stage_desc, true) {
                    Ok(stage) => stage,
                    Err(e) => {
                        device.lock_life().schedule_resource_destruction(
                            queue::TempResource::Buffer(Arc::new(buffer)),
                            !0,
                        );
                        break e;
                    }
                };
                let mapping = match unsafe {
                    device
                        .raw
                        .as_ref()
                        .unwrap()
                        .map_buffer(stage.raw.as_ref().unwrap(), 0..stage.size)
                } {
                    Ok(mapping) => mapping,
                    Err(e) => {
                        let mut life_lock = device.lock_life();
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(Arc::new(buffer)),
                            !0,
                        );
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(Arc::new(stage)),
                            !0,
                        );
                        break DeviceError::from(e).into();
                    }
                };

                assert_eq!(buffer.size % wgt::COPY_BUFFER_ALIGNMENT, 0);
                // Zero initialize memory and then mark both staging and buffer as initialized
                // (it's guaranteed that this is the case by the time the buffer is usable)
                unsafe { ptr::write_bytes(mapping.ptr.as_ptr(), 0, buffer.size as usize) };
                buffer.initialization_status.write().drain(0..buffer.size);
                stage.initialization_status.write().drain(0..buffer.size);

                *buffer.map_state.lock() = resource::BufferMapState::Init {
                    ptr: mapping.ptr,
                    needs_flush: !mapping.is_coherent,
                    stage_buffer: Arc::new(stage),
                };
                hal::BufferUses::COPY_DST
            };

            let (id, resource) = fid.assign(buffer);
            log::info!("Created buffer {:?} with {:?}", id, desc);

            device
                .trackers
                .lock()
                .buffers
                .insert_single(id, resource, buffer_use);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
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
    pub fn create_buffer_error<A: HalApi>(&self, id_in: Input<G, id::BufferId>, label: Label) {
        let hub = A::hub(self);
        let fid = hub.buffers.prepare(id_in);

        fid.assign_error(label.borrow_or_default());
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See `create_buffer_error` for more context and explaination.
    pub fn create_texture_error<A: HalApi>(&self, id_in: Input<G, id::TextureId>, label: Label) {
        let hub = A::hub(self);
        let fid = hub.textures.prepare(id_in);

        fid.assign_error(label.borrow_or_default());
    }

    #[cfg(feature = "replay")]
    pub fn device_wait_for_buffer<A: HalApi>(
        &self,
        device_id: DeviceId,
        buffer_id: id::BufferId,
    ) -> Result<(), WaitIdleError> {
        let hub = A::hub(self);
        let device_guard = hub.devices.read();
        let last_submission = {
            let buffer_guard = hub.buffers.write();
            match buffer_guard.get(buffer_id) {
                Ok(buffer) => buffer.info.submission_index(),
                Err(_) => return Ok(()),
            }
        };

        device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?
            .wait_for_submit(last_submission)
    }

    #[doc(hidden)]
    pub fn device_set_buffer_sub_data<A: HalApi>(
        &self,
        device_id: DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &[u8],
    ) -> BufferAccessResult {
        profiling::scope!("Device::set_buffer_sub_data");

        let hub = A::hub(self);

        let device = hub
            .devices
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_WRITE)?;
        //assert!(buffer isn't used by the GPU);

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            let data_path = trace.make_binary("bin", data);
            trace.add(trace::Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: offset..offset + data.len() as BufferAddress,
                queued: false,
            });
        }

        let raw_buf = buffer.raw.as_ref().unwrap();
        unsafe {
            let mapping = device
                .raw
                .as_ref()
                .unwrap()
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
            if !mapping.is_coherent {
                device
                    .raw
                    .as_ref()
                    .unwrap()
                    .flush_mapped_ranges(raw_buf, iter::once(offset..offset + data.len() as u64));
            }
            device
                .raw
                .as_ref()
                .unwrap()
                .unmap_buffer(raw_buf)
                .map_err(DeviceError::from)?;
        }

        Ok(())
    }

    #[doc(hidden)]
    pub fn device_get_buffer_sub_data<A: HalApi>(
        &self,
        device_id: DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &mut [u8],
    ) -> BufferAccessResult {
        profiling::scope!("Device::get_buffer_sub_data");

        let hub = A::hub(self);

        let device = hub
            .devices
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_READ)?;
        //assert!(buffer isn't used by the GPU);

        let raw_buf = buffer.raw.as_ref().unwrap();
        unsafe {
            let mapping = device
                .raw
                .as_ref()
                .unwrap()
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            if !mapping.is_coherent {
                device.raw.as_ref().unwrap().invalidate_mapped_ranges(
                    raw_buf,
                    iter::once(offset..offset + data.len() as u64),
                );
            }
            ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), data.as_mut_ptr(), data.len());
            device
                .raw
                .as_ref()
                .unwrap()
                .unmap_buffer(raw_buf)
                .map_err(DeviceError::from)?;
        }

        Ok(())
    }

    pub fn buffer_label<A: HalApi>(&self, id: id::BufferId) -> String {
        A::hub(self).buffers.label_for_resource(id)
    }

    pub fn buffer_destroy<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("Buffer::destroy");

        let map_closure;
        // Restrict the locks to this scope.
        {
            let hub = A::hub(self);

            //TODO: lock pending writes separately, keep the device read-only

            log::info!("Buffer {:?} is destroyed", buffer_id);
            let buffer = hub
                .buffers
                .get(buffer_id)
                .map_err(|_| resource::DestroyError::Invalid)?;

            let device = &buffer.device;

            map_closure = match &*buffer.map_state.lock() {
                &BufferMapState::Waiting(..) // To get the proper callback behavior.
                | &BufferMapState::Init { .. }
                | &BufferMapState::Active { .. }
                => {
                    self.buffer_unmap_inner(buffer_id, &buffer, device)
                        .unwrap_or(None)
                }
                _ => None,
            };

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::FreeBuffer(buffer_id));
            }
            if buffer.raw.is_none() {
                return Err(resource::DestroyError::AlreadyDestroyed);
            }

            let temp = queue::TempResource::Buffer(buffer.clone());
            let mut pending_writes = device.pending_writes.lock();
            let pending_writes = pending_writes.as_mut().unwrap();
            if pending_writes.dst_buffers.contains_key(&buffer_id) {
                pending_writes.temp_resources.push(temp);
            } else {
                let last_submit_index = buffer.info.submission_index();
                device
                    .lock_life()
                    .schedule_resource_destruction(temp, last_submit_index);
            }
        }

        // Note: outside the scope where locks are held when calling the callback
        if let Some((operation, status)) = map_closure {
            operation.callback.call(status);
        }

        Ok(())
    }

    pub fn buffer_drop<A: HalApi>(&self, buffer_id: id::BufferId, wait: bool) {
        profiling::scope!("Buffer::drop");
        log::debug!("buffer {:?} is dropped", buffer_id);

        let hub = A::hub(self);
        let mut buffer_guard = hub.buffers.write();

        let (last_submit_index, buffer) = {
            match buffer_guard.get(buffer_id) {
                Ok(buffer) => {
                    let last_submit_index = buffer.info.submission_index();
                    (last_submit_index, buffer)
                }
                Err(_) => {
                    hub.buffers.unregister_locked(buffer_id, &mut *buffer_guard);
                    return;
                }
            }
        };

        let device = &buffer.device;
        {
            let mut life_lock = device.lock_life();
            if device
                .pending_writes
                .lock()
                .as_ref()
                .unwrap()
                .dst_buffers
                .contains_key(&buffer_id)
            {
                life_lock.future_suspected_buffers.push(buffer.clone());
            } else {
                life_lock.suspected_resources.buffers.push(buffer.clone());
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for buffer {:?}: {:?}", buffer_id, e),
            }
        }
    }

    pub fn device_create_texture<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Input<G, id::TextureId>,
        idtv_in: Option<Input<G, id::TextureViewId>>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = A::hub(self);

        let fid = hub.textures.prepare(id_in);
        let mut fid_tv = idtv_in
            .as_ref()
            .map(|id| hub.texture_views.prepare(id.clone()));

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(
                    fid.id(),
                    fid_tv.as_ref().map(|id| id.id()),
                    desc.clone(),
                ));
            }

            let adapter = hub.adapters.get(device.adapter_id.0).unwrap();
            let texture = match device.create_texture(device_id, &adapter, desc) {
                Ok(texture) => texture,
                Err(error) => break error,
            };
            let (id, resource) = fid.assign(texture);
            log::info!("Created texture {:?} with {:?}", id, desc);

            if let TextureClearMode::RenderPass {
                ref mut clear_views,
                is_color: _,
            } = *resource.clear_mode.write()
            {
                if idtv_in.is_some() {
                    let dimension = match desc.dimension {
                        wgt::TextureDimension::D1 => wgt::TextureViewDimension::D1,
                        wgt::TextureDimension::D2 => wgt::TextureViewDimension::D2,
                        wgt::TextureDimension::D3 => unreachable!(),
                    };

                    for mip_level in 0..desc.mip_level_count {
                        for array_layer in 0..desc.size.depth_or_array_layers {
                            let descriptor = resource::TextureViewDescriptor {
                                label: Some(Cow::Borrowed("(wgpu internal) clear texture view")),
                                format: Some(desc.format),
                                dimension: Some(dimension),
                                range: wgt::ImageSubresourceRange {
                                    aspect: wgt::TextureAspect::All,
                                    base_mip_level: mip_level,
                                    mip_level_count: Some(1),
                                    base_array_layer: array_layer,
                                    array_layer_count: Some(1),
                                },
                            };

                            let texture_view = device
                                .create_texture_view(&resource, id.0, &descriptor)
                                .unwrap();
                            let fid_tv = if fid_tv.is_some() {
                                fid_tv.take().unwrap()
                            } else {
                                hub.texture_views.prepare(idtv_in.clone().unwrap())
                            };
                            let (tv_id, texture_view) = fid_tv.assign(texture_view);
                            log::info!("Created texture view {:?} for texture {:?}", tv_id, id);

                            clear_views.push(texture_view.clone());

                            device
                                .trackers
                                .lock()
                                .views
                                .insert_single(tv_id, texture_view);
                        }
                    }
                }
            }

            device.trackers.lock().textures.insert_single(
                id.0,
                resource,
                hal::TextureUses::UNINITIALIZED,
            );

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    /// # Safety
    ///
    /// - `hal_texture` must be created from `device_id` corresponding raw handle.
    /// - `hal_texture` must be created respecting `desc`
    /// - `hal_texture` must be initialized
    pub unsafe fn create_texture_from_hal<A: HalApi>(
        &self,
        hal_texture: A::Texture,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Input<G, id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = A::hub(self);

        let fid = hub.textures.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };

            // NB: Any change done through the raw texture handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(fid.id(), None, desc.clone()));
            }

            let adapter = hub.adapters.get(device.adapter_id.0).unwrap();

            let format_features = match device
                .describe_format_features(&adapter, desc.format)
                .map_err(|error| resource::CreateTextureError::MissingFeatures(desc.format, error))
            {
                Ok(features) => features,
                Err(error) => break error,
            };

            let mut texture = device.create_texture_from_hal(
                hal_texture,
                conv::map_texture_usage(desc.usage, desc.format.into()),
                device_id,
                desc,
                format_features,
                resource::TextureClearMode::None,
            );
            if desc.usage.contains(wgt::TextureUsages::COPY_DST) {
                texture.hal_usage |= hal::TextureUses::COPY_DST;
            }

            texture.initialization_status =
                RwLock::new(TextureInitTracker::new(desc.mip_level_count, 0));

            let (id, resource) = fid.assign(texture);
            log::info!("Created texture {:?} with {:?}", id, desc);

            device.trackers.lock().textures.insert_single(
                id.0,
                resource,
                hal::TextureUses::UNINITIALIZED,
            );

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn texture_label<A: HalApi>(&self, id: id::TextureId) -> String {
        A::hub(self).textures.label_for_resource(id)
    }

    pub fn texture_destroy<A: HalApi>(
        &self,
        texture_id: id::TextureId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("Texture::destroy");

        let hub = A::hub(self);

        log::info!("Buffer {:?} is destroyed", texture_id);
        let texture = hub
            .textures
            .get(texture_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &texture.device;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(trace::Action::FreeTexture(texture_id));
        }

        let last_submit_index = texture.info.submission_index();

        let mut clear_views = match std::mem::replace(
            &mut *texture.clear_mode.write(),
            resource::TextureClearMode::None,
        ) {
            resource::TextureClearMode::BufferCopy => SmallVec::new(),
            resource::TextureClearMode::RenderPass { clear_views, .. } => clear_views,
            resource::TextureClearMode::None => SmallVec::new(),
        };

        match *texture.inner.as_ref().unwrap() {
            resource::TextureInner::Native { ref raw } => {
                if !raw.is_none() {
                    let temp = queue::TempResource::Texture(texture.clone(), clear_views);
                    let mut pending_writes = device.pending_writes.lock();
                    let pending_writes = pending_writes.as_mut().unwrap();
                    if pending_writes.dst_textures.contains_key(&texture_id) {
                        pending_writes.temp_resources.push(temp);
                    } else {
                        device
                            .lock_life()
                            .schedule_resource_destruction(temp, last_submit_index);
                    }
                } else {
                    return Err(resource::DestroyError::AlreadyDestroyed);
                }
            }
            resource::TextureInner::Surface { .. } => {
                clear_views.clear();
            }
        }

        Ok(())
    }

    pub fn texture_drop<A: HalApi>(&self, texture_id: id::TextureId, wait: bool) {
        profiling::scope!("Texture::drop");
        log::debug!("texture {:?} is dropped", texture_id);

        let hub = A::hub(self);
        let mut texture_guard = hub.textures.write();

        let (last_submit_index, texture) = {
            match texture_guard.get(texture_id) {
                Ok(texture) => {
                    let last_submit_index = texture.info.submission_index();
                    (last_submit_index, texture)
                }
                Err(_) => {
                    hub.textures
                        .unregister_locked(texture_id, &mut *texture_guard);
                    return;
                }
            }
        };

        let device = &texture.device;
        {
            let mut life_lock = device.lock_life();
            if device
                .pending_writes
                .lock()
                .as_ref()
                .unwrap()
                .dst_textures
                .contains_key(&texture_id)
            {
                life_lock.future_suspected_textures.push(texture.clone());
            } else {
                life_lock.suspected_resources.textures.push(texture.clone());
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for texture {:?}: {:?}", texture_id, e),
            }
        }
    }

    pub fn texture_create_view<A: HalApi>(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: Input<G, id::TextureViewId>,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        profiling::scope!("Texture::create_view");

        let hub = A::hub(self);

        let fid = hub.texture_views.prepare(id_in);

        let error = loop {
            let texture = match hub.textures.get(texture_id) {
                Ok(texture) => texture,
                Err(_) => break resource::CreateTextureViewError::InvalidTexture,
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

            let view = match device.create_texture_view(&texture, texture_id, desc) {
                Ok(view) => view,
                Err(e) => break e,
            };
            let (id, resource) = fid.assign(view);

            device.trackers.lock().views.insert_single(id, resource);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn texture_view_label<A: HalApi>(&self, id: id::TextureViewId) -> String {
        A::hub(self).texture_views.label_for_resource(id)
    }

    pub fn texture_view_drop<A: HalApi>(
        &self,
        texture_view_id: id::TextureViewId,
        wait: bool,
    ) -> Result<(), resource::TextureViewDestroyError> {
        profiling::scope!("TextureView::drop");
        log::debug!("texture view {:?} is dropped", texture_view_id);

        let hub = A::hub(self);
        let mut texture_view_guard = hub.texture_views.write();

        let (last_submit_index, view) = {
            match texture_view_guard.get(texture_view_id) {
                Ok(view) => {
                    let last_submit_index = view.info.submission_index();
                    (last_submit_index, view)
                }
                Err(_) => {
                    hub.texture_views
                        .unregister_locked(texture_view_id, &mut *texture_view_guard);
                    return Ok(());
                }
            }
        };

        view.device
            .lock_life()
            .suspected_resources
            .texture_views
            .push(view.clone());

        if wait {
            match view.device.wait_for_submit(last_submit_index) {
                Ok(()) => (),
                Err(e) => log::error!(
                    "Failed to wait for texture view {:?}: {:?}",
                    texture_view_id,
                    e
                ),
            }
        }
        Ok(())
    }

    pub fn device_create_sampler<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Input<G, id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        profiling::scope!("Device::create_sampler");

        let hub = A::hub(self);
        let fid = hub.samplers.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateSampler(fid.id(), desc.clone()));
            }

            let sampler = match device.create_sampler(desc) {
                Ok(sampler) => sampler,
                Err(e) => break e,
            };

            let (id, resource) = fid.assign(sampler);

            device.trackers.lock().samplers.insert_single(id, resource);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn sampler_label<A: HalApi>(&self, id: id::SamplerId) -> String {
        A::hub(self).samplers.label_for_resource(id)
    }

    pub fn sampler_drop<A: HalApi>(&self, sampler_id: id::SamplerId) {
        profiling::scope!("Sampler::drop");
        log::debug!("sampler {:?} is dropped", sampler_id);

        let hub = A::hub(self);
        let mut sampler_guard = hub.samplers.write();

        let sampler = {
            match sampler_guard.get(sampler_id) {
                Ok(sampler) => sampler,
                Err(_) => {
                    hub.samplers
                        .unregister_locked(sampler_id, &mut *sampler_guard);
                    return;
                }
            }
        };

        sampler
            .device
            .lock_life()
            .suspected_resources
            .samplers
            .push(sampler.clone());
    }

    pub fn device_create_bind_group_layout<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        profiling::scope!("Device::create_bind_group_layout");

        let hub = A::hub(self);
        let fid = hub.bind_group_layouts.prepare(id_in);

        let error = 'outer: loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroupLayout(fid.id(), desc.clone()));
            }

            let mut entry_map = FastHashMap::default();
            for entry in desc.entries.iter() {
                if entry.binding > device.limits.max_bindings_per_bind_group {
                    break 'outer binding_model::CreateBindGroupLayoutError::InvalidBindingIndex {
                        binding: entry.binding,
                        maximum: device.limits.max_bindings_per_bind_group,
                    };
                }
                if entry_map.insert(entry.binding, *entry).is_some() {
                    break 'outer binding_model::CreateBindGroupLayoutError::ConflictBinding(
                        entry.binding,
                    );
                }
            }

            // If there is an equivalent BGL, just bump the refcount and return it.
            // This is only applicable for identity filters that are generating new IDs,
            // so their inputs are `PhantomData` of size 0.
            if mem::size_of::<Input<G, id::BindGroupLayoutId>>() == 0 {
                let bgl_guard = hub.bind_group_layouts.read();
                if let Some(id) =
                    Device::deduplicate_bind_group_layout(device_id, &entry_map, &*bgl_guard)
                {
                    return (id, None);
                }
            }

            let layout =
                match device.create_bind_group_layout(desc.label.borrow_option(), entry_map) {
                    Ok(layout) => layout,
                    Err(e) => break e,
                };

            let (id, _) = fid.assign(layout);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn bind_group_layout_label<A: HalApi>(&self, id: id::BindGroupLayoutId) -> String {
        A::hub(self).bind_group_layouts.label_for_resource(id)
    }

    pub fn bind_group_layout_drop<A: HalApi>(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("BindGroupLayout::drop");
        log::debug!("bind group layout {:?} is dropped", bind_group_layout_id);

        let hub = A::hub(self);
        let mut bind_group_layout_guard = hub.bind_group_layouts.write();

        let layout = {
            match bind_group_layout_guard.get(bind_group_layout_id) {
                Ok(layout) => layout,
                Err(_) => {
                    hub.bind_group_layouts
                        .unregister_locked(bind_group_layout_id, &mut *bind_group_layout_guard);
                    return;
                }
            }
        };

        layout
            .device
            .lock_life()
            .suspected_resources
            .bind_group_layouts
            .push(layout.clone());
    }

    pub fn device_create_pipeline_layout<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Input<G, id::PipelineLayoutId>,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        profiling::scope!("Device::create_pipeline_layout");

        let hub = A::hub(self);
        let fid = hub.pipeline_layouts.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreatePipelineLayout(fid.id(), desc.clone()));
            }

            let layout = {
                let bgl_guard = hub.bind_group_layouts.read();
                match device.create_pipeline_layout(desc, &*bgl_guard) {
                    Ok(layout) => layout,
                    Err(e) => break e,
                }
            };

            let (id, _) = fid.assign(layout);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn pipeline_layout_label<A: HalApi>(&self, id: id::PipelineLayoutId) -> String {
        A::hub(self).pipeline_layouts.label_for_resource(id)
    }

    pub fn pipeline_layout_drop<A: HalApi>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");
        log::debug!("pipeline layout {:?} is dropped", pipeline_layout_id);

        let hub = A::hub(self);
        let mut pipeline_layout_guard = hub.pipeline_layouts.write();
        let layout = {
            match pipeline_layout_guard.get(pipeline_layout_id) {
                Ok(layout) => layout,
                Err(_) => {
                    hub.pipeline_layouts
                        .unregister_locked(pipeline_layout_id, &mut *pipeline_layout_guard);
                    return;
                }
            }
        };

        layout
            .device
            .lock_life()
            .suspected_resources
            .pipeline_layouts
            .push(layout.clone());
    }

    pub fn device_create_bind_group<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Input<G, id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("Device::create_bind_group");

        let hub = A::hub(self);
        let fid = hub.bind_groups.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroup(fid.id(), desc.clone()));
            }

            let bind_group_layout = match hub.bind_group_layouts.get(desc.layout) {
                Ok(layout) => layout,
                Err(_) => break binding_model::CreateBindGroupError::InvalidLayout,
            };
            let bind_group = match device.create_bind_group(&bind_group_layout, desc, hub) {
                Ok(bind_group) => bind_group,
                Err(e) => break e,
            };
            let (id, resource) = fid.assign(bind_group);
            log::debug!("Bind group {:?}", id,);

            device
                .trackers
                .lock()
                .bind_groups
                .insert_single(id, resource);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn bind_group_label<A: HalApi>(&self, id: id::BindGroupId) -> String {
        A::hub(self).bind_groups.label_for_resource(id)
    }

    pub fn bind_group_drop<A: HalApi>(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("BindGroup::drop");
        log::debug!("bind group {:?} is dropped", bind_group_id);

        let hub = A::hub(self);
        let mut bind_group_guard = hub.bind_groups.write();

        let bind_group = {
            match bind_group_guard.get(bind_group_id) {
                Ok(bind_group) => bind_group,
                Err(_) => {
                    hub.bind_groups
                        .unregister_locked(bind_group_id, &mut *bind_group_guard);
                    return;
                }
            }
        };

        bind_group
            .device
            .lock_life()
            .suspected_resources
            .bind_groups
            .push(bind_group.clone());
    }

    pub fn device_create_shader_module<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: pipeline::ShaderModuleSource,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = A::hub(self);
        let fid = hub.shader_modules.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let data = match source {
                    #[cfg(feature = "wgsl")]
                    pipeline::ShaderModuleSource::Wgsl(ref code) => {
                        trace.make_binary("wgsl", code.as_bytes())
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
                Err(e) => break e,
            };
            let (id, _) = fid.assign(shader);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    // Unsafe-ness of internal calls has little to do with unsafe-ness of this.
    #[allow(unused_unsafe)]
    /// # Safety
    ///
    /// This function passes SPIR-V binary to the backend as-is and can potentially result in a
    /// driver crash.
    pub unsafe fn device_create_shader_module_spirv<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: Cow<[u32]>,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = A::hub(self);
        let fid = hub.shader_modules.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let data = trace.make_binary("spv", unsafe {
                    std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * 4)
                });
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader = match unsafe { device.create_shader_module_spirv(desc, &source) } {
                Ok(shader) => shader,
                Err(e) => break e,
            };
            let (id, _) = fid.assign(shader);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn shader_module_label<A: HalApi>(&self, id: id::ShaderModuleId) -> String {
        A::hub(self).shader_modules.label_for_resource(id)
    }

    pub fn shader_module_drop<A: HalApi>(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");
        log::debug!("shader module {:?} is dropped", shader_module_id);

        let hub = A::hub(self);
        hub.shader_modules.unregister(shader_module_id);
    }

    pub fn device_create_command_encoder<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Input<G, id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = A::hub(self);
        let fid = hub.command_buffers.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid,
            };
            let encoder = match device
                .command_allocator
                .lock()
                .as_mut()
                .unwrap()
                .acquire_encoder(device.raw.as_ref().unwrap(), device.queue.as_ref().unwrap())
            {
                Ok(raw) => raw,
                Err(_) => break DeviceError::OutOfMemory,
            };
            let command_buffer = command::CommandBuffer::new(
                encoder,
                &device,
                #[cfg(feature = "trace")]
                device.trace.lock().is_some(),
                &desc.label,
            );

            let (id, _) = fid.assign(command_buffer);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn command_buffer_label<A: HalApi>(&self, id: id::CommandBufferId) -> String {
        A::hub(self).command_buffers.label_for_resource(id)
    }

    pub fn command_encoder_drop<A: HalApi>(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");
        log::debug!("command encoder {:?} is dropped", command_encoder_id);

        let hub = A::hub(self);

        let cmd_buf = hub.command_buffers.unregister(command_encoder_id).unwrap();
        cmd_buf
            .device
            .untrack(&cmd_buf.data.lock().as_ref().unwrap().trackers);
    }

    pub fn command_buffer_drop<A: HalApi>(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        log::debug!("command buffer {:?} is dropped", command_buffer_id);
        self.command_encoder_drop::<A>(command_buffer_id)
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
    ) -> (
        id::RenderBundleEncoderId,
        Option<command::CreateRenderBundleError>,
    ) {
        profiling::scope!("Device::create_render_bundle_encoder");
        let (encoder, error) = match command::RenderBundleEncoder::new(desc, device_id, None) {
            Ok(encoder) => (encoder, None),
            Err(e) => (command::RenderBundleEncoder::dummy(device_id), Some(e)),
        };
        (Box::into_raw(Box::new(encoder)), error)
    }

    pub fn render_bundle_encoder_finish<A: HalApi>(
        &self,
        bundle_encoder: command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Input<G, id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("RenderBundleEncoder::finish");

        let hub = A::hub(self);

        let fid = hub.render_bundles.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(bundle_encoder.parent()) {
                Ok(device) => device,
                Err(_) => break command::RenderBundleError::INVALID_DEVICE,
            };
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
                Err(e) => break e,
            };

            log::debug!("Render bundle");
            let (id, resource) = fid.assign(render_bundle);

            device.trackers.lock().bundles.insert_single(id, resource);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn render_bundle_label<A: HalApi>(&self, id: id::RenderBundleId) -> String {
        A::hub(self).render_bundles.label_for_resource(id)
    }

    pub fn render_bundle_drop<A: HalApi>(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("RenderBundle::drop");
        log::debug!("render bundle {:?} is dropped", render_bundle_id);
        let hub = A::hub(self);
        let mut bundle_guard = hub.render_bundles.write();

        let bundle = {
            match bundle_guard.get(render_bundle_id) {
                Ok(bundle) => bundle,
                Err(_) => {
                    hub.render_bundles
                        .unregister_locked(render_bundle_id, &mut *bundle_guard);
                    return;
                }
            }
        };

        hub.devices
            .get(bundle.device_id.0)
            .unwrap()
            .lock_life()
            .suspected_resources
            .render_bundles
            .push(bundle.clone());
    }

    pub fn device_create_query_set<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: Input<G, id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("Device::create_query_set");

        let hub = A::hub(self);
        let fid = hub.query_sets.prepare(id_in);

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateQuerySet {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let query_set = match device.create_query_set(desc) {
                Ok(query_set) => query_set,
                Err(err) => break err,
            };

            let (id, resource) = fid.assign(query_set);

            device
                .trackers
                .lock()
                .query_sets
                .insert_single(id, resource);

            return (id.0, None);
        };

        let id = fid.assign_error("");
        (id, Some(error))
    }

    pub fn query_set_drop<A: HalApi>(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");
        log::debug!("query set {:?} is dropped", query_set_id);

        let hub = A::hub(self);
        let query_set_guard = hub.query_sets.read();

        let query_set = {
            let query_set = query_set_guard.get(query_set_id).unwrap();
            query_set
        };

        let device = &query_set.device;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(trace::Action::DestroyQuerySet(query_set_id));
        }

        device
            .lock_life()
            .suspected_resources
            .query_sets
            .push(query_set.clone());
    }

    pub fn query_set_label<A: HalApi>(&self, id: id::QuerySetId) -> String {
        A::hub(self).query_sets.label_for_resource(id)
    }

    pub fn device_create_render_pipeline<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Input<G, id::RenderPipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("Device::create_render_pipeline");

        let hub = A::hub(self);

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            let adapter = hub.adapters.get(device.adapter_id.0).unwrap();
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let pipeline =
                match device.create_render_pipeline(&adapter, desc, implicit_context, hub) {
                    Ok(pair) => pair,
                    Err(e) => break e,
                };

            let (id, resource) = fid.assign(pipeline);
            log::info!("Created render pipeline {:?} with {:?}", id, desc);

            device
                .trackers
                .lock()
                .render_pipelines
                .insert_single(id, resource);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);
        let pipeline_layout_guard = hub.pipeline_layouts.read();

        let error = loop {
            let pipeline = match hub.render_pipelines.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };
            let id = match pipeline_layout_guard[pipeline.layout_id]
                .bind_group_layout_ids
                .get(index as usize)
            {
                Some(id) => id,
                None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
            };

            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>");
        (id, Some(error))
    }

    pub fn render_pipeline_label<A: HalApi>(&self, id: id::RenderPipelineId) -> String {
        A::hub(self).render_pipelines.label_for_resource(id)
    }

    pub fn render_pipeline_drop<A: HalApi>(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("RenderPipeline::drop");
        log::debug!("render pipeline {:?} is dropped", render_pipeline_id);
        let hub = A::hub(self);
        let mut pipeline_guard = hub.render_pipelines.write();

        let (pipeline, layout_id) = {
            match pipeline_guard.get(render_pipeline_id) {
                Ok(pipeline) => (pipeline, pipeline.layout_id),
                Err(_) => {
                    hub.render_pipelines
                        .unregister_locked(render_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };
        let device = &pipeline.device;
        let mut life_lock = device.lock_life();
        life_lock
            .suspected_resources
            .render_pipelines
            .push(pipeline.clone());
        let layout = hub.pipeline_layouts.get(layout_id.0).unwrap();
        life_lock.suspected_resources.pipeline_layouts.push(layout);
    }

    pub fn device_create_compute_pipeline<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: Input<G, id::ComputePipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::ComputePipelineId,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        profiling::scope!("Device::create_compute_pipeline");

        let hub = A::hub(self);

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

        let error = loop {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateComputePipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let pipeline = match device.create_compute_pipeline(desc, implicit_context, hub) {
                Ok(pair) => pair,
                Err(e) => break e,
            };

            let (id, resource) = fid.assign(pipeline);
            log::info!("Created compute pipeline {:?} with {:?}", id, desc);

            device
                .trackers
                .lock()
                .compute_pipelines
                .insert_single(id, resource);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);
        let pipeline_layout_guard = hub.pipeline_layouts.read();

        let error = loop {
            let pipeline_guard = hub.compute_pipelines.read();

            let pipeline = match pipeline_guard.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };
            let id = match pipeline_layout_guard[pipeline.layout_id]
                .bind_group_layout_ids
                .get(index as usize)
            {
                Some(id) => id,
                None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
            };

            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>");
        (id, Some(error))
    }

    pub fn compute_pipeline_label<A: HalApi>(&self, id: id::ComputePipelineId) -> String {
        A::hub(self).compute_pipelines.label_for_resource(id)
    }

    pub fn compute_pipeline_drop<A: HalApi>(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("ComputePipeline::drop");
        log::debug!("compute pipeline {:?} is dropped", compute_pipeline_id);
        let hub = A::hub(self);
        let mut pipeline_guard = hub.compute_pipelines.write();

        let (pipeline, layout_id) = {
            match pipeline_guard.get(compute_pipeline_id) {
                Ok(pipeline) => (pipeline, pipeline.layout_id),
                Err(_) => {
                    hub.compute_pipelines
                        .unregister_locked(compute_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };
        let device = &pipeline.device;
        let mut life_lock = device.lock_life();
        life_lock
            .suspected_resources
            .compute_pipelines
            .push(pipeline.clone());
        let layout = hub.pipeline_layouts.get(layout_id.0).unwrap();
        life_lock.suspected_resources.pipeline_layouts.push(layout);
    }

    pub fn surface_configure<A: HalApi>(
        &self,
        surface_id: SurfaceId,
        device_id: DeviceId,
        config: &wgt::SurfaceConfiguration<Vec<TextureFormat>>,
    ) -> Option<present::ConfigureSurfaceError> {
        use hal::{Adapter as _, Surface as _};
        use present::ConfigureSurfaceError as E;
        profiling::scope!("surface_configure");

        fn validate_surface_configuration(
            config: &mut hal::SurfaceConfiguration,
            caps: &hal::SurfaceCapabilities,
        ) -> Result<(), E> {
            let width = config.extent.width;
            let height = config.extent.height;
            if width < caps.extents.start().width
                || width > caps.extents.end().width
                || height < caps.extents.start().height
                || height > caps.extents.end().height
            {
                log::warn!(
                    "Requested size {}x{} is outside of the supported range: {:?}",
                    width,
                    height,
                    caps.extents
                );
            }
            if !caps.present_modes.contains(&config.present_mode) {
                let new_mode = 'b: loop {
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

                    for &fallback in fallbacks {
                        if caps.present_modes.contains(&fallback) {
                            break 'b fallback;
                        }
                    }

                    unreachable!("Fallback system failed to choose present mode. This is a bug. Mode: {:?}, Options: {:?}", config.present_mode, &caps.present_modes);
                };

                log::info!(
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
                let new_alpha_mode = 'alpha: loop {
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

                log::info!(
                    "Automatically choosing alpha mode by rule {:?}. Chose {new_alpha_mode:?}",
                    config.composite_alpha_mode
                );
                config.composite_alpha_mode = new_alpha_mode;
            }
            if !caps.usage.contains(config.usage) {
                return Err(E::UnsupportedUsage);
            }
            if width == 0 || height == 0 {
                return Err(E::ZeroArea);
            }
            Ok(())
        }

        log::info!("configuring surface with {:?}", config);
        let hub = A::hub(self);

        let surface_guard = self.surfaces.read();
        let adapter_guard = hub.adapters.read();
        let device_guard = hub.devices.read();

        let error = 'outer: loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::ConfigureSurface(surface_id, config.clone()));
            }

            let surface = match surface_guard.get(surface_id) {
                Ok(surface) => surface,
                Err(_) => break E::InvalidSurface,
            };

            let caps = unsafe {
                let suf = A::get_surface(surface);
                let adapter = &adapter_guard[device.adapter_id];
                match adapter
                    .raw
                    .adapter
                    .surface_capabilities(suf.unwrap().raw.as_ref())
                {
                    Some(caps) => caps,
                    None => break E::UnsupportedQueueFamily,
                }
            };

            let mut hal_view_formats = vec![];
            for format in config.view_formats.iter() {
                if *format == config.format {
                    continue;
                }
                if !caps.formats.contains(&config.format) {
                    break 'outer E::UnsupportedFormat {
                        requested: config.format,
                        available: caps.formats.clone(),
                    };
                }
                if config.format.remove_srgb_suffix() != format.remove_srgb_suffix() {
                    break 'outer E::InvalidViewFormat(*format, config.format);
                }
                hal_view_formats.push(*format);
            }

            if !hal_view_formats.is_empty() {
                if let Err(missing_flag) =
                    device.require_downlevel_flags(wgt::DownlevelFlags::SURFACE_VIEW_FORMATS)
                {
                    break 'outer E::MissingDownlevelFlags(missing_flag);
                }
            }

            let num_frames = present::DESIRED_NUM_FRAMES
                .clamp(*caps.swap_chain_sizes.start(), *caps.swap_chain_sizes.end());
            let mut hal_config = hal::SurfaceConfiguration {
                swap_chain_size: num_frames,
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

            if let Err(error) = validate_surface_configuration(&mut hal_config, &caps) {
                break error;
            }

            match unsafe {
                A::get_surface(surface)
                    .unwrap()
                    .raw
                    .configure(device.raw.as_ref().unwrap(), &hal_config)
            } {
                Ok(()) => (),
                Err(error) => {
                    break match error {
                        hal::SurfaceError::Outdated | hal::SurfaceError::Lost => E::InvalidSurface,
                        hal::SurfaceError::Device(error) => E::Device(error.into()),
                        hal::SurfaceError::Other(message) => {
                            log::error!("surface configuration failed: {}", message);
                            E::InvalidSurface
                        }
                    }
                }
            }

            if let Some(present) = surface.presentation.lock().take() {
                if present.acquired_texture.is_some() {
                    break E::PreviousOutputExists;
                }
            }
            let mut presentation = surface.presentation.lock();
            *presentation = Some(present::Presentation {
                device_id: id::Valid(device_id),
                config: config.clone(),
                num_frames,
                acquired_texture: None,
            });

            return None;
        };

        Some(error)
    }

    #[cfg(feature = "replay")]
    /// Only triange suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<A: HalApi>(&self, device_id: DeviceId) -> Result<(), InvalidDevice> {
        let hub = A::hub(self);

        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;
        device.lock_life().triage_suspected(
            hub,
            &device.trackers,
            #[cfg(feature = "trace")]
            None,
        );
        Ok(())
    }

    /// Check `device_id` for freeable resources and completed buffer mappings.
    ///
    /// Return `queue_empty` indicating whether there are more queue submissions still in flight.
    pub fn device_poll<A: HalApi>(
        &self,
        device_id: DeviceId,
        maintain: wgt::Maintain<queue::WrappedSubmissionIndex>,
    ) -> Result<bool, WaitIdleError> {
        let (closures, queue_empty) = {
            if let wgt::Maintain::WaitForSubmissionIndex(submission_index) = maintain {
                if submission_index.queue_id != device_id {
                    return Err(WaitIdleError::WrongSubmissionIndex(
                        submission_index.queue_id,
                        device_id,
                    ));
                }
            }

            let hub = A::hub(self);
            hub.devices
                .get(device_id)
                .map_err(|_| DeviceError::Invalid)?
                .maintain(hub, maintain)?
        };

        closures.fire();

        Ok(queue_empty)
    }

    /// Poll all devices belonging to the backend `A`.
    ///
    /// If `force_wait` is true, block until all buffer mappings are done.
    ///
    /// Return `all_queue_empty` indicating whether there are more queue
    /// submissions still in flight.
    fn poll_device<A: HalApi>(
        &self,
        force_wait: bool,
        closures: &mut UserClosures,
    ) -> Result<bool, WaitIdleError> {
        profiling::scope!("poll_device");

        let hub = A::hub(self);
        let mut devices_to_drop = vec![];
        let mut all_queue_empty = true;
        {
            let device_guard = hub.devices.read();

            for (id, device) in device_guard.iter(A::VARIANT) {
                let maintain = if force_wait {
                    wgt::Maintain::Wait
                } else {
                    wgt::Maintain::Poll
                };
                let (cbs, queue_empty) = device.maintain(hub, maintain)?;
                all_queue_empty = all_queue_empty && queue_empty;

                // If the device's own `RefCount` is the only one left, and
                // its submission queue is empty, then it can be freed.
                if queue_empty && device.is_unique() {
                    devices_to_drop.push(id);
                }
                closures.extend(cbs);
            }
        }

        for device_id in devices_to_drop {
            self.exit_device::<A>(device_id);
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
        let mut closures = UserClosures::default();
        let mut all_queue_empty = true;

        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        {
            all_queue_empty =
                self.poll_device::<hal::api::Vulkan>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            all_queue_empty =
                self.poll_device::<hal::api::Metal>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(all(feature = "dx12", windows))]
        {
            all_queue_empty =
                self.poll_device::<hal::api::Dx12>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(all(feature = "dx11", windows))]
        {
            all_queue_empty =
                self.poll_device::<hal::api::Dx11>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(feature = "gles")]
        {
            all_queue_empty =
                self.poll_device::<hal::api::Gles>(force_wait, &mut closures)? && all_queue_empty;
        }

        closures.fire();

        Ok(all_queue_empty)
    }

    pub fn device_label<A: HalApi>(&self, id: DeviceId) -> String {
        A::hub(self).devices.label_for_resource(id)
    }

    pub fn device_start_capture<A: HalApi>(&self, id: DeviceId) {
        let hub = A::hub(self);
        if let Ok(device) = hub.devices.get(id) {
            unsafe { device.raw.as_ref().unwrap().start_capture() };
        }
    }

    pub fn device_stop_capture<A: HalApi>(&self, id: DeviceId) {
        let hub = A::hub(self);
        if let Ok(device) = hub.devices.get(id) {
            unsafe { device.raw.as_ref().unwrap().stop_capture() };
        }
    }

    pub fn device_drop<A: HalApi>(&self, device_id: DeviceId) {
        profiling::scope!("Device::drop");
        log::debug!("device {:?} is dropped", device_id);
    }

    /// Exit the unreferenced, inactive device `device_id`.
    fn exit_device<A: HalApi>(&self, device_id: DeviceId) {
        let hub = A::hub(self);
        let mut free_adapter_id = None;
        {
            let device = hub.devices.unregister(device_id);
            if let Some(device) = device {
                // The things `Device::prepare_to_die` takes care are mostly
                // unnecessary here. We know our queue is empty, so we don't
                // need to wait for submissions or triage them. We know we were
                // just polled, so `life_tracker.free_resources` is empty.
                debug_assert!(device.lock_life().queue_empty());
                device.pending_writes.lock().as_mut().unwrap().deactivate();

                let adapter = hub.adapters.get(device.adapter_id.0).unwrap();
                // Adapter is only referenced by the device and itself.
                // This isn't a robust way to destroy them, we should find a better one.
                if adapter.is_unique() {
                    free_adapter_id = Some(device.adapter_id.0);
                }

                drop(device);
            }
        }

        // Free the adapter now that we've dropped the `Device`.
        if let Some(free_adapter_id) = free_adapter_id {
            let _ = hub.adapters.unregister(free_adapter_id);
        }
    }

    pub fn buffer_map_async<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        range: Range<BufferAddress>,
        op: BufferMapOperation,
    ) -> BufferAccessResult {
        // User callbacks must not be called while holding buffer_map_async_inner's locks, so we
        // defer the error callback if it needs to be called immediately (typically when running
        // into errors).
        if let Err((op, err)) = self.buffer_map_async_inner::<A>(buffer_id, range, op) {
            op.callback.call(Err(err.clone()));

            return Err(err);
        }

        Ok(())
    }

    // Returns the mapping callback in case of error so that the callback can be fired outside
    // of the locks that are held in this function.
    fn buffer_map_async_inner<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        range: Range<BufferAddress>,
        op: BufferMapOperation,
    ) -> Result<(), (BufferMapOperation, BufferAccessError)> {
        profiling::scope!("Buffer::map_async");

        let hub = A::hub(self);
        let buffer_guard = hub.buffers.read();

        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsages::MAP_READ, hal::BufferUses::MAP_READ),
            HostMap::Write => (wgt::BufferUsages::MAP_WRITE, hal::BufferUses::MAP_WRITE),
        };

        if range.start % wgt::MAP_ALIGNMENT != 0 || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err((op, BufferAccessError::UnalignedRange));
        }

        let (device, buffer) = {
            let buffer = buffer_guard
                .get(buffer_id)
                .map_err(|_| BufferAccessError::Invalid);

            let buffer = match buffer {
                Ok(b) => b,
                Err(e) => {
                    return Err((op, e));
                }
            };

            if let Err(e) = check_buffer_usage(buffer.usage, pub_usage) {
                return Err((op, e.into()));
            }

            if range.start > range.end {
                return Err((
                    op,
                    BufferAccessError::NegativeRange {
                        start: range.start,
                        end: range.end,
                    },
                ));
            }
            if range.end > buffer.size {
                return Err((
                    op,
                    BufferAccessError::OutOfBoundsOverrun {
                        index: range.end,
                        max: buffer.size,
                    },
                ));
            }
            let mut map_state = buffer.map_state.lock();
            *map_state = match *map_state {
                resource::BufferMapState::Init { .. } | resource::BufferMapState::Active { .. } => {
                    return Err((op, BufferAccessError::AlreadyMapped));
                }
                resource::BufferMapState::Waiting(_) => {
                    return Err((op, BufferAccessError::MapAlreadyPending));
                }
                resource::BufferMapState::Idle => {
                    resource::BufferMapState::Waiting(resource::BufferPendingMapping {
                        range,
                        op,
                        _parent_buffer: buffer.clone(),
                    })
                }
            };
            log::debug!("Buffer {:?} map state -> Waiting", buffer_id);

            let device = &buffer.device;

            {
                let mut trackers = device.as_ref().trackers.lock();

                trackers
                    .buffers
                    .set_single(&*buffer_guard, buffer_id, internal_use);
                //TODO: Check if draining ALL buffers is correct!
                trackers.buffers.drain();
            }

            (device, buffer)
        };

        device.lock_life().map(buffer);

        Ok(())
    }

    pub fn buffer_get_mapped_range<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(*mut u8, u64), BufferAccessError> {
        profiling::scope!("Buffer::get_mapped_range");

        let hub = A::hub(self);

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;

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
            resource::BufferMapState::Init { ref ptr, .. } => {
                // offset (u64) can not be < 0, so no need to validate the lower bound
                if offset + range_size > buffer.size {
                    return Err(BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: buffer.size,
                    });
                }
                unsafe { Ok((ptr.as_ptr().offset(offset as isize), range_size)) }
            }
            resource::BufferMapState::Active {
                ref ptr, ref range, ..
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
                // rather thant the beginning of the buffer.
                let relative_offset = (offset - range.start) as isize;
                unsafe { Ok((ptr.as_ptr().offset(relative_offset), range_size)) }
            }
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                Err(BufferAccessError::NotMapped)
            }
        }
    }

    fn buffer_unmap_inner<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        buffer: &Arc<Buffer<A>>,
        device: &Device<A>,
    ) -> Result<Option<BufferMapPendingClosure>, BufferAccessError> {
        log::debug!("Buffer {:?} map state -> Idle", buffer_id);
        match mem::replace(
            &mut *buffer.map_state.lock(),
            resource::BufferMapState::Idle,
        ) {
            resource::BufferMapState::Init {
                ptr,
                stage_buffer,
                needs_flush,
            } => {
                #[cfg(feature = "trace")]
                if let Some(ref mut trace) = *device.trace.lock() {
                    let data = trace.make_binary("bin", unsafe {
                        std::slice::from_raw_parts(ptr.as_ptr(), buffer.size as usize)
                    });
                    trace.add(trace::Action::WriteBuffer {
                        id: buffer_id,
                        data,
                        range: 0..buffer.size,
                        queued: true,
                    });
                }
                let _ = ptr;
                if needs_flush {
                    unsafe {
                        device.raw.as_ref().unwrap().flush_mapped_ranges(
                            stage_buffer.raw.as_ref().unwrap(),
                            iter::once(0..buffer.size),
                        );
                    }
                }

                let raw_buf = buffer.raw.as_ref().ok_or(BufferAccessError::Destroyed)?;

                buffer.info.use_at(
                    device
                        .active_submission_index
                        .fetch_add(1, Ordering::Relaxed),
                );
                let region = wgt::BufferSize::new(buffer.size).map(|size| hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                });
                let transition_src = hal::BufferBarrier {
                    buffer: stage_buffer.raw.as_ref().unwrap(),
                    usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                };
                let transition_dst = hal::BufferBarrier {
                    buffer: raw_buf,
                    usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                };
                let mut pending_writes = device.pending_writes.lock();
                let pending_writes = pending_writes.as_mut().unwrap();
                let encoder = pending_writes.activate();
                unsafe {
                    encoder.transition_buffers(
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    if buffer.size > 0 {
                        encoder.copy_buffer_to_buffer(
                            stage_buffer.raw.as_ref().unwrap(),
                            raw_buf,
                            region.into_iter(),
                        );
                    }
                }
                pending_writes.consume_temp(queue::TempResource::Buffer(stage_buffer));
                pending_writes.dst_buffers.insert(buffer_id, buffer.clone());
            }
            resource::BufferMapState::Idle => {
                return Err(BufferAccessError::NotMapped);
            }
            resource::BufferMapState::Waiting(pending) => {
                return Ok(Some((pending.op, Err(BufferAccessError::MapAborted))));
            }
            resource::BufferMapState::Active { ptr, range, host } => {
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    if let Some(ref mut trace) = *device.trace.lock() {
                        let size = range.end - range.start;
                        let data = trace.make_binary("bin", unsafe {
                            std::slice::from_raw_parts(ptr.as_ptr(), size as usize)
                        });
                        trace.add(trace::Action::WriteBuffer {
                            id: buffer_id,
                            data,
                            range: range.clone(),
                            queued: false,
                        });
                    }
                    let _ = (ptr, range);
                }
                unsafe {
                    device
                        .raw
                        .as_ref()
                        .unwrap()
                        .unmap_buffer(buffer.raw.as_ref().unwrap())
                        .map_err(DeviceError::from)?
                };
            }
        }
        Ok(None)
    }

    pub fn buffer_unmap<A: HalApi>(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        profiling::scope!("unmap", "Buffer");

        let closure;
        {
            // Restrict the locks to this scope.
            let hub = A::hub(self);

            let buffer = hub
                .buffers
                .get(buffer_id)
                .map_err(|_| BufferAccessError::Invalid)?;
            let device = &buffer.device;

            closure = self.buffer_unmap_inner(buffer_id, &buffer, device)
        }

        // Note: outside the scope where locks are held when calling the callback
        if let Some((operation, status)) = closure? {
            operation.callback.call(status);
        }
        Ok(())
    }
}
