#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    api_log, binding_model, command, conv,
    device::{
        bgl, life::WaitIdleError, map_buffer, queue, DeviceError, DeviceLostClosure,
        DeviceLostReason, HostMap, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL,
    },
    global::Global,
    hal_api::HalApi,
    id::{self, AdapterId, DeviceId, QueueId, SurfaceId},
    init_tracker::TextureInitTracker,
    instance::{self, Adapter, Surface},
    lock::{rank, RwLock},
    pipeline, present,
    resource::{
        self, BufferAccessError, BufferAccessResult, BufferMapOperation, CreateBufferError,
        Resource,
    },
    validation::check_buffer_usage,
    Label, LabelHelpers as _,
};

use arrayvec::ArrayVec;
use hal::Device as _;

use wgt::{BufferAddress, TextureFormat};

use std::{
    borrow::Cow,
    iter, ptr,
    sync::{atomic::Ordering, Arc},
};

use super::{ImplicitPipelineIds, InvalidDevice, UserClosures};

impl Global {
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

            let usages = conv::map_texture_usage_from_hal(hal_caps.usage);

            Ok(wgt::SurfaceCapabilities {
                formats: hal_caps.formats,
                present_modes: hal_caps.present_modes,
                alpha_modes: hal_caps.composite_alpha_modes,
                usages,
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
        if !device.is_valid() {
            return Err(InvalidDevice);
        }

        Ok(device.features)
    }

    pub fn device_limits<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::Limits, InvalidDevice> {
        let hub = A::hub(self);

        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.is_valid() {
            return Err(InvalidDevice);
        }

        Ok(device.limits.clone())
    }

    pub fn device_downlevel_properties<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidDevice> {
        let hub = A::hub(self);

        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.is_valid() {
            return Err(InvalidDevice);
        }

        Ok(device.downlevel.clone())
    }

    pub fn device_create_buffer<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Option<id::BufferId>,
    ) -> (id::BufferId, Option<CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = A::hub(self);
        let fid = hub.buffers.prepare(id_in);

        let mut to_destroy: ArrayVec<resource::Buffer<A>, 2> = ArrayVec::new();
        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => {
                    break 'error DeviceError::Invalid.into();
                }
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            if desc.usage.is_empty() {
                // Per spec, `usage` must not be zero.
                break 'error CreateBufferError::InvalidUsage(desc.usage);
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                let mut desc = desc.clone();
                let mapped_at_creation = std::mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsages::COPY_DST;
                }
                trace.add(trace::Action::CreateBuffer(fid.id(), desc));
            }

            let buffer = match device.create_buffer(desc, false) {
                Ok(buffer) => buffer,
                Err(e) => {
                    break 'error e;
                }
            };

            let buffer_use = if !desc.mapped_at_creation {
                hal::BufferUses::empty()
            } else if desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // buffer is mappable, so we are just doing that at start
                let map_size = buffer.size;
                let ptr = if map_size == 0 {
                    ptr::NonNull::dangling()
                } else {
                    let snatch_guard = device.snatchable_lock.read();
                    match map_buffer(
                        device.raw(),
                        &buffer,
                        0,
                        map_size,
                        HostMap::Write,
                        &snatch_guard,
                    ) {
                        Ok(ptr) => ptr,
                        Err(e) => {
                            to_destroy.push(buffer);
                            break 'error e.into();
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
                let stage = match device.create_buffer(&stage_desc, true) {
                    Ok(stage) => Arc::new(stage),
                    Err(e) => {
                        to_destroy.push(buffer);
                        break 'error e;
                    }
                };

                let snatch_guard = device.snatchable_lock.read();
                let stage_raw = stage.raw(&snatch_guard).unwrap();
                let mapping = match unsafe { device.raw().map_buffer(stage_raw, 0..stage.size) } {
                    Ok(mapping) => mapping,
                    Err(e) => {
                        to_destroy.push(buffer);
                        break 'error CreateBufferError::Device(e.into());
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
                    stage_buffer: stage,
                };
                hal::BufferUses::COPY_DST
            };

            let (id, resource) = fid.assign(Arc::new(buffer));
            api_log!("Device::create_buffer({desc:?}) -> {id:?}");

            device
                .trackers
                .lock()
                .buffers
                .insert_single(resource, buffer_use);

            return (id, None);
        };

        // Error path

        for buffer in to_destroy {
            let device = Arc::clone(&buffer.device);
            device
                .lock_life()
                .schedule_resource_destruction(queue::TempResource::Buffer(Arc::new(buffer)), !0);
        }

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
    pub fn create_buffer_error<A: HalApi>(&self, id_in: Option<id::BufferId>, label: Label) {
        let hub = A::hub(self);
        let fid = hub.buffers.prepare(id_in);

        fid.assign_error(label.borrow_or_default());
    }

    pub fn create_render_bundle_error<A: HalApi>(
        &self,
        id_in: Option<id::RenderBundleId>,
        label: Label,
    ) {
        let hub = A::hub(self);
        let fid = hub.render_bundles.prepare(id_in);

        fid.assign_error(label.borrow_or_default());
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See `create_buffer_error` for more context and explanation.
    pub fn create_texture_error<A: HalApi>(&self, id_in: Option<id::TextureId>, label: Label) {
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

        let last_submission = {
            let buffer_guard = hub.buffers.write();
            match buffer_guard.get(buffer_id) {
                Ok(buffer) => buffer.info.submission_index(),
                Err(_) => return Ok(()),
            }
        };

        hub.devices
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
        let snatch_guard = device.snatchable_lock.read();
        if !device.is_valid() {
            return Err(DeviceError::Lost.into());
        }

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer_id, buffer.usage, wgt::BufferUsages::MAP_WRITE)?;
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

        let raw_buf = buffer
            .raw(&snatch_guard)
            .ok_or(BufferAccessError::Destroyed)?;
        unsafe {
            let mapping = device
                .raw()
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
            if !mapping.is_coherent {
                device
                    .raw()
                    .flush_mapped_ranges(raw_buf, iter::once(offset..offset + data.len() as u64));
            }
            device
                .raw()
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
        if !device.is_valid() {
            return Err(DeviceError::Lost.into());
        }

        let snatch_guard = device.snatchable_lock.read();

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer_id, buffer.usage, wgt::BufferUsages::MAP_READ)?;
        //assert!(buffer isn't used by the GPU);

        let raw_buf = buffer
            .raw(&snatch_guard)
            .ok_or(BufferAccessError::Destroyed)?;
        unsafe {
            let mapping = device
                .raw()
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            if !mapping.is_coherent {
                device.raw().invalidate_mapped_ranges(
                    raw_buf,
                    iter::once(offset..offset + data.len() as u64),
                );
            }
            ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), data.as_mut_ptr(), data.len());
            device
                .raw()
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
        api_log!("Buffer::destroy {buffer_id:?}");

        let hub = A::hub(self);

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let _ = buffer.unmap();

        buffer.destroy()
    }

    pub fn buffer_drop<A: HalApi>(&self, buffer_id: id::BufferId, wait: bool) {
        profiling::scope!("Buffer::drop");
        api_log!("Buffer::drop {buffer_id:?}");

        let hub = A::hub(self);

        let buffer = match hub.buffers.unregister(buffer_id) {
            Some(buffer) => buffer,
            None => {
                return;
            }
        };

        let _ = buffer.unmap();

        let last_submit_index = buffer.info.submission_index();

        let device = buffer.device.clone();

        if device
            .pending_writes
            .lock()
            .as_ref()
            .unwrap()
            .dst_buffers
            .contains_key(&buffer_id)
        {
            device.lock_life().future_suspected_buffers.push(buffer);
        } else {
            device
                .lock_life()
                .suspected_resources
                .buffers
                .insert(buffer.info.tracker_index(), buffer);
        }

        if wait {
            match device.wait_for_submit(last_submit_index) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for buffer {:?}: {}", buffer_id, e),
            }
        }
    }

    pub fn device_create_texture<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Option<id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = A::hub(self);

        let fid = hub.textures.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let texture = match device.create_texture(&device.adapter, desc) {
                Ok(texture) => texture,
                Err(error) => break 'error error,
            };

            let (id, resource) = fid.assign(Arc::new(texture));
            api_log!("Device::create_texture({desc:?}) -> {id:?}");

            device
                .trackers
                .lock()
                .textures
                .insert_single(resource, hal::TextureUses::UNINITIALIZED);

            return (id, None);
        };

        log::error!("Device::create_texture error: {error}");

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
        id_in: Option<id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture_from_hal");

        let hub = A::hub(self);

        let fid = hub.textures.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            // NB: Any change done through the raw texture handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let format_features = match device
                .describe_format_features(&device.adapter, desc.format)
                .map_err(|error| resource::CreateTextureError::MissingFeatures(desc.format, error))
            {
                Ok(features) => features,
                Err(error) => break 'error error,
            };

            let mut texture = device.create_texture_from_hal(
                hal_texture,
                conv::map_texture_usage(desc.usage, desc.format.into()),
                desc,
                format_features,
                resource::TextureClearMode::None,
            );
            if desc.usage.contains(wgt::TextureUsages::COPY_DST) {
                texture.hal_usage |= hal::TextureUses::COPY_DST;
            }

            texture.initialization_status = RwLock::new(
                rank::TEXTURE_INITIALIZATION_STATUS,
                TextureInitTracker::new(desc.mip_level_count, 0),
            );

            let (id, resource) = fid.assign(Arc::new(texture));
            api_log!("Device::create_texture({desc:?}) -> {id:?}");

            device
                .trackers
                .lock()
                .textures
                .insert_single(resource, hal::TextureUses::UNINITIALIZED);

            return (id, None);
        };

        log::error!("Device::create_texture error: {error}");

        let id = fid.assign_error(desc.label.borrow_or_default());
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

        let hub = A::hub(self);
        let fid = hub.buffers.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            // NB: Any change done through the raw buffer handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(trace) = device.trace.lock().as_mut() {
                trace.add(trace::Action::CreateBuffer(fid.id(), desc.clone()));
            }

            let buffer = device.create_buffer_from_hal(hal_buffer, desc);

            let (id, buffer) = fid.assign(Arc::new(buffer));
            api_log!("Device::create_buffer -> {id:?}");

            device
                .trackers
                .lock()
                .buffers
                .insert_single(buffer, hal::BufferUses::empty());

            return (id, None);
        };

        log::error!("Device::create_buffer error: {error}");

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
        api_log!("Texture::destroy {texture_id:?}");

        let hub = A::hub(self);

        let texture = hub
            .textures
            .get(texture_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        texture.destroy()
    }

    pub fn texture_drop<A: HalApi>(&self, texture_id: id::TextureId, wait: bool) {
        profiling::scope!("Texture::drop");
        api_log!("Texture::drop {texture_id:?}");

        let hub = A::hub(self);

        if let Some(texture) = hub.textures.unregister(texture_id) {
            let last_submit_index = texture.info.submission_index();

            let device = &texture.device;
            {
                if device
                    .pending_writes
                    .lock()
                    .as_ref()
                    .unwrap()
                    .dst_textures
                    .contains_key(&texture_id)
                {
                    device
                        .lock_life()
                        .future_suspected_textures
                        .push(texture.clone());
                } else {
                    device
                        .lock_life()
                        .suspected_resources
                        .textures
                        .insert(texture.info.tracker_index(), texture.clone());
                }
            }

            if wait {
                match device.wait_for_submit(last_submit_index) {
                    Ok(()) => (),
                    Err(e) => log::error!("Failed to wait for texture {texture_id:?}: {e}"),
                }
            }
        }
    }

    #[allow(unused_unsafe)]
    pub fn texture_create_view<A: HalApi>(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: Option<id::TextureViewId>,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        profiling::scope!("Texture::create_view");

        let hub = A::hub(self);

        let fid = hub.texture_views.prepare(id_in);

        let error = 'error: {
            let texture = match hub.textures.get(texture_id) {
                Ok(texture) => texture,
                Err(_) => break 'error resource::CreateTextureViewError::InvalidTexture,
            };
            let device = &texture.device;
            {
                let snatch_guard = device.snatchable_lock.read();
                if texture.is_destroyed(&snatch_guard) {
                    break 'error resource::CreateTextureViewError::InvalidTexture;
                }
            }
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateTextureView {
                    id: fid.id(),
                    parent_id: texture_id,
                    desc: desc.clone(),
                });
            }

            let view = match unsafe { device.create_texture_view(&texture, desc) } {
                Ok(view) => view,
                Err(e) => break 'error e,
            };

            let (id, resource) = fid.assign(Arc::new(view));

            {
                let mut views = texture.views.lock();
                views.push(Arc::downgrade(&resource));
            }

            api_log!("Texture::create_view({texture_id:?}) -> {id:?}");
            device.trackers.lock().views.insert_single(resource);
            return (id, None);
        };

        log::error!("Texture::create_view({texture_id:?}) error: {error}");
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
        api_log!("TextureView::drop {texture_view_id:?}");

        let hub = A::hub(self);

        if let Some(view) = hub.texture_views.unregister(texture_view_id) {
            let last_submit_index = view.info.submission_index();

            view.device
                .lock_life()
                .suspected_resources
                .texture_views
                .insert(view.info.tracker_index(), view.clone());

            if wait {
                match view.device.wait_for_submit(last_submit_index) {
                    Ok(()) => (),
                    Err(e) => {
                        log::error!("Failed to wait for texture view {texture_view_id:?}: {e}")
                    }
                }
            }
        }
        Ok(())
    }

    pub fn device_create_sampler<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Option<id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        profiling::scope!("Device::create_sampler");

        let hub = A::hub(self);
        let fid = hub.samplers.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateSampler(fid.id(), desc.clone()));
            }

            let sampler = match device.create_sampler(desc) {
                Ok(sampler) => sampler,
                Err(e) => break 'error e,
            };

            let (id, resource) = fid.assign(Arc::new(sampler));
            api_log!("Device::create_sampler -> {id:?}");
            device.trackers.lock().samplers.insert_single(resource);

            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn sampler_label<A: HalApi>(&self, id: id::SamplerId) -> String {
        A::hub(self).samplers.label_for_resource(id)
    }

    pub fn sampler_drop<A: HalApi>(&self, sampler_id: id::SamplerId) {
        profiling::scope!("Sampler::drop");
        api_log!("Sampler::drop {sampler_id:?}");

        let hub = A::hub(self);

        if let Some(sampler) = hub.samplers.unregister(sampler_id) {
            sampler
                .device
                .lock_life()
                .suspected_resources
                .samplers
                .insert(sampler.info.tracker_index(), sampler.clone());
        }
    }

    pub fn device_create_bind_group_layout<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        profiling::scope!("Device::create_bind_group_layout");

        let hub = A::hub(self);
        let fid = hub.bind_group_layouts.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroupLayout(fid.id(), desc.clone()));
            }

            let entry_map = match bgl::EntryMap::from_entries(&device.limits, &desc.entries) {
                Ok(map) => map,
                Err(e) => break 'error e,
            };

            // Currently we make a distinction between fid.assign and fid.assign_existing. This distinction is incorrect,
            // but see https://github.com/gfx-rs/wgpu/issues/4912.
            //
            // `assign` also registers the ID with the resource info, so it can be automatically reclaimed. This needs to
            // happen with a mutable reference, which means it can only happen on creation.
            //
            // Because we need to call `assign` inside the closure (to get mut access), we need to "move" the future id into the closure.
            // Rust cannot figure out at compile time that we only ever consume the ID once, so we need to move the check
            // to runtime using an Option.
            let mut fid = Some(fid);

            // The closure might get called, and it might give us an ID. Side channel it out of the closure.
            let mut id = None;

            let bgl_result = device.bgl_pool.get_or_init(entry_map, |entry_map| {
                let bgl =
                    device.create_bind_group_layout(&desc.label, entry_map, bgl::Origin::Pool)?;

                let (id_inner, arc) = fid.take().unwrap().assign(Arc::new(bgl));
                id = Some(id_inner);

                Ok(arc)
            });

            let layout = match bgl_result {
                Ok(layout) => layout,
                Err(e) => break 'error e,
            };

            // If the ID was not assigned, and we survived the above check,
            // it means that the bind group layout already existed and we need to call `assign_existing`.
            //
            // Calling this function _will_ leak the ID. See https://github.com/gfx-rs/wgpu/issues/4912.
            if id.is_none() {
                id = Some(fid.take().unwrap().assign_existing(&layout))
            }

            api_log!("Device::create_bind_group_layout -> {id:?}");
            return (id.unwrap(), None);
        };

        let fid = hub.bind_group_layouts.prepare(id_in);
        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn bind_group_layout_label<A: HalApi>(&self, id: id::BindGroupLayoutId) -> String {
        A::hub(self).bind_group_layouts.label_for_resource(id)
    }

    pub fn bind_group_layout_drop<A: HalApi>(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("BindGroupLayout::drop");
        api_log!("BindGroupLayout::drop {bind_group_layout_id:?}");

        let hub = A::hub(self);

        if let Some(layout) = hub.bind_group_layouts.unregister(bind_group_layout_id) {
            layout
                .device
                .lock_life()
                .suspected_resources
                .bind_group_layouts
                .insert(layout.info.tracker_index(), layout.clone());
        }
    }

    pub fn device_create_pipeline_layout<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Option<id::PipelineLayoutId>,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        profiling::scope!("Device::create_pipeline_layout");

        let hub = A::hub(self);
        let fid = hub.pipeline_layouts.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreatePipelineLayout(fid.id(), desc.clone()));
            }

            let layout = match device.create_pipeline_layout(desc, &hub.bind_group_layouts) {
                Ok(layout) => layout,
                Err(e) => break 'error e,
            };

            let (id, _) = fid.assign(Arc::new(layout));
            api_log!("Device::create_pipeline_layout -> {id:?}");
            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn pipeline_layout_label<A: HalApi>(&self, id: id::PipelineLayoutId) -> String {
        A::hub(self).pipeline_layouts.label_for_resource(id)
    }

    pub fn pipeline_layout_drop<A: HalApi>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");
        api_log!("PipelineLayout::drop {pipeline_layout_id:?}");

        let hub = A::hub(self);
        if let Some(layout) = hub.pipeline_layouts.unregister(pipeline_layout_id) {
            layout
                .device
                .lock_life()
                .suspected_resources
                .pipeline_layouts
                .insert(layout.info.tracker_index(), layout.clone());
        }
    }

    pub fn device_create_bind_group<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Option<id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("Device::create_bind_group");

        let hub = A::hub(self);
        let fid = hub.bind_groups.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateBindGroup(fid.id(), desc.clone()));
            }

            let bind_group_layout = match hub.bind_group_layouts.get(desc.layout) {
                Ok(layout) => layout,
                Err(..) => break 'error binding_model::CreateBindGroupError::InvalidLayout,
            };

            if bind_group_layout.device.as_info().id() != device.as_info().id() {
                break 'error DeviceError::WrongDevice.into();
            }

            let bind_group = match device.create_bind_group(&bind_group_layout, desc, hub) {
                Ok(bind_group) => bind_group,
                Err(e) => break 'error e,
            };

            let (id, resource) = fid.assign(Arc::new(bind_group));

            let weak_ref = Arc::downgrade(&resource);
            for range in &resource.used_texture_ranges {
                range.texture.bind_groups.lock().push(weak_ref.clone());
            }
            for range in &resource.used_buffer_ranges {
                range.buffer.bind_groups.lock().push(weak_ref.clone());
            }

            api_log!("Device::create_bind_group -> {id:?}");

            device.trackers.lock().bind_groups.insert_single(resource);
            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn bind_group_label<A: HalApi>(&self, id: id::BindGroupId) -> String {
        A::hub(self).bind_groups.label_for_resource(id)
    }

    pub fn bind_group_drop<A: HalApi>(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("BindGroup::drop");
        api_log!("BindGroup::drop {bind_group_id:?}");

        let hub = A::hub(self);

        if let Some(bind_group) = hub.bind_groups.unregister(bind_group_id) {
            bind_group
                .device
                .lock_life()
                .suspected_resources
                .bind_groups
                .insert(bind_group.info.tracker_index(), bind_group.clone());
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
    pub fn device_create_shader_module<A: HalApi>(
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

        let hub = A::hub(self);
        let fid = hub.shader_modules.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

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

            let (id, _) = fid.assign(Arc::new(shader));
            api_log!("Device::create_shader_module -> {id:?}");
            return (id, None);
        };

        log::error!("Device::create_shader_module error: {error}");

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
        id_in: Option<id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = A::hub(self);
        let fid = hub.shader_modules.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

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
                Err(e) => break 'error e,
            };
            let (id, _) = fid.assign(Arc::new(shader));
            api_log!("Device::create_shader_module_spirv -> {id:?}");
            return (id, None);
        };

        log::error!("Device::create_shader_module_spirv error: {error}");

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn shader_module_label<A: HalApi>(&self, id: id::ShaderModuleId) -> String {
        A::hub(self).shader_modules.label_for_resource(id)
    }

    pub fn shader_module_drop<A: HalApi>(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");
        api_log!("ShaderModule::drop {shader_module_id:?}");

        let hub = A::hub(self);
        hub.shader_modules.unregister(shader_module_id);
    }

    pub fn device_create_command_encoder<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Option<id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = A::hub(self);
        let fid = hub
            .command_buffers
            .prepare(id_in.map(|id| id.into_command_buffer_id()));

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid,
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost;
            }
            let Some(queue) = device.get_queue() else {
                break 'error DeviceError::InvalidQueueId;
            };
            let encoder = match device
                .command_allocator
                .acquire_encoder(device.raw(), queue.raw.as_ref().unwrap())
            {
                Ok(raw) => raw,
                Err(_) => break 'error DeviceError::OutOfMemory,
            };
            let command_buffer = command::CommandBuffer::new(
                encoder,
                &device,
                #[cfg(feature = "trace")]
                device.trace.lock().is_some(),
                desc.label.to_hal(device.instance_flags).map(str::to_owned),
            );

            let (id, _) = fid.assign(Arc::new(command_buffer));
            api_log!("Device::create_command_encoder -> {id:?}");
            return (id.into_command_encoder_id(), None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id.into_command_encoder_id(), Some(error))
    }

    pub fn command_buffer_label<A: HalApi>(&self, id: id::CommandBufferId) -> String {
        A::hub(self).command_buffers.label_for_resource(id)
    }

    pub fn command_encoder_drop<A: HalApi>(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");
        api_log!("CommandEncoder::drop {command_encoder_id:?}");

        let hub = A::hub(self);

        if let Some(cmd_buf) = hub
            .command_buffers
            .unregister(command_encoder_id.into_command_buffer_id())
        {
            cmd_buf.data.lock().as_mut().unwrap().encoder.discard();
            cmd_buf
                .device
                .untrack(&cmd_buf.data.lock().as_ref().unwrap().trackers);
        }
    }

    pub fn command_buffer_drop<A: HalApi>(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        api_log!("CommandBuffer::drop {command_buffer_id:?}");
        self.command_encoder_drop::<A>(command_buffer_id.into_command_encoder_id())
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

    pub fn render_bundle_encoder_finish<A: HalApi>(
        &self,
        bundle_encoder: command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Option<id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("RenderBundleEncoder::finish");

        let hub = A::hub(self);

        let fid = hub.render_bundles.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(bundle_encoder.parent()) {
                Ok(device) => device,
                Err(_) => break 'error command::RenderBundleError::INVALID_DEVICE,
            };
            if !device.is_valid() {
                break 'error command::RenderBundleError::INVALID_DEVICE;
            }

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

            let (id, resource) = fid.assign(Arc::new(render_bundle));
            api_log!("RenderBundleEncoder::finish -> {id:?}");
            device.trackers.lock().bundles.insert_single(resource);
            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn render_bundle_label<A: HalApi>(&self, id: id::RenderBundleId) -> String {
        A::hub(self).render_bundles.label_for_resource(id)
    }

    pub fn render_bundle_drop<A: HalApi>(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("RenderBundle::drop");
        api_log!("RenderBundle::drop {render_bundle_id:?}");

        let hub = A::hub(self);

        if let Some(bundle) = hub.render_bundles.unregister(render_bundle_id) {
            bundle
                .device
                .lock_life()
                .suspected_resources
                .render_bundles
                .insert(bundle.info.tracker_index(), bundle.clone());
        }
    }

    pub fn device_create_query_set<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: Option<id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("Device::create_query_set");

        let hub = A::hub(self);
        let fid = hub.query_sets.prepare(id_in);

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

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

            let (id, resource) = fid.assign(Arc::new(query_set));
            api_log!("Device::create_query_set -> {id:?}");
            device.trackers.lock().query_sets.insert_single(resource);

            return (id, None);
        };

        let id = fid.assign_error("");
        (id, Some(error))
    }

    pub fn query_set_drop<A: HalApi>(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");
        api_log!("QuerySet::drop {query_set_id:?}");

        let hub = A::hub(self);

        if let Some(query_set) = hub.query_sets.unregister(query_set_id) {
            let device = &query_set.device;

            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::DestroyQuerySet(query_set_id));
            }

            device
                .lock_life()
                .suspected_resources
                .query_sets
                .insert(query_set.info.tracker_index(), query_set.clone());
        }
    }

    pub fn query_set_label<A: HalApi>(&self, id: id::QuerySetId) -> String {
        A::hub(self).query_sets.label_for_resource(id)
    }

    pub fn device_create_render_pipeline<A: HalApi>(
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

        let hub = A::hub(self);

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));
        let implicit_error_context = implicit_context.clone();

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }
            #[cfg(feature = "trace")]
            if let Some(ref mut trace) = *device.trace.lock() {
                trace.add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let pipeline =
                match device.create_render_pipeline(&device.adapter, desc, implicit_context, hub) {
                    Ok(pair) => pair,
                    Err(e) => break 'error e,
                };

            let (id, resource) = fid.assign(Arc::new(pipeline));
            api_log!("Device::create_render_pipeline -> {id:?}");

            device
                .trackers
                .lock()
                .render_pipelines
                .insert_single(resource);

            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layout. We have to remove any existing entries first.
        let mut pipeline_layout_guard = hub.pipeline_layouts.write();
        let mut bgl_guard = hub.bind_group_layouts.write();
        if let Some(ref ids) = implicit_error_context {
            if pipeline_layout_guard.contains(ids.root_id) {
                pipeline_layout_guard.remove(ids.root_id);
            }
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            for &bgl_id in ids.group_ids.iter() {
                if bgl_guard.contains(bgl_id) {
                    bgl_guard.remove(bgl_id);
                }
                bgl_guard.insert_error(bgl_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            }
        }

        log::error!("Device::create_render_pipeline error: {error}");

        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);

        let error = 'error: {
            let pipeline = match hub.render_pipelines.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break 'error binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };
            let id = match pipeline.layout.bind_group_layouts.get(index as usize) {
                Some(bg) => hub.bind_group_layouts.prepare(id_in).assign_existing(bg),
                None => {
                    break 'error binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index)
                }
            };
            return (id, None);
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
        api_log!("RenderPipeline::drop {render_pipeline_id:?}");

        let hub = A::hub(self);

        if let Some(pipeline) = hub.render_pipelines.unregister(render_pipeline_id) {
            let device = &pipeline.device;
            let mut life_lock = device.lock_life();
            life_lock
                .suspected_resources
                .render_pipelines
                .insert(pipeline.info.tracker_index(), pipeline.clone());

            life_lock.suspected_resources.pipeline_layouts.insert(
                pipeline.layout.info.tracker_index(),
                pipeline.layout.clone(),
            );
        }
    }

    pub fn device_create_compute_pipeline<A: HalApi>(
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

        let hub = A::hub(self);

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));
        let implicit_error_context = implicit_context.clone();

        let error = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

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
                Err(e) => break 'error e,
            };

            let (id, resource) = fid.assign(Arc::new(pipeline));
            api_log!("Device::create_compute_pipeline -> {id:?}");

            device
                .trackers
                .lock()
                .compute_pipelines
                .insert_single(resource);
            return (id, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layout. We have to remove any existing entries first.
        let mut pipeline_layout_guard = hub.pipeline_layouts.write();
        let mut bgl_guard = hub.bind_group_layouts.write();
        if let Some(ref ids) = implicit_error_context {
            if pipeline_layout_guard.contains(ids.root_id) {
                pipeline_layout_guard.remove(ids.root_id);
            }
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            for &bgl_id in ids.group_ids.iter() {
                if bgl_guard.contains(bgl_id) {
                    bgl_guard.remove(bgl_id);
                }
                bgl_guard.insert_error(bgl_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            }
        }
        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: Option<id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);

        let error = 'error: {
            let pipeline = match hub.compute_pipelines.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break 'error binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };

            let id = match pipeline.layout.bind_group_layouts.get(index as usize) {
                Some(bg) => hub.bind_group_layouts.prepare(id_in).assign_existing(bg),
                None => {
                    break 'error binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index)
                }
            };

            return (id, None);
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
        api_log!("ComputePipeline::drop {compute_pipeline_id:?}");

        let hub = A::hub(self);

        if let Some(pipeline) = hub.compute_pipelines.unregister(compute_pipeline_id) {
            let device = &pipeline.device;
            let mut life_lock = device.lock_life();
            life_lock
                .suspected_resources
                .compute_pipelines
                .insert(pipeline.info.tracker_index(), pipeline.clone());
            life_lock.suspected_resources.pipeline_layouts.insert(
                pipeline.layout.info.tracker_index(),
                pipeline.layout.clone(),
            );
        }
    }

    /// # Safety
    /// The `data` argument of `desc` must have been returned by
    /// [Self::pipeline_cache_get_data] for the same adapter
    pub unsafe fn device_create_pipeline_cache<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &pipeline::PipelineCacheDescriptor<'_>,
        id_in: Option<id::PipelineCacheId>,
    ) -> (
        id::PipelineCacheId,
        Option<pipeline::CreatePipelineCacheError>,
    ) {
        profiling::scope!("Device::create_pipeline_cache");

        let hub = A::hub(self);

        let fid = hub.pipeline_caches.prepare(id_in);
        let error: pipeline::CreatePipelineCacheError = 'error: {
            let device = match hub.devices.get(device_id) {
                Ok(device) => device,
                // TODO: Handle error properly
                Err(crate::storage::InvalidId) => break 'error DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }
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
                    let (id, _) = fid.assign(Arc::new(cache));
                    api_log!("Device::create_pipeline_cache -> {id:?}");
                    return (id, None);
                }
                Err(e) => break 'error e,
            }
        };

        let id = fid.assign_error(desc.label.borrow_or_default());

        (id, Some(error))
    }

    pub fn pipeline_cache_drop<A: HalApi>(&self, pipeline_cache_id: id::PipelineCacheId) {
        profiling::scope!("PipelineCache::drop");
        api_log!("PipelineCache::drop {pipeline_cache_id:?}");

        let hub = A::hub(self);

        if let Some(cache) = hub.pipeline_caches.unregister(pipeline_cache_id) {
            drop(cache)
        }
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
                return Err(E::UnsupportedUsage);
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
                let hub = A::hub(self);
                let surface_guard = self.surfaces.read();
                let device_guard = hub.devices.read();

                let device = match device_guard.get(device_id) {
                    Ok(device) => device,
                    Err(_) => break 'error DeviceError::Invalid.into(),
                };
                if !device.is_valid() {
                    break 'error DeviceError::Lost.into();
                }

                #[cfg(feature = "trace")]
                if let Some(ref mut trace) = *device.trace.lock() {
                    trace.add(trace::Action::ConfigureSurface(surface_id, config.clone()));
                }

                let surface = match surface_guard.get(surface_id) {
                    Ok(surface) => surface,
                    Err(_) => break 'error E::InvalidSurface,
                };

                let caps = unsafe {
                    let suf = A::surface_as_hal(surface);
                    let adapter = &device.adapter;
                    match adapter.raw.adapter.surface_capabilities(suf.unwrap()) {
                        Some(caps) => caps,
                        None => break 'error E::UnsupportedQueueFamily,
                    }
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

                match unsafe {
                    A::surface_as_hal(surface)
                        .unwrap()
                        .configure(device.raw(), &hal_config)
                } {
                    Ok(()) => (),
                    Err(error) => {
                        break 'error match error {
                            hal::SurfaceError::Outdated | hal::SurfaceError::Lost => {
                                E::InvalidSurface
                            }
                            hal::SurfaceError::Device(error) => E::Device(error.into()),
                            hal::SurfaceError::Other(message) => {
                                log::error!("surface configuration failed: {}", message);
                                E::InvalidSurface
                            }
                        }
                    }
                }

                let mut presentation = surface.presentation.lock();
                *presentation = Some(present::Presentation {
                    device: super::any_device::AnyDevice::new(device.clone()),
                    config: config.clone(),
                    acquired_texture: None,
                });
            }

            user_callbacks.fire();
            return None;
        };

        Some(error)
    }

    #[cfg(feature = "replay")]
    /// Only triage suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<A: HalApi>(&self, device_id: DeviceId) -> Result<(), InvalidDevice> {
        let hub = A::hub(self);

        let device = hub.devices.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.is_valid() {
            return Err(InvalidDevice);
        }
        device.lock_life().triage_suspected(&device.trackers);
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
        api_log!("Device::poll");

        let hub = A::hub(self);
        let device = hub
            .devices
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;

        if let wgt::Maintain::WaitForSubmissionIndex(submission_index) = maintain {
            if submission_index.queue_id != device_id.into_queue_id() {
                return Err(WaitIdleError::WrongSubmissionIndex(
                    submission_index.queue_id,
                    device_id,
                ));
            }
        }

        let DevicePoll {
            closures,
            queue_empty,
        } = Self::poll_single_device(&device, maintain)?;

        closures.fire();

        Ok(queue_empty)
    }

    fn poll_single_device<A: HalApi>(
        device: &crate::device::Device<A>,
        maintain: wgt::Maintain<queue::WrappedSubmissionIndex>,
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

    /// Poll all devices belonging to the backend `A`.
    ///
    /// If `force_wait` is true, block until all buffer mappings are done.
    ///
    /// Return `all_queue_empty` indicating whether there are more queue
    /// submissions still in flight.
    fn poll_all_devices_of_api<A: HalApi>(
        &self,
        force_wait: bool,
        closures: &mut UserClosures,
    ) -> Result<bool, WaitIdleError> {
        profiling::scope!("poll_device");

        let hub = A::hub(self);
        let mut all_queue_empty = true;
        {
            let device_guard = hub.devices.read();

            for (_id, device) in device_guard.iter(A::VARIANT) {
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
        let mut all_queue_empty = true;

        #[cfg(vulkan)]
        {
            all_queue_empty &=
                self.poll_all_devices_of_api::<hal::api::Vulkan>(force_wait, &mut closures)?;
        }
        #[cfg(metal)]
        {
            all_queue_empty &=
                self.poll_all_devices_of_api::<hal::api::Metal>(force_wait, &mut closures)?;
        }
        #[cfg(dx12)]
        {
            all_queue_empty &=
                self.poll_all_devices_of_api::<hal::api::Dx12>(force_wait, &mut closures)?;
        }
        #[cfg(gles)]
        {
            all_queue_empty &=
                self.poll_all_devices_of_api::<hal::api::Gles>(force_wait, &mut closures)?;
        }

        closures.fire();

        Ok(all_queue_empty)
    }

    pub fn device_label<A: HalApi>(&self, id: DeviceId) -> String {
        A::hub(self).devices.label_for_resource(id)
    }

    pub fn device_start_capture<A: HalApi>(&self, id: DeviceId) {
        api_log!("Device::start_capture");

        let hub = A::hub(self);

        if let Ok(device) = hub.devices.get(id) {
            if !device.is_valid() {
                return;
            }
            unsafe { device.raw().start_capture() };
        }
    }

    pub fn device_stop_capture<A: HalApi>(&self, id: DeviceId) {
        api_log!("Device::stop_capture");

        let hub = A::hub(self);

        if let Ok(device) = hub.devices.get(id) {
            if !device.is_valid() {
                return;
            }
            unsafe { device.raw().stop_capture() };
        }
    }

    // This is a test-only function to force the device into an
    // invalid state by inserting an error value in its place in
    // the registry.
    pub fn device_make_invalid<A: HalApi>(&self, device_id: DeviceId) {
        let hub = A::hub(self);
        hub.devices
            .force_replace_with_error(device_id, "Made invalid.");
    }

    pub fn pipeline_cache_get_data<A: HalApi>(&self, id: id::PipelineCacheId) -> Option<Vec<u8>> {
        use crate::pipeline_cache;
        api_log!("PipelineCache::get_data");
        let hub = A::hub(self);

        if let Ok(cache) = hub.pipeline_caches.get(id) {
            // TODO: Is this check needed?
            if !cache.device.is_valid() {
                return None;
            }
            if let Some(raw_cache) = cache.raw.as_ref() {
                let mut vec = unsafe { cache.device.raw().pipeline_cache_get_data(raw_cache) }?;
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
        }
        None
    }

    pub fn device_drop<A: HalApi>(&self, device_id: DeviceId) {
        profiling::scope!("Device::drop");
        api_log!("Device::drop {device_id:?}");

        let hub = A::hub(self);
        if let Some(device) = hub.devices.unregister(device_id) {
            let device_lost_closure = device.lock_life().device_lost_closure.take();
            if let Some(closure) = device_lost_closure {
                closure.call(DeviceLostReason::Dropped, String::from("Device dropped."));
            }

            // The things `Device::prepare_to_die` takes care are mostly
            // unnecessary here. We know our queue is empty, so we don't
            // need to wait for submissions or triage them. We know we were
            // just polled, so `life_tracker.free_resources` is empty.
            debug_assert!(device.lock_life().queue_empty());
            {
                let mut pending_writes = device.pending_writes.lock();
                let pending_writes = pending_writes.as_mut().unwrap();
                pending_writes.deactivate();
            }

            drop(device);
        }
    }

    // This closure will be called exactly once during "lose the device",
    // or when it is replaced.
    pub fn device_set_device_lost_closure<A: HalApi>(
        &self,
        device_id: DeviceId,
        device_lost_closure: DeviceLostClosure,
    ) {
        let hub = A::hub(self);

        if let Ok(Some(device)) = hub.devices.try_get(device_id) {
            let mut life_tracker = device.lock_life();
            if let Some(existing_closure) = life_tracker.device_lost_closure.take() {
                // It's important to not hold the lock while calling the closure.
                drop(life_tracker);
                existing_closure.call(DeviceLostReason::ReplacedCallback, "".to_string());
                life_tracker = device.lock_life();
            }
            life_tracker.device_lost_closure = Some(device_lost_closure);
        } else {
            // No device? Okay. Just like we have to call any existing closure
            // before we drop it, we need to call this closure before we exit
            // this function, because there's no device that is ever going to
            // call it.
            device_lost_closure.call(DeviceLostReason::DeviceInvalid, "".to_string());
        }
    }

    pub fn device_destroy<A: HalApi>(&self, device_id: DeviceId) {
        api_log!("Device::destroy {device_id:?}");

        let hub = A::hub(self);

        if let Ok(device) = hub.devices.get(device_id) {
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
            device.valid.store(false, Ordering::Relaxed);
        }
    }

    pub fn device_mark_lost<A: HalApi>(&self, device_id: DeviceId, message: &str) {
        api_log!("Device::mark_lost {device_id:?}");

        let hub = A::hub(self);

        if let Ok(device) = hub.devices.get(device_id) {
            device.lose(message);
        }
    }

    pub fn queue_drop<A: HalApi>(&self, queue_id: QueueId) {
        profiling::scope!("Queue::drop");
        api_log!("Queue::drop {queue_id:?}");

        let hub = A::hub(self);
        if let Some(queue) = hub.queues.unregister(queue_id) {
            drop(queue);
        }
    }

    pub fn buffer_map_async<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
        op: BufferMapOperation,
    ) -> BufferAccessResult {
        api_log!("Buffer::map_async {buffer_id:?} offset {offset:?} size {size:?} op: {op:?}");

        // User callbacks must not be called while holding buffer_map_async_inner's locks, so we
        // defer the error callback if it needs to be called immediately (typically when running
        // into errors).
        if let Err((mut operation, err)) =
            self.buffer_map_async_inner::<A>(buffer_id, offset, size, op)
        {
            if let Some(callback) = operation.callback.take() {
                callback.call(Err(err.clone()));
            }
            log::error!("Buffer::map_async error: {err}");
            return Err(err);
        }

        Ok(())
    }

    // Returns the mapping callback in case of error so that the callback can be fired outside
    // of the locks that are held in this function.
    fn buffer_map_async_inner<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
        op: BufferMapOperation,
    ) -> Result<(), (BufferMapOperation, BufferAccessError)> {
        profiling::scope!("Buffer::map_async");

        let hub = A::hub(self);

        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsages::MAP_READ, hal::BufferUses::MAP_READ),
            HostMap::Write => (wgt::BufferUsages::MAP_WRITE, hal::BufferUses::MAP_WRITE),
        };

        let buffer = {
            let buffer = hub.buffers.get(buffer_id);

            let buffer = match buffer {
                Ok(b) => b,
                Err(_) => {
                    return Err((op, BufferAccessError::Invalid));
                }
            };
            {
                let snatch_guard = buffer.device.snatchable_lock.read();
                if buffer.is_destroyed(&snatch_guard) {
                    return Err((op, BufferAccessError::Destroyed));
                }
            }

            let range_size = if let Some(size) = size {
                size
            } else if offset > buffer.size {
                0
            } else {
                buffer.size - offset
            };

            if offset % wgt::MAP_ALIGNMENT != 0 {
                return Err((op, BufferAccessError::UnalignedOffset { offset }));
            }
            if range_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err((op, BufferAccessError::UnalignedRangeSize { range_size }));
            }

            let range = offset..(offset + range_size);

            if range.start % wgt::MAP_ALIGNMENT != 0 || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0
            {
                return Err((op, BufferAccessError::UnalignedRange));
            }

            let device = &buffer.device;
            if !device.is_valid() {
                return Err((op, DeviceError::Lost.into()));
            }

            if let Err(e) = check_buffer_usage(buffer.info.id(), buffer.usage, pub_usage) {
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

            {
                let map_state = &mut *buffer.map_state.lock();
                *map_state = match *map_state {
                    resource::BufferMapState::Init { .. }
                    | resource::BufferMapState::Active { .. } => {
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
            }

            let snatch_guard = buffer.device.snatchable_lock.read();

            {
                let mut trackers = buffer.device.as_ref().trackers.lock();
                trackers.buffers.set_single(&buffer, internal_use);
                //TODO: Check if draining ALL buffers is correct!
                let _ = trackers.buffers.drain_transitions(&snatch_guard);
            }

            drop(snatch_guard);

            buffer
        };

        buffer.device.lock_life().map(&buffer);

        Ok(())
    }

    pub fn buffer_get_mapped_range<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(*mut u8, u64), BufferAccessError> {
        profiling::scope!("Buffer::get_mapped_range");
        api_log!("Buffer::get_mapped_range {buffer_id:?} offset {offset:?} size {size:?}");

        let hub = A::hub(self);

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;

        {
            let snatch_guard = buffer.device.snatchable_lock.read();
            if buffer.is_destroyed(&snatch_guard) {
                return Err(BufferAccessError::Destroyed);
            }
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
                // rather than the beginning of the buffer.
                let relative_offset = (offset - range.start) as isize;
                unsafe { Ok((ptr.as_ptr().offset(relative_offset), range_size)) }
            }
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                Err(BufferAccessError::NotMapped)
            }
        }
    }
    pub fn buffer_unmap<A: HalApi>(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        profiling::scope!("unmap", "Buffer");
        api_log!("Buffer::unmap {buffer_id:?}");

        let hub = A::hub(self);

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;

        let snatch_guard = buffer.device.snatchable_lock.read();
        if buffer.is_destroyed(&snatch_guard) {
            return Err(BufferAccessError::Destroyed);
        }
        drop(snatch_guard);

        if !buffer.device.is_valid() {
            return Err(DeviceError::Lost.into());
        }

        buffer.unmap()
    }
}

struct DevicePoll {
    closures: UserClosures,
    queue_empty: bool,
}
