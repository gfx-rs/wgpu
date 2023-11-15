#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{self, BindGroupLayout},
    command, conv,
    device::{
        life::WaitIdleError, map_buffer, queue, Device, DeviceError, DeviceLostClosure, HostMap,
    },
    global::Global,
    hal_api::HalApi,
    hub::Token,
    id::{self, AdapterId, DeviceId, SurfaceId},
    identity::{GlobalIdentityHandlerFactory, Input},
    init_tracker::TextureInitTracker,
    instance::{self, Adapter, Surface},
    pipeline, present,
    resource::{self, Buffer, BufferAccessResult, BufferMapState},
    resource::{BufferAccessError, BufferMapOperation, TextureClearMode},
    storage::InvalidId,
    validation::check_buffer_usage,
    FastHashMap, Label, LabelHelpers as _, Stored,
};

use hal::{CommandEncoder as _, Device as _};
use smallvec::SmallVec;

use wgt::{BufferAddress, TextureFormat};

use std::{borrow::Cow, iter, mem, ops::Range, ptr};

use super::{
    BufferMapPendingClosure, ImplicitPipelineIds, InvalidDevice, UserClosures, IMPLICIT_FAILURE,
};

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_is_surface_supported<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        surface_id: SurfaceId,
    ) -> Result<bool, instance::IsSurfaceSupportedError> {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let (adapter_guard, mut _token) = hub.adapters.read(&mut token);
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
        let mut token = Token::root();

        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let (adapter_guard, mut _token) = hub.adapters.read(&mut token);
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
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.valid {
            return Err(InvalidDevice);
        }

        Ok(device.features)
    }

    pub fn device_limits<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::Limits, InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.valid {
            return Err(InvalidDevice);
        }

        Ok(device.limits.clone())
    }

    pub fn device_downlevel_properties<A: HalApi>(
        &self,
        device_id: DeviceId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.valid {
            return Err(InvalidDevice);
        }

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
        let mut token = Token::root();
        let fid = hub.buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            if desc.usage.is_empty() {
                // Per spec, `usage` must not be zero.
                break resource::CreateBufferError::InvalidUsage(desc.usage);
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                let mut desc = desc.clone();
                let mapped_at_creation = mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsages::COPY_DST;
                }
                trace
                    .lock()
                    .add(trace::Action::CreateBuffer(fid.id(), desc));
            }

            let mut buffer = match device.create_buffer(device_id, desc, false) {
                Ok(buffer) => buffer,
                Err(e) => break e,
            };
            let ref_count = buffer.life_guard.add_ref();

            let buffer_use = if !desc.mapped_at_creation {
                hal::BufferUses::empty()
            } else if desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // buffer is mappable, so we are just doing that at start
                let map_size = buffer.size;
                let ptr = if map_size == 0 {
                    std::ptr::NonNull::dangling()
                } else {
                    match map_buffer(&device.raw, &mut buffer, 0, map_size, HostMap::Write) {
                        Ok(ptr) => ptr,
                        Err(e) => {
                            let raw = buffer.raw.unwrap();
                            device.lock_life(&mut token).schedule_resource_destruction(
                                queue::TempResource::Buffer(raw),
                                !0,
                            );
                            break e.into();
                        }
                    }
                };
                buffer.map_state = resource::BufferMapState::Active {
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
                let mut stage = match device.create_buffer(device_id, &stage_desc, true) {
                    Ok(stage) => stage,
                    Err(e) => {
                        let raw = buffer.raw.unwrap();
                        device
                            .lock_life(&mut token)
                            .schedule_resource_destruction(queue::TempResource::Buffer(raw), !0);
                        break e;
                    }
                };
                let stage_buffer = stage.raw.unwrap();
                let mapping = match unsafe { device.raw.map_buffer(&stage_buffer, 0..stage.size) } {
                    Ok(mapping) => mapping,
                    Err(e) => {
                        let raw = buffer.raw.unwrap();
                        let mut life_lock = device.lock_life(&mut token);
                        life_lock
                            .schedule_resource_destruction(queue::TempResource::Buffer(raw), !0);
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(stage_buffer),
                            !0,
                        );
                        break DeviceError::from(e).into();
                    }
                };

                assert_eq!(buffer.size % wgt::COPY_BUFFER_ALIGNMENT, 0);
                // Zero initialize memory and then mark both staging and buffer as initialized
                // (it's guaranteed that this is the case by the time the buffer is usable)
                unsafe { ptr::write_bytes(mapping.ptr.as_ptr(), 0, buffer.size as usize) };
                buffer.initialization_status.drain(0..buffer.size);
                stage.initialization_status.drain(0..buffer.size);

                buffer.map_state = resource::BufferMapState::Init {
                    ptr: mapping.ptr,
                    needs_flush: !mapping.is_coherent,
                    stage_buffer,
                };
                hal::BufferUses::COPY_DST
            };

            let id = fid.assign(buffer, &mut token);
            log::trace!("Device::create_buffer -> {:?}", id.0);

            device
                .trackers
                .lock()
                .buffers
                .insert_single(id, ref_count, buffer_use);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        let mut token = Token::root();
        let fid = hub.buffers.prepare(id_in);

        fid.assign_error(label.borrow_or_default(), &mut token);
    }

    pub fn create_render_bundle_error<A: HalApi>(
        &self,
        id_in: Input<G, id::RenderBundleId>,
        label: Label,
    ) {
        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.render_bundles.prepare(id_in);

        let (_, mut token) = hub.devices.read(&mut token);
        fid.assign_error(label.borrow_or_default(), &mut token);
    }

    /// Assign `id_in` an error with the given `label`.
    ///
    /// See `create_buffer_error` for more context and explaination.
    pub fn create_texture_error<A: HalApi>(&self, id_in: Input<G, id::TextureId>, label: Label) {
        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(id_in);

        fid.assign_error(label.borrow_or_default(), &mut token);
    }

    #[cfg(feature = "replay")]
    pub fn device_wait_for_buffer<A: HalApi>(
        &self,
        device_id: DeviceId,
        buffer_id: id::BufferId,
    ) -> Result<(), WaitIdleError> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let last_submission = {
            let (buffer_guard, _) = hub.buffers.write(&mut token);
            match buffer_guard.get(buffer_id) {
                Ok(buffer) => buffer.life_guard.life_count(),
                Err(_) => return Ok(()),
            }
        };

        device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?
            .wait_for_submit(last_submission, &mut token)
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
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        if !device.valid {
            return Err(DeviceError::Lost.into());
        }
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_WRITE)?;
        //assert!(buffer isn't used by the GPU);

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
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
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
            if !mapping.is_coherent {
                device
                    .raw
                    .flush_mapped_ranges(raw_buf, iter::once(offset..offset + data.len() as u64));
            }
            device
                .raw
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
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        if !device.valid {
            return Err(DeviceError::Lost.into());
        }
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_READ)?;
        //assert!(buffer isn't used by the GPU);

        let raw_buf = buffer.raw.as_ref().unwrap();
        unsafe {
            let mapping = device
                .raw
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            if !mapping.is_coherent {
                device.raw.invalidate_mapped_ranges(
                    raw_buf,
                    iter::once(offset..offset + data.len() as u64),
                );
            }
            ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), data.as_mut_ptr(), data.len());
            device
                .raw
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
            let mut token = Token::root();

            //TODO: lock pending writes separately, keep the device read-only
            let (mut device_guard, mut token) = hub.devices.write(&mut token);

            log::trace!("Buffer::destroy {buffer_id:?}");
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = buffer_guard
                .get_and_mark_destroyed(buffer_id)
                .map_err(|_| resource::DestroyError::Invalid)?;

            let device = &mut device_guard[buffer.device_id.value];

            map_closure = match &buffer.map_state {
                &BufferMapState::Waiting(..) // To get the proper callback behavior.
                | &BufferMapState::Init { .. }
                | &BufferMapState::Active { .. }
                => {
                    self.buffer_unmap_inner(buffer_id, buffer, device)
                        .unwrap_or(None)
                }
                _ => None,
            };

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::FreeBuffer(buffer_id));
            }

            let raw = buffer
                .raw
                .take()
                .ok_or(resource::DestroyError::AlreadyDestroyed)?;
            let temp = queue::TempResource::Buffer(raw);

            if device.pending_writes.dst_buffers.contains(&buffer_id) {
                device.pending_writes.temp_resources.push(temp);
            } else {
                let last_submit_index = buffer.life_guard.life_count();
                drop(buffer_guard);
                device
                    .lock_life(&mut token)
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
        log::trace!("Buffer::drop {buffer_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (ref_count, last_submit_index, device_id) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            match buffer_guard.get_occupied_or_destroyed_mut(buffer_id) {
                Ok(buffer) => {
                    let ref_count = buffer.life_guard.ref_count.take().unwrap();
                    let last_submit_index = buffer.life_guard.life_count();
                    (ref_count, last_submit_index, buffer.device_id.value)
                }
                Err(InvalidId) => {
                    hub.buffers.unregister_locked(buffer_id, &mut *buffer_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            if device.pending_writes.dst_buffers.contains(&buffer_id) {
                life_lock.future_suspected_buffers.push(Stored {
                    value: id::Valid(buffer_id),
                    ref_count,
                });
            } else {
                drop(ref_count);
                life_lock
                    .suspected_resources
                    .buffers
                    .push(id::Valid(buffer_id));
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
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
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(id_in);

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let adapter = &adapter_guard[device.adapter_id.value];
            let texture = match device.create_texture(device_id, adapter, desc) {
                Ok(texture) => texture,
                Err(error) => break error,
            };
            let ref_count = texture.life_guard.add_ref();

            let id = fid.assign(texture, &mut token);
            log::trace!("Device::create_texture -> {:?}", id.0);

            device.trackers.lock().textures.insert_single(
                id.0,
                ref_count,
                hal::TextureUses::UNINITIALIZED,
            );

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        profiling::scope!("Device::create_texture_from_hal");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(id_in);

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            // NB: Any change done through the raw texture handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let adapter = &adapter_guard[device.adapter_id.value];

            let format_features = match device
                .describe_format_features(adapter, desc.format)
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
                TextureClearMode::None,
            );
            if desc.usage.contains(wgt::TextureUsages::COPY_DST) {
                texture.hal_usage |= hal::TextureUses::COPY_DST;
            }

            texture.initialization_status = TextureInitTracker::new(desc.mip_level_count, 0);

            let ref_count = texture.life_guard.add_ref();

            let id = fid.assign(texture, &mut token);
            log::trace!("Device::create_texture -> {:?}", id.0);

            device.trackers.lock().textures.insert_single(
                id.0,
                ref_count,
                hal::TextureUses::UNINITIALIZED,
            );

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        id_in: Input<G, id::BufferId>,
    ) -> (id::BufferId, Option<resource::CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            // NB: Any change done through the raw buffer handle will not be
            // recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateBuffer(fid.id(), desc.clone()));
            }

            let mut buffer = device.create_buffer_from_hal(hal_buffer, device_id, desc);

            // Assume external buffers are initialized
            buffer.initialization_status = crate::init_tracker::BufferInitTracker::new(0);

            let ref_count = buffer.life_guard.add_ref();

            let id = fid.assign(buffer, &mut token);
            log::trace!("Device::create_buffer -> {:?}", id.0);

            device
                .trackers
                .lock()
                .buffers
                .insert_single(id, ref_count, hal::BufferUses::empty());

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        log::trace!("Texture::destroy {texture_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        //TODO: lock pending writes separately, keep the device read-only
        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        let (mut texture_guard, _) = hub.textures.write(&mut token);
        let texture = texture_guard
            .get_and_mark_destroyed(texture_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut device_guard[texture.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(trace::Action::FreeTexture(texture_id));
        }

        let last_submit_index = texture.life_guard.life_count();

        let clear_views = match std::mem::replace(&mut texture.clear_mode, TextureClearMode::None) {
            TextureClearMode::BufferCopy => SmallVec::new(),
            TextureClearMode::RenderPass { clear_views, .. } => clear_views,
            TextureClearMode::None => SmallVec::new(),
        };

        match texture.inner {
            resource::TextureInner::Native { ref mut raw } => {
                let raw = raw.take().ok_or(resource::DestroyError::AlreadyDestroyed)?;
                let temp = queue::TempResource::Texture(raw, clear_views);

                if device.pending_writes.dst_textures.contains(&texture_id) {
                    device.pending_writes.temp_resources.push(temp);
                } else {
                    drop(texture_guard);
                    device
                        .lock_life(&mut token)
                        .schedule_resource_destruction(temp, last_submit_index);
                }
            }
            resource::TextureInner::Surface { .. } => {
                for clear_view in clear_views {
                    unsafe {
                        device.raw.destroy_texture_view(clear_view);
                    }
                }
                // TODO?
            }
        }

        Ok(())
    }

    pub fn texture_drop<A: HalApi>(&self, texture_id: id::TextureId, wait: bool) {
        profiling::scope!("Texture::drop");
        log::trace!("Texture::drop {texture_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (ref_count, last_submit_index, device_id) = {
            let (mut texture_guard, _) = hub.textures.write(&mut token);
            match texture_guard.get_occupied_or_destroyed_mut(texture_id) {
                Ok(texture) => {
                    let ref_count = texture.life_guard.ref_count.take().unwrap();
                    let last_submit_index = texture.life_guard.life_count();
                    (ref_count, last_submit_index, texture.device_id.value)
                }
                Err(InvalidId) => {
                    hub.textures
                        .unregister_locked(texture_id, &mut *texture_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            if device.pending_writes.dst_textures.contains(&texture_id) {
                life_lock.future_suspected_textures.push(Stored {
                    value: id::Valid(texture_id),
                    ref_count,
                });
            } else {
                drop(ref_count);
                life_lock
                    .suspected_resources
                    .textures
                    .push(id::Valid(texture_id));
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
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
        let mut token = Token::root();
        let fid = hub.texture_views.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let error = loop {
            let texture = match texture_guard.get(texture_id) {
                Ok(texture) => texture,
                Err(_) => break resource::CreateTextureViewError::InvalidTexture,
            };
            let device = &device_guard[texture.device_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateTextureView {
                    id: fid.id(),
                    parent_id: texture_id,
                    desc: desc.clone(),
                });
            }

            let view = match device.create_texture_view(texture, texture_id, desc) {
                Ok(view) => view,
                Err(e) => break e,
            };
            let ref_count = view.life_guard.add_ref();
            let id = fid.assign(view, &mut token);

            device.trackers.lock().views.insert_single(id, ref_count);

            log::trace!("Texture::create_view {:?} -> {:?}", texture_id, id.0);

            return (id.0, None);
        };

        log::error!("Texture::create_view {:?} error {:?}", texture_id, error);

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        log::trace!("TextureView::drop {:?}", texture_view_id);

        let hub = A::hub(self);
        let mut token = Token::root();

        let (last_submit_index, device_id) = {
            let (mut texture_view_guard, _) = hub.texture_views.write(&mut token);

            match texture_view_guard.get_mut(texture_view_id) {
                Ok(view) => {
                    let _ref_count = view.life_guard.ref_count.take();
                    let last_submit_index = view.life_guard.life_count();
                    (last_submit_index, view.device_id.value)
                }
                Err(InvalidId) => {
                    hub.texture_views
                        .unregister_locked(texture_view_id, &mut *texture_view_guard);
                    return Ok(());
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        device
            .lock_life(&mut token)
            .suspected_resources
            .texture_views
            .push(id::Valid(texture_view_id));

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
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
        let mut token = Token::root();
        let fid = hub.samplers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateSampler(fid.id(), desc.clone()));
            }

            let sampler = match device.create_sampler(device_id, desc) {
                Ok(sampler) => sampler,
                Err(e) => break e,
            };
            let ref_count = sampler.life_guard.add_ref();
            let id = fid.assign(sampler, &mut token);

            device.trackers.lock().samplers.insert_single(id, ref_count);

            log::trace!("Device::create_sampler -> {:?}", id.0);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn sampler_label<A: HalApi>(&self, id: id::SamplerId) -> String {
        A::hub(self).samplers.label_for_resource(id)
    }

    pub fn sampler_drop<A: HalApi>(&self, sampler_id: id::SamplerId) {
        profiling::scope!("Sampler::drop");
        log::trace!("Sampler::drop {sampler_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut sampler_guard, _) = hub.samplers.write(&mut token);
            match sampler_guard.get_mut(sampler_id) {
                Ok(sampler) => {
                    sampler.life_guard.ref_count.take();
                    sampler.device_id.value
                }
                Err(InvalidId) => {
                    hub.samplers
                        .unregister_locked(sampler_id, &mut *sampler_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .samplers
            .push(id::Valid(sampler_id));
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

        let mut token = Token::root();
        let hub = A::hub(self);
        let fid = hub.bind_group_layouts.prepare(id_in);

        let error = 'outer: loop {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateBindGroupLayout(fid.id(), desc.clone()));
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

            let mut compatible_layout = None;
            let layout = {
                let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
                if let Some(id) =
                    Device::deduplicate_bind_group_layout(device_id, &entry_map, &*bgl_guard)
                {
                    // If there is an equivalent BGL, just bump the refcount and return it.
                    // This is only applicable if ids are generated in wgpu. In practice:
                    //  - wgpu users take this branch and return the existing
                    //    id without using the indirection layer in BindGroupLayout.
                    //  - Other users like gecko or the replay tool use don't take
                    //    the branch and instead rely on the indirection to use the
                    //    proper bind group layout id.
                    if G::ids_are_generated_in_wgpu() {
                        log::trace!("Device::create_bind_group_layout (duplicate of {id:?})");
                        return (id, None);
                    }

                    compatible_layout = Some(id::Valid(id));
                }

                if let Some(original_id) = compatible_layout {
                    let original = &bgl_guard[original_id];
                    BindGroupLayout {
                        device_id: original.device_id.clone(),
                        inner: crate::binding_model::BglOrDuplicate::Duplicate(original_id),
                        multi_ref_count: crate::MultiRefCount::new(),
                    }
                } else {
                    match device.create_bind_group_layout(device_id, &desc.label, entry_map) {
                        Ok(layout) => layout,
                        Err(e) => break e,
                    }
                }
            };

            let id = fid.assign(layout, &mut token);

            if let Some(dupe) = compatible_layout {
                log::trace!(
                    "Device::create_bind_group_layout (duplicate of {dupe:?}) -> {:?}",
                    id.0
                );
            } else {
                log::trace!("Device::create_bind_group_layout -> {:?}", id.0);
            }

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn bind_group_layout_label<A: HalApi>(&self, id: id::BindGroupLayoutId) -> String {
        A::hub(self).bind_group_layouts.label_for_resource(id)
    }

    pub fn bind_group_layout_drop<A: HalApi>(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("BindGroupLayout::drop");
        log::trace!("BindGroupLayout::drop {:?}", bind_group_layout_id);

        let hub = A::hub(self);
        let mut token = Token::root();
        let device_id = {
            let (mut bind_group_layout_guard, _) = hub.bind_group_layouts.write(&mut token);
            match bind_group_layout_guard.get_mut(bind_group_layout_id) {
                Ok(layout) => layout.device_id.value,
                Err(InvalidId) => {
                    hub.bind_group_layouts
                        .unregister_locked(bind_group_layout_id, &mut *bind_group_layout_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .bind_group_layouts
            .push(id::Valid(bind_group_layout_id));
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
        let mut token = Token::root();
        let fid = hub.pipeline_layouts.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreatePipelineLayout(fid.id(), desc.clone()));
            }

            let layout = {
                let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
                match device.create_pipeline_layout(device_id, desc, &*bgl_guard) {
                    Ok(layout) => layout,
                    Err(e) => break e,
                }
            };

            let id = fid.assign(layout, &mut token);

            log::trace!("Device::create_pipeline_layout -> {:?}", id.0);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn pipeline_layout_label<A: HalApi>(&self, id: id::PipelineLayoutId) -> String {
        A::hub(self).pipeline_layouts.label_for_resource(id)
    }

    pub fn pipeline_layout_drop<A: HalApi>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");
        log::trace!("PipelineLayout::drop {:?}", pipeline_layout_id);

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_id, ref_count) = {
            let (mut pipeline_layout_guard, _) = hub.pipeline_layouts.write(&mut token);
            match pipeline_layout_guard.get_mut(pipeline_layout_id) {
                Ok(layout) => (
                    layout.device_id.value,
                    layout.life_guard.ref_count.take().unwrap(),
                ),
                Err(InvalidId) => {
                    hub.pipeline_layouts
                        .unregister_locked(pipeline_layout_id, &mut *pipeline_layout_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .pipeline_layouts
            .push(Stored {
                value: id::Valid(pipeline_layout_id),
                ref_count,
            });
    }

    pub fn device_create_bind_group<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Input<G, id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("Device::create_bind_group");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.bind_groups.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (bind_group_layout_guard, mut token) = hub.bind_group_layouts.read(&mut token);

        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateBindGroup(fid.id(), desc.clone()));
            }

            let mut bind_group_layout = match bind_group_layout_guard.get(desc.layout) {
                Ok(layout) => layout,
                Err(..) => break binding_model::CreateBindGroupError::InvalidLayout,
            };

            if bind_group_layout.device_id.value.0 != device_id {
                break DeviceError::WrongDevice.into();
            }

            let mut layout_id = id::Valid(desc.layout);
            if let Some(id) = bind_group_layout.as_duplicate() {
                layout_id = id;
                bind_group_layout = &bind_group_layout_guard[id];
            }

            let bind_group = match device.create_bind_group(
                device_id,
                bind_group_layout,
                layout_id,
                desc,
                hub,
                &mut token,
            ) {
                Ok(bind_group) => bind_group,
                Err(e) => break e,
            };
            let ref_count = bind_group.life_guard.add_ref();

            let id = fid.assign(bind_group, &mut token);

            log::trace!("Device::create_bind_group -> {:?}", id.0);

            device
                .trackers
                .lock()
                .bind_groups
                .insert_single(id, ref_count);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn bind_group_label<A: HalApi>(&self, id: id::BindGroupId) -> String {
        A::hub(self).bind_groups.label_for_resource(id)
    }

    pub fn bind_group_drop<A: HalApi>(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("BindGroup::drop");
        log::trace!("BindGroup::drop {:?}", bind_group_id);

        let hub = A::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut bind_group_guard, _) = hub.bind_groups.write(&mut token);
            match bind_group_guard.get_mut(bind_group_id) {
                Ok(bind_group) => {
                    bind_group.life_guard.ref_count.take();
                    bind_group.device_id.value
                }
                Err(InvalidId) => {
                    hub.bind_groups
                        .unregister_locked(bind_group_id, &mut *bind_group_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .bind_groups
            .push(id::Valid(bind_group_id));
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
        let mut token = Token::root();
        let fid = hub.shader_modules.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                let mut trace = trace.lock();
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

            let shader = match device.create_shader_module(device_id, desc, source) {
                Ok(shader) => shader,
                Err(e) => break e,
            };
            let id = fid.assign(shader, &mut token);

            log::trace!("Device::create_shader_module -> {:?}", id.0);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
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
        let mut token = Token::root();
        let fid = hub.shader_modules.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                let mut trace = trace.lock();
                let data = trace.make_binary("spv", unsafe {
                    std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * 4)
                });
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader =
                match unsafe { device.create_shader_module_spirv(device_id, desc, &source) } {
                    Ok(shader) => shader,
                    Err(e) => break e,
                };
            let id = fid.assign(shader, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn shader_module_label<A: HalApi>(&self, id: id::ShaderModuleId) -> String {
        A::hub(self).shader_modules.label_for_resource(id)
    }

    pub fn shader_module_drop<A: HalApi>(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");
        log::trace!("ShaderModule::drop {:?}", shader_module_id);

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (module, _) = hub.shader_modules.unregister(shader_module_id, &mut token);
        if let Some(module) = module {
            let device = &device_guard[module.device_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::DestroyShaderModule(shader_module_id));
            }
            unsafe {
                device.raw.destroy_shader_module(module.raw);
            }
        }
    }

    pub fn device_create_command_encoder<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Input<G, id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.command_buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid,
            };
            if !device.valid {
                break DeviceError::Lost;
            }

            let dev_stored = Stored {
                value: id::Valid(device_id),
                ref_count: device.life_guard.add_ref(),
            };
            let encoder = match device
                .command_allocator
                .lock()
                .acquire_encoder(&device.raw, &device.queue)
            {
                Ok(raw) => raw,
                Err(_) => break DeviceError::OutOfMemory,
            };
            let command_buffer = command::CommandBuffer::new(
                encoder,
                dev_stored,
                device.limits.clone(),
                device.downlevel.clone(),
                device.features,
                #[cfg(feature = "trace")]
                device.trace.is_some(),
                desc.label
                    .to_hal(device.instance_flags)
                    .map(|s| s.to_string()),
            );

            let id = fid.assign(command_buffer, &mut token);

            log::trace!("Device::create_command_encoder -> {:?}", id.0);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn command_buffer_label<A: HalApi>(&self, id: id::CommandBufferId) -> String {
        A::hub(self).command_buffers.label_for_resource(id)
    }

    pub fn command_encoder_drop<A: HalApi>(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");
        log::trace!("CommandEncoder::drop {:?}", command_encoder_id);

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (cmdbuf, _) = hub
            .command_buffers
            .unregister(command_encoder_id, &mut token);
        if let Some(cmdbuf) = cmdbuf {
            let device = &mut device_guard[cmdbuf.device_id.value];
            device.untrack::<G>(hub, &cmdbuf.trackers, &mut token);
            device.destroy_command_buffer(cmdbuf);
        }
    }

    pub fn command_buffer_drop<A: HalApi>(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        log::trace!("CommandBuffer::drop {:?}", command_buffer_id);
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
        log::trace!("Device::device_create_render_bundle_encoder");
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
        let mut token = Token::root();
        let fid = hub.render_bundles.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(bundle_encoder.parent()) {
                Ok(device) => device,
                Err(_) => break command::RenderBundleError::INVALID_DEVICE,
            };
            if !device.valid {
                break command::RenderBundleError::INVALID_DEVICE;
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateRenderBundle {
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

            let render_bundle = match bundle_encoder.finish(desc, device, hub, &mut token) {
                Ok(bundle) => bundle,
                Err(e) => break e,
            };

            log::debug!("Render bundle");
            let ref_count = render_bundle.life_guard.add_ref();
            let id = fid.assign(render_bundle, &mut token);

            device.trackers.lock().bundles.insert_single(id, ref_count);

            log::trace!("RenderBundleEncoder::finish -> {:?}", id.0);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn render_bundle_label<A: HalApi>(&self, id: id::RenderBundleId) -> String {
        A::hub(self).render_bundles.label_for_resource(id)
    }

    pub fn render_bundle_drop<A: HalApi>(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("RenderBundle::drop");
        log::trace!("RenderBundle::drop {:?}", render_bundle_id);
        let hub = A::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device_id = {
            let (mut bundle_guard, _) = hub.render_bundles.write(&mut token);
            match bundle_guard.get_mut(render_bundle_id) {
                Ok(bundle) => {
                    bundle.life_guard.ref_count.take();
                    bundle.device_id.value
                }
                Err(InvalidId) => {
                    hub.render_bundles
                        .unregister_locked(render_bundle_id, &mut *bundle_guard);
                    return;
                }
            }
        };

        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .render_bundles
            .push(id::Valid(render_bundle_id));
    }

    pub fn device_create_query_set<A: HalApi>(
        &self,
        device_id: DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: Input<G, id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("Device::create_query_set");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.query_sets.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateQuerySet {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let query_set = match device.create_query_set(device_id, desc) {
                Ok(query_set) => query_set,
                Err(err) => break err,
            };

            let ref_count = query_set.life_guard.add_ref();
            let id = fid.assign(query_set, &mut token);

            device
                .trackers
                .lock()
                .query_sets
                .insert_single(id, ref_count);

            return (id.0, None);
        };

        let id = fid.assign_error("", &mut token);

        log::trace!("Device::create_query_set -> {:?}", id);

        (id, Some(error))
    }

    pub fn query_set_drop<A: HalApi>(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");
        log::trace!("QuerySet::drop {query_set_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut query_set_guard, _) = hub.query_sets.write(&mut token);
            let query_set = query_set_guard.get_mut(query_set_id).unwrap();
            query_set.life_guard.ref_count.take();
            query_set.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace
                .lock()
                .add(trace::Action::DestroyQuerySet(query_set_id));
        }

        device
            .lock_life(&mut token)
            .suspected_resources
            .query_sets
            .push(id::Valid(query_set_id));
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
        let mut token = Token::root();

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));
        let implicit_error_context = implicit_context.clone();

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            let adapter = &adapter_guard[device.adapter_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let pipeline = match device.create_render_pipeline(
                device_id,
                adapter,
                desc,
                implicit_context,
                hub,
                &mut token,
            ) {
                Ok(pair) => pair,
                Err(e) => break e,
            };
            let ref_count = pipeline.life_guard.add_ref();

            let id = fid.assign(pipeline, &mut token);
            log::trace!("Device::create_render_pipeline -> {:?}", id.0);

            device
                .trackers
                .lock()
                .render_pipelines
                .insert_single(id, ref_count);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layout. We have to remove any existing entries first.
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(&mut token);
        let (mut bgl_guard, _token) = hub.bind_group_layouts.write(&mut token);
        if let Some(ref ids) = implicit_error_context {
            if pipeline_layout_guard.contains(ids.root_id) {
                pipeline_layout_guard.remove(ids.root_id);
            }
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_FAILURE);
            for &bgl_id in ids.group_ids.iter() {
                if bgl_guard.contains(bgl_id) {
                    bgl_guard.remove(bgl_id);
                }
                bgl_guard.insert_error(bgl_id, IMPLICIT_FAILURE);
            }
        }

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
        let mut token = Token::root();

        let error = loop {
            let device_id;
            let id;

            {
                let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);

                let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
                let (_, mut token) = hub.bind_groups.read(&mut token);
                let (pipeline_guard, _) = hub.render_pipelines.read(&mut token);

                let pipeline = match pipeline_guard.get(pipeline_id) {
                    Ok(pipeline) => pipeline,
                    Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
                };
                id = match pipeline_layout_guard[pipeline.layout_id.value]
                    .bind_group_layout_ids
                    .get(index as usize)
                {
                    Some(id) => *id,
                    None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
                };

                let layout = &bgl_guard[id];
                layout.multi_ref_count.inc();

                if G::ids_are_generated_in_wgpu() {
                    return (id.0, None);
                }

                device_id = layout.device_id.clone();
            }

            // The ID is provided externally, so we must create a new bind group layout
            // with the given ID as a duplicate of the existing one.
            let new_layout = BindGroupLayout {
                device_id,
                inner: crate::binding_model::BglOrDuplicate::<A>::Duplicate(id),
                multi_ref_count: crate::MultiRefCount::new(),
            };

            let fid = hub.bind_group_layouts.prepare(id_in);
            let id = fid.assign(new_layout, &mut token);

            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>", &mut token);
        (id, Some(error))
    }

    pub fn render_pipeline_label<A: HalApi>(&self, id: id::RenderPipelineId) -> String {
        A::hub(self).render_pipelines.label_for_resource(id)
    }

    pub fn render_pipeline_drop<A: HalApi>(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("RenderPipeline::drop");
        log::trace!("RenderPipeline::drop {:?}", render_pipeline_id);
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.render_pipelines.write(&mut token);
            match pipeline_guard.get_mut(render_pipeline_id) {
                Ok(pipeline) => {
                    pipeline.life_guard.ref_count.take();
                    (pipeline.device_id.value, pipeline.layout_id.clone())
                }
                Err(InvalidId) => {
                    hub.render_pipelines
                        .unregister_locked(render_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .render_pipelines
            .push(id::Valid(render_pipeline_id));
        life_lock
            .suspected_resources
            .pipeline_layouts
            .push(layout_id);
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
        let mut token = Token::root();

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));
        let implicit_error_context = implicit_context.clone();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.valid {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateComputePipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }
            let pipeline = match device.create_compute_pipeline(
                device_id,
                desc,
                implicit_context,
                hub,
                &mut token,
            ) {
                Ok(pair) => pair,
                Err(e) => break e,
            };
            let ref_count = pipeline.life_guard.add_ref();

            let id = fid.assign(pipeline, &mut token);
            log::trace!("Device::create_compute_pipeline -> {:?}", id.0);

            device
                .trackers
                .lock()
                .compute_pipelines
                .insert_single(id, ref_count);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);

        // We also need to assign errors to the implicit pipeline layout and the
        // implicit bind group layout. We have to remove any existing entries first.
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(&mut token);
        let (mut bgl_guard, _token) = hub.bind_group_layouts.write(&mut token);
        if let Some(ref ids) = implicit_error_context {
            if pipeline_layout_guard.contains(ids.root_id) {
                pipeline_layout_guard.remove(ids.root_id);
            }
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_FAILURE);
            for &bgl_id in ids.group_ids.iter() {
                if bgl_guard.contains(bgl_id) {
                    bgl_guard.remove(bgl_id);
                }
                bgl_guard.insert_error(bgl_id, IMPLICIT_FAILURE);
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
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);
        let mut token = Token::root();

        let error = loop {
            let device_id;
            let id;

            {
                let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);

                let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
                let (_, mut token) = hub.bind_groups.read(&mut token);
                let (pipeline_guard, _) = hub.compute_pipelines.read(&mut token);

                let pipeline = match pipeline_guard.get(pipeline_id) {
                    Ok(pipeline) => pipeline,
                    Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
                };
                id = match pipeline_layout_guard[pipeline.layout_id.value]
                    .bind_group_layout_ids
                    .get(index as usize)
                {
                    Some(id) => *id,
                    None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
                };

                let layout = &bgl_guard[id];
                layout.multi_ref_count.inc();

                if G::ids_are_generated_in_wgpu() {
                    return (id.0, None);
                }

                device_id = layout.device_id.clone();
            }

            // The ID is provided externally, so we must create a new bind group layout
            // with the given ID as a duplicate of the existing one.
            let new_layout = BindGroupLayout {
                device_id,
                inner: crate::binding_model::BglOrDuplicate::<A>::Duplicate(id),
                multi_ref_count: crate::MultiRefCount::new(),
            };

            let fid = hub.bind_group_layouts.prepare(id_in);
            let id = fid.assign(new_layout, &mut token);

            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>", &mut token);
        (id, Some(error))
    }

    pub fn compute_pipeline_label<A: HalApi>(&self, id: id::ComputePipelineId) -> String {
        A::hub(self).compute_pipelines.label_for_resource(id)
    }

    pub fn compute_pipeline_drop<A: HalApi>(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("ComputePipeline::drop");
        log::trace!("ComputePipeline::drop {:?}", compute_pipeline_id);
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.compute_pipelines.write(&mut token);
            match pipeline_guard.get_mut(compute_pipeline_id) {
                Ok(pipeline) => {
                    pipeline.life_guard.ref_count.take();
                    (pipeline.device_id.value, pipeline.layout_id.clone())
                }
                Err(InvalidId) => {
                    hub.compute_pipelines
                        .unregister_locked(compute_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .compute_pipelines
            .push(id::Valid(compute_pipeline_id));
        life_lock
            .suspected_resources
            .pipeline_layouts
            .push(layout_id);
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

        let error = 'outer: loop {
            // User callbacks must not be called while we are holding locks.
            let user_callbacks;
            {
                let hub = A::hub(self);
                let mut token = Token::root();

                let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
                let (adapter_guard, mut token) = hub.adapters.read(&mut token);
                let (device_guard, mut token) = hub.devices.read(&mut token);

                let device = match device_guard.get(device_id) {
                    Ok(device) => device,
                    Err(_) => break DeviceError::Invalid.into(),
                };
                if !device.valid {
                    break DeviceError::Lost.into();
                }

                #[cfg(feature = "trace")]
                if let Some(ref trace) = device.trace {
                    trace
                        .lock()
                        .add(trace::Action::ConfigureSurface(surface_id, config.clone()));
                }

                let surface = match surface_guard.get_mut(surface_id) {
                    Ok(surface) => surface,
                    Err(_) => break E::InvalidSurface,
                };

                let caps = unsafe {
                    let suf = A::get_surface(surface);
                    let adapter = &adapter_guard[device.adapter_id.value];
                    match adapter.raw.adapter.surface_capabilities(&suf.unwrap().raw) {
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
                            available: caps.formats,
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

                // Wait for all work to finish before configuring the surface.
                match device.maintain(hub, wgt::Maintain::Wait, &mut token) {
                    Ok((closures, _)) => {
                        user_callbacks = closures;
                    }
                    Err(e) => {
                        break e.into();
                    }
                }

                // All textures must be destroyed before the surface can be re-configured.
                if let Some(present) = surface.presentation.take() {
                    if present.acquired_texture.is_some() {
                        break E::PreviousOutputExists;
                    }
                }

                // TODO: Texture views may still be alive that point to the texture.
                // this will allow the user to render to the surface texture, long after
                // it has been removed.
                //
                // https://github.com/gfx-rs/wgpu/issues/4105

                match unsafe {
                    A::get_surface_mut(surface)
                        .unwrap()
                        .raw
                        .configure(&device.raw, &hal_config)
                } {
                    Ok(()) => (),
                    Err(error) => {
                        break match error {
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

                surface.presentation = Some(present::Presentation {
                    device_id: Stored {
                        value: id::Valid(device_id),
                        ref_count: device.life_guard.add_ref(),
                    },
                    config: config.clone(),
                    num_frames,
                    acquired_texture: None,
                });
            }

            user_callbacks.fire();

            return None;
        };

        Some(error)
    }

    #[cfg(feature = "replay")]
    /// Only triange suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<A: HalApi>(&self, device_id: DeviceId) -> Result<(), InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        if !device.valid {
            return Err(InvalidDevice);
        }
        device.lock_life(&mut token).triage_suspected(
            hub,
            &device.trackers,
            #[cfg(feature = "trace")]
            None,
            &mut token,
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
        log::trace!("Device::poll");

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
            let mut token = Token::root();
            let (device_guard, mut token) = hub.devices.read(&mut token);
            device_guard
                .get(device_id)
                .map_err(|_| DeviceError::Invalid)?
                .maintain(hub, maintain, &mut token)?
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
    fn poll_devices<A: HalApi>(
        &self,
        force_wait: bool,
        closures: &mut UserClosures,
    ) -> Result<bool, WaitIdleError> {
        profiling::scope!("poll_devices");

        let hub = A::hub(self);
        let mut devices_to_drop = vec![];
        let mut all_queue_empty = true;
        {
            let mut token = Token::root();
            let (device_guard, mut token) = hub.devices.read(&mut token);

            for (id, device) in device_guard.iter(A::VARIANT) {
                let maintain = if force_wait {
                    wgt::Maintain::Wait
                } else {
                    wgt::Maintain::Poll
                };
                let (cbs, queue_empty) = device.maintain(hub, maintain, &mut token)?;
                all_queue_empty = all_queue_empty && queue_empty;

                // If the device's own `RefCount` clone is the only one left, and
                // its submission queue is empty, then it can be freed.
                if queue_empty && device.ref_count.load() == 1 {
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
            all_queue_empty = self.poll_devices::<hal::api::Vulkan>(force_wait, &mut closures)?
                && all_queue_empty;
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            all_queue_empty =
                self.poll_devices::<hal::api::Metal>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(all(feature = "dx12", windows))]
        {
            all_queue_empty =
                self.poll_devices::<hal::api::Dx12>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(all(feature = "dx11", windows))]
        {
            all_queue_empty =
                self.poll_devices::<hal::api::Dx11>(force_wait, &mut closures)? && all_queue_empty;
        }
        #[cfg(feature = "gles")]
        {
            all_queue_empty =
                self.poll_devices::<hal::api::Gles>(force_wait, &mut closures)? && all_queue_empty;
        }

        closures.fire();

        Ok(all_queue_empty)
    }

    pub fn device_label<A: HalApi>(&self, id: DeviceId) -> String {
        A::hub(self).devices.label_for_resource(id)
    }

    pub fn device_start_capture<A: HalApi>(&self, id: DeviceId) {
        log::trace!("Device::start_capture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        if let Ok(device) = device_guard.get(id) {
            if !device.valid {
                return;
            }
            unsafe { device.raw.start_capture() };
        }
    }

    pub fn device_stop_capture<A: HalApi>(&self, id: DeviceId) {
        log::trace!("Device::stop_capture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        if let Ok(device) = device_guard.get(id) {
            if !device.valid {
                return;
            }
            unsafe { device.raw.stop_capture() };
        }
    }

    pub fn device_drop<A: HalApi>(&self, device_id: DeviceId) {
        profiling::scope!("Device::drop");
        log::trace!("Device::drop {device_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        // For now, just drop the `RefCount` in `device.life_guard`, which
        // stands for the user's reference to the device. We'll take care of
        // cleaning up the device when we're polled, once its queue submissions
        // have completed and it is no longer needed by other resources.
        let (mut device_guard, _) = hub.devices.write(&mut token);
        if let Ok(device) = device_guard.get_mut(device_id) {
            device.life_guard.ref_count.take().unwrap();
        }
    }

    pub fn device_set_device_lost_closure<A: HalApi>(
        &self,
        device_id: DeviceId,
        device_lost_closure: DeviceLostClosure,
    ) {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        if let Ok(device) = device_guard.get_mut(device_id) {
            let mut life_tracker = device.lock_life(&mut token);
            life_tracker.device_lost_closure = Some(device_lost_closure);
        }
    }

    pub fn device_destroy<A: HalApi>(&self, device_id: DeviceId) {
        log::trace!("Device::destroy {device_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, _) = hub.devices.write(&mut token);
        if let Ok(device) = device_guard.get_mut(device_id) {
            // Follow the steps at
            // https://gpuweb.github.io/gpuweb/#dom-gpudevice-destroy.

            // The last part of destroy is to lose the device. The spec says
            // delay that until all "currently-enqueued operations on any
            // queue on this device are completed." This is accomplished by
            // setting valid to false, and then relying upon maintain to
            // check for empty queues and a DeviceLostClosure. At that time,
            // the DeviceLostClosure will be called with "destroyed" as the
            // reason.
            device.valid = false;
        }
    }

    pub fn device_mark_lost<A: HalApi>(&self, device_id: DeviceId, message: &str) {
        log::trace!("Device::mark_lost {device_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        if let Ok(device) = device_guard.get_mut(device_id) {
            device.lose(&mut token, message);
        }
    }

    /// Exit the unreferenced, inactive device `device_id`.
    fn exit_device<A: HalApi>(&self, device_id: DeviceId) {
        let hub = A::hub(self);
        let mut token = Token::root();
        let mut free_adapter_id = None;
        {
            let (device, mut _token) = hub.devices.unregister(device_id, &mut token);
            if let Some(mut device) = device {
                // The things `Device::prepare_to_die` takes care are mostly
                // unnecessary here. We know our queue is empty, so we don't
                // need to wait for submissions or triage them. We know we were
                // just polled, so `life_tracker.free_resources` is empty.
                debug_assert!(device.lock_life(&mut _token).queue_empty());
                device.pending_writes.deactivate();

                // Adapter is only referenced by the device and itself.
                // This isn't a robust way to destroy them, we should find a better one.
                if device.adapter_id.ref_count.load() == 1 {
                    free_adapter_id = Some(device.adapter_id.value.0);
                }

                device.dispose();
            }
        }

        // Free the adapter now that we've dropped the `Device` token.
        if let Some(free_adapter_id) = free_adapter_id {
            let _ = hub.adapters.unregister(free_adapter_id, &mut token);
        }
    }

    pub fn buffer_map_async<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        range: Range<BufferAddress>,
        op: BufferMapOperation,
    ) -> BufferAccessResult {
        log::trace!("Buffer::map_async {buffer_id:?}");

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
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsages::MAP_READ, hal::BufferUses::MAP_READ),
            HostMap::Write => (wgt::BufferUsages::MAP_WRITE, hal::BufferUses::MAP_WRITE),
        };

        if range.start % wgt::MAP_ALIGNMENT != 0 || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err((op, BufferAccessError::UnalignedRange));
        }

        let (device_id, ref_count) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = buffer_guard
                .get_mut(buffer_id)
                .map_err(|_| BufferAccessError::Invalid);

            let buffer = match buffer {
                Ok(b) => b,
                Err(e) => {
                    return Err((op, e));
                }
            };

            let device = &device_guard[buffer.device_id.value];
            if !device.valid {
                return Err((op, DeviceError::Lost.into()));
            }

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

            buffer.map_state = match buffer.map_state {
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
                        _parent_ref_count: buffer.life_guard.add_ref(),
                    })
                }
            };
            log::debug!("Buffer {:?} map state -> Waiting", buffer_id);

            let ret = (buffer.device_id.value, buffer.life_guard.add_ref());

            let mut trackers = device.trackers.lock();
            trackers
                .buffers
                .set_single(&*buffer_guard, buffer_id, internal_use);
            trackers.buffers.drain();

            ret
        };

        let device = &device_guard[device_id];
        // Validity of device was confirmed in the code block that set device_id.
        device
            .lock_life(&mut token)
            .map(id::Valid(buffer_id), ref_count);

        Ok(())
    }

    pub fn buffer_get_mapped_range<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(*mut u8, u64), BufferAccessError> {
        profiling::scope!("Buffer::get_mapped_range");
        log::trace!("Buffer::get_mapped_range {buffer_id:?}");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        let buffer = buffer_guard
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

        match buffer.map_state {
            resource::BufferMapState::Init { ptr, .. } => {
                // offset (u64) can not be < 0, so no need to validate the lower bound
                if offset + range_size > buffer.size {
                    return Err(BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: buffer.size,
                    });
                }
                unsafe { Ok((ptr.as_ptr().offset(offset as isize), range_size)) }
            }
            resource::BufferMapState::Active { ptr, ref range, .. } => {
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
        buffer: &mut Buffer<A>,
        device: &mut Device<A>,
    ) -> Result<Option<BufferMapPendingClosure>, BufferAccessError> {
        log::debug!("Buffer {:?} map state -> Idle", buffer_id);
        match mem::replace(&mut buffer.map_state, resource::BufferMapState::Idle) {
            resource::BufferMapState::Init {
                ptr,
                stage_buffer,
                needs_flush,
            } => {
                #[cfg(feature = "trace")]
                if let Some(ref trace) = device.trace {
                    let mut trace = trace.lock();
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
                        device
                            .raw
                            .flush_mapped_ranges(&stage_buffer, iter::once(0..buffer.size));
                    }
                }

                let raw_buf = buffer.raw.as_ref().ok_or(BufferAccessError::Destroyed)?;

                buffer.life_guard.use_at(device.active_submission_index + 1);
                let region = wgt::BufferSize::new(buffer.size).map(|size| hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                });
                let transition_src = hal::BufferBarrier {
                    buffer: &stage_buffer,
                    usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                };
                let transition_dst = hal::BufferBarrier {
                    buffer: raw_buf,
                    usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                };
                let encoder = device.pending_writes.activate();
                unsafe {
                    encoder.transition_buffers(
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    if buffer.size > 0 {
                        encoder.copy_buffer_to_buffer(&stage_buffer, raw_buf, region.into_iter());
                    }
                }
                device
                    .pending_writes
                    .consume_temp(queue::TempResource::Buffer(stage_buffer));
                device.pending_writes.dst_buffers.insert(buffer_id);
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
                    if let Some(ref trace) = device.trace {
                        let mut trace = trace.lock();
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
                        .unmap_buffer(buffer.raw.as_ref().unwrap())
                        .map_err(DeviceError::from)?
                };
            }
        }
        Ok(None)
    }

    pub fn buffer_unmap<A: HalApi>(&self, buffer_id: id::BufferId) -> BufferAccessResult {
        profiling::scope!("unmap", "Buffer");
        log::trace!("Buffer::unmap {buffer_id:?}");

        let closure;
        {
            // Restrict the locks to this scope.
            let hub = A::hub(self);
            let mut token = Token::root();

            let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = buffer_guard
                .get_mut(buffer_id)
                .map_err(|_| BufferAccessError::Invalid)?;
            let device = &mut device_guard[buffer.device_id.value];
            if !device.valid {
                return Err(DeviceError::Lost.into());
            }

            closure = self.buffer_unmap_inner(buffer_id, buffer, device)
        }

        // Note: outside the scope where locks are held when calling the callback
        if let Some((operation, status)) = closure? {
            operation.callback.call(status);
        }
        Ok(())
    }
}
