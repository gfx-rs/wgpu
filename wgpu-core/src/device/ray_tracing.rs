#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    device::{queue::TempResource, Device, DeviceError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{self, BlasId, TlasId},
    ray_tracing::{getRawTlasInstanceSize, CreateBlasError, CreateTlasError},
    resource, LabelHelpers, LifeGuard, Stored,
};

use hal::{AccelerationStructureTriangleIndices, Device as _};
use parking_lot::Mutex;

impl<A: HalApi> Device<A> {
    // TODO:
    // validation
    // comments/documentation
    fn create_blas(
        &self,
        self_id: id::DeviceId,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<resource::Blas<A>, CreateBlasError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        let size_info = match &sizes {
            &wgt::BlasGeometrySizeDescriptors::Triangles { ref desc } => {
                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<A>>::with_capacity(desc.len());
                for x in desc {
                    if x.index_count.is_some() != x.index_format.is_some() {
                        return Err(CreateBlasError::Unimplemented);
                    }
                    // TODO more validation
                    let indices =
                        x.index_count
                            .map(|count| AccelerationStructureTriangleIndices::<A> {
                                format: x.index_format.unwrap(),
                                buffer: None,
                                offset: 0,
                                count,
                            });
                    entries.push(hal::AccelerationStructureTriangles::<A> {
                        vertex_buffer: None,
                        vertex_format: x.vertex_format,
                        first_vertex: 0,
                        vertex_count: x.vertex_count,
                        vertex_stride: 0,
                        indices,
                        transform: None,
                        flags: x.flags,
                    });
                }
                unsafe {
                    self.raw.get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::Triangles(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
        };

        let raw = unsafe {
            self.raw
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: blas_desc.label.borrow_option(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let handle = unsafe { self.raw.get_acceleration_structure_device_address(&raw) };

        Ok(resource::Blas {
            raw: Some(raw),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(blas_desc.label.borrow_or_default()),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
            handle,
            built_index: None,
        })
    }

    // TODO:
    // validation
    // comments/documentation
    fn create_tlas(
        &self,
        self_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
    ) -> Result<resource::Tlas<A>, CreateTlasError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        // TODO validate max_instances
        let size_info = unsafe {
            self.raw.get_acceleration_structure_build_sizes(
                &hal::GetAccelerationStructureBuildSizesDescriptor {
                    entries: &hal::AccelerationStructureEntries::Instances(
                        hal::AccelerationStructureInstances {
                            buffer: None,
                            offset: 0,
                            count: desc.max_instances,
                        },
                    ),
                    flags: desc.flags,
                },
            )
        };

        let raw = unsafe {
            self.raw
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: desc.label.borrow_option(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::TopLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let instance_buffer_size = getRawTlasInstanceSize::<A>();
        let instance_buffer = unsafe {
            self.raw.create_buffer(&hal::BufferDescriptor {
                label: Some("(wgpu-core) instances_buffer"),
                size: instance_buffer_size as u64,
                usage: hal::BufferUses::COPY_DST
                    | hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                memory_flags: hal::MemoryFlags::PREFER_COHERENT,
            })
        }
        .map_err(DeviceError::from)?;

        Ok(resource::Tlas {
            raw: Some(raw),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            size_info,
            flags: desc.flags,
            update_mode: desc.update_mode,
            built_index: None,
            dependencies: Vec::new(),
            instance_buffer: Some(instance_buffer),
        })
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    // TODO:
    // tracing
    // comments/documentation
    pub fn device_create_blas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
        id_in: Input<G, BlasId>,
    ) -> (BlasId, Option<u64>, Option<CreateBlasError>) {
        profiling::scope!("Device::create_blas");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.blas_s.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateBlas {
                    id: fid.id(),
                    desc: desc.clone(),
                    sizes: sizes.clone(),
                });
            }

            let blas = match device.create_blas(device_id, desc, sizes) {
                Ok(blas) => blas,
                Err(e) => break e,
            };
            let handle = blas.handle;

            let ref_count = blas.life_guard.add_ref();

            let id = fid.assign(blas, &mut token);
            log::info!("Created blas {:?} with {:?}", id, desc);

            device.trackers.lock().blas_s.insert_single(id, ref_count);

            return (id.0, Some(handle), None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, None, Some(error))
    }

    // TODO:
    // tracing
    // comments/documentation
    pub fn device_create_tlas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
        id_in: Input<G, TlasId>,
    ) -> (TlasId, Option<CreateTlasError>) {
        profiling::scope!("Device::create_tlas");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.tlas_s.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateTlas {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let tlas = match device.create_tlas(device_id, desc) {
                Ok(tlas) => tlas,
                Err(e) => break e,
            };
            let ref_count = tlas.life_guard.add_ref();

            let id = fid.assign(tlas, &mut token);
            log::info!("Created blas {:?} with {:?}", id, desc);

            device.trackers.lock().tlas_s.insert_single(id, ref_count);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn blas_destroy<A: HalApi>(&self, blas_id: BlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Blas::destroy");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        log::info!("Blas {:?} is destroyed", blas_id);
        let (mut blas_guard, _) = hub.blas_s.write(&mut token);
        let blas = blas_guard
            .get_mut(blas_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut device_guard[blas.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(trace::Action::FreeBlas(blas_id));
        }

        let raw = blas
            .raw
            .take()
            .ok_or(resource::DestroyError::AlreadyDestroyed)?;
        let temp = TempResource::AccelerationStructure(raw);
        {
            let last_submit_index = blas.life_guard.life_count();
            drop(blas_guard);
            device
                .lock_life(&mut token)
                .schedule_resource_destruction(temp, last_submit_index);
        }

        Ok(())
    }

    pub fn blas_drop<A: HalApi>(&self, blas_id: BlasId, wait: bool) {
        profiling::scope!("Blas::drop");
        log::debug!("blas {:?} is dropped", blas_id);

        let hub = A::hub(self);
        let mut token = Token::root();

        let (ref_count, last_submit_index, device_id) = {
            let (mut blas_guard, _) = hub.blas_s.write(&mut token);
            match blas_guard.get_mut(blas_id) {
                Ok(blas) => {
                    let ref_count = blas.life_guard.ref_count.take().unwrap();
                    let last_submit_index = blas.life_guard.life_count();
                    (ref_count, last_submit_index, blas.device_id.value)
                }
                Err(crate::hub::InvalidId) => {
                    hub.blas_s.unregister_locked(blas_id, &mut *blas_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            drop(ref_count);
            life_lock
                .suspected_resources
                .blas_s
                .push(id::Valid(blas_id));
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for blas {:?}: {:?}", blas_id, e),
            }
        }
    }

    pub fn tlas_destroy<A: HalApi>(&self, tlas_id: TlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Tlas::destroy");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        log::info!("Tlas {:?} is destroyed", tlas_id);
        let (mut tlas_guard, _) = hub.tlas_s.write(&mut token);
        let tlas = tlas_guard
            .get_mut(tlas_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut device_guard[tlas.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(trace::Action::FreeTlas(tlas_id));
        }

        let raw = tlas
            .raw
            .take()
            .ok_or(resource::DestroyError::AlreadyDestroyed)?;

        let temp = TempResource::AccelerationStructure(raw);

        let raw_instance_buffer = tlas.instance_buffer.take();
        let temp_instance_buffer = raw_instance_buffer.map(|e| TempResource::Buffer(e));
        {
            let last_submit_index = tlas.life_guard.life_count();
            drop(tlas_guard);
            let guard = &mut device.lock_life(&mut token);

            guard.schedule_resource_destruction(temp, last_submit_index);
            if let Some(temp_instance_buffer) = temp_instance_buffer {
                guard.schedule_resource_destruction(temp_instance_buffer, last_submit_index);
            }
        }

        Ok(())
    }

    pub fn tlas_drop<A: HalApi>(&self, tlas_id: TlasId, wait: bool) {
        profiling::scope!("Tlas::drop");
        log::debug!("tlas {:?} is dropped", tlas_id);

        let hub = A::hub(self);
        let mut token = Token::root();

        let (ref_count, last_submit_index, device_id) = {
            let (mut tlas_guard, _) = hub.tlas_s.write(&mut token);
            match tlas_guard.get_mut(tlas_id) {
                Ok(tlas) => {
                    let ref_count = tlas.life_guard.ref_count.take().unwrap();
                    let last_submit_index = tlas.life_guard.life_count();
                    (ref_count, last_submit_index, tlas.device_id.value)
                }
                Err(crate::hub::InvalidId) => {
                    hub.tlas_s.unregister_locked(tlas_id, &mut *tlas_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            drop(ref_count);
            life_lock
                .suspected_resources
                .tlas_s
                .push(id::Valid(tlas_id));
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for tlas {:?}: {:?}", tlas_id, e),
            }
        }
    }
}
