#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    device::{queue::TempResource, Device, DeviceError},
    global::Global,
    hal_api::HalApi,
    id::{self, BlasId, TlasId},
    ray_tracing::{get_raw_tlas_instance_size, CreateBlasError, CreateTlasError},
    resource, LabelHelpers,
};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

use crate::resource::{ResourceInfo, StagingBuffer};
use hal::{AccelerationStructureTriangleIndices, Device as _};

impl<A: HalApi> Device<A> {
    fn create_blas(
        self: &Arc<Self>,
        self_id: id::DeviceId,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<resource::Blas<A>, CreateBlasError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { desc } => {
                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<A>>::with_capacity(desc.len());
                for x in desc {
                    if x.index_count.is_some() != x.index_format.is_some() {
                        return Err(CreateBlasError::MissingIndexData);
                    }
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
                    self.raw().get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::Triangles(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
        };

        let raw = unsafe {
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: blas_desc.label.borrow_option(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let handle = unsafe { self.raw().get_acceleration_structure_device_address(&raw) };

        Ok(resource::Blas {
            raw: Some(raw),
            device: self.clone(),
            info: ResourceInfo::new(
                blas_desc
                    .label
                    .to_hal(self.instance_flags)
                    .unwrap_or("<BindGroupLayoyt>"),
            ),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
            handle,
            built_index: RwLock::new(None),
        })
    }

    fn create_tlas(
        self: &Arc<Self>,
        self_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
    ) -> Result<resource::Tlas<A>, CreateTlasError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        let size_info = unsafe {
            self.raw().get_acceleration_structure_build_sizes(
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
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: desc.label.borrow_option(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::TopLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let instance_buffer_size =
            get_raw_tlas_instance_size::<A>() * std::cmp::max(desc.max_instances, 1) as usize;
        let instance_buffer = unsafe {
            self.raw().create_buffer(&hal::BufferDescriptor {
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
            device: self.clone(),
            info: ResourceInfo::new(
                desc.label
                    .to_hal(self.instance_flags)
                    .unwrap_or("<BindGroupLayoyt>"),
            ),
            size_info,
            flags: desc.flags,
            update_mode: desc.update_mode,
            built_index: RwLock::new(None),
            dependencies: RwLock::new(Vec::new()),
            instance_buffer: RwLock::new(Some(instance_buffer)),
            max_instance_count: desc.max_instances,
        })
    }
}

impl Global {
    pub fn device_create_blas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
        id_in: Option<BlasId>,
    ) -> (BlasId, Option<u64>, Option<CreateBlasError>) {
        profiling::scope!("Device::create_blas");

        let hub = A::hub(self);
        let fid = hub.blas_s.prepare(id_in);

        let device_guard = hub.devices.read();
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            if !device.is_valid() {
                break DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(trace) = device.trace.lock().as_mut() {
                trace.add(trace::Action::CreateBlas {
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

            let (id, resource) = fid.assign(blas);
            log::info!("Created blas {:?} with {:?}", id, desc);

            device.trackers.lock().blas_s.insert_single(id, resource);

            return (id, Some(handle), None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, None, Some(error))
    }

    pub fn device_create_tlas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
        id_in: Option<TlasId>,
    ) -> (TlasId, Option<CreateTlasError>) {
        profiling::scope!("Device::create_tlas");

        let hub = A::hub(self);
        let fid = hub.tlas_s.prepare(id_in);

        let device_guard = hub.devices.read();
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(trace) = device.trace.lock().as_mut() {
                trace.add(trace::Action::CreateTlas {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let tlas = match device.create_tlas(device_id, desc) {
                Ok(tlas) => tlas,
                Err(e) => break e,
            };

            let id = fid.assign(tlas);
            log::info!("Created tlas {:?} with {:?}", id.0, desc);

            device.trackers.lock().tlas_s.insert_single(id.0, id.1);

            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default());
        (id, Some(error))
    }

    pub fn blas_destroy<A: HalApi>(&self, blas_id: BlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Blas::destroy");

        let hub = A::hub(self);

        let device_guard = hub.devices.write();

        log::info!("Blas {:?} is destroyed", blas_id);
        let blas_guard = hub.blas_s.write();
        let blas = blas_guard
            .get(blas_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = device_guard.get(blas.device.info.id()).unwrap();

        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeBlas(blas_id));
        }

        let temp = TempResource::Blas(blas.clone());
        {
            let last_submit_index = blas.info.submission_index();
            drop(blas_guard);
            device
                .lock_life()
                .schedule_resource_destruction(temp, last_submit_index);
        }

        Ok(())
    }

    pub fn blas_drop<A: HalApi>(&self, blas_id: BlasId, wait: bool) {
        profiling::scope!("Blas::drop");
        log::debug!("blas {:?} is dropped", blas_id);

        let hub = A::hub(self);

        if let Some(blas) = hub.blas_s.unregister(blas_id) {
            let last_submit_index = blas.info.submission_index();

            blas.device
                .lock_life()
                .suspected_resources
                .blas_s
                .insert(blas_id, blas.clone());

            if wait {
                match blas.device.wait_for_submit(last_submit_index) {
                    Ok(()) => (),
                    Err(e) => log::error!("Failed to wait for blas {:?}: {:?}", blas_id, e),
                }
            }
        }
    }

    pub fn tlas_destroy<A: HalApi>(&self, tlas_id: TlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Tlas::destroy");

        let hub = A::hub(self);

        log::info!("Tlas {:?} is destroyed", tlas_id);
        let tlas_guard = hub.tlas_s.write();
        let tlas = tlas_guard
            .get(tlas_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut tlas.device.clone();

        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeTlas(tlas_id));
        }

        let temp = TempResource::Tlas(tlas.clone());

        let raw_instance_buffer = tlas.instance_buffer.write().take();
        let temp_instance_buffer = match raw_instance_buffer {
            None => None,
            Some(e) => {
                let size = get_raw_tlas_instance_size::<A>() as u64
                    * std::cmp::max(tlas.max_instance_count, 1) as u64;
                let mapping = unsafe {
                    device
                        .raw()
                        .map_buffer(&e, 0..size)
                        .map_err(|_| resource::DestroyError::Invalid)?
                };
                Some(TempResource::StagingBuffer(Arc::new(StagingBuffer {
                    raw: Mutex::new(Some(e)),
                    device: device.clone(),
                    size,
                    info: ResourceInfo::new("Raytracing scratch buffer"),
                    is_coherent: mapping.is_coherent,
                })))
            }
        };
        {
            let last_submit_index = tlas.info.submission_index();
            drop(tlas_guard);
            let guard = &mut device.lock_life();

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

        if let Some(tlas) = hub.tlas_s.unregister(tlas_id) {
            let last_submit_index = tlas.info.submission_index();

            tlas.device
                .lock_life()
                .suspected_resources
                .tlas_s
                .insert(tlas_id, tlas.clone());

            if wait {
                match tlas.device.wait_for_submit(last_submit_index) {
                    Ok(()) => (),
                    Err(e) => log::error!("Failed to wait for blas {:?}: {:?}", tlas_id, e),
                }
            }
        }
    }
}
