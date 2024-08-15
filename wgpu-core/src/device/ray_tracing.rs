use std::mem::ManuallyDrop;
use std::sync::Arc;

use hal::AccelerationStructureTriangleIndices;

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::lock::rank;
use crate::resource::TrackingData;
use crate::{
    device::{queue::TempResource, Device, DeviceError},
    global::Global,
    id::{self, BlasId, TlasId},
    lock::RwLock,
    ray_tracing::{get_raw_tlas_instance_size, CreateBlasError, CreateTlasError},
    resource, LabelHelpers,
};

impl Device {
    fn create_blas(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<Arc<resource::Blas>, CreateBlasError> {
        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { desc } => {
                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<dyn hal::DynBuffer>>::with_capacity(
                        desc.len(),
                    );
                for x in desc {
                    if x.index_count.is_some() != x.index_format.is_some() {
                        return Err(CreateBlasError::MissingIndexData);
                    }
                    let indices =
                        x.index_count
                            .map(|count| AccelerationStructureTriangleIndices::<
                                dyn hal::DynBuffer,
                            > {
                                format: x.index_format.unwrap(),
                                buffer: None,
                                offset: 0,
                                count,
                            });
                    entries.push(hal::AccelerationStructureTriangles::<dyn hal::DynBuffer> {
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
                    label: blas_desc.label.as_deref(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let handle = unsafe {
            self.raw()
                .get_acceleration_structure_device_address(raw.as_ref())
        };

        Ok(Arc::new(resource::Blas {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
            handle,
            label: blas_desc.label.to_string(),
            built_index: RwLock::new(rank::BLAS_BUILT_INDEX, None),
            tracking_data: TrackingData::new(self.tracker_indices.blas_s.clone()),
        }))
    }

    fn create_tlas(
        self: &Arc<Self>,
        desc: &resource::TlasDescriptor,
    ) -> Result<Arc<resource::Tlas>, CreateTlasError> {
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
                    label: desc.label.as_deref(),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::TopLevel,
                })
        }
        .map_err(DeviceError::from)?;

        let instance_buffer_size = get_raw_tlas_instance_size(self.backend())
            * std::cmp::max(desc.max_instances, 1) as usize;
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

        Ok(Arc::new(resource::Tlas {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            size_info,
            flags: desc.flags,
            update_mode: desc.update_mode,
            built_index: RwLock::new(rank::TLAS_BUILT_INDEX, None),
            dependencies: RwLock::new(rank::TLAS_DEPENDENCIES, Vec::new()),
            instance_buffer: ManuallyDrop::new(instance_buffer),
            label: desc.label.to_string(),
            max_instance_count: desc.max_instances,
            tracking_data: TrackingData::new(self.tracker_indices.tlas_s.clone()),
        }))
    }
}

impl Global {
    pub fn device_create_blas(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
        id_in: Option<BlasId>,
    ) -> (BlasId, Option<u64>, Option<CreateBlasError>) {
        profiling::scope!("Device::create_blas");

        let hub = &self.hub;
        let fid = hub.blas_s.prepare(device_id.backend(), id_in);

        let device_guard = hub.devices.read();
        let error = 'error: {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::InvalidDeviceId.into(),
            };
            if !device.is_valid() {
                break 'error DeviceError::Lost.into();
            }

            #[cfg(feature = "trace")]
            if let Some(trace) = device.trace.lock().as_mut() {
                trace.add(trace::Action::CreateBlas {
                    id: fid.id(),
                    desc: desc.clone(),
                    sizes: sizes.clone(),
                });
            }

            let blas = match device.create_blas(desc, sizes) {
                Ok(blas) => blas,
                Err(e) => break 'error e,
            };
            let handle = blas.handle;

            let id = fid.assign(blas.clone());
            log::info!("Created blas {:?} with {:?}", id, desc);

            return (id, Some(handle), None);
        };

        let id = fid.assign_error();
        (id, None, Some(error))
    }

    pub fn device_create_tlas(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
        id_in: Option<TlasId>,
    ) -> (TlasId, Option<CreateTlasError>) {
        profiling::scope!("Device::create_tlas");

        let hub = &self.hub;
        let fid = hub.tlas_s.prepare(device_id.backend(), id_in);

        let device_guard = hub.devices.read();
        let error = 'error: {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break 'error DeviceError::InvalidDeviceId.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(trace) = device.trace.lock().as_mut() {
                trace.add(trace::Action::CreateTlas {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            let tlas = match device.create_tlas(desc) {
                Ok(tlas) => tlas,
                Err(e) => break 'error e,
            };

            let id = fid.assign(tlas.clone());
            log::info!("Created tlas {:?} with {:?}", id, desc);

            return (id, None);
        };

        let id = fid.assign_error();
        (id, Some(error))
    }

    pub fn blas_destroy(&self, blas_id: BlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Blas::destroy");

        let hub = &self.hub;

        log::info!("Blas {:?} is destroyed", blas_id);
        let blas_guard = hub.blas_s.write();
        let blas = blas_guard
            .get(blas_id)
            .map_err(|_| resource::DestroyError::Invalid)?
            .clone();
        drop(blas_guard);
        let device = &blas.device;

        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeBlas(blas_id));
        }

        let temp = TempResource::Blas(blas.clone());
        {
            let mut device_lock = device.lock_life();
            let last_submit_index = device_lock.get_blas_latest_submission_index(blas.as_ref());
            if let Some(last_submit_index) = last_submit_index {
                device_lock.schedule_resource_destruction(temp, last_submit_index);
            }
        }

        Ok(())
    }

    pub fn blas_drop(&self, blas_id: BlasId) {
        profiling::scope!("Blas::drop");
        log::debug!("blas {:?} is dropped", blas_id);

        let hub = &self.hub;

        let _blas = match hub.blas_s.unregister(blas_id) {
            Some(blas) => blas,
            None => {
                return;
            }
        };

        #[cfg(feature = "trace")]
        {
            let mut lock = _blas.device.trace.lock();

            if let Some(t) = lock.as_mut() {
                t.add(trace::Action::DestroyBlas(blas_id));
            }
        }
    }

    pub fn tlas_destroy(&self, tlas_id: TlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Tlas::destroy");

        let hub = &self.hub;

        log::info!("Tlas {:?} is destroyed", tlas_id);
        let tlas_guard = hub.tlas_s.write();
        let tlas = tlas_guard
            .get(tlas_id)
            .map_err(|_| resource::DestroyError::Invalid)?
            .clone();
        drop(tlas_guard);

        let device = &mut tlas.device.clone();

        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeTlas(tlas_id));
        }

        let temp = TempResource::Tlas(tlas.clone());
        {
            let mut device_lock = device.lock_life();
            let last_submit_index = device_lock.get_tlas_latest_submission_index(tlas.as_ref());
            if let Some(last_submit_index) = last_submit_index {
                device_lock.schedule_resource_destruction(temp, last_submit_index);
            }
        }

        Ok(())
    }

    pub fn tlas_drop(&self, tlas_id: TlasId) {
        profiling::scope!("Tlas::drop");
        log::debug!("tlas {:?} is dropped", tlas_id);

        let hub = &self.hub;

        let _tlas = match hub.tlas_s.unregister(tlas_id) {
            Some(tlas) => tlas,
            None => {
                return;
            }
        };

        #[cfg(feature = "trace")]
        {
            let mut lock = _tlas.device.trace.lock();

            if let Some(t) = lock.as_mut() {
                t.add(trace::Action::DestroyTlas(tlas_id));
            }
        }
    }
}
