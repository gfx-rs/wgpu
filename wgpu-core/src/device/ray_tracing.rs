use std::mem::ManuallyDrop;
use std::sync::Arc;

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::lock::{rank, Mutex};
use crate::resource::{Fallible, TrackingData};
use crate::snatch::Snatchable;
use crate::weak_vec::WeakVec;
use crate::{
    device::{Device, DeviceError},
    global::Global,
    id::{self, BlasId, TlasId},
    lock::RwLock,
    ray_tracing::{CreateBlasError, CreateTlasError},
    resource, LabelHelpers,
};
use hal::AccelerationStructureTriangleIndices;
use wgt::Features;

impl Device {
    fn create_blas(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<Arc<resource::Blas>, CreateBlasError> {
        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { descriptors } => {
                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<dyn hal::DynBuffer>>::with_capacity(
                        descriptors.len(),
                    );
                for desc in descriptors {
                    if desc.index_count.is_some() != desc.index_format.is_some() {
                        return Err(CreateBlasError::MissingIndexData);
                    }
                    let indices =
                        desc.index_count
                            .map(|count| AccelerationStructureTriangleIndices::<
                                dyn hal::DynBuffer,
                            > {
                                format: desc.index_format.unwrap(),
                                buffer: None,
                                offset: 0,
                                count,
                            });
                    if !self
                        .features
                        .allowed_vertex_formats_for_blas()
                        .contains(&desc.vertex_format)
                    {
                        return Err(CreateBlasError::InvalidVertexFormat(
                            desc.vertex_format,
                            self.features.allowed_vertex_formats_for_blas(),
                        ));
                    }
                    entries.push(hal::AccelerationStructureTriangles::<dyn hal::DynBuffer> {
                        vertex_buffer: None,
                        vertex_format: desc.vertex_format,
                        first_vertex: 0,
                        vertex_count: desc.vertex_count,
                        vertex_stride: 0,
                        indices,
                        transform: None,
                        flags: desc.flags,
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
        .map_err(DeviceError::from_hal)?;

        let handle = unsafe {
            self.raw()
                .get_acceleration_structure_device_address(raw.as_ref())
        };

        Ok(Arc::new(resource::Blas {
            raw: Snatchable::new(raw),
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
        .map_err(DeviceError::from_hal)?;

        let instance_buffer_size =
            self.alignments.raw_tlas_instance_size * desc.max_instances.max(1) as usize;
        let instance_buffer = unsafe {
            self.raw().create_buffer(&hal::BufferDescriptor {
                label: Some("(wgpu-core) instances_buffer"),
                size: instance_buffer_size as u64,
                usage: hal::BufferUses::COPY_DST
                    | hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                memory_flags: hal::MemoryFlags::PREFER_COHERENT,
            })
        }
        .map_err(DeviceError::from_hal)?;

        Ok(Arc::new(resource::Tlas {
            raw: Snatchable::new(raw),
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
            bind_groups: Mutex::new(rank::TLAS_BIND_GROUPS, WeakVec::new()),
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
        let fid = hub.blas_s.prepare(id_in);

        let device_guard = hub.devices.read();
        let error = 'error: {
            let device = device_guard.get(device_id);
            match device.check_is_valid() {
                Ok(_) => {}
                Err(err) => break 'error CreateBlasError::Device(err),
            };

            if !device
                .features
                .contains(Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            {
                break 'error CreateBlasError::MissingFeature;
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

            let id = fid.assign(Fallible::Valid(blas.clone()));
            log::info!("Created blas {:?} with {:?}", id, desc);

            return (id, Some(handle), None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(error.to_string())));
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
        let fid = hub.tlas_s.prepare(id_in);

        let device_guard = hub.devices.read();
        let error = 'error: {
            let device = device_guard.get(device_id);
            match device.check_is_valid() {
                Ok(_) => {}
                Err(e) => break 'error CreateTlasError::Device(e),
            }

            if !device
                .features
                .contains(Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)
            {
                break 'error CreateTlasError::MissingFeature;
            }

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

            let id = fid.assign(Fallible::Valid(tlas));
            log::info!("Created tlas {:?} with {:?}", id, desc);

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(error.to_string())));
        (id, Some(error))
    }

    pub fn blas_destroy(&self, blas_id: BlasId) -> Result<(), resource::DestroyError> {
        profiling::scope!("Blas::destroy");
        log::info!("Blas::destroy {blas_id:?}");

        let hub = &self.hub;

        let blas = hub.blas_s.get(blas_id).get()?;
        let _device = &blas.device;

        #[cfg(feature = "trace")]
        if let Some(trace) = _device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeBlas(blas_id));
        }

        blas.destroy()
    }

    pub fn blas_drop(&self, blas_id: BlasId) {
        profiling::scope!("Blas::drop");
        log::debug!("blas {:?} is dropped", blas_id);

        let hub = &self.hub;

        let _blas = match hub.blas_s.remove(blas_id).get() {
            Ok(blas) => blas,
            Err(_) => {
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
            .get()
            .map_err(resource::DestroyError::InvalidResource)?
            .clone();
        drop(tlas_guard);

        let _device = &mut tlas.device.clone();

        #[cfg(feature = "trace")]
        if let Some(trace) = _device.trace.lock().as_mut() {
            trace.add(trace::Action::FreeTlas(tlas_id));
        }

        tlas.destroy()
    }

    pub fn tlas_drop(&self, tlas_id: TlasId) {
        profiling::scope!("Tlas::drop");
        log::debug!("tlas {:?} is dropped", tlas_id);

        let hub = &self.hub;

        let _tlas = match hub.tlas_s.remove(tlas_id).get() {
            Ok(tlas) => tlas,
            Err(_) => {
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
