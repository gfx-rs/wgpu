use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::api_log;
#[cfg(feature = "trace")]
use crate::device::trace;
use crate::lock::rank;
use crate::resource::{Fallible, TrackingData};
use crate::snatch::Snatchable;
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
        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)?;

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
        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE)?;

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
                usage: wgt::BufferUses::COPY_DST
                    | wgt::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
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

        let fid = self.hub.blas_s.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

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

            let id = fid.assign(Fallible::Valid(blas));
            api_log!("Device::create_blas -> {id:?}");

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

        let fid = self.hub.tlas_s.prepare(id_in);

        let error = 'error: {
            let device = self.hub.devices.get(device_id);

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
            api_log!("Device::create_tlas -> {id:?}");

            return (id, None);
        };

        let id = fid.assign(Fallible::Invalid(Arc::new(error.to_string())));
        (id, Some(error))
    }

    pub fn blas_drop(&self, blas_id: BlasId) {
        profiling::scope!("Blas::drop");
        api_log!("Blas::drop {blas_id:?}");

        let _blas = self.hub.blas_s.remove(blas_id);

        #[cfg(feature = "trace")]
        if let Ok(blas) = _blas.get() {
            if let Some(t) = blas.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyBlas(blas_id));
            }
        }
    }

    pub fn tlas_drop(&self, tlas_id: TlasId) {
        profiling::scope!("Tlas::drop");
        api_log!("Tlas::drop {tlas_id:?}");

        let _tlas = self.hub.tlas_s.remove(tlas_id);

        #[cfg(feature = "trace")]
        if let Ok(tlas) = _tlas.get() {
            if let Some(t) = tlas.device.trace.lock().as_mut() {
                t.add(trace::Action::DestroyTlas(tlas_id));
            }
        }
    }
}
