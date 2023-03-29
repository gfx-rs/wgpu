#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::{clear_texture, CommandBuffer, CommandEncoderError},
    conv,
    device::{Device, DeviceError, MissingDownlevelFlags},
    error::{ErrorFormatter, PrettyError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Storage, Token},
    id::{self, BlasId, BufferId, CommandEncoderId, TlasId},
    init_tracker::{
        has_copy_partial_init_tracker_coverage, MemoryInitKind, TextureInitRange,
        TextureInitTrackerAction,
    },
    resource::{self, Texture, TextureErrorDimension},
    track::TextureSelector,
    LabelHelpers, LifeGuard, Stored,
};

use arrayvec::ArrayVec;
use hal::{AccelerationStructureTriangleIndices, CommandEncoder as _, Device as _};
use thiserror::Error;
use wgt::{BufferAddress, BufferUsages, Extent3d, TextureUsages};

use std::iter;

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum BuildAccelerationStructureError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("Buffer {0:?} is missing `BLAS_INPUT` usage flag")]
    MissingBlasInputUsageFlag(BufferId),
    #[error("Blas {0:?} is invalid or destroyed")]
    InvalidBlas(BlasId),
    #[error("Tlas {0:?} is invalid or destroyed")]
    InvalidTlas(TlasId),
    #[error("Buffer {0:?} is missing `TLAS_INPUT` usage flag")]
    MissingTlasInputUsageFlag(BufferId),
}

impl<A: HalApi> Device<A> {
    fn create_blas(
        &self,
        self_id: id::DeviceId,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<resource::Blas<A>, resource::CreateBlasError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { desc } => {
                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<A>>::with_capacity(desc.len());
                for x in desc {
                    if x.index_count.is_some() != x.index_format.is_some() {
                        return Err(resource::CreateBlasError::Unimplemented);
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
                        indices: indices,
                        transform: None,
                        flags: x.flags,
                    });
                }
                unsafe {
                    self.raw.get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::Triangles(&entries),
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

        Ok(resource::Blas {
            raw: raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(blas_desc.label.borrow_or_default()),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
        })
    }

    fn create_tlas(
        &self,
        self_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
    ) -> Result<resource::Tlas<A>, resource::CreateTlasError> {
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
                    format: hal::AccelerationStructureFormat::BottomLevel,
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
        })
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn device_create_blas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
        id_in: Input<G, BlasId>,
    ) -> (BlasId, Option<u64>, Option<resource::CreateBlasError>) {
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
            // #[cfg(feature = "trace")]
            // if let Some(ref trace) = device.trace {
            //     let mut desc = desc.clone();
            //     trace
            //         .lock()
            //         .add(trace::Action::CreateBlas(fid.id(), desc));
            // }

            let blas = match device.create_blas(device_id, desc, sizes) {
                Ok(blas) => blas,
                Err(e) => break e,
            };

            let handle = unsafe{
                device.raw.get_acceleration_structure_device_address(&blas.raw)
            };

            let ref_count = blas.life_guard.add_ref();

            let id = fid.assign(blas, &mut token);
            log::info!("Created blas {:?} with {:?}", id, desc);

            device.trackers.lock().blas_s.insert_single(id, ref_count);

            return (id.0, Some(handle), None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, None, Some(error))
    }

    pub fn device_create_tlas<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
        id_in: Input<G, id::TlasId>,
    ) -> (id::TlasId, Option<resource::CreateTlasError>) {
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
            // #[cfg(feature = "trace")]
            // if let Some(ref trace) = device.trace {
            //     let mut desc = desc.clone();
            //     trace
            //         .lock()
            //         .add(trace::Action::CreateTlas(fid.id(), desc));
            // }

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
}

#[derive(Debug)]
pub struct BlasTriangleGeometry<'a> {
    pub size: &'a wgt::BlasTriangleGeometrySizeDescriptor,
    pub vertex_buffer: BufferId,
    pub index_buffer: Option<BufferId>,
    pub transform_buffer: Option<BufferId>,
    pub first_vertex: u32,
    pub vertex_stride: BufferAddress,
    pub index_buffer_offset: Option<BufferAddress>,
    pub transform_buffer_offset: Option<BufferAddress>,
}

pub enum BlasGeometries<'a> {
    TriangleGeometries(Box<dyn Iterator<Item = BlasTriangleGeometry<'a>> + 'a>),
}

pub struct BlasBuildEntry<'a> {
    pub blas_id: BlasId,
    pub geometries: BlasGeometries<'a>,
}

pub struct TlasBuildEntry {
    pub tlas_id: TlasId,
    pub instance_buffer_id: BufferId,
    pub instance_count: u32,
}

pub enum BlasGeometriesStorage<'a> {
    TriangleGeometries(Vec<BlasTriangleGeometry<'a>>),
}

pub struct BlasBuildEntryStorage<'a> {
    pub blas_id: BlasId,
    pub geometries: BlasGeometriesStorage<'a>,
}
