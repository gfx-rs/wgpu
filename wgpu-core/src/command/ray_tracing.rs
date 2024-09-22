#[cfg(feature = "trace")]
use crate::device::trace;
#[cfg(feature = "trace")]
use crate::ray_tracing::{
    TlasInstance, TraceBlasBuildEntry, TraceBlasGeometries, TraceBlasTriangleGeometry,
    TraceTlasInstance, TraceTlasPackage,
};
use crate::{
    device::queue::TempResource,
    global::Global,
    id::CommandEncoderId,
    init_tracker::MemoryInitKind,
    lock::RwLockReadGuard,
    ray_tracing::{
        tlas_instance_into_bytes, BlasAction, BlasBuildEntry, BlasGeometries,
        BuildAccelerationStructureError, TlasAction, TlasBuildEntry, TlasPackage,
        ValidateBlasActionsError, ValidateTlasActionsError,
    },
    resource::{Blas, Tlas},
    FastHashSet,
};

use wgt::{math::align_to, BufferAddress, BufferUsages, Features};

use super::{BakedCommands, CommandBufferMutable};
use crate::ray_tracing::BlasTriangleGeometry;
use crate::resource::{AccelerationStructure, Buffer, Fallible, Labeled, StagingBuffer, Trackable};
use crate::scratch::ScratchBuffer;
use crate::snatch::SnatchGuard;
use crate::storage::Storage;
use crate::track::PendingTransition;
use hal::BufferUses;
use std::ops::Deref;
use std::sync::Arc;
use std::{cmp::max, num::NonZeroU64, ops::Range};

type BufferStorage<'a> = Vec<(
    Arc<Buffer>,
    Option<PendingTransition<BufferUses>>,
    Option<(Arc<Buffer>, Option<PendingTransition<BufferUses>>)>,
    Option<(Arc<Buffer>, Option<PendingTransition<BufferUses>>)>,
    BlasTriangleGeometry<'a>,
    Option<Arc<Blas>>,
)>;

type BlasStorage<'a> = Vec<(
    Arc<Blas>,
    hal::AccelerationStructureEntries<'a, dyn hal::DynBuffer>,
    u64,
)>;

// TODO: Get this from the device (e.g. VkPhysicalDeviceAccelerationStructurePropertiesKHR.minAccelerationStructureScratchOffsetAlignment) this is currently the largest possible some devices have 0, 64, 128 (lower limits) so this could create excess allocation (Note: dx12 has 256).
const SCRATCH_BUFFER_ALIGNMENT: u32 = 256;

impl Global {
    pub fn command_encoder_build_acceleration_structures_unsafe_tlas<'a>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
        tlas_iter: impl Iterator<Item = TlasBuildEntry>,
    ) -> Result<(), BuildAccelerationStructureError> {
        profiling::scope!("CommandEncoder::build_acceleration_structures_unsafe_tlas");

        let hub = &self.hub;

        let cmd_buf = hub
            .command_buffers
            .get(command_encoder_id.into_command_buffer_id());

        let buffer_guard = hub.buffers.read();
        let blas_guard = hub.blas_s.read();
        let tlas_guard = hub.tlas_s.read();

        let device = &cmd_buf.device;

        if !device
            .features
            .contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
        {
            return Err(BuildAccelerationStructureError::MissingFeature);
        }

        let build_command_index = NonZeroU64::new(
            device
                .last_acceleration_structure_build_command_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
        )
        .unwrap();

        #[cfg(feature = "trace")]
        let trace_blas: Vec<TraceBlasBuildEntry> = blas_iter
            .map(|blas_entry| {
                let geometries = match blas_entry.geometries {
                    BlasGeometries::TriangleGeometries(triangle_geometries) => {
                        TraceBlasGeometries::TriangleGeometries(
                            triangle_geometries
                                .map(|tg| TraceBlasTriangleGeometry {
                                    size: tg.size.clone(),
                                    vertex_buffer: tg.vertex_buffer,
                                    index_buffer: tg.index_buffer,
                                    transform_buffer: tg.transform_buffer,
                                    first_vertex: tg.first_vertex,
                                    vertex_stride: tg.vertex_stride,
                                    index_buffer_offset: tg.index_buffer_offset,
                                    transform_buffer_offset: tg.transform_buffer_offset,
                                })
                                .collect(),
                        )
                    }
                };
                TraceBlasBuildEntry {
                    blas_id: blas_entry.blas_id,
                    geometries,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        let trace_tlas: Vec<TlasBuildEntry> = tlas_iter.collect();
        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.data.lock().as_mut().unwrap().commands {
            list.push(trace::Command::BuildAccelerationStructuresUnsafeTlas {
                blas: trace_blas.clone(),
                tlas: trace_tlas.clone(),
            });
            if !trace_tlas.is_empty() {
                log::warn!("a trace of command_encoder_build_acceleration_structures_unsafe_tlas containing a tlas build is not replayable!");
            }
        }

        #[cfg(feature = "trace")]
        let blas_iter = trace_blas.iter().map(|blas_entry| {
            let geometries = match &blas_entry.geometries {
                TraceBlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg| BlasTriangleGeometry {
                        size: &tg.size,
                        vertex_buffer: tg.vertex_buffer,
                        index_buffer: tg.index_buffer,
                        transform_buffer: tg.transform_buffer,
                        first_vertex: tg.first_vertex,
                        vertex_stride: tg.vertex_stride,
                        index_buffer_offset: tg.index_buffer_offset,
                        transform_buffer_offset: tg.transform_buffer_offset,
                    });
                    BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            BlasBuildEntry {
                blas_id: blas_entry.blas_id,
                geometries,
            }
        });

        #[cfg(feature = "trace")]
        let tlas_iter = trace_tlas.iter();

        let mut input_barriers = Vec::<hal::BufferBarrier<dyn hal::DynBuffer>>::new();
        let mut buf_storage = BufferStorage::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = BlasStorage::new();
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        iter_blas(
            blas_iter,
            cmd_buf_data,
            build_command_index,
            &buffer_guard,
            &blas_guard,
            &mut buf_storage,
        )?;

        let snatch_guard = device.snatchable_lock.read();
        iter_buffers(
            &mut buf_storage,
            &snatch_guard,
            &mut input_barriers,
            cmd_buf_data,
            &buffer_guard,
            &mut scratch_buffer_blas_size,
            &mut blas_storage,
        )?;

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(
            Arc<Tlas>,
            hal::AccelerationStructureEntries<dyn hal::DynBuffer>,
            u64,
        )>::new();
        let mut tlas_buf_storage = Vec::<(
            Arc<Buffer>,
            Option<PendingTransition<BufferUses>>,
            TlasBuildEntry,
        )>::new();

        for entry in tlas_iter {
            let instance_buffer = match buffer_guard.get(entry.instance_buffer_id).get() {
                Ok(buffer) => buffer,
                Err(_) => {
                    return Err(BuildAccelerationStructureError::InvalidBufferId);
                }
            };
            let data = cmd_buf_data.trackers.buffers.set_single(
                &instance_buffer,
                BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            );
            tlas_buf_storage.push((instance_buffer.clone(), data, entry.clone()));
        }

        for tlas_buf in &mut tlas_buf_storage {
            let entry = &tlas_buf.2;
            let instance_buffer = {
                let (instance_buffer, instance_pending) = (&mut tlas_buf.0, &mut tlas_buf.1);
                let instance_raw = instance_buffer.raw.get(&snatch_guard).ok_or(
                    BuildAccelerationStructureError::InvalidBuffer(instance_buffer.error_ident()),
                )?;
                if !instance_buffer.usage.contains(BufferUsages::TLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingTlasInputUsageFlag(
                        instance_buffer.error_ident(),
                    ));
                }
                if let Some(barrier) = instance_pending
                    .take()
                    .map(|pending| pending.into_hal(instance_buffer, &snatch_guard))
                {
                    input_barriers.push(barrier);
                }
                instance_raw
            };

            let tlas = tlas_guard
                .get(entry.tlas_id)
                .get()
                .map_err(|_| BuildAccelerationStructureError::InvalidTlasId)?;
            cmd_buf_data.trackers.tlas_s.set_single(tlas.clone());

            cmd_buf_data.tlas_actions.push(TlasAction {
                tlas: tlas.clone(),
                kind: crate::ray_tracing::TlasActionKind::Build {
                    build_index: build_command_index,
                    dependencies: Vec::new(),
                },
            });

            let scratch_buffer_offset = scratch_buffer_tlas_size;
            scratch_buffer_tlas_size += align_to(
                tlas.size_info.build_scratch_size as u32,
                SCRATCH_BUFFER_ALIGNMENT,
            ) as u64;

            tlas_storage.push((
                tlas,
                hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
                    buffer: Some(instance_buffer.as_ref()),
                    offset: 0,
                    count: entry.instance_count,
                }),
                scratch_buffer_offset,
            ));
        }

        let scratch_size =
            match wgt::BufferSize::new(max(scratch_buffer_blas_size, scratch_buffer_tlas_size)) {
                None => return Ok(()),
                Some(size) => size,
            };

        let scratch_buffer =
            ScratchBuffer::new(device, scratch_size).map_err(crate::device::DeviceError::from)?;

        let scratch_buffer_barrier = hal::BufferBarrier::<dyn hal::DynBuffer> {
            buffer: scratch_buffer.raw(),
            usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH
                ..BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
        };

        let blas_descriptors = blas_storage
            .iter()
            .map(|storage| map_blas(storage, scratch_buffer.raw()));

        let tlas_descriptors = tlas_storage
            .iter()
            .map(|(tlas, entries, scratch_buffer_offset)| {
                if tlas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                    log::info!("only rebuild implemented")
                }
                hal::BuildAccelerationStructureDescriptor {
                    entries,
                    mode: hal::AccelerationStructureBuildMode::Build,
                    flags: tlas.flags,
                    source_acceleration_structure: None,
                    destination_acceleration_structure: tlas.raw(),
                    scratch_buffer: scratch_buffer.raw(),
                    scratch_buffer_offset: *scratch_buffer_offset,
                }
            });

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf_data.encoder.open(device)?;

        build_blas(
            cmd_buf_raw,
            blas_present,
            tlas_present,
            input_barriers,
            &blas_descriptors.collect::<Vec<_>>(),
            scratch_buffer_barrier,
        );

        if tlas_present {
            unsafe {
                cmd_buf_raw.build_acceleration_structures(&tlas_descriptors.collect::<Vec<_>>());

                cmd_buf_raw.place_acceleration_structure_barrier(
                    hal::AccelerationStructureBarrier {
                        usage: hal::AccelerationStructureUses::BUILD_OUTPUT
                            ..hal::AccelerationStructureUses::SHADER_INPUT,
                    },
                );
            }
        }

        device
            .pending_writes
            .lock()
            .consume_temp(TempResource::ScratchBuffer(scratch_buffer));

        Ok(())
    }

    pub fn command_encoder_build_acceleration_structures<'a>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
        tlas_iter: impl Iterator<Item = TlasPackage<'a>>,
    ) -> Result<(), BuildAccelerationStructureError> {
        profiling::scope!("CommandEncoder::build_acceleration_structures");

        let hub = &self.hub;

        let cmd_buf = hub
            .command_buffers
            .get(command_encoder_id.into_command_buffer_id());

        let buffer_guard = hub.buffers.read();
        let blas_guard = hub.blas_s.read();
        let tlas_guard = hub.tlas_s.read();

        let device = &cmd_buf.device;

        if !device
            .features
            .contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE)
        {
            return Err(BuildAccelerationStructureError::MissingFeature);
        }

        let build_command_index = NonZeroU64::new(
            device
                .last_acceleration_structure_build_command_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
        )
        .unwrap();

        #[cfg(feature = "trace")]
        let trace_blas: Vec<TraceBlasBuildEntry> = blas_iter
            .map(|blas_entry| {
                let geometries = match blas_entry.geometries {
                    BlasGeometries::TriangleGeometries(triangle_geometries) => {
                        TraceBlasGeometries::TriangleGeometries(
                            triangle_geometries
                                .map(|tg| TraceBlasTriangleGeometry {
                                    size: tg.size.clone(),
                                    vertex_buffer: tg.vertex_buffer,
                                    index_buffer: tg.index_buffer,
                                    transform_buffer: tg.transform_buffer,
                                    first_vertex: tg.first_vertex,
                                    vertex_stride: tg.vertex_stride,
                                    index_buffer_offset: tg.index_buffer_offset,
                                    transform_buffer_offset: tg.transform_buffer_offset,
                                })
                                .collect(),
                        )
                    }
                };
                TraceBlasBuildEntry {
                    blas_id: blas_entry.blas_id,
                    geometries,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        let trace_tlas: Vec<TraceTlasPackage> = tlas_iter
            .map(|package: TlasPackage| {
                let instances = package
                    .instances
                    .map(|instance| {
                        instance.map(|instance| TraceTlasInstance {
                            blas_id: instance.blas_id,
                            transform: *instance.transform,
                            custom_index: instance.custom_index,
                            mask: instance.mask,
                        })
                    })
                    .collect();
                TraceTlasPackage {
                    tlas_id: package.tlas_id,
                    instances,
                    lowest_unmodified: package.lowest_unmodified,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.data.lock().as_mut().unwrap().commands {
            list.push(trace::Command::BuildAccelerationStructures {
                blas: trace_blas.clone(),
                tlas: trace_tlas.clone(),
            });
        }

        #[cfg(feature = "trace")]
        let blas_iter = trace_blas.iter().map(|blas_entry| {
            let geometries = match &blas_entry.geometries {
                TraceBlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg| BlasTriangleGeometry {
                        size: &tg.size,
                        vertex_buffer: tg.vertex_buffer,
                        index_buffer: tg.index_buffer,
                        transform_buffer: tg.transform_buffer,
                        first_vertex: tg.first_vertex,
                        vertex_stride: tg.vertex_stride,
                        index_buffer_offset: tg.index_buffer_offset,
                        transform_buffer_offset: tg.transform_buffer_offset,
                    });
                    BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            BlasBuildEntry {
                blas_id: blas_entry.blas_id,
                geometries,
            }
        });

        #[cfg(feature = "trace")]
        let tlas_iter = trace_tlas.iter().map(|tlas_package| {
            let instances = tlas_package.instances.iter().map(|instance| {
                instance.as_ref().map(|instance| TlasInstance {
                    blas_id: instance.blas_id,
                    transform: &instance.transform,
                    custom_index: instance.custom_index,
                    mask: instance.mask,
                })
            });
            TlasPackage {
                tlas_id: tlas_package.tlas_id,
                instances: Box::new(instances),
                lowest_unmodified: tlas_package.lowest_unmodified,
            }
        });

        let mut input_barriers = Vec::<hal::BufferBarrier<dyn hal::DynBuffer>>::new();
        let mut buf_storage = BufferStorage::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = BlasStorage::new();
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        iter_blas(
            blas_iter,
            cmd_buf_data,
            build_command_index,
            &buffer_guard,
            &blas_guard,
            &mut buf_storage,
        )?;

        let snatch_guard = device.snatchable_lock.read();
        iter_buffers(
            &mut buf_storage,
            &snatch_guard,
            &mut input_barriers,
            cmd_buf_data,
            &buffer_guard,
            &mut scratch_buffer_blas_size,
            &mut blas_storage,
        )?;
        let mut tlas_lock_store = Vec::<(Option<TlasPackage>, Arc<Tlas>)>::new();

        for package in tlas_iter {
            let tlas = tlas_guard
                .get(package.tlas_id)
                .get()
                .map_err(|_| BuildAccelerationStructureError::InvalidTlasId)?;

            cmd_buf_data.trackers.tlas_s.set_single(tlas.clone());
            tlas_lock_store.push((Some(package), tlas.clone()))
        }

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(
            &Tlas,
            hal::AccelerationStructureEntries<dyn hal::DynBuffer>,
            u64,
            Range<usize>,
        )>::new();
        let mut instance_buffer_staging_source = Vec::<u8>::new();

        for (package, tlas) in &mut tlas_lock_store {
            let package = package.take().unwrap();

            let scratch_buffer_offset = scratch_buffer_tlas_size;
            scratch_buffer_tlas_size += align_to(
                tlas.size_info.build_scratch_size as u32,
                SCRATCH_BUFFER_ALIGNMENT,
            ) as u64;

            let first_byte_index = instance_buffer_staging_source.len();

            let mut dependencies = Vec::new();

            let mut instance_count = 0;
            for instance in package.instances.flatten() {
                if instance.custom_index >= (1u32 << 24u32) {
                    return Err(BuildAccelerationStructureError::TlasInvalidCustomIndex(
                        tlas.error_ident(),
                    ));
                }
                let blas = blas_guard
                    .get(instance.blas_id)
                    .get()
                    .map_err(|_| BuildAccelerationStructureError::InvalidBlasIdForInstance)?
                    .clone();

                cmd_buf_data.trackers.blas_s.set_single(blas.clone());

                instance_buffer_staging_source.extend(tlas_instance_into_bytes(
                    &instance,
                    blas.handle,
                    device.backend(),
                ));

                instance_count += 1;

                dependencies.push(blas.clone());

                cmd_buf_data.blas_actions.push(BlasAction {
                    blas: blas.clone(),
                    kind: crate::ray_tracing::BlasActionKind::Use,
                });
            }

            cmd_buf_data.tlas_actions.push(TlasAction {
                tlas: tlas.clone(),
                kind: crate::ray_tracing::TlasActionKind::Build {
                    build_index: build_command_index,
                    dependencies,
                },
            });

            if instance_count > tlas.max_instance_count {
                return Err(BuildAccelerationStructureError::TlasInstanceCountExceeded(
                    tlas.error_ident(),
                    instance_count,
                    tlas.max_instance_count,
                ));
            }

            tlas_storage.push((
                tlas,
                hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
                    buffer: Some(tlas.instance_buffer.as_ref()),
                    offset: 0,
                    count: instance_count,
                }),
                scratch_buffer_offset,
                first_byte_index..instance_buffer_staging_source.len(),
            ));
        }

        let scratch_size =
            match wgt::BufferSize::new(max(scratch_buffer_blas_size, scratch_buffer_tlas_size)) {
                // if the size is zero there is nothing to build
                None => return Ok(()),
                Some(size) => size,
            };

        let scratch_buffer =
            ScratchBuffer::new(device, scratch_size).map_err(crate::device::DeviceError::from)?;

        let scratch_buffer_barrier = hal::BufferBarrier::<dyn hal::DynBuffer> {
            buffer: scratch_buffer.raw(),
            usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH
                ..BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
        };

        let blas_descriptors = blas_storage
            .iter()
            .map(|storage| map_blas(storage, scratch_buffer.raw()));

        let mut tlas_descriptors = Vec::with_capacity(tlas_storage.len());

        for &(tlas, ref entries, ref scratch_buffer_offset, _) in &tlas_storage {
            if tlas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                log::info!("only rebuild implemented")
            }
            tlas_descriptors.push(hal::BuildAccelerationStructureDescriptor {
                entries,
                mode: hal::AccelerationStructureBuildMode::Build,
                flags: tlas.flags,
                source_acceleration_structure: None,
                destination_acceleration_structure: tlas.raw.as_ref(),
                scratch_buffer: scratch_buffer.raw(),
                scratch_buffer_offset: *scratch_buffer_offset,
            })
        }

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf_data.encoder.open(device)?;

        build_blas(
            cmd_buf_raw,
            blas_present,
            tlas_present,
            input_barriers,
            &blas_descriptors.collect::<Vec<_>>(),
            scratch_buffer_barrier,
        );

        if tlas_present {
            let staging_buffer = if !instance_buffer_staging_source.is_empty() {
                let mut staging_buffer = StagingBuffer::new(
                    device,
                    wgt::BufferSize::new(instance_buffer_staging_source.len() as u64).unwrap(),
                )
                .map_err(crate::device::DeviceError::from)?;
                staging_buffer.write(&instance_buffer_staging_source);
                let flushed = staging_buffer.flush();
                Some(flushed)
            } else {
                None
            };

            unsafe {
                if let Some(ref staging_buffer) = staging_buffer {
                    cmd_buf_raw.transition_buffers(&[hal::BufferBarrier::<dyn hal::DynBuffer> {
                        buffer: staging_buffer.raw(),
                        usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                    }]);
                }
            }

            let mut instance_buffer_barriers = Vec::new();
            for &(tlas, _, _, ref range) in &tlas_storage {
                let size = match wgt::BufferSize::new((range.end - range.start) as u64) {
                    None => continue,
                    Some(size) => size,
                };
                instance_buffer_barriers.push(hal::BufferBarrier::<dyn hal::DynBuffer> {
                    buffer: tlas.instance_buffer.as_ref(),
                    usage: BufferUses::COPY_DST..BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                });
                unsafe {
                    cmd_buf_raw.transition_buffers(&[hal::BufferBarrier::<dyn hal::DynBuffer> {
                        buffer: tlas.instance_buffer.as_ref(),
                        usage: hal::BufferUses::MAP_READ..hal::BufferUses::COPY_DST,
                    }]);
                    let temp = hal::BufferCopy {
                        src_offset: range.start as u64,
                        dst_offset: 0,
                        size,
                    };
                    cmd_buf_raw.copy_buffer_to_buffer(
                        // the range whose size we just checked end is at (at that point in time) instance_buffer_staging_source.len()
                        // and since instance_buffer_staging_source doesn't shrink we can un wrap this without a panic
                        staging_buffer.as_ref().unwrap().raw(),
                        tlas.instance_buffer.as_ref(),
                        &[temp],
                    );
                }
            }

            unsafe {
                cmd_buf_raw.transition_buffers(&instance_buffer_barriers);

                cmd_buf_raw.build_acceleration_structures(&tlas_descriptors);

                cmd_buf_raw.place_acceleration_structure_barrier(
                    hal::AccelerationStructureBarrier {
                        usage: hal::AccelerationStructureUses::BUILD_OUTPUT
                            ..hal::AccelerationStructureUses::SHADER_INPUT,
                    },
                );
            }

            if let Some(staging_buffer) = staging_buffer {
                device
                    .pending_writes
                    .lock()
                    .consume_temp(TempResource::StagingBuffer(staging_buffer));
            }
        }

        device
            .pending_writes
            .lock()
            .consume_temp(TempResource::ScratchBuffer(scratch_buffer));

        Ok(())
    }
}

impl BakedCommands {
    // makes sure a blas is build before it is used
    pub(crate) fn validate_blas_actions(&mut self) -> Result<(), ValidateBlasActionsError> {
        profiling::scope!("CommandEncoder::[submission]::validate_blas_actions");
        let mut built = FastHashSet::default();
        for action in self.blas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::BlasActionKind::Build(id) => {
                    built.insert(action.blas.tracker_index());
                    *action.blas.built_index.write() = Some(id);
                }
                crate::ray_tracing::BlasActionKind::Use => {
                    if !built.contains(&action.blas.tracker_index())
                        && (*action.blas.built_index.read()).is_none()
                    {
                        return Err(ValidateBlasActionsError::UsedUnbuilt(
                            action.blas.error_ident(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    // makes sure a tlas is build before it is used
    pub(crate) fn validate_tlas_actions(&mut self) -> Result<(), ValidateTlasActionsError> {
        profiling::scope!("CommandEncoder::[submission]::validate_tlas_actions");
        for action in self.tlas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::TlasActionKind::Build {
                    build_index,
                    dependencies,
                } => {
                    *action.tlas.built_index.write() = Some(build_index);
                    *action.tlas.dependencies.write() = dependencies;
                }
                crate::ray_tracing::TlasActionKind::Use => {
                    let tlas_build_index = action.tlas.built_index.read();
                    let dependencies = action.tlas.dependencies.read();

                    if (*tlas_build_index).is_none() {
                        return Err(ValidateTlasActionsError::UsedUnbuilt(
                            action.tlas.error_ident(),
                        ));
                    }
                    for blas in dependencies.deref() {
                        let blas_build_index = *blas.built_index.read();
                        if blas_build_index.is_none() {
                            return Err(ValidateTlasActionsError::UsedUnbuilt(
                                action.tlas.error_ident(),
                            ));
                        }
                        if blas_build_index.unwrap() > tlas_build_index.unwrap() {
                            return Err(ValidateTlasActionsError::BlasNewerThenTlas(
                                blas.error_ident(),
                                action.tlas.error_ident(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

///iterates over the blas iterator, and it's geometry, pushing the buffers into a storage vector (and also some validation).
fn iter_blas<'a>(
    blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
    cmd_buf_data: &mut CommandBufferMutable,
    build_command_index: NonZeroU64,
    buffer_guard: &RwLockReadGuard<Storage<Fallible<Buffer>>>,
    blas_guard: &RwLockReadGuard<Storage<Fallible<Blas>>>,
    buf_storage: &mut BufferStorage<'a>,
) -> Result<(), BuildAccelerationStructureError> {
    for entry in blas_iter {
        let blas = blas_guard
            .get(entry.blas_id)
            .get()
            .map_err(|_| BuildAccelerationStructureError::InvalidBlasId)?;
        cmd_buf_data.trackers.blas_s.set_single(blas.clone());

        cmd_buf_data.blas_actions.push(BlasAction {
            blas: blas.clone(),
            kind: crate::ray_tracing::BlasActionKind::Build(build_command_index),
        });

        match entry.geometries {
            BlasGeometries::TriangleGeometries(triangle_geometries) => {
                for (i, mesh) in triangle_geometries.enumerate() {
                    let size_desc = match &blas.sizes {
                        wgt::BlasGeometrySizeDescriptors::Triangles { descriptors } => descriptors,
                    };
                    if i >= size_desc.len() {
                        return Err(BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                            blas.error_ident(),
                        ));
                    }
                    let size_desc = &size_desc[i];

                    if size_desc.flags != mesh.size.flags {
                        return Err(BuildAccelerationStructureError::IncompatibleBlasFlags(
                            blas.error_ident(),
                            size_desc.flags,
                            mesh.size.flags,
                        ));
                    }

                    if size_desc.vertex_count < mesh.size.vertex_count {
                        return Err(
                            BuildAccelerationStructureError::IncompatibleBlasVertexCount(
                                blas.error_ident(),
                                size_desc.vertex_count,
                                mesh.size.vertex_count,
                            ),
                        );
                    }

                    if size_desc.vertex_format != mesh.size.vertex_format {
                        return Err(BuildAccelerationStructureError::DifferentBlasVertexFormats(
                            blas.error_ident(),
                            size_desc.vertex_format,
                            mesh.size.vertex_format,
                        ));
                    }

                    match (size_desc.index_count, mesh.size.index_count) {
                        (Some(_), None) | (None, Some(_)) => {
                            return Err(
                                BuildAccelerationStructureError::BlasIndexCountProvidedMismatch(
                                    blas.error_ident(),
                                ),
                            )
                        }
                        (Some(create), Some(build)) if create < build => {
                            return Err(
                                BuildAccelerationStructureError::IncompatibleBlasIndexCount(
                                    blas.error_ident(),
                                    create,
                                    build,
                                ),
                            )
                        }
                        _ => {}
                    }

                    if size_desc.index_format != mesh.size.index_format {
                        return Err(BuildAccelerationStructureError::DifferentBlasIndexFormats(
                            blas.error_ident(),
                            size_desc.index_format,
                            mesh.size.index_format,
                        ));
                    }

                    if size_desc.index_count.is_some() && mesh.index_buffer.is_none() {
                        return Err(BuildAccelerationStructureError::MissingIndexBuffer(
                            blas.error_ident(),
                        ));
                    }
                    let vertex_buffer = match buffer_guard.get(mesh.vertex_buffer).get() {
                        Ok(buffer) => buffer,
                        Err(_) => return Err(BuildAccelerationStructureError::InvalidBufferId),
                    };
                    let vertex_pending = cmd_buf_data.trackers.buffers.set_single(
                        &vertex_buffer,
                        BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    );
                    let index_data = if let Some(index_id) = mesh.index_buffer {
                        let index_buffer = match buffer_guard.get(index_id).get() {
                            Ok(buffer) => buffer,
                            Err(_) => return Err(BuildAccelerationStructureError::InvalidBufferId),
                        };
                        if mesh.index_buffer_offset.is_none()
                            || mesh.size.index_count.is_none()
                            || mesh.size.index_count.is_none()
                        {
                            return Err(BuildAccelerationStructureError::MissingAssociatedData(
                                index_buffer.error_ident(),
                            ));
                        }
                        let data = cmd_buf_data.trackers.buffers.set_single(
                            &index_buffer,
                            hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                        );
                        Some((index_buffer.clone(), data))
                    } else {
                        None
                    };
                    let transform_data = if let Some(transform_id) = mesh.transform_buffer {
                        let transform_buffer = match buffer_guard.get(transform_id).get() {
                            Ok(buffer) => buffer,
                            Err(_) => return Err(BuildAccelerationStructureError::InvalidBufferId),
                        };
                        if mesh.transform_buffer_offset.is_none() {
                            return Err(BuildAccelerationStructureError::MissingAssociatedData(
                                transform_buffer.error_ident(),
                            ));
                        }
                        let data = cmd_buf_data.trackers.buffers.set_single(
                            &transform_buffer,
                            BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                        );
                        Some((transform_buffer, data))
                    } else {
                        None
                    };
                    buf_storage.push((
                        vertex_buffer.clone(),
                        vertex_pending,
                        index_data,
                        transform_data,
                        mesh,
                        None,
                    ));
                }

                if let Some(last) = buf_storage.last_mut() {
                    last.5 = Some(blas.clone());
                }
            }
        }
    }
    Ok(())
}

/// Iterates over the buffers generated in [iter_blas], convert the barriers into hal barriers, and the triangles into [hal::AccelerationStructureEntries] (and also some validation).
fn iter_buffers<'a, 'b>(
    buf_storage: &'a mut BufferStorage<'b>,
    snatch_guard: &'a SnatchGuard,
    input_barriers: &mut Vec<hal::BufferBarrier<'a, dyn hal::DynBuffer>>,
    cmd_buf_data: &mut CommandBufferMutable,
    buffer_guard: &RwLockReadGuard<Storage<Fallible<Buffer>>>,
    scratch_buffer_blas_size: &mut u64,
    blas_storage: &mut BlasStorage<'a>,
) -> Result<(), BuildAccelerationStructureError> {
    let mut triangle_entries =
        Vec::<hal::AccelerationStructureTriangles<dyn hal::DynBuffer>>::new();
    for buf in buf_storage {
        let mesh = &buf.4;
        let vertex_buffer = {
            let vertex_buffer = buf.0.as_ref();
            let vertex_raw = vertex_buffer.raw.get(snatch_guard).ok_or(
                BuildAccelerationStructureError::InvalidBuffer(vertex_buffer.error_ident()),
            )?;
            if !vertex_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                    vertex_buffer.error_ident(),
                ));
            }
            if let Some(barrier) = buf
                .1
                .take()
                .map(|pending| pending.into_hal(vertex_buffer, snatch_guard))
            {
                input_barriers.push(barrier);
            }
            if vertex_buffer.size
                < (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride
            {
                return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                    vertex_buffer.error_ident(),
                    vertex_buffer.size,
                    (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride,
                ));
            }
            let vertex_buffer_offset = mesh.first_vertex as u64 * mesh.vertex_stride;
            cmd_buf_data.buffer_memory_init_actions.extend(
                vertex_buffer.initialization_status.read().create_action(
                    &buffer_guard
                        .get(mesh.vertex_buffer)
                        .get()
                        .map_err(|_| BuildAccelerationStructureError::InvalidBufferId)?,
                    vertex_buffer_offset
                        ..(vertex_buffer_offset
                            + mesh.size.vertex_count as u64 * mesh.vertex_stride),
                    MemoryInitKind::NeedsInitializedMemory,
                ),
            );
            vertex_raw
        };
        let index_buffer = if let Some((ref mut index_buffer, ref mut index_pending)) = buf.2 {
            let index_raw = index_buffer.raw.get(snatch_guard).ok_or(
                BuildAccelerationStructureError::InvalidBuffer(index_buffer.error_ident()),
            )?;
            if !index_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                    index_buffer.error_ident(),
                ));
            }
            if let Some(barrier) = index_pending
                .take()
                .map(|pending| pending.into_hal(index_buffer, snatch_guard))
            {
                input_barriers.push(barrier);
            }
            let index_stride = match mesh.size.index_format.unwrap() {
                wgt::IndexFormat::Uint16 => 2,
                wgt::IndexFormat::Uint32 => 4,
            };
            if mesh.index_buffer_offset.unwrap() % index_stride != 0 {
                return Err(BuildAccelerationStructureError::UnalignedIndexBufferOffset(
                    index_buffer.error_ident(),
                ));
            }
            let index_buffer_size = mesh.size.index_count.unwrap() as u64 * index_stride;

            if mesh.size.index_count.unwrap() % 3 != 0 {
                return Err(BuildAccelerationStructureError::InvalidIndexCount(
                    index_buffer.error_ident(),
                    mesh.size.index_count.unwrap(),
                ));
            }
            if index_buffer.size
                < mesh.size.index_count.unwrap() as u64 * index_stride
                    + mesh.index_buffer_offset.unwrap()
            {
                return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                    index_buffer.error_ident(),
                    index_buffer.size,
                    mesh.size.index_count.unwrap() as u64 * index_stride
                        + mesh.index_buffer_offset.unwrap(),
                ));
            }

            cmd_buf_data.buffer_memory_init_actions.extend(
                index_buffer.initialization_status.read().create_action(
                    index_buffer,
                    mesh.index_buffer_offset.unwrap()
                        ..(mesh.index_buffer_offset.unwrap() + index_buffer_size),
                    MemoryInitKind::NeedsInitializedMemory,
                ),
            );
            Some(index_raw)
        } else {
            None
        };
        let transform_buffer = if let Some((ref mut transform_buffer, ref mut transform_pending)) =
            buf.3
        {
            if mesh.transform_buffer_offset.is_none() {
                return Err(BuildAccelerationStructureError::MissingAssociatedData(
                    transform_buffer.error_ident(),
                ));
            }
            let transform_raw = transform_buffer.raw.get(snatch_guard).ok_or(
                BuildAccelerationStructureError::InvalidBuffer(transform_buffer.error_ident()),
            )?;
            if !transform_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                    transform_buffer.error_ident(),
                ));
            }
            if let Some(barrier) = transform_pending
                .take()
                .map(|pending| pending.into_hal(transform_buffer, snatch_guard))
            {
                input_barriers.push(barrier);
            }
            if mesh.transform_buffer_offset.unwrap() % wgt::TRANSFORM_BUFFER_ALIGNMENT != 0 {
                return Err(
                    BuildAccelerationStructureError::UnalignedTransformBufferOffset(
                        transform_buffer.error_ident(),
                    ),
                );
            }
            if transform_buffer.size < 48 + mesh.transform_buffer_offset.unwrap() {
                return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                    transform_buffer.error_ident(),
                    transform_buffer.size,
                    48 + mesh.transform_buffer_offset.unwrap(),
                ));
            }
            cmd_buf_data.buffer_memory_init_actions.extend(
                transform_buffer.initialization_status.read().create_action(
                    transform_buffer,
                    mesh.transform_buffer_offset.unwrap()..(mesh.index_buffer_offset.unwrap() + 48),
                    MemoryInitKind::NeedsInitializedMemory,
                ),
            );
            Some(transform_raw)
        } else {
            None
        };

        let triangles = hal::AccelerationStructureTriangles {
            vertex_buffer: Some(vertex_buffer.as_ref()),
            vertex_format: mesh.size.vertex_format,
            first_vertex: mesh.first_vertex,
            vertex_count: mesh.size.vertex_count,
            vertex_stride: mesh.vertex_stride,
            indices: index_buffer.map(|index_buffer| hal::AccelerationStructureTriangleIndices::<
                dyn hal::DynBuffer,
            > {
                format: mesh.size.index_format.unwrap(),
                buffer: Some(index_buffer.as_ref()),
                offset: mesh.index_buffer_offset.unwrap() as u32,
                count: mesh.size.index_count.unwrap(),
            }),
            transform: transform_buffer.map(|transform_buffer| {
                hal::AccelerationStructureTriangleTransform {
                    buffer: transform_buffer.as_ref(),
                    offset: mesh.transform_buffer_offset.unwrap() as u32,
                }
            }),
            flags: mesh.size.flags,
        };
        triangle_entries.push(triangles);
        if let Some(blas) = buf.5.take() {
            let scratch_buffer_offset = *scratch_buffer_blas_size;
            *scratch_buffer_blas_size += align_to(
                blas.size_info.build_scratch_size as u32,
                SCRATCH_BUFFER_ALIGNMENT,
            ) as u64;

            blas_storage.push((
                blas,
                hal::AccelerationStructureEntries::Triangles(triangle_entries),
                scratch_buffer_offset,
            ));
            triangle_entries = Vec::new();
        }
    }
    Ok(())
}

fn map_blas<'a>(
    storage: &'a (
        Arc<Blas>,
        hal::AccelerationStructureEntries<dyn hal::DynBuffer>,
        BufferAddress,
    ),
    scratch_buffer: &'a dyn hal::DynBuffer,
) -> hal::BuildAccelerationStructureDescriptor<
    'a,
    dyn hal::DynBuffer,
    dyn hal::DynAccelerationStructure,
> {
    let (blas, entries, scratch_buffer_offset) = storage;
    if blas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
        log::info!("only rebuild implemented")
    }
    hal::BuildAccelerationStructureDescriptor {
        entries,
        mode: hal::AccelerationStructureBuildMode::Build,
        flags: blas.flags,
        source_acceleration_structure: None,
        destination_acceleration_structure: blas.raw.as_ref(),
        scratch_buffer,
        scratch_buffer_offset: *scratch_buffer_offset,
    }
}

fn build_blas<'a>(
    cmd_buf_raw: &mut dyn hal::DynCommandEncoder,
    blas_present: bool,
    tlas_present: bool,
    input_barriers: Vec<hal::BufferBarrier<dyn hal::DynBuffer>>,
    blas_descriptors: &[hal::BuildAccelerationStructureDescriptor<
        'a,
        dyn hal::DynBuffer,
        dyn hal::DynAccelerationStructure,
    >],
    scratch_buffer_barrier: hal::BufferBarrier<dyn hal::DynBuffer>,
) {
    unsafe {
        cmd_buf_raw.transition_buffers(&input_barriers);
    }

    if blas_present {
        unsafe {
            cmd_buf_raw.place_acceleration_structure_barrier(hal::AccelerationStructureBarrier {
                usage: hal::AccelerationStructureUses::BUILD_INPUT
                    ..hal::AccelerationStructureUses::BUILD_OUTPUT,
            });

            cmd_buf_raw.build_acceleration_structures(blas_descriptors);
        }
    }

    if blas_present && tlas_present {
        unsafe {
            cmd_buf_raw.transition_buffers(&[scratch_buffer_barrier]);
        }
    }

    let mut source_usage = hal::AccelerationStructureUses::empty();
    let mut destination_usage = hal::AccelerationStructureUses::empty();
    if blas_present {
        source_usage |= hal::AccelerationStructureUses::BUILD_OUTPUT;
        destination_usage |= hal::AccelerationStructureUses::BUILD_INPUT
    }
    if tlas_present {
        source_usage |= hal::AccelerationStructureUses::SHADER_INPUT;
        destination_usage |= hal::AccelerationStructureUses::BUILD_OUTPUT;
    }
    unsafe {
        cmd_buf_raw.place_acceleration_structure_barrier(hal::AccelerationStructureBarrier {
            usage: source_usage..destination_usage,
        });
    }
}