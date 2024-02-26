use crate::{
    command::CommandBuffer,
    device::queue::TempResource,
    global::Global,
    hal_api::HalApi,
    id::CommandEncoderId,
    init_tracker::MemoryInitKind,
    ray_tracing::{
        tlas_instance_into_bytes, BlasAction, BlasBuildEntry, BlasGeometries,
        BuildAccelerationStructureError, TlasAction, TlasBuildEntry, TlasPackage,
        ValidateBlasActionsError, ValidateTlasActionsError,
    },
    resource::{Blas, Tlas},
    storage::Storage,
    FastHashSet,
};

use wgt::{math::align_to, BufferUsages};

use crate::ray_tracing::BlasTriangleGeometry;
use crate::resource::{Buffer, Resource, ResourceInfo, StagingBuffer};
use crate::track::PendingTransition;
use hal::{BufferUses, CommandEncoder, Device};
use parking_lot::{Mutex, RwLockReadGuard};
use std::ops::Deref;
use std::sync::Arc;
use std::{cmp::max, iter, num::NonZeroU64, ops::Range, ptr};

use super::BakedCommands;

// This should be queried from the device, maybe the the hal api should pre aline it, since I am unsure how else we can idiomatically get this value.
const SCRATCH_BUFFER_ALIGNMENT: u32 = 256;

impl Global {
    pub fn command_encoder_build_acceleration_structures_unsafe_tlas<'a, A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
        tlas_iter: impl Iterator<Item = TlasBuildEntry>,
    ) -> Result<(), BuildAccelerationStructureError> {
        profiling::scope!("CommandEncoder::build_acceleration_structures_unsafe_tlas");

        let hub = A::hub(self);

        let device_guard = hub.devices.write();
        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let buffer_guard = hub.buffers.read();
        let blas_guard = hub.blas_s.read();
        let tlas_guard = hub.tlas_s.read();

        let device = &mut device_guard.get(cmd_buf.device.as_info().id()).unwrap();

        let build_command_index = NonZeroU64::new(
            device
                .last_acceleration_structure_build_command_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
        )
        .unwrap();

        #[cfg(feature = "trace")]
        let trace_blas: Vec<crate::ray_tracing::TraceBlasBuildEntry> = blas_iter
            .map(|x| {
                let geometries = match x.geometries {
                    BlasGeometries::TriangleGeometries(triangle_geometries) => {
                        crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                            triangle_geometries
                                .map(|tg| crate::ray_tracing::TraceBlasTriangleGeometry {
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
                crate::ray_tracing::TraceBlasBuildEntry {
                    blas_id: x.blas_id,
                    geometries,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        let trace_tlas: Vec<TlasBuildEntry> = tlas_iter.collect();
        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.data.lock().as_mut().unwrap().commands {
            list.push(
                crate::device::trace::Command::BuildAccelerationStructuresUnsafeTlas {
                    blas: trace_blas.clone(),
                    tlas: trace_tlas.clone(),
                },
            );
            if !trace_tlas.is_empty() {
                log::warn!("a trace of command_encoder_build_acceleration_structures_unsafe_tlas containing a tlas build is not replayable!");
            }
        }

        #[cfg(feature = "trace")]
        let blas_iter = trace_blas.iter().map(|x| {
            let geometries = match &x.geometries {
                crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                    triangle_geometries,
                ) => {
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
                blas_id: x.blas_id,
                geometries,
            }
        });

        #[cfg(feature = "trace")]
        let tlas_iter = trace_tlas.iter();

        let mut input_barriers = Vec::<hal::BufferBarrier<A>>::new();
        let mut buf_storage = Vec::<(
            Arc<Buffer<A>>,
            Option<PendingTransition<BufferUses>>,
            Option<(Arc<Buffer<A>>, Option<PendingTransition<BufferUses>>)>,
            Option<(Arc<Buffer<A>>, Option<PendingTransition<BufferUses>>)>,
            BlasTriangleGeometry,
            Option<Arc<Blas<A>>>,
        )>::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = Vec::<(&Blas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();
        for entry in blas_iter {
            let blas = cmd_buf_data
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            if blas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidBlas(entry.blas_id));
            }

            cmd_buf_data.blas_actions.push(BlasAction {
                id: entry.blas_id,
                kind: crate::ray_tracing::BlasActionKind::Build(build_command_index),
            });

            match entry.geometries {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    for (i, mesh) in triangle_geometries.enumerate() {
                        let size_desc = match &blas.sizes {
                            wgt::BlasGeometrySizeDescriptors::Triangles { desc } => desc,
                        };
                        if i >= size_desc.len() {
                            return Err(
                                BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                                    entry.blas_id,
                                ),
                            );
                        }
                        let size_desc = &size_desc[i];

                        if size_desc.flags != mesh.size.flags
                            || size_desc.vertex_count < mesh.size.vertex_count
                            || size_desc.vertex_format != mesh.size.vertex_format
                            || size_desc.index_count.is_none() != mesh.size.index_count.is_none()
                            || (size_desc.index_count.is_none()
                                || size_desc.index_count.unwrap() < mesh.size.index_count.unwrap())
                            || size_desc.index_format.is_none() != mesh.size.index_format.is_none()
                            || (size_desc.index_format.is_none()
                                || size_desc.index_format.unwrap()
                                    != mesh.size.index_format.unwrap())
                        {
                            return Err(
                                BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                                    entry.blas_id,
                                ),
                            );
                        }

                        if size_desc.index_count.is_some() && mesh.index_buffer.is_none() {
                            return Err(BuildAccelerationStructureError::MissingIndexBuffer(
                                entry.blas_id,
                            ));
                        }
                        let (vertex_buffer, vertex_pending) = cmd_buf_data
                            .trackers
                            .buffers
                            .set_single(
                                match buffer_guard.get(mesh.vertex_buffer) {
                                    Ok(buffer) => buffer,
                                    Err(_) => {
                                        return Err(BuildAccelerationStructureError::InvalidBuffer(
                                            mesh.vertex_buffer,
                                        ))
                                    }
                                },
                                BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                            )
                            .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                mesh.vertex_buffer,
                            ))?;
                        let index_data = if let Some(index_id) = mesh.index_buffer {
                            if mesh.index_buffer_offset.is_none()
                                || mesh.size.index_count.is_none()
                                || mesh.size.index_count.is_none()
                            {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        index_id,
                                    ),
                                );
                            }
                            let data = cmd_buf_data
                                .trackers
                                .buffers
                                .set_single(
                                    match buffer_guard.get(index_id) {
                                        Ok(buffer) => buffer,
                                        Err(_) => {
                                            return Err(
                                                BuildAccelerationStructureError::InvalidBuffer(
                                                    index_id,
                                                ),
                                            )
                                        }
                                    },
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            Some(data)
                        } else {
                            None
                        };
                        let transform_data = if let Some(transform_id) = mesh.transform_buffer {
                            if mesh.transform_buffer_offset.is_none() {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        transform_id,
                                    ),
                                );
                            }
                            let data = cmd_buf_data
                                .trackers
                                .buffers
                                .set_single(
                                    match buffer_guard.get(transform_id) {
                                        Ok(buffer) => buffer,
                                        Err(_) => {
                                            return Err(
                                                BuildAccelerationStructureError::InvalidBuffer(
                                                    transform_id,
                                                ),
                                            )
                                        }
                                    },
                                    BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    transform_id,
                                ))?;
                            Some(data)
                        } else {
                            None
                        };

                        buf_storage.push((
                            vertex_buffer,
                            vertex_pending,
                            index_data,
                            transform_data,
                            mesh,
                            None,
                        ))
                    }
                    if let Some(last) = buf_storage.last_mut() {
                        last.5 = Some(blas.clone());
                    }
                }
            }
        }

        let mut triangle_entries = Vec::<hal::AccelerationStructureTriangles<A>>::new();
        let snatch_guard = device.snatchable_lock.read();
        for buf in &mut buf_storage {
            let mesh = &buf.4;
            let vertex_buffer = {
                let vertex_buffer = buf.0.as_ref();
                let vertex_raw = vertex_buffer.raw.get(&snatch_guard).ok_or(
                    BuildAccelerationStructureError::InvalidBuffer(mesh.vertex_buffer),
                )?;
                if !vertex_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                        mesh.vertex_buffer,
                    ));
                }
                if let Some(barrier) = buf
                    .1
                    .take()
                    .map(|pending| pending.into_hal(vertex_buffer, &snatch_guard))
                {
                    input_barriers.push(barrier);
                }
                if vertex_buffer.size
                    < (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride
                {
                    return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                        mesh.vertex_buffer,
                        vertex_buffer.size,
                        (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride,
                    ));
                }
                let vertex_buffer_offset = mesh.first_vertex as u64 * mesh.vertex_stride;
                cmd_buf_data.buffer_memory_init_actions.extend(
                    vertex_buffer.initialization_status.read().create_action(
                        buffer_guard.get(mesh.vertex_buffer).unwrap(),
                        vertex_buffer_offset
                            ..(vertex_buffer_offset
                                + mesh.size.vertex_count as u64 * mesh.vertex_stride),
                        MemoryInitKind::NeedsInitializedMemory,
                    ),
                );
                vertex_raw
            };
            let index_buffer = if let Some((ref mut index_buffer, ref mut index_pending)) = buf.2 {
                let index_id = mesh.index_buffer.as_ref().unwrap();
                let index_raw = index_buffer
                    .raw
                    .get(&snatch_guard)
                    .ok_or(BuildAccelerationStructureError::InvalidBuffer(*index_id))?;
                if !index_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                        *index_id,
                    ));
                }
                if let Some(barrier) = index_pending
                    .take()
                    .map(|pending| pending.into_hal(index_buffer, &snatch_guard))
                {
                    input_barriers.push(barrier);
                }
                let index_stride = match mesh.size.index_format.unwrap() {
                    wgt::IndexFormat::Uint16 => 2,
                    wgt::IndexFormat::Uint32 => 4,
                };
                if mesh.index_buffer_offset.unwrap() % index_stride != 0 {
                    return Err(BuildAccelerationStructureError::UnalignedIndexBufferOffset(
                        *index_id,
                    ));
                }
                let index_buffer_size = mesh.size.index_count.unwrap() as u64 * index_stride;

                if mesh.size.index_count.unwrap() % 3 != 0 {
                    return Err(BuildAccelerationStructureError::InvalidIndexCount(
                        *index_id,
                        mesh.size.index_count.unwrap(),
                    ));
                }
                if index_buffer.size
                    < mesh.size.index_count.unwrap() as u64 * index_stride
                        + mesh.index_buffer_offset.unwrap()
                {
                    return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                        *index_id,
                        index_buffer.size,
                        mesh.size.index_count.unwrap() as u64 * index_stride
                            + mesh.index_buffer_offset.unwrap(),
                    ));
                }

                cmd_buf_data.buffer_memory_init_actions.extend(
                    index_buffer.initialization_status.read().create_action(
                        match buffer_guard.get(*index_id) {
                            Ok(buffer) => buffer,
                            Err(_) => {
                                return Err(BuildAccelerationStructureError::InvalidBuffer(
                                    *index_id,
                                ))
                            }
                        },
                        mesh.index_buffer_offset.unwrap()
                            ..(mesh.index_buffer_offset.unwrap() + index_buffer_size),
                        MemoryInitKind::NeedsInitializedMemory,
                    ),
                );
                Some(index_raw)
            } else {
                None
            };
            let transform_buffer =
                if let Some((ref mut transform_buffer, ref mut transform_pending)) = buf.3 {
                    let transform_id = mesh.transform_buffer.as_ref().unwrap();
                    if mesh.transform_buffer_offset.is_none() {
                        return Err(BuildAccelerationStructureError::MissingAssociatedData(
                            *transform_id,
                        ));
                    }
                    let transform_raw = transform_buffer.raw.get(&snatch_guard).ok_or(
                        BuildAccelerationStructureError::InvalidBuffer(*transform_id),
                    )?;
                    if !transform_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                        return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                            *transform_id,
                        ));
                    }
                    if let Some(barrier) = transform_pending
                        .take()
                        .map(|pending| pending.into_hal(transform_buffer, &snatch_guard))
                    {
                        input_barriers.push(barrier);
                    }
                    if mesh.transform_buffer_offset.unwrap() % wgt::TRANSFORM_BUFFER_ALIGNMENT != 0
                    {
                        return Err(
                            BuildAccelerationStructureError::UnalignedTransformBufferOffset(
                                *transform_id,
                            ),
                        );
                    }
                    if transform_buffer.size < 48 + mesh.transform_buffer_offset.unwrap() {
                        return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                            *transform_id,
                            transform_buffer.size,
                            48 + mesh.transform_buffer_offset.unwrap(),
                        ));
                    }
                    cmd_buf_data.buffer_memory_init_actions.extend(
                        transform_buffer.initialization_status.read().create_action(
                            buffer_guard.get(*transform_id).unwrap(),
                            mesh.transform_buffer_offset.unwrap()
                                ..(mesh.index_buffer_offset.unwrap() + 48),
                            MemoryInitKind::NeedsInitializedMemory,
                        ),
                    );
                    Some(transform_raw)
                } else {
                    None
                };

            let triangles = hal::AccelerationStructureTriangles {
                vertex_buffer: Some(vertex_buffer),
                vertex_format: mesh.size.vertex_format,
                first_vertex: mesh.first_vertex,
                vertex_count: mesh.size.vertex_count,
                vertex_stride: mesh.vertex_stride,
                indices: index_buffer.map(|index_buffer| {
                    hal::AccelerationStructureTriangleIndices::<A> {
                        format: mesh.size.index_format.unwrap(),
                        buffer: Some(index_buffer),
                        offset: mesh.index_buffer_offset.unwrap() as u32,
                        count: mesh.size.index_count.unwrap(),
                    }
                }),
                transform: transform_buffer.map(|transform_buffer| {
                    hal::AccelerationStructureTriangleTransform {
                        buffer: transform_buffer,
                        offset: mesh.transform_buffer_offset.unwrap() as u32,
                    }
                }),
                flags: mesh.size.flags,
            };
            triangle_entries.push(triangles);
            if let Some(blas) = buf.5.as_ref() {
                let scratch_buffer_offset = scratch_buffer_blas_size;
                scratch_buffer_blas_size += align_to(
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

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(&Tlas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();
        let mut tlas_buf_storage = Vec::<(
            Arc<Buffer<A>>,
            Option<PendingTransition<BufferUses>>,
            TlasBuildEntry,
        )>::new();

        for entry in tlas_iter {
            let data = cmd_buf_data
                .trackers
                .buffers
                .set_single(
                    match buffer_guard.get(entry.instance_buffer_id) {
                        Ok(buffer) => buffer,
                        Err(_) => {
                            return Err(BuildAccelerationStructureError::InvalidBuffer(
                                entry.instance_buffer_id,
                            ));
                        }
                    },
                    BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                )
                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                    entry.instance_buffer_id,
                ))?;
            tlas_buf_storage.push((data.0, data.1, entry.clone()));
        }

        for tlas_buf in &mut tlas_buf_storage {
            let entry = &tlas_buf.2;
            let instance_buffer = {
                let (instance_buffer, instance_pending) = (&mut tlas_buf.0, &mut tlas_buf.1);
                let instance_raw = instance_buffer.raw.get(&snatch_guard).ok_or(
                    BuildAccelerationStructureError::InvalidBuffer(entry.instance_buffer_id),
                )?;
                if !instance_buffer.usage.contains(BufferUsages::TLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingTlasInputUsageFlag(
                        entry.instance_buffer_id,
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

            let tlas = cmd_buf_data
                .trackers
                .tlas_s
                .add_single(&tlas_guard, entry.tlas_id)
                .ok_or(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id))?;

            if tlas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id));
            }

            cmd_buf_data.tlas_actions.push(TlasAction {
                id: entry.tlas_id,
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
                    buffer: Some(instance_buffer),
                    offset: 0,
                    count: entry.instance_count,
                }),
                scratch_buffer_offset,
            ));
        }

        if max(scratch_buffer_blas_size, scratch_buffer_tlas_size) == 0 {
            return Ok(());
        }

        let scratch_buffer = unsafe {
            device
                .raw()
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("(wgpu) scratch buffer"),
                    size: max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                    usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .unwrap()
        };

        let scratch_buffer_barrier = hal::BufferBarrier::<A> {
            buffer: &scratch_buffer,
            usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH
                ..hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
        };

        let blas_descriptors =
            blas_storage
                .iter()
                .map(|&(blas, ref entries, ref scratch_buffer_offset)| {
                    if blas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                        log::info!("only rebuild implemented")
                    }
                    hal::BuildAccelerationStructureDescriptor {
                        entries,
                        mode: hal::AccelerationStructureBuildMode::Build,
                        flags: blas.flags,
                        source_acceleration_structure: None,
                        destination_acceleration_structure: blas.raw.as_ref().unwrap(),
                        scratch_buffer: &scratch_buffer,
                        scratch_buffer_offset: *scratch_buffer_offset,
                    }
                });

        let tlas_descriptors =
            tlas_storage
                .iter()
                .map(|&(tlas, ref entries, ref scratch_buffer_offset)| {
                    if tlas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                        log::info!("only rebuild implemented")
                    }
                    hal::BuildAccelerationStructureDescriptor {
                        entries,
                        mode: hal::AccelerationStructureBuildMode::Build,
                        flags: tlas.flags,
                        source_acceleration_structure: None,
                        destination_acceleration_structure: tlas.raw.as_ref().unwrap(),
                        scratch_buffer: &scratch_buffer,
                        scratch_buffer_offset: *scratch_buffer_offset,
                    }
                });

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf_data.encoder.open()?;
        unsafe {
            cmd_buf_raw.transition_buffers(input_barriers.into_iter());

            if blas_present {
                cmd_buf_raw.place_acceleration_structure_barrier(
                    hal::AccelerationStructureBarrier {
                        usage: hal::AccelerationStructureUses::BUILD_INPUT
                            ..hal::AccelerationStructureUses::BUILD_OUTPUT,
                    },
                );

                cmd_buf_raw
                    .build_acceleration_structures(blas_storage.len() as u32, blas_descriptors);
            }

            if blas_present && tlas_present {
                cmd_buf_raw.transition_buffers(iter::once(scratch_buffer_barrier));
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

            cmd_buf_raw.place_acceleration_structure_barrier(hal::AccelerationStructureBarrier {
                usage: source_usage..destination_usage,
            });

            if tlas_present {
                cmd_buf_raw
                    .build_acceleration_structures(tlas_storage.len() as u32, tlas_descriptors);

                cmd_buf_raw.place_acceleration_structure_barrier(
                    hal::AccelerationStructureBarrier {
                        usage: hal::AccelerationStructureUses::BUILD_OUTPUT
                            ..hal::AccelerationStructureUses::SHADER_INPUT,
                    },
                );
            }
        }
        let scratch_mapping = unsafe {
            device
                .raw()
                .map_buffer(
                    &scratch_buffer,
                    0..max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                )
                .map_err(crate::device::DeviceError::from)?
        };
        device
            .pending_writes
            .lock()
            .as_mut()
            .unwrap()
            .temp_resources
            .push(TempResource::StagingBuffer(Arc::new(StagingBuffer {
                raw: Mutex::new(Some(scratch_buffer)),
                device: device.clone(),
                size: max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                info: ResourceInfo::new("Raytracing scratch buffer"),
                is_coherent: scratch_mapping.is_coherent,
            })));

        Ok(())
    }

    pub fn command_encoder_build_acceleration_structures<'a, A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
        tlas_iter: impl Iterator<Item = TlasPackage<'a>>,
    ) -> Result<(), BuildAccelerationStructureError> {
        profiling::scope!("CommandEncoder::build_acceleration_structures");

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let buffer_guard = hub.buffers.read();
        let blas_guard = hub.blas_s.read();
        let tlas_guard = hub.tlas_s.read();

        let device = &cmd_buf.device;

        let build_command_index = NonZeroU64::new(
            device
                .last_acceleration_structure_build_command_index
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
        )
        .unwrap();

        #[cfg(feature = "trace")]
        let trace_blas: Vec<crate::ray_tracing::TraceBlasBuildEntry> = blas_iter
            .map(|x| {
                let geometries = match x.geometries {
                    BlasGeometries::TriangleGeometries(triangle_geometries) => {
                        crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                            triangle_geometries
                                .map(|tg| crate::ray_tracing::TraceBlasTriangleGeometry {
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
                crate::ray_tracing::TraceBlasBuildEntry {
                    blas_id: x.blas_id,
                    geometries,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        let trace_tlas: Vec<crate::ray_tracing::TraceTlasPackage> = tlas_iter
            .map(|x: TlasPackage| {
                let instances = x
                    .instances
                    .map(|instance| {
                        instance.map(|instance| crate::ray_tracing::TraceTlasInstance {
                            blas_id: instance.blas_id,
                            transform: *instance.transform,
                            custom_index: instance.custom_index,
                            mask: instance.mask,
                        })
                    })
                    .collect();
                crate::ray_tracing::TraceTlasPackage {
                    tlas_id: x.tlas_id,
                    instances,
                    lowest_unmodified: x.lowest_unmodified,
                }
            })
            .collect();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.data.lock().as_mut().unwrap().commands {
            list.push(crate::device::trace::Command::BuildAccelerationStructures {
                blas: trace_blas.clone(),
                tlas: trace_tlas.clone(),
            });
        }

        #[cfg(feature = "trace")]
        let blas_iter = trace_blas.iter().map(|x| {
            let geometries = match &x.geometries {
                crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                    triangle_geometries,
                ) => {
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
                blas_id: x.blas_id,
                geometries,
            }
        });

        #[cfg(feature = "trace")]
        let tlas_iter = trace_tlas.iter().map(|x| {
            let instances = x.instances.iter().map(|instance| {
                instance
                    .as_ref()
                    .map(|instance| crate::ray_tracing::TlasInstance {
                        blas_id: instance.blas_id,
                        transform: &instance.transform,
                        custom_index: instance.custom_index,
                        mask: instance.mask,
                    })
            });
            TlasPackage {
                tlas_id: x.tlas_id,
                instances: Box::new(instances),
                lowest_unmodified: x.lowest_unmodified,
            }
        });

        let mut input_barriers = Vec::<hal::BufferBarrier<A>>::new();
        let mut buf_storage = Vec::<(
            Arc<Buffer<A>>,
            Option<PendingTransition<BufferUses>>,
            Option<(Arc<Buffer<A>>, Option<PendingTransition<BufferUses>>)>,
            Option<(Arc<Buffer<A>>, Option<PendingTransition<BufferUses>>)>,
            BlasTriangleGeometry,
            Option<Arc<Blas<A>>>,
        )>::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = Vec::<(&Blas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();
        for entry in blas_iter {
            let blas = cmd_buf_data
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            if blas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidBlas(entry.blas_id));
            }

            cmd_buf_data.blas_actions.push(BlasAction {
                id: entry.blas_id,
                kind: crate::ray_tracing::BlasActionKind::Build(build_command_index),
            });

            match entry.geometries {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    for (i, mesh) in triangle_geometries.enumerate() {
                        let size_desc = match &blas.sizes {
                            wgt::BlasGeometrySizeDescriptors::Triangles { desc } => desc,
                        };
                        if i >= size_desc.len() {
                            return Err(
                                BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                                    entry.blas_id,
                                ),
                            );
                        }
                        let size_desc = &size_desc[i];

                        if size_desc.flags != mesh.size.flags
                            || size_desc.vertex_count < mesh.size.vertex_count
                            || size_desc.vertex_format != mesh.size.vertex_format
                            || size_desc.index_count.is_none() != mesh.size.index_count.is_none()
                            || (size_desc.index_count.is_none()
                                || size_desc.index_count.unwrap() < mesh.size.index_count.unwrap())
                            || size_desc.index_format.is_none() != mesh.size.index_format.is_none()
                            || (size_desc.index_format.is_none()
                                || size_desc.index_format.unwrap()
                                    != mesh.size.index_format.unwrap())
                        {
                            return Err(
                                BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                                    entry.blas_id,
                                ),
                            );
                        }

                        if size_desc.index_count.is_some() && mesh.index_buffer.is_none() {
                            return Err(BuildAccelerationStructureError::MissingIndexBuffer(
                                entry.blas_id,
                            ));
                        }
                        let (vertex_buffer, vertex_pending) = cmd_buf_data
                            .trackers
                            .buffers
                            .set_single(
                                match buffer_guard.get(mesh.vertex_buffer) {
                                    Ok(buffer) => buffer,
                                    Err(_) => {
                                        return Err(BuildAccelerationStructureError::InvalidBuffer(
                                            mesh.vertex_buffer,
                                        ))
                                    }
                                },
                                BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                            )
                            .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                mesh.vertex_buffer,
                            ))?;
                        let index_data = if let Some(index_id) = mesh.index_buffer {
                            if mesh.index_buffer_offset.is_none()
                                || mesh.size.index_count.is_none()
                                || mesh.size.index_count.is_none()
                            {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        index_id,
                                    ),
                                );
                            }
                            let data = cmd_buf_data
                                .trackers
                                .buffers
                                .set_single(
                                    match buffer_guard.get(index_id) {
                                        Ok(buffer) => buffer,
                                        Err(_) => {
                                            return Err(
                                                BuildAccelerationStructureError::InvalidBuffer(
                                                    index_id,
                                                ),
                                            )
                                        }
                                    },
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            Some(data)
                        } else {
                            None
                        };
                        let transform_data = if let Some(transform_id) = mesh.transform_buffer {
                            if mesh.transform_buffer_offset.is_none() {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        transform_id,
                                    ),
                                );
                            }
                            let data = cmd_buf_data
                                .trackers
                                .buffers
                                .set_single(
                                    match buffer_guard.get(transform_id) {
                                        Ok(buffer) => buffer,
                                        Err(_) => {
                                            return Err(
                                                BuildAccelerationStructureError::InvalidBuffer(
                                                    transform_id,
                                                ),
                                            )
                                        }
                                    },
                                    BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    transform_id,
                                ))?;
                            Some(data)
                        } else {
                            None
                        };

                        buf_storage.push((
                            vertex_buffer,
                            vertex_pending,
                            index_data,
                            transform_data,
                            mesh,
                            None,
                        ))
                    }

                    if let Some(last) = buf_storage.last_mut() {
                        last.5 = Some(blas.clone());
                    }
                }
            }
        }

        let mut triangle_entries = Vec::<hal::AccelerationStructureTriangles<A>>::new();
        let snatch_guard = device.snatchable_lock.read();
        for buf in &mut buf_storage {
            let mesh = &buf.4;
            let vertex_buffer = {
                let vertex_buffer = buf.0.as_ref();
                let vertex_raw = vertex_buffer.raw.get(&snatch_guard).ok_or(
                    BuildAccelerationStructureError::InvalidBuffer(mesh.vertex_buffer),
                )?;
                if !vertex_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                        mesh.vertex_buffer,
                    ));
                }
                if let Some(barrier) = buf
                    .1
                    .take()
                    .map(|pending| pending.into_hal(vertex_buffer, &snatch_guard))
                {
                    input_barriers.push(barrier);
                }
                if vertex_buffer.size
                    < (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride
                {
                    return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                        mesh.vertex_buffer,
                        vertex_buffer.size,
                        (mesh.size.vertex_count + mesh.first_vertex) as u64 * mesh.vertex_stride,
                    ));
                }
                let vertex_buffer_offset = mesh.first_vertex as u64 * mesh.vertex_stride;
                cmd_buf_data.buffer_memory_init_actions.extend(
                    vertex_buffer.initialization_status.read().create_action(
                        buffer_guard.get(mesh.vertex_buffer).unwrap(),
                        vertex_buffer_offset
                            ..(vertex_buffer_offset
                                + mesh.size.vertex_count as u64 * mesh.vertex_stride),
                        MemoryInitKind::NeedsInitializedMemory,
                    ),
                );
                vertex_raw
            };
            let index_buffer = if let Some((ref mut index_buffer, ref mut index_pending)) = buf.2 {
                let index_id = mesh.index_buffer.as_ref().unwrap();
                let index_raw = index_buffer
                    .raw
                    .get(&snatch_guard)
                    .ok_or(BuildAccelerationStructureError::InvalidBuffer(*index_id))?;
                if !index_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                        *index_id,
                    ));
                }
                if let Some(barrier) = index_pending
                    .take()
                    .map(|pending| pending.into_hal(index_buffer, &snatch_guard))
                {
                    input_barriers.push(barrier);
                }
                let index_stride = match mesh.size.index_format.unwrap() {
                    wgt::IndexFormat::Uint16 => 2,
                    wgt::IndexFormat::Uint32 => 4,
                };
                if mesh.index_buffer_offset.unwrap() % index_stride != 0 {
                    return Err(BuildAccelerationStructureError::UnalignedIndexBufferOffset(
                        *index_id,
                    ));
                }
                let index_buffer_size = mesh.size.index_count.unwrap() as u64 * index_stride;

                if mesh.size.index_count.unwrap() % 3 != 0 {
                    return Err(BuildAccelerationStructureError::InvalidIndexCount(
                        *index_id,
                        mesh.size.index_count.unwrap(),
                    ));
                }
                if index_buffer.size
                    < mesh.size.index_count.unwrap() as u64 * index_stride
                        + mesh.index_buffer_offset.unwrap()
                {
                    return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                        *index_id,
                        index_buffer.size,
                        mesh.size.index_count.unwrap() as u64 * index_stride
                            + mesh.index_buffer_offset.unwrap(),
                    ));
                }

                cmd_buf_data.buffer_memory_init_actions.extend(
                    index_buffer.initialization_status.read().create_action(
                        match buffer_guard.get(*index_id) {
                            Ok(buffer) => buffer,
                            Err(_) => {
                                return Err(BuildAccelerationStructureError::InvalidBuffer(
                                    *index_id,
                                ))
                            }
                        },
                        mesh.index_buffer_offset.unwrap()
                            ..(mesh.index_buffer_offset.unwrap() + index_buffer_size),
                        MemoryInitKind::NeedsInitializedMemory,
                    ),
                );
                Some(index_raw)
            } else {
                None
            };
            let transform_buffer =
                if let Some((ref mut transform_buffer, ref mut transform_pending)) = buf.3 {
                    let transform_id = mesh.transform_buffer.as_ref().unwrap();
                    if mesh.transform_buffer_offset.is_none() {
                        return Err(BuildAccelerationStructureError::MissingAssociatedData(
                            *transform_id,
                        ));
                    }
                    let transform_raw = transform_buffer.raw.get(&snatch_guard).ok_or(
                        BuildAccelerationStructureError::InvalidBuffer(*transform_id),
                    )?;
                    if !transform_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                        return Err(BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                            *transform_id,
                        ));
                    }
                    if let Some(barrier) = transform_pending
                        .take()
                        .map(|pending| pending.into_hal(transform_buffer, &snatch_guard))
                    {
                        input_barriers.push(barrier);
                    }
                    if mesh.transform_buffer_offset.unwrap() % wgt::TRANSFORM_BUFFER_ALIGNMENT != 0
                    {
                        return Err(
                            BuildAccelerationStructureError::UnalignedTransformBufferOffset(
                                *transform_id,
                            ),
                        );
                    }
                    if transform_buffer.size < 48 + mesh.transform_buffer_offset.unwrap() {
                        return Err(BuildAccelerationStructureError::InsufficientBufferSize(
                            *transform_id,
                            transform_buffer.size,
                            48 + mesh.transform_buffer_offset.unwrap(),
                        ));
                    }
                    cmd_buf_data.buffer_memory_init_actions.extend(
                        transform_buffer.initialization_status.read().create_action(
                            buffer_guard.get(*transform_id).unwrap(),
                            mesh.transform_buffer_offset.unwrap()
                                ..(mesh.index_buffer_offset.unwrap() + 48),
                            MemoryInitKind::NeedsInitializedMemory,
                        ),
                    );
                    Some(transform_raw)
                } else {
                    None
                };

            let triangles = hal::AccelerationStructureTriangles {
                vertex_buffer: Some(vertex_buffer),
                vertex_format: mesh.size.vertex_format,
                first_vertex: mesh.first_vertex,
                vertex_count: mesh.size.vertex_count,
                vertex_stride: mesh.vertex_stride,
                indices: index_buffer.map(|index_buffer| {
                    hal::AccelerationStructureTriangleIndices::<A> {
                        format: mesh.size.index_format.unwrap(),
                        buffer: Some(index_buffer),
                        offset: mesh.index_buffer_offset.unwrap() as u32,
                        count: mesh.size.index_count.unwrap(),
                    }
                }),
                transform: transform_buffer.map(|transform_buffer| {
                    hal::AccelerationStructureTriangleTransform {
                        buffer: transform_buffer,
                        offset: mesh.transform_buffer_offset.unwrap() as u32,
                    }
                }),
                flags: mesh.size.flags,
            };
            triangle_entries.push(triangles);
            if let Some(blas) = buf.5.as_ref() {
                let scratch_buffer_offset = scratch_buffer_blas_size;
                scratch_buffer_blas_size += align_to(
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

        let mut tlas_lock_store = Vec::<(
            RwLockReadGuard<Option<A::Buffer>>,
            Option<TlasPackage>,
            Arc<Tlas<A>>,
        )>::new();

        for package in tlas_iter {
            let tlas = cmd_buf_data
                .trackers
                .tlas_s
                .add_single(&tlas_guard, package.tlas_id)
                .ok_or(BuildAccelerationStructureError::InvalidTlas(
                    package.tlas_id,
                ))?;
            tlas_lock_store.push((tlas.instance_buffer.read(), Some(package), tlas.clone()))
        }

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(
            &Tlas<A>,
            hal::AccelerationStructureEntries<A>,
            u64,
            Range<usize>,
        )>::new();
        let mut instance_buffer_staging_source = Vec::<u8>::new();

        for entry in &mut tlas_lock_store {
            let package = entry.1.take().unwrap();
            let tlas = &entry.2;
            if tlas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidTlas(
                    package.tlas_id,
                ));
            }

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
                        package.tlas_id,
                    ));
                }
                let blas = cmd_buf_data
                    .trackers
                    .blas_s
                    .add_single(&blas_guard, instance.blas_id)
                    .ok_or(BuildAccelerationStructureError::InvalidBlasForInstance(
                        instance.blas_id,
                    ))?;

                instance_buffer_staging_source
                    .extend(tlas_instance_into_bytes::<A>(&instance, blas.handle));

                instance_count += 1;

                dependencies.push(instance.blas_id);

                cmd_buf_data.blas_actions.push(BlasAction {
                    id: instance.blas_id,
                    kind: crate::ray_tracing::BlasActionKind::Use,
                });
            }

            cmd_buf_data.tlas_actions.push(TlasAction {
                id: package.tlas_id,
                kind: crate::ray_tracing::TlasActionKind::Build {
                    build_index: build_command_index,
                    dependencies,
                },
            });

            if instance_count > tlas.max_instance_count {
                return Err(BuildAccelerationStructureError::TlasInstanceCountExceeded(
                    package.tlas_id,
                    instance_count,
                    tlas.max_instance_count,
                ));
            }

            tlas_storage.push((
                tlas,
                hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
                    buffer: Some(entry.0.as_ref().unwrap()),
                    offset: 0,
                    count: instance_count,
                }),
                scratch_buffer_offset,
                first_byte_index..instance_buffer_staging_source.len(),
            ));
        }

        if max(scratch_buffer_blas_size, scratch_buffer_tlas_size) == 0 {
            return Ok(());
        }

        let scratch_buffer = unsafe {
            device
                .raw()
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("(wgpu) scratch buffer"),
                    size: max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                    usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH | BufferUses::MAP_WRITE,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .map_err(crate::device::DeviceError::from)?
        };
        let staging_buffer = if !instance_buffer_staging_source.is_empty() {
            unsafe {
                let staging_buffer = device
                    .raw()
                    .create_buffer(&hal::BufferDescriptor {
                        label: Some("(wgpu) instance staging buffer"),
                        size: instance_buffer_staging_source.len() as u64,
                        usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
                        memory_flags: hal::MemoryFlags::empty(),
                    })
                    .map_err(crate::device::DeviceError::from)?;
                let mapping = device
                    .raw()
                    .map_buffer(
                        &staging_buffer,
                        0..instance_buffer_staging_source.len() as u64,
                    )
                    .map_err(crate::device::DeviceError::from)?;
                ptr::copy_nonoverlapping(
                    instance_buffer_staging_source.as_ptr(),
                    mapping.ptr.as_ptr(),
                    instance_buffer_staging_source.len(),
                );
                device
                    .raw()
                    .unmap_buffer(&staging_buffer)
                    .map_err(crate::device::DeviceError::from)?;
                assert!(mapping.is_coherent);
                let buf = StagingBuffer {
                    raw: Mutex::new(Some(staging_buffer)),
                    device: device.clone(),
                    size: instance_buffer_staging_source.len() as u64,
                    info: ResourceInfo::new("Raytracing staging buffer"),
                    is_coherent: mapping.is_coherent,
                };
                let staging_fid = hub.staging_buffers.request();
                let stage_buf = staging_fid.init(buf);
                Some(stage_buf)
            }
        } else {
            None
        };

        let blas_descriptors =
            blas_storage
                .iter()
                .map(|&(blas, ref entries, ref scratch_buffer_offset)| {
                    if blas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                        log::info!("only rebuild implemented")
                    }
                    hal::BuildAccelerationStructureDescriptor {
                        entries,
                        mode: hal::AccelerationStructureBuildMode::Build,
                        flags: blas.flags,
                        source_acceleration_structure: None,
                        destination_acceleration_structure: blas.raw.as_ref().unwrap(),
                        scratch_buffer: &scratch_buffer,
                        scratch_buffer_offset: *scratch_buffer_offset,
                    }
                });

        let tlas_descriptors = tlas_storage.iter().map(
            |&(tlas, ref entries, ref scratch_buffer_offset, ref _range)| {
                if tlas.update_mode == wgt::AccelerationStructureUpdateMode::PreferUpdate {
                    log::info!("only rebuild implemented")
                }
                hal::BuildAccelerationStructureDescriptor {
                    entries,
                    mode: hal::AccelerationStructureBuildMode::Build,
                    flags: tlas.flags,
                    source_acceleration_structure: None,
                    destination_acceleration_structure: tlas.raw.as_ref().unwrap(),
                    scratch_buffer: &scratch_buffer,
                    scratch_buffer_offset: *scratch_buffer_offset,
                }
            },
        );

        let scratch_buffer_barrier = hal::BufferBarrier::<A> {
            buffer: &scratch_buffer,
            usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH
                ..BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
        };

        let mut lock_vec = Vec::<Option<RwLockReadGuard<Option<<A>::Buffer>>>>::new();

        for tlas in &tlas_storage {
            let size = (tlas.3.end - tlas.3.start) as u64;
            lock_vec.push(if size == 0 {
                None
            } else {
                Some(tlas.0.instance_buffer.read())
            })
        }

        let instance_buffer_barriers = lock_vec.iter().filter_map(|lock| {
            lock.as_ref().map(|lock| hal::BufferBarrier::<A> {
                buffer: lock.as_ref().unwrap(),
                usage: BufferUses::COPY_DST..BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            })
        });

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf_data.encoder.open()?;

        unsafe {
            cmd_buf_raw.transition_buffers(input_barriers.into_iter());
        }

        if blas_present {
            unsafe {
                cmd_buf_raw.place_acceleration_structure_barrier(
                    hal::AccelerationStructureBarrier {
                        usage: hal::AccelerationStructureUses::BUILD_INPUT
                            ..hal::AccelerationStructureUses::BUILD_OUTPUT,
                    },
                );

                cmd_buf_raw
                    .build_acceleration_structures(blas_storage.len() as u32, blas_descriptors);
            }
        }

        if blas_present && tlas_present {
            unsafe {
                cmd_buf_raw.transition_buffers(iter::once(scratch_buffer_barrier));
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

        if tlas_present {
            unsafe {
                if let Some(ref staging_buffer) = staging_buffer {
                    cmd_buf_raw.transition_buffers(iter::once(hal::BufferBarrier::<A> {
                        buffer: staging_buffer.raw.lock().as_ref().unwrap(),
                        usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                    }));
                }
            }

            for &(tlas, ref _entries, ref _scratch_buffer_offset, ref range) in &tlas_storage {
                let size = (range.end - range.start) as u64;
                if size == 0 {
                    continue;
                }
                unsafe {
                    let temp = hal::BufferCopy {
                        src_offset: range.start as u64,
                        dst_offset: 0,
                        size: NonZeroU64::new(size).unwrap(),
                    };
                    cmd_buf_raw.copy_buffer_to_buffer(
                        staging_buffer
                            .as_ref()
                            .unwrap()
                            .raw
                            .lock()
                            .as_ref()
                            .unwrap(),
                        tlas.instance_buffer.read().as_ref().unwrap(),
                        iter::once(temp),
                    );
                }
            }

            unsafe {
                cmd_buf_raw.transition_buffers(instance_buffer_barriers);

                cmd_buf_raw
                    .build_acceleration_structures(tlas_storage.len() as u32, tlas_descriptors);

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
                    .as_mut()
                    .unwrap()
                    .temp_resources
                    .push(TempResource::StagingBuffer(staging_buffer));
            }
        }
        let scratch_mapping = unsafe {
            device
                .raw()
                .map_buffer(
                    &scratch_buffer,
                    0..max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                )
                .map_err(crate::device::DeviceError::from)?
        };

        let buf = StagingBuffer {
            raw: Mutex::new(Some(scratch_buffer)),
            device: device.clone(),
            size: max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
            info: ResourceInfo::new("Ratracing scratch buffer"),
            is_coherent: scratch_mapping.is_coherent,
        };
        let staging_fid = hub.staging_buffers.request();
        let stage_buf = staging_fid.init(buf);

        device
            .pending_writes
            .lock()
            .as_mut()
            .unwrap()
            .temp_resources
            .push(TempResource::StagingBuffer(stage_buf));

        Ok(())
    }
}

impl<A: HalApi> BakedCommands<A> {
    // makes sure a blas is build before it is used
    pub(crate) fn validate_blas_actions(
        &mut self,
        blas_guard: &mut Storage<Blas<A>>,
    ) -> Result<(), ValidateBlasActionsError> {
        profiling::scope!("CommandEncoder::[submission]::validate_blas_actions");
        let mut built = FastHashSet::default();
        for action in self.blas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::BlasActionKind::Build(id) => {
                    built.insert(action.id);
                    let blas = blas_guard
                        .get(action.id)
                        .map_err(|_| ValidateBlasActionsError::InvalidBlas(action.id))?;
                    *blas.built_index.write() = Some(id);
                }
                crate::ray_tracing::BlasActionKind::Use => {
                    if !built.contains(&action.id) {
                        let blas = blas_guard
                            .get(action.id)
                            .map_err(|_| ValidateBlasActionsError::InvalidBlas(action.id))?;
                        if (*blas.built_index.read()).is_none() {
                            return Err(ValidateBlasActionsError::UsedUnbuilt(action.id));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // makes sure a tlas is build before it is used
    pub(crate) fn validate_tlas_actions(
        &mut self,
        blas_guard: &Storage<Blas<A>>,
        tlas_guard: &mut Storage<Tlas<A>>,
    ) -> Result<(), ValidateTlasActionsError> {
        profiling::scope!("CommandEncoder::[submission]::validate_tlas_actions");
        for action in self.tlas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::TlasActionKind::Build {
                    build_index,
                    dependencies,
                } => {
                    let tlas = tlas_guard
                        .get(action.id)
                        .map_err(|_| ValidateTlasActionsError::InvalidTlas(action.id))?;

                    *tlas.built_index.write() = Some(build_index);
                    *tlas.dependencies.write() = dependencies;
                }
                crate::ray_tracing::TlasActionKind::Use => {
                    let tlas = tlas_guard
                        .get(action.id)
                        .map_err(|_| ValidateTlasActionsError::InvalidTlas(action.id))?;

                    let tlas_build_index = tlas.built_index.read();
                    let dependencies = tlas.dependencies.read();

                    if (*tlas_build_index).is_none() {
                        return Err(ValidateTlasActionsError::UsedUnbuilt(action.id));
                    }
                    for dependency in dependencies.deref() {
                        let blas = blas_guard.get(*dependency).map_err(|_| {
                            ValidateTlasActionsError::InvalidBlas(*dependency, action.id)
                        })?;
                        let blas_build_index = *blas.built_index.read();
                        if blas_build_index.is_none() {
                            return Err(ValidateTlasActionsError::UsedUnbuilt(action.id));
                        }
                        if blas_build_index.unwrap() > tlas_build_index.unwrap() {
                            return Err(ValidateTlasActionsError::BlasNewerThenTlas(
                                *dependency,
                                action.id,
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
