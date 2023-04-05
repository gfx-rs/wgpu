use crate::{
    command::CommandBuffer,
    device::queue::TempResource,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Storage, Token},
    id::{BlasId, CommandEncoderId, TlasId},
    init_tracker::MemoryInitKind,
    ray_tracing::{
        tlas_instance_into_bytes, BlasAction, BlasBuildEntry, BlasGeometries,
        BuildAccelerationStructureError, TlasAction, TlasBuildEntry, TlasInstance, TlasPackage,
        TraceTlasInstance, ValidateBlasActionsError, ValidateTlasActionsError,
    },
    resource::{Blas, StagingBuffer, Tlas},
    FastHashMap, FastHashSet,
};

use hal::{CommandEncoder, Device};
use wgt::BufferUsages;

use std::{cmp::max, iter, num::NonZeroU64, ops::Range, ptr};

use super::BakedCommands;

// TODO: a lot
impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_build_acceleration_structures_unsafe_tlas<'a, A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a>>,
        tlas_iter: impl Iterator<Item = TlasBuildEntry>,
    ) -> Result<(), BuildAccelerationStructureError> {
        profiling::scope!("CommandEncoder::build_acceleration_structures_unsafe_tlas");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (blas_guard, mut token) = hub.blas_s.read(&mut token);
        let (tlas_guard, _) = hub.tlas_s.read(&mut token);

        let device = &mut device_guard[cmd_buf.device_id.value];

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
        let mut trace_tlas: Vec<TlasBuildEntry> = tlas_iter.collect();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
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
        let blas_iter = (&trace_blas).into_iter().map(|x| {
            let geometries = match &x.geometries {
                crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                    triangle_geometries,
                ) => {
                    let iter = triangle_geometries.into_iter().map(|tg| {
                        crate::ray_tracing::BlasTriangleGeometry {
                            size: &tg.size,
                            vertex_buffer: tg.vertex_buffer,
                            index_buffer: tg.index_buffer,
                            transform_buffer: tg.transform_buffer,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                            transform_buffer_offset: tg.transform_buffer_offset,
                        }
                    });
                    BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            BlasBuildEntry {
                blas_id: x.blas_id,
                geometries: geometries,
            }
        });

        #[cfg(feature = "trace")]
        let tlas_iter = (&mut trace_tlas).iter();

        let mut input_barriers = Vec::<hal::BufferBarrier<A>>::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = Vec::<(&Blas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();

        for entry in blas_iter {
            let blas = cmd_buf
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            if blas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidBlas(entry.blas_id));
            }

            cmd_buf.blas_actions.push(BlasAction {
                id: entry.blas_id,
                kind: crate::ray_tracing::BlasActionKind::Build(build_command_index),
            });

            match entry.geometries {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let mut triangle_entries = Vec::<hal::AccelerationStructureTriangles<A>>::new();

                    for (i, mesh) in triangle_geometries.enumerate() {
                        let size_desc = match &blas.sizes {
                            &wgt::BlasGeometrySizeDescriptors::Triangles { ref desc } => desc,
                            // _ => {
                            //     return Err(
                            //         BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                            //             entry.blas_id,
                            //         ),
                            //     )
                            // }
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

                        let vertex_buffer = {
                            let (vertex_buffer, vertex_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    mesh.vertex_buffer,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    mesh.vertex_buffer,
                                ))?;
                            let vertex_raw = vertex_buffer.raw.as_ref().ok_or(
                                BuildAccelerationStructureError::InvalidBuffer(mesh.vertex_buffer),
                            )?;
                            if !vertex_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        mesh.vertex_buffer,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                vertex_pending.map(|pending| pending.into_hal(vertex_buffer))
                            {
                                input_barriers.push(barrier);
                            }
                            if mesh.size.vertex_count as i64 - mesh.first_vertex as i64 <= 0 {
                                return Err(BuildAccelerationStructureError::EmptyVertexBuffer(
                                    mesh.vertex_buffer,
                                ));
                            }
                            if vertex_buffer.size
                                < (mesh.size.vertex_count + mesh.first_vertex) as u64
                                    * mesh.vertex_stride
                            {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        mesh.vertex_buffer,
                                        vertex_buffer.size,
                                        (mesh.size.vertex_count + mesh.first_vertex) as u64
                                            * mesh.vertex_stride,
                                    ),
                                );
                            }
                            let vertex_buffer_offset =
                                mesh.first_vertex as u64 * mesh.vertex_stride;
                            cmd_buf.buffer_memory_init_actions.extend(
                                vertex_buffer.initialization_status.create_action(
                                    mesh.vertex_buffer,
                                    vertex_buffer_offset
                                        ..(vertex_buffer_offset
                                            + mesh.size.vertex_count as u64 * mesh.vertex_stride),
                                    MemoryInitKind::NeedsInitializedMemory,
                                ),
                            );
                            vertex_raw
                        };
                        let index_buffer = if let Some(index_id) = mesh.index_buffer {
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
                            let (index_buffer, index_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    index_id,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            let index_raw = index_buffer
                                .raw
                                .as_ref()
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            if !index_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        index_id,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                index_pending.map(|pending| pending.into_hal(index_buffer))
                            {
                                input_barriers.push(barrier);
                            }
                            let index_stride = match mesh.size.index_format.unwrap() {
                                wgt::IndexFormat::Uint16 => 2,
                                wgt::IndexFormat::Uint32 => 4,
                            };
                            let index_buffer_size =
                                mesh.size.index_count.unwrap() as u64 * index_stride;

                            if mesh.size.index_count.unwrap() as u64
                                - mesh.index_buffer_offset.unwrap() / index_stride
                                < 3
                            {
                                return Err(BuildAccelerationStructureError::EmptyIndexBuffer(
                                    index_id,
                                ));
                            }
                            if mesh.size.index_count.unwrap() % 3 != 0 {
                                return Err(BuildAccelerationStructureError::InvalidIndexCount(
                                    index_id,
                                    mesh.size.index_count.unwrap(),
                                ));
                            }
                            if index_buffer.size
                                < mesh.size.index_count.unwrap() as u64 * index_stride
                                    + mesh.index_buffer_offset.unwrap()
                            {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        index_id,
                                        index_buffer.size,
                                        mesh.size.index_count.unwrap() as u64 * index_stride
                                            + mesh.index_buffer_offset.unwrap(),
                                    ),
                                );
                            }

                            cmd_buf.buffer_memory_init_actions.extend(
                                index_buffer.initialization_status.create_action(
                                    index_id,
                                    mesh.index_buffer_offset.unwrap()
                                        ..(mesh.index_buffer_offset.unwrap() + index_buffer_size),
                                    MemoryInitKind::NeedsInitializedMemory,
                                ),
                            );
                            Some(index_raw)
                        } else {
                            None
                        };
                        let transform_buffer = if let Some(transform_id) = mesh.transform_buffer {
                            if mesh.transform_buffer_offset.is_none() {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        transform_id,
                                    ),
                                );
                            }
                            let (transform_buffer, transform_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    transform_id,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    transform_id,
                                ))?;
                            let transform_raw = transform_buffer.raw.as_ref().ok_or(
                                BuildAccelerationStructureError::InvalidBuffer(transform_id),
                            )?;
                            if !transform_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        transform_id,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                transform_pending.map(|pending| pending.into_hal(transform_buffer))
                            {
                                input_barriers.push(barrier);
                            }

                            if transform_buffer.size < 48 + mesh.transform_buffer_offset.unwrap() {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        transform_id,
                                        transform_buffer.size,
                                        48 + mesh.transform_buffer_offset.unwrap(),
                                    ),
                                );
                            }
                            cmd_buf.buffer_memory_init_actions.extend(
                                transform_buffer.initialization_status.create_action(
                                    transform_id,
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
                                hal::AccelerationStructureTriangleIndices {
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
                    }

                    let scratch_buffer_offset = scratch_buffer_blas_size;
                    scratch_buffer_blas_size += blas.size_info.build_scratch_size; // TODO Alignment

                    blas_storage.push((
                        blas,
                        hal::AccelerationStructureEntries::Triangles(triangle_entries),
                        scratch_buffer_offset,
                    ))
                }
            }
        }

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(&Tlas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();

        for entry in tlas_iter {
            let instance_buffer = {
                let (instance_buffer, instance_pending) = cmd_buf
                    .trackers
                    .buffers
                    .set_single(
                        &*buffer_guard,
                        entry.instance_buffer_id,
                        hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    )
                    .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                        entry.instance_buffer_id,
                    ))?;
                let instance_raw = instance_buffer.raw.as_ref().ok_or(
                    BuildAccelerationStructureError::InvalidBuffer(entry.instance_buffer_id),
                )?;
                if !instance_buffer.usage.contains(BufferUsages::TLAS_INPUT) {
                    return Err(BuildAccelerationStructureError::MissingTlasInputUsageFlag(
                        entry.instance_buffer_id,
                    ));
                }
                if let Some(barrier) =
                    instance_pending.map(|pending| pending.into_hal(instance_buffer))
                {
                    input_barriers.push(barrier);
                }
                instance_raw
            };

            let tlas = cmd_buf
                .trackers
                .tlas_s
                .add_single(&tlas_guard, entry.tlas_id)
                .ok_or(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id))?;

            if tlas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id));
            }

            cmd_buf.tlas_actions.push(TlasAction {
                id: entry.tlas_id,
                kind: crate::ray_tracing::TlasActionKind::Build {
                    build_index: build_command_index,
                    dependencies: Vec::new(),
                },
            });

            let scratch_buffer_offset = scratch_buffer_tlas_size;
            scratch_buffer_tlas_size += tlas.size_info.build_scratch_size; // TODO Alignment

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
                .raw
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
                        mode: hal::AccelerationStructureBuildMode::Build, // TODO
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
                        mode: hal::AccelerationStructureBuildMode::Build, // TODO
                        flags: tlas.flags,
                        source_acceleration_structure: None,
                        destination_acceleration_structure: tlas.raw.as_ref().unwrap(),
                        scratch_buffer: &scratch_buffer,
                        scratch_buffer_offset: *scratch_buffer_offset,
                    }
                });

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf.encoder.open();
        unsafe {
            cmd_buf_raw.transition_buffers(input_barriers.into_iter());

            if blas_present {
                cmd_buf_raw
                    .build_acceleration_structures(blas_storage.len() as u32, blas_descriptors);
            }

            if blas_present && tlas_present {
                cmd_buf_raw.transition_buffers(iter::once(scratch_buffer_barrier));
            }

            if tlas_present {
                cmd_buf_raw
                    .build_acceleration_structures(tlas_storage.len() as u32, tlas_descriptors);
            }
        }

        device
            .pending_writes
            .temp_resources
            .push(TempResource::Buffer(scratch_buffer));

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
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (blas_guard, mut token) = hub.blas_s.read(&mut token);
        let (tlas_guard, _) = hub.tlas_s.read(&mut token);

        let device = &mut device_guard[cmd_buf.device_id.value];

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
                        if let Some(instance) = instance {
                            Some(TraceTlasInstance {
                                blas_id: instance.blas_id,
                                transform: instance.transform.clone(),
                                custom_index: instance.custom_index,
                                mask: instance.mask,
                            })
                        } else {
                            None
                        }
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
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(crate::device::trace::Command::BuildAccelerationStructures {
                blas: trace_blas.clone(),
                tlas: trace_tlas.clone(),
            });
        }

        #[cfg(feature = "trace")]
        let blas_iter = (&trace_blas).into_iter().map(|x| {
            let geometries = match &x.geometries {
                crate::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                    triangle_geometries,
                ) => {
                    let iter = triangle_geometries.into_iter().map(|tg| {
                        crate::ray_tracing::BlasTriangleGeometry {
                            size: &tg.size,
                            vertex_buffer: tg.vertex_buffer,
                            index_buffer: tg.index_buffer,
                            transform_buffer: tg.transform_buffer,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                            transform_buffer_offset: tg.transform_buffer_offset,
                        }
                    });
                    BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            BlasBuildEntry {
                blas_id: x.blas_id,
                geometries: geometries,
            }
        });

        // dbg!(&trace_tlas);

        #[cfg(feature = "trace")]
        let tlas_iter = (&trace_tlas).into_iter().map(|x| {
            let instances = x.instances.iter().map(|instance| {
                if let Some(instance) = instance {
                    Some(TlasInstance {
                        blas_id: instance.blas_id,
                        transform: &instance.transform,
                        custom_index: instance.custom_index,
                        mask: instance.mask,
                    })
                } else {
                    None
                }
            });
            TlasPackage {
                tlas_id: x.tlas_id,
                instances: Box::new(instances),
                lowest_unmodified: x.lowest_unmodified,
            }
        });

        let mut input_barriers = Vec::<hal::BufferBarrier<A>>::new();

        let mut scratch_buffer_blas_size = 0;
        let mut blas_storage = Vec::<(&Blas<A>, hal::AccelerationStructureEntries<A>, u64)>::new();

        for entry in blas_iter {
            let blas = cmd_buf
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            if blas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidBlas(entry.blas_id));
            }

            cmd_buf.blas_actions.push(BlasAction {
                id: entry.blas_id,
                kind: crate::ray_tracing::BlasActionKind::Build(build_command_index),
            });

            match entry.geometries {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let mut triangle_entries = Vec::<hal::AccelerationStructureTriangles<A>>::new();

                    for (i, mesh) in triangle_geometries.enumerate() {
                        let size_desc = match &blas.sizes {
                            &wgt::BlasGeometrySizeDescriptors::Triangles { ref desc } => desc,
                            // _ => {
                            //     return Err(
                            //         BuildAccelerationStructureError::IncompatibleBlasBuildSizes(
                            //             entry.blas_id,
                            //         ),
                            //     )
                            // }
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

                        let vertex_buffer = {
                            let (vertex_buffer, vertex_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    mesh.vertex_buffer,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    mesh.vertex_buffer,
                                ))?;
                            let vertex_raw = vertex_buffer.raw.as_ref().ok_or(
                                BuildAccelerationStructureError::InvalidBuffer(mesh.vertex_buffer),
                            )?;
                            if !vertex_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        mesh.vertex_buffer,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                vertex_pending.map(|pending| pending.into_hal(vertex_buffer))
                            {
                                input_barriers.push(barrier);
                            }
                            if mesh.size.vertex_count as i64 - mesh.first_vertex as i64 <= 0 {
                                return Err(BuildAccelerationStructureError::EmptyVertexBuffer(
                                    mesh.vertex_buffer,
                                ));
                            }
                            if vertex_buffer.size
                                < (mesh.size.vertex_count + mesh.first_vertex) as u64
                                    * mesh.vertex_stride
                            {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        mesh.vertex_buffer,
                                        vertex_buffer.size,
                                        (mesh.size.vertex_count + mesh.first_vertex) as u64
                                            * mesh.vertex_stride,
                                    ),
                                );
                            }
                            let vertex_buffer_offset =
                                mesh.first_vertex as u64 * mesh.vertex_stride;
                            cmd_buf.buffer_memory_init_actions.extend(
                                vertex_buffer.initialization_status.create_action(
                                    mesh.vertex_buffer,
                                    vertex_buffer_offset
                                        ..(vertex_buffer_offset
                                            + mesh.size.vertex_count as u64 * mesh.vertex_stride),
                                    MemoryInitKind::NeedsInitializedMemory,
                                ),
                            );
                            vertex_raw
                        };
                        let index_buffer = if let Some(index_id) = mesh.index_buffer {
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
                            let (index_buffer, index_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    index_id,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            let index_raw = index_buffer
                                .raw
                                .as_ref()
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(index_id))?;
                            if !index_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        index_id,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                index_pending.map(|pending| pending.into_hal(index_buffer))
                            {
                                input_barriers.push(barrier);
                            }
                            let index_stride = match mesh.size.index_format.unwrap() {
                                wgt::IndexFormat::Uint16 => 2,
                                wgt::IndexFormat::Uint32 => 4,
                            };
                            let index_buffer_size =
                                mesh.size.index_count.unwrap() as u64 * index_stride;

                            if mesh.size.index_count.unwrap() as u64
                                - mesh.index_buffer_offset.unwrap() / index_stride
                                < 3
                            {
                                return Err(BuildAccelerationStructureError::EmptyIndexBuffer(
                                    index_id,
                                ));
                            }
                            if mesh.size.index_count.unwrap() % 3 != 0 {
                                return Err(BuildAccelerationStructureError::InvalidIndexCount(
                                    index_id,
                                    mesh.size.index_count.unwrap(),
                                ));
                            }
                            if index_buffer.size
                                < mesh.size.index_count.unwrap() as u64 * index_stride
                                    + mesh.index_buffer_offset.unwrap()
                            {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        index_id,
                                        index_buffer.size,
                                        mesh.size.index_count.unwrap() as u64 * index_stride
                                            + mesh.index_buffer_offset.unwrap(),
                                    ),
                                );
                            }

                            cmd_buf.buffer_memory_init_actions.extend(
                                index_buffer.initialization_status.create_action(
                                    index_id,
                                    mesh.index_buffer_offset.unwrap()
                                        ..(mesh.index_buffer_offset.unwrap() + index_buffer_size),
                                    MemoryInitKind::NeedsInitializedMemory,
                                ),
                            );
                            Some(index_raw)
                        } else {
                            None
                        };
                        let transform_buffer = if let Some(transform_id) = mesh.transform_buffer {
                            if mesh.transform_buffer_offset.is_none() {
                                return Err(
                                    BuildAccelerationStructureError::MissingAssociatedData(
                                        transform_id,
                                    ),
                                );
                            }
                            let (transform_buffer, transform_pending) = cmd_buf
                                .trackers
                                .buffers
                                .set_single(
                                    &*buffer_guard,
                                    transform_id,
                                    hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                                )
                                .ok_or(BuildAccelerationStructureError::InvalidBuffer(
                                    transform_id,
                                ))?;
                            let transform_raw = transform_buffer.raw.as_ref().ok_or(
                                BuildAccelerationStructureError::InvalidBuffer(transform_id),
                            )?;
                            if !transform_buffer.usage.contains(BufferUsages::BLAS_INPUT) {
                                return Err(
                                    BuildAccelerationStructureError::MissingBlasInputUsageFlag(
                                        transform_id,
                                    ),
                                );
                            }
                            if let Some(barrier) =
                                transform_pending.map(|pending| pending.into_hal(transform_buffer))
                            {
                                input_barriers.push(barrier);
                            }

                            if transform_buffer.size < 48 + mesh.transform_buffer_offset.unwrap() {
                                return Err(
                                    BuildAccelerationStructureError::InsufficientBufferSize(
                                        transform_id,
                                        transform_buffer.size,
                                        48 + mesh.transform_buffer_offset.unwrap(),
                                    ),
                                );
                            }
                            cmd_buf.buffer_memory_init_actions.extend(
                                transform_buffer.initialization_status.create_action(
                                    transform_id,
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
                                hal::AccelerationStructureTriangleIndices {
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
                    }

                    let scratch_buffer_offset = scratch_buffer_blas_size;
                    scratch_buffer_blas_size += blas.size_info.build_scratch_size; // TODO Alignment

                    blas_storage.push((
                        blas,
                        hal::AccelerationStructureEntries::Triangles(triangle_entries),
                        scratch_buffer_offset,
                    ))
                }
            }
        }

        let mut scratch_buffer_tlas_size = 0;
        let mut tlas_storage = Vec::<(
            &Tlas<A>,
            hal::AccelerationStructureEntries<A>,
            u64,
            Range<usize>,
        )>::new();
        let mut instance_buffer_staging_source = Vec::<u8>::new();

        for entry in tlas_iter {
            let tlas = cmd_buf
                .trackers
                .tlas_s
                .add_single(&tlas_guard, entry.tlas_id)
                .ok_or(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id))?;

            if tlas.raw.is_none() {
                return Err(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id));
            }

            let scratch_buffer_offset = scratch_buffer_tlas_size;
            scratch_buffer_tlas_size += tlas.size_info.build_scratch_size; // TODO Alignment

            let first_byte_index = instance_buffer_staging_source.len();

            let mut dependencies = Vec::new();

            let mut instance_count = 0;
            for instance in entry.instances {
                if let Some(instance) = instance {
                    // TODO validation
                    let blas = cmd_buf
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

                    cmd_buf.blas_actions.push(BlasAction {
                        id: instance.blas_id,
                        kind: crate::ray_tracing::BlasActionKind::Use,
                    });
                }
            }

            cmd_buf.tlas_actions.push(TlasAction {
                id: entry.tlas_id,
                kind: crate::ray_tracing::TlasActionKind::Build {
                    build_index: build_command_index,
                    dependencies,
                },
            });

            tlas_storage.push((
                tlas,
                hal::AccelerationStructureEntries::Instances(hal::AccelerationStructureInstances {
                    buffer: Some(tlas.instance_buffer.as_ref().unwrap()),
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
                .raw
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("(wgpu) scratch buffer"),
                    size: max(scratch_buffer_blas_size, scratch_buffer_tlas_size),
                    usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .map_err(crate::device::DeviceError::from)?
        };

        let staging_buffer = if instance_buffer_staging_source.len() > 0 {
            unsafe {
                let staging_buffer = device
                    .raw
                    .create_buffer(&hal::BufferDescriptor {
                        label: Some("(wgpu) instance staging buffer"),
                        size: instance_buffer_staging_source.len() as u64,
                        usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
                        memory_flags: hal::MemoryFlags::empty(),
                    })
                    .map_err(crate::device::DeviceError::from)?;

                let mapping = device
                    .raw
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
                    .raw
                    .unmap_buffer(&staging_buffer)
                    .map_err(crate::device::DeviceError::from)?;
                assert!(mapping.is_coherent);

                Some(staging_buffer)
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
                        mode: hal::AccelerationStructureBuildMode::Build, // TODO
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
                    mode: hal::AccelerationStructureBuildMode::Build, // TODO
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
            usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH
                ..hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
        };

        let instance_buffer_barriers = tlas_storage.iter().filter_map(
            |&(tlas, ref _entries, ref _scratch_buffer_offset, ref range)| {
                let size = (range.end - range.start) as u64;
                if size == 0 {
                    None
                } else {
                    Some(hal::BufferBarrier::<A> {
                        buffer: tlas.instance_buffer.as_ref().unwrap(),
                        usage: hal::BufferUses::COPY_DST
                            ..hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    })
                }
            },
        );

        let blas_present = !blas_storage.is_empty();
        let tlas_present = !tlas_storage.is_empty();

        let cmd_buf_raw = cmd_buf.encoder.open();

        unsafe {
            cmd_buf_raw.transition_buffers(input_barriers.into_iter());
        }

        if blas_present {
            unsafe {
                cmd_buf_raw
                    .build_acceleration_structures(blas_storage.len() as u32, blas_descriptors);
            }
        }

        if blas_present && tlas_present {
            unsafe {
                cmd_buf_raw.transition_buffers(iter::once(scratch_buffer_barrier));
            }
        }

        if tlas_present {
            unsafe {
                cmd_buf_raw.transition_buffers(iter::once(hal::BufferBarrier::<A> {
                    buffer: staging_buffer.as_ref().unwrap(),
                    usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                }));
            }

            for (tlas, _entries, _scratch_buffer_offset, range) in &tlas_storage {
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
                        staging_buffer.as_ref().unwrap(),
                        tlas.instance_buffer.as_ref().unwrap(),
                        iter::once(temp),
                    );
                }
            }

            unsafe {
                cmd_buf_raw.transition_buffers(instance_buffer_barriers);
            }

            unsafe {
                cmd_buf_raw
                    .build_acceleration_structures(tlas_storage.len() as u32, tlas_descriptors);
            }

            device
                .pending_writes
                .temp_resources
                .push(TempResource::Buffer(staging_buffer.unwrap()));
        }

        device
            .pending_writes
            .temp_resources
            .push(TempResource::Buffer(scratch_buffer));

        Ok(())
    }
}

impl<A: HalApi> BakedCommands<A> {
    // makes sure a blas is build before it is used
    pub(crate) fn validate_blas_actions(
        &mut self,
        blas_guard: &mut Storage<Blas<A>, BlasId>,
    ) -> Result<(), ValidateBlasActionsError> {
        let mut built = FastHashSet::default();
        for action in self.blas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::BlasActionKind::Build(id) => {
                    built.insert(action.id);
                    let blas = blas_guard
                        .get_mut(action.id)
                        .map_err(|_| ValidateBlasActionsError::InvalidBlas(action.id))?;
                    blas.built_index = Some(id);
                }
                crate::ray_tracing::BlasActionKind::Use => {
                    if !built.contains(&action.id) {
                        let blas = blas_guard
                            .get(action.id)
                            .map_err(|_| ValidateBlasActionsError::InvalidBlas(action.id))?;
                        if blas.built_index == None {
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
        blas_guard: &Storage<Blas<A>, BlasId>,
        tlas_guard: &mut Storage<Tlas<A>, TlasId>,
    ) -> Result<(), ValidateTlasActionsError> {
        for action in self.tlas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::TlasActionKind::Build {
                    build_index,
                    dependencies,
                } => {
                    let tlas = tlas_guard
                        .get_mut(action.id)
                        .map_err(|_| ValidateTlasActionsError::InvalidTlas(action.id))?;

                    tlas.built_index = Some(build_index);
                    tlas.dependencies = dependencies;
                }
                crate::ray_tracing::TlasActionKind::Use => {
                    let tlas = tlas_guard
                        .get(action.id)
                        .map_err(|_| ValidateTlasActionsError::InvalidTlas(action.id))?;

                    let tlas_build_index = tlas.built_index;
                    let dependencies = &tlas.dependencies;

                    if tlas_build_index == None {
                        return Err(ValidateTlasActionsError::UsedUnbuilt(action.id));
                    }
                    for dependency in dependencies {
                        let blas = blas_guard.get(*dependency).map_err(|_| {
                            ValidateTlasActionsError::InvalidBlas(*dependency, action.id)
                        })?;
                        let blas_build_index = blas.built_index;
                        if blas_build_index == None {
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
