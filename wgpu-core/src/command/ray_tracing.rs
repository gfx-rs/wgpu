#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::{clear_texture, CommandBuffer, CommandEncoderError},
    conv,
    device::{queue::TempResource, Device, DeviceError, MissingDownlevelFlags},
    error::{ErrorFormatter, PrettyError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Storage, Token},
    id::{self, BlasId, BufferId, CommandEncoderId, TlasId},
    init_tracker::{
        has_copy_partial_init_tracker_coverage, MemoryInitKind, TextureInitRange,
        TextureInitTrackerAction,
    },
    ray_tracing::{
        BlasBuildEntry, BlasBuildEntryStorage, BlasGeometriesStorage,
        BuildAccelerationStructureError, TlasBuildEntry,
    },
    resource::{self, Blas, Texture, TextureErrorDimension, Tlas},
    track::TextureSelector,
    LabelHelpers, LifeGuard, Stored,
};

use arrayvec::ArrayVec;
use hal::{
    AccelerationStructureInstances, AccelerationStructureTriangleIndices,
    AccelerationStructureTriangleTransform, CommandEncoder as _, Device as _,
};
use thiserror::Error;
use wgt::{BufferAddress, BufferUsages, Extent3d, TextureUsages};

use std::{borrow::Borrow, cmp::max, iter};

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

        let blas_storage: Vec<_> = blas_iter
            .into_iter()
            .map(|e| {
                let geometries = match e.geometries {
                    crate::ray_tracing::BlasGeometries::TriangleGeometries(triangle_geometries) => {
                        BlasGeometriesStorage::TriangleGeometries(triangle_geometries.collect())
                    }
                };
                BlasBuildEntryStorage {
                    blas_id: e.blas_id,
                    geometries,
                }
            })
            .collect();

        let tlas_storage: Vec<_> = tlas_iter.collect();

        // #[cfg(feature = "trace")]
        // if let Some(ref mut list) = cmd_buf.commands {
        //     list.push(TraceCommand::CopyBufferToBuffer {
        //         src: source,
        //         src_offset: source_offset,
        //         dst: destination,
        //         dst_offset: destination_offset,
        //         size,
        //     });
        // }

        let mut input_barriers = Vec::<hal::BufferBarrier<A>>::new();

        let mut triangle_geometry_storage =
            Vec::<Vec<hal::AccelerationStructureTriangles<A>>>::new();

        for entry in &blas_storage {
            match &entry.geometries {
                BlasGeometriesStorage::TriangleGeometries(triangle_geometries) => {
                    triangle_geometry_storage
                        .push(Vec::<hal::AccelerationStructureTriangles<A>>::new());

                    for mesh in triangle_geometries {
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
                            let index_buffer_size = mesh.size.index_count.unwrap() as u64
                                * match mesh.size.index_format.unwrap() {
                                    wgt::IndexFormat::Uint16 => 2,
                                    wgt::IndexFormat::Uint32 => 4,
                                };
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
                                AccelerationStructureTriangleIndices {
                                    format: mesh.size.index_format.unwrap(),
                                    buffer: Some(index_buffer),
                                    offset: mesh.index_buffer_offset.unwrap() as u32,
                                    count: mesh.size.index_count.unwrap(),
                                }
                            }),
                            transform: transform_buffer.map(|transform_buffer| {
                                AccelerationStructureTriangleTransform {
                                    buffer: transform_buffer,
                                    offset: mesh.transform_buffer_offset.unwrap() as u32,
                                }
                            }),
                            flags: mesh.size.flags,
                        };
                        triangle_geometry_storage
                            .last_mut()
                            .unwrap()
                            .push(triangles);
                    }
                }
            };
        }

        let mut blas_entry_storage = Vec::<hal::AccelerationStructureEntries<A>>::new();

        let mut scratch_buffer_blas_size = 0;
        let mut scratch_buffer_blas_offsets = Vec::<u64>::new();

        let mut blas_refs = Vec::<&Blas<A>>::new();

        let mut triangel_geometrie_counter = 0;
        for entry in &blas_storage {
            match &entry.geometries {
                BlasGeometriesStorage::TriangleGeometries(_) => {
                    blas_entry_storage.push(hal::AccelerationStructureEntries::Triangles(
                        &triangle_geometry_storage[triangel_geometrie_counter],
                    ));
                    triangel_geometrie_counter += 1;
                }
            }

            let blas = cmd_buf
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            scratch_buffer_blas_offsets.push(scratch_buffer_blas_size);
            scratch_buffer_blas_size += blas.size_info.build_scratch_size; // TODO Alignment

            blas_refs.push(blas);
        }

        let mut tlas_refs = Vec::<&Tlas<A>>::new();
        let mut scratch_buffer_tlas_size = 0;
        let mut scratch_buffer_tlas_offsets = Vec::<u64>::new();
        let mut tlas_entry_storage = Vec::<hal::AccelerationStructureEntries<A>>::new();
        for entry in &tlas_storage {
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

            tlas_entry_storage.push(hal::AccelerationStructureEntries::Instances(
                AccelerationStructureInstances {
                    buffer: Some(instance_buffer),
                    offset: 0,
                    count: entry.instance_count,
                },
            ));

            let tlas = cmd_buf
                .trackers
                .tlas_s
                .add_single(&tlas_guard, entry.tlas_id)
                .ok_or(BuildAccelerationStructureError::InvalidTlas(entry.tlas_id))?;

            scratch_buffer_tlas_offsets.push(scratch_buffer_tlas_size);
            scratch_buffer_tlas_size += tlas.size_info.build_scratch_size; // TODO Alignment

            tlas_refs.push(tlas);
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

        let mut blas_descriptors =
            Vec::<hal::BuildAccelerationStructureDescriptor<A>>::with_capacity(blas_storage.len());
        for (i, _entry) in blas_storage.iter().enumerate() {
            let blas = blas_refs[i];
            blas_descriptors.push(hal::BuildAccelerationStructureDescriptor {
                entries: &blas_entry_storage[i],
                mode: hal::AccelerationStructureBuildMode::Build, // TODO
                flags: blas.flags,
                source_acceleration_structure: None,
                destination_acceleration_structure: blas.raw.as_ref().unwrap(),
                scratch_buffer: &scratch_buffer,
            })
        }

        let mut blas_descriptor_references =
            Vec::<&hal::BuildAccelerationStructureDescriptor<A>>::with_capacity(blas_storage.len());
        for (i, _entry) in blas_storage.iter().enumerate() {
            blas_descriptor_references.push(&blas_descriptors[i]);
        }

        let mut tlas_descriptors =
            Vec::<hal::BuildAccelerationStructureDescriptor<A>>::with_capacity(tlas_storage.len());
        for (i, _entry) in tlas_storage.iter().enumerate() {
            let tlas = tlas_refs[i];
            tlas_descriptors.push(hal::BuildAccelerationStructureDescriptor {
                entries: &tlas_entry_storage[i],
                mode: hal::AccelerationStructureBuildMode::Build, // TODO
                flags: tlas.flags,
                source_acceleration_structure: None,
                destination_acceleration_structure: tlas.raw.as_ref().unwrap(),
                scratch_buffer: &scratch_buffer,
            })
        }

        let mut tlas_descriptor_references =
            Vec::<&hal::BuildAccelerationStructureDescriptor<A>>::with_capacity(tlas_storage.len());
        for (i, _entry) in tlas_storage.iter().enumerate() {
            tlas_descriptor_references.push(&tlas_descriptors[i]);
        }

        let cmd_buf_raw = cmd_buf.encoder.open();
        unsafe {
            cmd_buf_raw.transition_buffers(input_barriers.into_iter());
            if !blas_descriptor_references.is_empty() {
                cmd_buf_raw.build_acceleration_structures(&blas_descriptor_references);
            }
            cmd_buf_raw.transition_buffers(iter::once(scratch_buffer_barrier));
            if !tlas_descriptor_references.is_empty() {
                cmd_buf_raw.build_acceleration_structures(&tlas_descriptor_references);
            }
        }

        device
            .pending_writes
            .temp_resources
            .push(TempResource::Buffer(scratch_buffer));

        Ok(())
    }
}
