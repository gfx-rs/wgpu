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
    resource::{self, Texture, TextureErrorDimension},
    track::TextureSelector,
    LabelHelpers, LifeGuard, Stored,
};

use arrayvec::ArrayVec;
use hal::{
    AccelerationStructureTriangleIndices, AccelerationStructureTriangleTransform,
    CommandEncoder as _, Device as _,
};
use thiserror::Error;
use wgt::{BufferAddress, BufferUsages, Extent3d, TextureUsages};

use std::{borrow::Borrow, iter};

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

        let mut barriers = Vec::<hal::BufferBarrier<A>>::new();

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
                                barriers.push(barrier);
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
                                barriers.push(barrier);
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
                        let transform_buffer = if let Some(transform_id) = mesh.index_buffer {
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
                                barriers.push(barrier);
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

        let mut ac_entry_storage = Vec::<hal::AccelerationStructureEntries<A>>::new();

        let mut triangel_geometrie_counter = 0;
        for entry in &blas_storage {
            match &entry.geometries {
                BlasGeometriesStorage::TriangleGeometries(_) => {
                    ac_entry_storage.push(hal::AccelerationStructureEntries::Triangles(
                        &triangle_geometry_storage[triangel_geometrie_counter],
                    ));
                    triangel_geometrie_counter += 1;
                }
            }
        }

        let scratch_size = 1000000;

        let scratch_buffer = unsafe {
            device
                .raw
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("(wgpu) scratch buffer"),
                    size: scratch_size,
                    usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .unwrap()
        };

        let mut blas_descriptors =
            Vec::<hal::BuildAccelerationStructureDescriptor<A>>::with_capacity(blas_storage.len());

        for (i, entry) in blas_storage.iter().enumerate() {
            let blas = cmd_buf
                .trackers
                .blas_s
                .add_single(&blas_guard, entry.blas_id)
                .ok_or(BuildAccelerationStructureError::InvalidBlas(entry.blas_id))?;

            blas_descriptors.push(hal::BuildAccelerationStructureDescriptor {
                entries: &ac_entry_storage[i],
                mode: hal::AccelerationStructureBuildMode::Build, // TODO
                flags: blas.flags,
                source_acceleration_structure: None,
                destination_acceleration_structure: blas.raw.as_ref().unwrap(),
                scratch_buffer: &scratch_buffer,
            })
        }

        device
            .pending_writes
            .temp_resources
            .push(TempResource::Buffer(scratch_buffer));

        let cmd_buf_raw = cmd_buf.encoder.open();
        unsafe {
            cmd_buf_raw.transition_buffers(barriers.into_iter());
        }
        Ok(())

        // todo!()
    }
}
