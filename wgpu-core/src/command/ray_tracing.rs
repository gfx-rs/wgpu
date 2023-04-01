// #[cfg(feature = "trace")]
// use crate::device::trace::Command as TraceCommand;
use crate::{
    command::CommandBuffer,
    device::queue::TempResource,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Storage, Token},
    id::{BlasId, CommandEncoderId, TlasId},
    init_tracker::MemoryInitKind,
    ray_tracing::{
        BlasAction, BlasBuildEntry, BlasGeometries, BuildAccelerationStructureError, TlasAction,
        TlasBuildEntry, ValidateBlasActionsError, ValidateTlasActionsError,
    },
    resource::{Blas, Tlas},
    FastHashSet,
};

use hal::{CommandEncoder, Device};
use wgt::BufferUsages;

use std::{cmp::max, iter};

use super::BakedCommands;

// TODO:
// tracing
// automatic build splitting (if to big for spec or scratch buffer)
// comments/documentation
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

        #[cfg(feature = "trace")]
        let blas_iter: Box<dyn Iterator<Item = _>> = if let Some(ref mut _list) = cmd_buf.commands {
            // Create temporary allocation, save trace and recreate iterator (same for tlas)
            Box::new(blas_iter.map(|x| x))
        } else {
            Box::new(blas_iter)
        };

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

            if !*blas.built.lock() {
                cmd_buf.blas_actions.push(BlasAction {
                    id: entry.blas_id,
                    kind: crate::ray_tracing::AccelerationStructureActionKind::Build,
                });
            }

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

            if !*tlas.built.lock() {
                cmd_buf.tlas_actions.push(TlasAction {
                    id: entry.tlas_id,
                    kind: crate::ray_tracing::AccelerationStructureActionKind::Build,
                });
            }

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
                    *blas.built.lock() = true;
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
                    *tlas.built.lock() = true;
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
}

impl<A: HalApi> BakedCommands<A> {
    // makes sure a blas is build before it is used
    pub(crate) fn validate_blas_actions(
        &mut self,
        blas_guard: &Storage<Blas<A>, BlasId>,
    ) -> Result<(), ValidateBlasActionsError> {
        let mut built = FastHashSet::default();
        for action in self.blas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::AccelerationStructureActionKind::Build => {
                    built.insert(action.id);
                }
                crate::ray_tracing::AccelerationStructureActionKind::Use => {
                    if !built.contains(&action.id) {
                        let blas = blas_guard
                            .get(action.id)
                            .map_err(|_| ValidateBlasActionsError::InvalidBlas(action.id))?;
                        if !*blas.built.lock() {
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
        tlas_guard: &Storage<Tlas<A>, TlasId>,
    ) -> Result<(), ValidateTlasActionsError> {
        let mut built = FastHashSet::default();
        for action in self.tlas_actions.drain(..) {
            match action.kind {
                crate::ray_tracing::AccelerationStructureActionKind::Build => {
                    built.insert(action.id);
                }
                crate::ray_tracing::AccelerationStructureActionKind::Use => {
                    if !built.contains(&action.id) {
                        let tlas = tlas_guard
                            .get(action.id)
                            .map_err(|_| ValidateTlasActionsError::InvalidTlas(action.id))?;
                        if !*tlas.built.lock() {
                            return Err(ValidateTlasActionsError::UsedUnbuilt(action.id));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
