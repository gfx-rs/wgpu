use alloc::boxed::Box;
use core::num::NonZeroU64;
use wgpu_sync::Arc;

use hal::{AccelerationStructureDescriptor, GetAccelerationStructureBuildSizesDescriptor};

use crate::{
    device::{
        queue::{self, Queue},
        Device,
    },
    hal_label,
    scratch::ScratchBuffer,
};

/// The optional acceleration structure to be used for an instance with a `None` Blas.
pub struct NullAccelerationStructure {
    /// A built bottom level acceleration structure. May be either triangles or AABBs.
    ///
    /// None means the null acceleration structure is fine (handle `0`).
    acceleration_structure: Option<EmptyAccelerationStructure>,
}

struct EmptyAccelerationStructure {
    acceleration_structure: Box<dyn hal::DynAccelerationStructure>,
    handle: u64,
}

impl NullAccelerationStructure {
    pub fn new(
        device: &Arc<Device>,
        queue: &Queue,
    ) -> Result<Self, NullAccelerationStructureInitError> {
        if device.alignments.null_acceleration_structure_allowed {
            return Ok(Self {
                acceleration_structure: None,
            });
        }

        unsafe {
            let entries = hal::AccelerationStructureEntries::Triangles(alloc::vec![
                hal::AccelerationStructureTriangles {
                    vertex_buffer: Some(device.zero_buffer.as_ref()),
                    vertex_format: wgt::VertexFormat::Float32x3,
                    first_vertex: 0,
                    vertex_count: 3,
                    vertex_stride: wgt::VertexFormat::Float32x3
                        .min_acceleration_structure_vertex_stride(),
                    indices: None,
                    transform: None,
                    flags: wgt::AccelerationStructureGeometryFlags::empty()
                }
            ]);

            let sizes = device.raw().get_acceleration_structure_build_sizes(
                &GetAccelerationStructureBuildSizesDescriptor {
                    entries: &entries,
                    flags: wgt::AccelerationStructureFlags::empty(),
                },
            );

            let acceleration_structure = device
                .raw()
                .create_acceleration_structure(&AccelerationStructureDescriptor {
                    label: hal_label(
                        Some("(wgpu internal) null acceleration structure"),
                        device.instance_flags,
                    ),
                    size: sizes.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                    allow_compaction: false,
                })
                .map_err(NullAccelerationStructureInitError::AccelerationStructure)?;

            let scratch = match ScratchBuffer::new(
                device,
                NonZeroU64::new(sizes.build_scratch_size)
                    .expect("scratch size should always be non-zero"),
            ) {
                Ok(buf) => buf,
                Err(e) => {
                    device.raw().destroy_acceleration_structure(acceleration_structure);
                    return Err(NullAccelerationStructureInitError::ScratchBuffer(e))
                }
            };

            let mut pending = queue.pending_writes.lock();
            let command_encoder = pending.activate();

            command_encoder.build_acceleration_structures(&[
                hal::BuildAccelerationStructureDescriptor {
                    entries: &entries,
                    mode: hal::AccelerationStructureBuildMode::Build,
                    flags: wgt::AccelerationStructureFlags::empty(),
                    source_acceleration_structure: None,
                    destination_acceleration_structure: acceleration_structure.as_ref(),
                    scratch_buffer: scratch.raw(),
                    scratch_buffer_offset: 0,
                },
            ]);

            command_encoder.place_acceleration_structure_barrier(
                hal::AccelerationStructureBarrier {
                    usage: hal::StateTransition {
                        from: hal::AccelerationStructureUses::BUILD_OUTPUT,
                        to: hal::AccelerationStructureUses::BUILD_INPUT,
                    },
                },
            );

            pending.consume_temp(queue::TempResource::ScratchBuffer(scratch));

            let handle = device
                .raw()
                .get_acceleration_structure_device_address(acceleration_structure.as_ref());

            Ok(Self {
                acceleration_structure: Some(EmptyAccelerationStructure {
                    acceleration_structure,
                    handle,
                }),
            })
        }
    }

    /// # Safety
    /// No command buffer may be using the acceleration structure.
    pub unsafe fn dispose(self, device: &dyn hal::DynDevice) {
        let Some(EmptyAccelerationStructure {
            acceleration_structure,
            ..
        }) = self.acceleration_structure
        else {
            // Nothing to dispose of
            return;
        };

        unsafe { device.destroy_acceleration_structure(acceleration_structure) };
    }

    pub fn handle(&self) -> u64 {
        self.acceleration_structure.as_ref().map_or(0, |a| a.handle)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum NullAccelerationStructureInitError {
    #[error("Failed to create the null acceleration structure: {0:?}")]
    AccelerationStructure(hal::DeviceError),
    #[error("Failed to create the scratch buffer to build: {0:?}")]
    ScratchBuffer(crate::device::DeviceError),
}
