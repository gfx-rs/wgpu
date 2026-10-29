use alloc::boxed::Box;
use core::num::NonZeroU64;
use wgpu_sync::Arc;

use hal::{AccelerationStructureDescriptor, GetAccelerationStructureBuildSizesDescriptor};
use wgt::math::AlignTo;

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
    acceleration_structure: Option<EmtpyAcclerationStructure>,
}

struct EmtpyAcclerationStructure {
    acceleration_structure: Box<dyn hal::DynAccelerationStructure>,
    handle: u64,
}

impl NullAccelerationStructure {
    pub fn new(
        device: &Arc<Device>,
        queue: &Queue,
    ) -> Result<Self, NullAccelerationStructureInitError> {
        unsafe {
            let entries = hal::AccelerationStructureEntries::Triangles(alloc::vec![]);

            let sizes = device.raw().get_acceleration_structure_build_sizes(
                &GetAccelerationStructureBuildSizesDescriptor {
                    entries: &entries,
                    flags: wgt::AccelerationStructureFlags::empty(),
                }
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

            let scratch = ScratchBuffer::new(
                device,
                NonZeroU64::new(
                    sizes.build_scratch_size.max(1).align_to(
                        device
                            .alignments
                            .ray_tracing_scratch_buffer_alignment
                            .into(),
                    ),
                )
                .expect(".max(1) to ensure it is non-zero"),
            )
            .map_err(NullAccelerationStructureInitError::ScratchBuffer)?;

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

            pending.consume_temp(queue::TempResource::ScratchBuffer(scratch));

            let handle = device
                .raw()
                .get_acceleration_structure_device_address(acceleration_structure.as_ref());

            Ok(Self {
                acceleration_structure: Some(EmtpyAcclerationStructure {
                    acceleration_structure,
                    handle,
                }),
            })
        }
    }

    /// # Safety
    /// No command buffer may be using the acceleration structure.
    pub unsafe fn dispose(self, device: &dyn hal::DynDevice) {
        let Some(EmtpyAcclerationStructure {
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
