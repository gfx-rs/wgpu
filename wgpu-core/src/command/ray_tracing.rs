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
    ray_tracing::{BlasBuildEntry, BuildAccelerationStructureError, TlasBuildEntry},
    resource::{self, Texture, TextureErrorDimension},
    track::TextureSelector,
    LabelHelpers, LifeGuard, Stored,
};

use arrayvec::ArrayVec;
use hal::{AccelerationStructureTriangleIndices, CommandEncoder as _, Device as _};
use thiserror::Error;
use wgt::{BufferAddress, BufferUsages, Extent3d, TextureUsages};

use std::iter;

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

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        let device = &device_guard[cmd_buf.device_id.value];

        let blas_build_descriptors = Vec::<hal::BuildAccelerationStructureDescriptor<A>>::new();

        let blas_build_entries = Vec::<hal::AccelerationStructureEntries<A>>::new();

        // for blas in blas_iter.into_iter(){
        //     match blas.geometries {
        //         crate::ray_tracing::BlasGeometries::TriangleGeometries(triangle_geometries) => {

        //         },
        //     }
        // }

        todo!()

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

        // let (src_buffer, src_pending) = cmd_buf
        //     .trackers
        //     .buffers
        //     .set_single(&*buffer_guard, source, hal::BufferUses::COPY_SRC)
        //     .ok_or(TransferError::InvalidBuffer(source))?;
        // let src_raw = src_buffer
        //     .raw
        //     .as_ref()
        //     .ok_or(TransferError::InvalidBuffer(source))?;
        // if !src_buffer.usage.contains(BufferUsages::COPY_SRC) {
        //     return Err(TransferError::MissingCopySrcUsageFlag.into());
        // }
        // // expecting only a single barrier
        // let src_barrier = src_pending.map(|pending| pending.into_hal(src_buffer));
    }
}
