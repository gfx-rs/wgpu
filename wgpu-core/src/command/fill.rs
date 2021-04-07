/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::CommandBuffer,
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id::{BufferId, CommandEncoderId, TextureId},
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::BufferUse,
};

use hal::command::CommandBuffer as _;
use thiserror::Error;
use wgt::{BufferAddress, BufferUsage};

/// Error encountered while attempting a clear.
#[derive(Clone, Debug, Error)]
pub enum FillError {
    #[error("to use fill_buffer/texture the BUFFER_AND_TEXTURE_FILL feature needs to be enabled")]
    MissingFillBufferTextureFeature,
    #[error("buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("buffer fill size {0:?} is not a multiple of 4")]
    UnalignedFillSize(BufferAddress),
    #[error("buffer offset {0:?} is not a multiple of 4")]
    UnalignedBufferOffset(BufferAddress),
    #[error("fill of {start_offset}..{end_offset} would end up overrunning the bounds of the buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error("destination buffer/texture is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_fill_buffer<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
        data: u32,
    ) -> Result<(), FillError> {
        profiling::scope!("CommandEncoder::fill_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)
            .map_err(|_| FillError::InvalidBuffer(dst))?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::FillBuffer {
                dst,
                offset,
                size,
                data,
            });
        }

        if !cmd_buf.support_fill_buffer_texture {
            return Err(FillError::MissingFillBufferTextureFeature);
        }

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, dst, (), BufferUse::COPY_DST)
            .map_err(FillError::InvalidBuffer)?;
        let &(ref dst_raw, _) = dst_buffer
            .raw
            .as_ref()
            .ok_or(FillError::InvalidBuffer(dst))?;
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            return Err(FillError::MissingCopyDstUsageFlag(Some(dst), None).into());
        }

        // Check if offset & size are valid.
        if offset % std::mem::size_of_val(&data) as BufferAddress != 0 {
            return Err(FillError::UnalignedBufferOffset(offset).into());
        }
        if let Some(size) = size {
            if size % std::mem::size_of_val(&data) as BufferAddress != 0 {
                return Err(FillError::UnalignedFillSize(size).into());
            }
            let destination_end_offset = offset + size;
            if destination_end_offset > dst_buffer.size {
                return Err(FillError::BufferOverrun {
                    start_offset: offset,
                    end_offset: destination_end_offset,
                    buffer_size: dst_buffer.size,
                }
                .into());
            }
        }

        let num_bytes_filled = size.unwrap_or(dst_buffer.size - offset);
        if num_bytes_filled == 0 {
            log::trace!("Ignoring fill_buffer of size 0");
            return Ok(());
        }

        // Mark dest as initialized.
        cmd_buf.buffer_memory_init_actions.extend(
            dst_buffer
                .initialization_status
                .check(offset..(offset + num_bytes_filled))
                .map(|range| MemoryInitTrackerAction {
                    id: dst,
                    range,
                    kind: MemoryInitKind::ImplicitlyInitialized,
                }),
        );

        // actual hal barrier & operation
        let dst_barrier = dst_pending
            .map(|pending| pending.into_hal(dst_buffer))
            .next();
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmd_buf_raw.pipeline_barrier(
                all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                dst_barrier.into_iter(),
            );
            cmd_buf_raw.fill_buffer(dst_raw, hal::buffer::SubRange { offset, size }, data);
        }
        Ok(())
    }
}
