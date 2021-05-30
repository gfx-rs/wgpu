/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::{num::NonZeroU32, ops::Range};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::CommandBuffer,
    conv,
    device::all_buffer_stages,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id::{BufferId, CommandEncoderId, TextureId},
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::{BufferUse, TextureUse},
    track::TextureSelector,
};

use hal::command::CommandBuffer as _;
use thiserror::Error;
use wgt::{
    BufferAddress, BufferSize, BufferUsage, ImageSubresourceRange, TextureAspect, TextureUsage,
};

/// Error encountered while attempting a clear.
#[derive(Clone, Debug, Error)]
pub enum ClearError {
    #[error("to use clear_buffer/texture the CLEAR_COMMANDS feature needs to be enabled")]
    MissingClearCommandsFeature,
    #[error("command encoder {0:?} is invalid")]
    InvalidCommandEncoder(CommandEncoderId),
    #[error("buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("buffer clear size {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedFillSize(BufferSize),
    #[error("buffer offset {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(BufferAddress),
    #[error("clear of {start_offset}..{end_offset} would end up overrunning the bounds of the buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error("destination buffer/texture is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
    #[error("texture lacks the aspects that were specified in the image subresource range. Texture has {texture_aspects:?}, specified was {subresource_range_aspects:?}")]
    MissingTextureAspect {
        texture_aspects: hal::format::Aspects,
        subresource_range_aspects: TextureAspect,
    },
    #[error("image subresource level range is outside of the texture's level range. texture range is {texture_level_range:?},  \
whereas subesource range specified start {subresource_base_mip_level} and count {subresource_mip_level_count:?}")]
    InvalidTextureLevelRange {
        texture_level_range: Range<hal::image::Level>,
        subresource_base_mip_level: u32,
        subresource_mip_level_count: Option<NonZeroU32>,
    },
    #[error("image subresource layer range is outside of the texture's layer range. texture range is {texture_layer_range:?},  \
whereas subesource range specified start {subresource_base_array_layer} and count {subresource_array_layer_count:?}")]
    InvalidTextureLayerRange {
        texture_layer_range: Range<hal::image::Layer>,
        subresource_base_array_layer: u32,
        subresource_array_layer_count: Option<NonZeroU32>,
    },
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_clear_buffer<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::fill_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::ClearBuffer { dst, offset, size });
        }

        if !cmd_buf.support_fill_buffer_texture {
            return Err(ClearError::MissingClearCommandsFeature);
        }

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, dst, (), BufferUse::COPY_DST)
            .map_err(ClearError::InvalidBuffer)?;
        let &(ref dst_raw, _) = dst_buffer
            .raw
            .as_ref()
            .ok_or(ClearError::InvalidBuffer(dst))?;
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            return Err(ClearError::MissingCopyDstUsageFlag(Some(dst), None));
        }

        // Check if offset & size are valid.
        if offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(ClearError::UnalignedBufferOffset(offset));
        }
        if let Some(size) = size {
            if size.get() % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(ClearError::UnalignedFillSize(size));
            }
            let destination_end_offset = offset + size.get();
            if destination_end_offset > dst_buffer.size {
                return Err(ClearError::BufferOverrun {
                    start_offset: offset,
                    end_offset: destination_end_offset,
                    buffer_size: dst_buffer.size,
                });
            }
        }

        let num_bytes_filled = size.map_or(dst_buffer.size - offset, |s| s.get());
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
            cmd_buf_raw.fill_buffer(
                dst_raw,
                hal::buffer::SubRange {
                    offset,
                    size: size.map(|s| s.get()),
                },
                0,
            );
        }
        Ok(())
    }

    pub fn command_encoder_clear_image<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: TextureId,
        subresource_range: ImageSubresourceRange,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::clear_image");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let (_, mut token) = hub.buffers.read(&mut token); // skip token
        let (texture_guard, _) = hub.textures.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::ClearImage {
                dst,
                subresource_range: subresource_range.clone(),
            });
        }

        if !cmd_buf.support_fill_buffer_texture {
            return Err(ClearError::MissingClearCommandsFeature);
        }

        let dst_texture = texture_guard
            .get(dst)
            .map_err(|_| ClearError::InvalidTexture(dst))?;

        // Check if subresource aspects are valid.
        let aspects = match subresource_range.aspect {
            wgt::TextureAspect::All => dst_texture.aspects,
            wgt::TextureAspect::DepthOnly => hal::format::Aspects::DEPTH,
            wgt::TextureAspect::StencilOnly => hal::format::Aspects::STENCIL,
        };
        if !dst_texture.aspects.contains(aspects) {
            return Err(ClearError::MissingTextureAspect {
                texture_aspects: dst_texture.aspects,
                subresource_range_aspects: subresource_range.aspect,
            });
        };
        // Check if subresource level range is valid
        let subresource_level_end = if let Some(count) = subresource_range.mip_level_count {
            (subresource_range.base_mip_level + count.get()) as u8
        } else {
            dst_texture.full_range.levels.end
        };
        if dst_texture.full_range.levels.start > subresource_range.base_mip_level as u8
            || dst_texture.full_range.levels.end < subresource_level_end
        {
            return Err(ClearError::InvalidTextureLevelRange {
                texture_level_range: dst_texture.full_range.levels.clone(),
                subresource_base_mip_level: subresource_range.base_mip_level,
                subresource_mip_level_count: subresource_range.mip_level_count,
            });
        }
        // Check if subresource layer range is valid
        let subresource_layer_end = if let Some(count) = subresource_range.array_layer_count {
            (subresource_range.base_array_layer + count.get()) as u16
        } else {
            dst_texture.full_range.layers.end
        };
        if dst_texture.full_range.layers.start > subresource_range.base_array_layer as u16
            || dst_texture.full_range.layers.end < subresource_layer_end
        {
            return Err(ClearError::InvalidTextureLayerRange {
                texture_layer_range: dst_texture.full_range.layers.clone(),
                subresource_base_array_layer: subresource_range.base_array_layer,
                subresource_array_layer_count: subresource_range.array_layer_count,
            });
        }

        // query from tracker with usage (and check usage)
        let (dst_texture, dst_pending) = cmd_buf
            .trackers
            .textures
            .use_replace(
                &*texture_guard,
                dst,
                TextureSelector {
                    levels: subresource_range.base_mip_level as u8..subresource_level_end,
                    layers: subresource_range.base_array_layer as u16..subresource_layer_end,
                },
                TextureUse::COPY_DST,
            )
            .map_err(ClearError::InvalidTexture)?;
        let &(ref dst_raw, _) = dst_texture
            .raw
            .as_ref()
            .ok_or(ClearError::InvalidTexture(dst))?;
        if !dst_texture.usage.contains(TextureUsage::COPY_DST) {
            return Err(ClearError::MissingCopyDstUsageFlag(None, Some(dst)));
        }

        // actual hal barrier & operation
        let dst_barrier = dst_pending
            .map(|pending| pending.into_hal(dst_texture))
            .next();
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmd_buf_raw.pipeline_barrier(
                all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                dst_barrier.into_iter(),
            );
            cmd_buf_raw.clear_image(
                dst_raw,
                hal::image::Layout::TransferDstOptimal,
                hal::command::ClearValue {
                    color: hal::command::ClearColor {
                        float32: conv::map_color_f32(&wgt::Color::TRANSPARENT),
                    },
                },
                std::iter::once(hal::image::SubresourceRange {
                    aspects,
                    level_start: subresource_range.base_mip_level as u8,
                    level_count: subresource_range.mip_level_count.map(|c| c.get() as u8),
                    layer_start: subresource_range.base_array_layer as u16,
                    layer_count: subresource_range.array_layer_count.map(|c| c.get() as u16),
                }),
            );
        }
        Ok(())
    }
}
