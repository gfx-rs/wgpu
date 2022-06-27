use std::{num::NonZeroU32, ops::Range};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::CommandBuffer,
    get_lowest_common_denom,
    hub::{self, Global, GlobalIdentityHandlerFactory, HalApi, Token},
    id::{BufferId, CommandEncoderId, DeviceId, TextureId, Valid},
    init_tracker::{MemoryInitKind, TextureInitRange},
    resource::{Texture, TextureClearMode},
    track::{TextureSelector, TextureTracker},
};

use hal::{auxil::align_to, CommandEncoder as _};
use thiserror::Error;
use wgt::{BufferAddress, BufferSize, BufferUsages, ImageSubresourceRange, TextureAspect};

/// Error encountered while attempting a clear.
#[derive(Clone, Debug, Error)]
pub enum ClearError {
    #[error("to use clear_texture the CLEAR_TEXTURE feature needs to be enabled")]
    MissingClearTextureFeature,
    #[error("command encoder {0:?} is invalid")]
    InvalidCommandEncoder(CommandEncoderId),
    #[error("device {0:?} is invalid")]
    InvalidDevice(DeviceId),
    #[error("buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("texture {0:?} can not be cleared")]
    NoValidTextureClearMode(TextureId),
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
    #[error("destination buffer is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
    #[error("texture lacks the aspects that were specified in the image subresource range. Texture with format {texture_format:?}, specified was {subresource_range_aspects:?}")]
    MissingTextureAspect {
        texture_format: wgt::TextureFormat,
        subresource_range_aspects: TextureAspect,
    },
    #[error("image subresource level range is outside of the texture's level range. texture range is {texture_level_range:?},  \
whereas subesource range specified start {subresource_base_mip_level} and count {subresource_mip_level_count:?}")]
    InvalidTextureLevelRange {
        texture_level_range: Range<u32>,
        subresource_base_mip_level: u32,
        subresource_mip_level_count: Option<NonZeroU32>,
    },
    #[error("image subresource layer range is outside of the texture's layer range. texture range is {texture_layer_range:?},  \
whereas subesource range specified start {subresource_base_array_layer} and count {subresource_array_layer_count:?}")]
    InvalidTextureLayerRange {
        texture_layer_range: Range<u32>,
        subresource_base_array_layer: u32,
        subresource_array_layer_count: Option<NonZeroU32>,
    },
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_clear_buffer<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::fill_buffer");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::ClearBuffer { dst, offset, size });
        }

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .set_single(&*buffer_guard, dst, hal::BufferUses::COPY_DST)
            .ok_or(ClearError::InvalidBuffer(dst))?;
        let dst_raw = dst_buffer
            .raw
            .as_ref()
            .ok_or(ClearError::InvalidBuffer(dst))?;
        if !dst_buffer.usage.contains(BufferUsages::COPY_DST) {
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

        let end = match size {
            Some(size) => offset + size.get(),
            None => dst_buffer.size,
        };
        if offset == end {
            log::trace!("Ignoring fill_buffer of size 0");
            return Ok(());
        }

        // Mark dest as initialized.
        cmd_buf
            .buffer_memory_init_actions
            .extend(dst_buffer.initialization_status.create_action(
                dst,
                offset..end,
                MemoryInitKind::ImplicitlyInitialized,
            ));
        // actual hal barrier & operation
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(dst_buffer));
        let cmd_buf_raw = cmd_buf.encoder.open();
        unsafe {
            cmd_buf_raw.transition_buffers(dst_barrier.into_iter());
            cmd_buf_raw.clear_buffer(dst_raw, offset..end);
        }
        Ok(())
    }

    pub fn command_encoder_clear_texture<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: TextureId,
        subresource_range: &ImageSubresourceRange,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::clear_texture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.write(&mut token);
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let (_, mut token) = hub.buffers.read(&mut token); // skip token
        let (texture_guard, _) = hub.textures.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::ClearTexture {
                dst,
                subresource_range: subresource_range.clone(),
            });
        }

        if !cmd_buf.support_clear_texture {
            return Err(ClearError::MissingClearTextureFeature);
        }

        let dst_texture = texture_guard
            .get(dst)
            .map_err(|_| ClearError::InvalidTexture(dst))?;

        // Check if subresource aspects are valid.
        let requested_aspects = hal::FormatAspects::from(subresource_range.aspect);
        let clear_aspects = hal::FormatAspects::from(dst_texture.desc.format) & requested_aspects;
        if clear_aspects.is_empty() {
            return Err(ClearError::MissingTextureAspect {
                texture_format: dst_texture.desc.format,
                subresource_range_aspects: subresource_range.aspect,
            });
        };

        // Check if subresource level range is valid
        let subresource_level_end = match subresource_range.mip_level_count {
            Some(count) => subresource_range.base_mip_level + count.get(),
            None => dst_texture.full_range.mips.end,
        };
        if dst_texture.full_range.mips.start > subresource_range.base_mip_level
            || dst_texture.full_range.mips.end < subresource_level_end
        {
            return Err(ClearError::InvalidTextureLevelRange {
                texture_level_range: dst_texture.full_range.mips.clone(),
                subresource_base_mip_level: subresource_range.base_mip_level,
                subresource_mip_level_count: subresource_range.mip_level_count,
            });
        }
        // Check if subresource layer range is valid
        let subresource_layer_end = match subresource_range.array_layer_count {
            Some(count) => subresource_range.base_array_layer + count.get(),
            None => dst_texture.full_range.layers.end,
        };
        if dst_texture.full_range.layers.start > subresource_range.base_array_layer
            || dst_texture.full_range.layers.end < subresource_layer_end
        {
            return Err(ClearError::InvalidTextureLayerRange {
                texture_layer_range: dst_texture.full_range.layers.clone(),
                subresource_base_array_layer: subresource_range.base_array_layer,
                subresource_array_layer_count: subresource_range.array_layer_count,
            });
        }

        let device = &device_guard[cmd_buf.device_id.value];

        clear_texture(
            &*texture_guard,
            Valid(dst),
            TextureInitRange {
                mip_range: subresource_range.base_mip_level..subresource_level_end,
                layer_range: subresource_range.base_array_layer..subresource_layer_end,
            },
            cmd_buf.encoder.open(),
            &mut cmd_buf.trackers.textures,
            &device.alignments,
            &device.zero_buffer,
        )
    }
}

pub(crate) fn clear_texture<A: HalApi>(
    storage: &hub::Storage<Texture<A>, TextureId>,
    dst_texture_id: Valid<TextureId>,
    range: TextureInitRange,
    encoder: &mut A::CommandEncoder,
    texture_tracker: &mut TextureTracker<A>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer,
) -> Result<(), ClearError> {
    let dst_texture = &storage[dst_texture_id];

    let dst_raw = dst_texture
        .inner
        .as_raw()
        .ok_or(ClearError::InvalidTexture(dst_texture_id.0))?;

    // Issue the right barrier.
    let clear_usage = match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => hal::TextureUses::COPY_DST,
        TextureClearMode::RenderPass {
            is_color: false, ..
        } => hal::TextureUses::DEPTH_STENCIL_WRITE,
        TextureClearMode::RenderPass { is_color: true, .. } => hal::TextureUses::COLOR_TARGET,
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(dst_texture_id.0));
        }
    };

    let selector = TextureSelector {
        mips: range.mip_range.clone(),
        layers: range.layer_range.clone(),
    };

    // If we're in a texture-init usecase, we know that the texture is already tracked since whatever caused the init requirement,
    // will have caused the usage tracker to be aware of the texture. Meaning, that it is safe to call call change_replace_tracked if the life_guard is already gone
    // (i.e. the user no longer holds on to this texture).
    // On the other hand, when coming via command_encoder_clear_texture, the life_guard is still there since in order to call it a texture object is needed.
    //
    // We could in theory distinguish these two scenarios in the internal clear_texture api in order to remove this check and call the cheaper change_replace_tracked whenever possible.
    let dst_barrier = texture_tracker
        .set_single(storage, dst_texture_id.0, selector, clear_usage)
        .unwrap()
        .1
        .map(|pending| pending.into_hal(dst_texture));
    unsafe {
        encoder.transition_textures(dst_barrier.into_iter());
    }

    // Record actual clearing
    match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => clear_texture_via_buffer_copies::<A>(
            &dst_texture.desc,
            alignments,
            zero_buffer,
            range,
            encoder,
            dst_raw,
        ),
        TextureClearMode::RenderPass { is_color, .. } => {
            clear_texture_via_render_passes(dst_texture, range, is_color, encoder)?
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(dst_texture_id.0));
        }
    }
    Ok(())
}

fn clear_texture_via_buffer_copies<A: hal::Api>(
    texture_desc: &wgt::TextureDescriptor<()>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer, // Buffer of size device::ZERO_BUFFER_SIZE
    range: TextureInitRange,
    encoder: &mut A::CommandEncoder,
    dst_raw: &A::Texture,
) {
    // Gather list of zero_buffer copies and issue a single command then to perform them
    let mut zero_buffer_copy_regions = Vec::new();
    let buffer_copy_pitch = alignments.buffer_copy_pitch.get() as u32;
    let format_desc = texture_desc.format.describe();

    let bytes_per_row_alignment =
        get_lowest_common_denom(buffer_copy_pitch, format_desc.block_size as u32);

    for mip_level in range.mip_range {
        let mut mip_size = texture_desc.mip_level_size(mip_level).unwrap();
        // Round to multiple of block size
        mip_size.width = align_to(mip_size.width, format_desc.block_dimensions.0 as u32);
        mip_size.height = align_to(mip_size.height, format_desc.block_dimensions.1 as u32);

        let bytes_per_row = align_to(
            mip_size.width / format_desc.block_dimensions.0 as u32 * format_desc.block_size as u32,
            bytes_per_row_alignment,
        );

        let max_rows_per_copy = crate::device::ZERO_BUFFER_SIZE as u32 / bytes_per_row;
        // round down to a multiple of rows needed by the texture format
        let max_rows_per_copy = max_rows_per_copy / format_desc.block_dimensions.1 as u32
            * format_desc.block_dimensions.1 as u32;
        assert!(max_rows_per_copy > 0, "Zero buffer size is too small to fill a single row of a texture with format {:?} and desc {:?}",
                texture_desc.format, texture_desc.size);

        let z_range = 0..(if texture_desc.dimension == wgt::TextureDimension::D3 {
            mip_size.depth_or_array_layers
        } else {
            1
        });

        for array_layer in range.layer_range.clone() {
            // TODO: Only doing one layer at a time for volume textures right now.
            for z in z_range.clone() {
                // May need multiple copies for each subresource! However, we assume that we never need to split a row.
                let mut num_rows_left = mip_size.height;
                while num_rows_left > 0 {
                    let num_rows = num_rows_left.min(max_rows_per_copy);

                    zero_buffer_copy_regions.push(hal::BufferTextureCopy {
                        buffer_layout: wgt::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: NonZeroU32::new(bytes_per_row),
                            rows_per_image: None,
                        },
                        texture_base: hal::TextureCopyBase {
                            mip_level,
                            array_layer,
                            origin: wgt::Origin3d {
                                x: 0, // Always full rows
                                y: mip_size.height - num_rows_left,
                                z,
                            },
                            aspect: hal::FormatAspects::all(),
                        },
                        size: hal::CopyExtent {
                            width: mip_size.width, // full row
                            height: num_rows,
                            depth: 1, // Only single slice of volume texture at a time right now
                        },
                    });

                    num_rows_left -= num_rows;
                }
            }
        }
    }

    unsafe {
        encoder.copy_buffer_to_texture(zero_buffer, dst_raw, zero_buffer_copy_regions.into_iter());
    }
}

fn clear_texture_via_render_passes<A: hal::Api>(
    dst_texture: &Texture<A>,
    range: TextureInitRange,
    is_color: bool,
    encoder: &mut A::CommandEncoder,
) -> Result<(), ClearError> {
    let extent_base = wgt::Extent3d {
        width: dst_texture.desc.size.width,
        height: dst_texture.desc.size.height,
        depth_or_array_layers: 1, // Only one layer or slice is cleared at a time.
    };

    let sample_count = dst_texture.desc.sample_count;
    let is_3d_texture = dst_texture.desc.dimension == wgt::TextureDimension::D3;
    for mip_level in range.mip_range {
        let extent = extent_base.mip_level_size(mip_level, is_3d_texture);
        let layer_or_depth_range = if dst_texture.desc.dimension == wgt::TextureDimension::D3 {
            // TODO: We assume that we're allowed to do clear operations on volume texture slices, this is not properly specified.
            0..extent.depth_or_array_layers
        } else {
            range.layer_range.clone()
        };
        for depth_or_layer in layer_or_depth_range {
            let color_attachments_tmp;
            let (color_attachments, depth_stencil_attachment) = if is_color {
                color_attachments_tmp = [Some(hal::ColorAttachment {
                    target: hal::Attachment {
                        view: dst_texture.get_clear_view(mip_level, depth_or_layer),
                        usage: hal::TextureUses::COLOR_TARGET,
                    },
                    resolve_target: None,
                    ops: hal::AttachmentOps::STORE,
                    clear_value: wgt::Color::TRANSPARENT,
                })];
                (&color_attachments_tmp[..], None)
            } else {
                (
                    &[][..],
                    Some(hal::DepthStencilAttachment {
                        target: hal::Attachment {
                            view: dst_texture.get_clear_view(mip_level, depth_or_layer),
                            usage: hal::TextureUses::DEPTH_STENCIL_WRITE,
                        },
                        depth_ops: hal::AttachmentOps::STORE,
                        stencil_ops: hal::AttachmentOps::STORE,
                        clear_value: (0.0, 0),
                    }),
                )
            };
            unsafe {
                encoder.begin_render_pass(&hal::RenderPassDescriptor {
                    label: Some("(wgpu internal) clear_texture clear pass"),
                    extent,
                    sample_count,
                    color_attachments,
                    depth_stencil_attachment,
                    multiview: None,
                });
                encoder.end_render_pass();
            }
        }
    }
    Ok(())
}
