use std::{ops::Range, sync::Arc};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    api_log,
    command::CommandEncoderError,
    device::DeviceError,
    get_lowest_common_denom,
    global::Global,
    id::{BufferId, CommandEncoderId, TextureId},
    init_tracker::{MemoryInitKind, TextureInitRange},
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled, MissingBufferUsageError,
        ParentDevice, ResourceErrorIdent, Texture, TextureClearMode,
    },
    snatch::SnatchGuard,
    track::{TextureSelector, TextureTrackerSetSingle},
};

use thiserror::Error;
use wgt::{math::align_to, BufferAddress, BufferUsages, ImageSubresourceRange, TextureAspect};

/// Error encountered while attempting a clear.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ClearError {
    #[error("To use clear_texture the CLEAR_TEXTURE feature needs to be enabled")]
    MissingClearTextureFeature,
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("{0} can not be cleared")]
    NoValidTextureClearMode(ResourceErrorIdent),
    #[error("Buffer clear size {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedFillSize(BufferAddress),
    #[error("Buffer offset {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(BufferAddress),
    #[error("Clear starts at offset {start_offset} with size of {requested_size}, but these added together exceed `u64::MAX`")]
    OffsetPlusSizeExceeds64BitBounds {
        start_offset: BufferAddress,
        requested_size: BufferAddress,
    },
    #[error("Clear of {start_offset}..{end_offset} would end up overrunning the bounds of the buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error("Texture lacks the aspects that were specified in the image subresource range. Texture with format {texture_format:?}, specified was {subresource_range_aspects:?}")]
    MissingTextureAspect {
        texture_format: wgt::TextureFormat,
        subresource_range_aspects: TextureAspect,
    },
    #[error("Image subresource level range is outside of the texture's level range. texture range is {texture_level_range:?},  \
whereas subesource range specified start {subresource_base_mip_level} and count {subresource_mip_level_count:?}")]
    InvalidTextureLevelRange {
        texture_level_range: Range<u32>,
        subresource_base_mip_level: u32,
        subresource_mip_level_count: Option<u32>,
    },
    #[error("Image subresource layer range is outside of the texture's layer range. texture range is {texture_layer_range:?},  \
whereas subesource range specified start {subresource_base_array_layer} and count {subresource_array_layer_count:?}")]
    InvalidTextureLayerRange {
        texture_layer_range: Range<u32>,
        subresource_base_array_layer: u32,
        subresource_array_layer_count: Option<u32>,
    },
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CommandEncoderError(#[from] CommandEncoderError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

impl Global {
    pub fn command_encoder_clear_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::clear_buffer");
        api_log!("CommandEncoder::clear_buffer {dst:?}");

        let hub = &self.hub;

        let cmd_buf = hub
            .command_buffers
            .get(command_encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.try_get()?;
        cmd_buf_data.check_recording()?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::ClearBuffer { dst, offset, size });
        }

        let dst_buffer = hub.buffers.get(dst).get()?;

        dst_buffer.same_device_as(cmd_buf.as_ref())?;

        let dst_pending = cmd_buf_data
            .trackers
            .buffers
            .set_single(&dst_buffer, hal::BufferUses::COPY_DST);

        let snatch_guard = dst_buffer.device.snatchable_lock.read();
        let dst_raw = dst_buffer.try_raw(&snatch_guard)?;
        dst_buffer.check_usage(BufferUsages::COPY_DST)?;

        // Check if offset & size are valid.
        if offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(ClearError::UnalignedBufferOffset(offset));
        }

        let size = size.unwrap_or(dst_buffer.size.saturating_sub(offset));
        if size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(ClearError::UnalignedFillSize(size));
        }
        let end_offset =
            offset
                .checked_add(size)
                .ok_or(ClearError::OffsetPlusSizeExceeds64BitBounds {
                    start_offset: offset,
                    requested_size: size,
                })?;
        if end_offset > dst_buffer.size {
            return Err(ClearError::BufferOverrun {
                start_offset: offset,
                end_offset,
                buffer_size: dst_buffer.size,
            });
        }

        if offset == end_offset {
            log::trace!("Ignoring fill_buffer of size 0");
            return Ok(());
        }

        // Mark dest as initialized.
        cmd_buf_data.buffer_memory_init_actions.extend(
            dst_buffer.initialization_status.read().create_action(
                &dst_buffer,
                offset..end_offset,
                MemoryInitKind::ImplicitlyInitialized,
            ),
        );

        // actual hal barrier & operation
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));
        let cmd_buf_raw = cmd_buf_data.encoder.open(&cmd_buf.device)?;
        unsafe {
            cmd_buf_raw.transition_buffers(dst_barrier.as_slice());
            cmd_buf_raw.clear_buffer(dst_raw, offset..end_offset);
        }
        Ok(())
    }

    pub fn command_encoder_clear_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: TextureId,
        subresource_range: &ImageSubresourceRange,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::clear_texture");
        api_log!("CommandEncoder::clear_texture {dst:?}");

        let hub = &self.hub;

        let cmd_buf = hub
            .command_buffers
            .get(command_encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.try_get()?;
        cmd_buf_data.check_recording()?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::ClearTexture {
                dst,
                subresource_range: *subresource_range,
            });
        }

        if !cmd_buf.support_clear_texture {
            return Err(ClearError::MissingClearTextureFeature);
        }

        let dst_texture = hub.textures.get(dst).get()?;

        dst_texture.same_device_as(cmd_buf.as_ref())?;

        // Check if subresource aspects are valid.
        let clear_aspects =
            hal::FormatAspects::new(dst_texture.desc.format, subresource_range.aspect);
        if clear_aspects.is_empty() {
            return Err(ClearError::MissingTextureAspect {
                texture_format: dst_texture.desc.format,
                subresource_range_aspects: subresource_range.aspect,
            });
        };

        // Check if subresource level range is valid
        let subresource_mip_range = subresource_range.mip_range(dst_texture.full_range.mips.end);
        if dst_texture.full_range.mips.start > subresource_mip_range.start
            || dst_texture.full_range.mips.end < subresource_mip_range.end
        {
            return Err(ClearError::InvalidTextureLevelRange {
                texture_level_range: dst_texture.full_range.mips.clone(),
                subresource_base_mip_level: subresource_range.base_mip_level,
                subresource_mip_level_count: subresource_range.mip_level_count,
            });
        }
        // Check if subresource layer range is valid
        let subresource_layer_range =
            subresource_range.layer_range(dst_texture.full_range.layers.end);
        if dst_texture.full_range.layers.start > subresource_layer_range.start
            || dst_texture.full_range.layers.end < subresource_layer_range.end
        {
            return Err(ClearError::InvalidTextureLayerRange {
                texture_layer_range: dst_texture.full_range.layers.clone(),
                subresource_base_array_layer: subresource_range.base_array_layer,
                subresource_array_layer_count: subresource_range.array_layer_count,
            });
        }

        let device = &cmd_buf.device;
        device.check_is_valid()?;
        let (encoder, tracker) = cmd_buf_data.open_encoder_and_tracker(&cmd_buf.device)?;

        let snatch_guard = device.snatchable_lock.read();
        clear_texture(
            &dst_texture,
            TextureInitRange {
                mip_range: subresource_mip_range,
                layer_range: subresource_layer_range,
            },
            encoder,
            &mut tracker.textures,
            &device.alignments,
            device.zero_buffer.as_ref(),
            &snatch_guard,
        )
    }
}

pub(crate) fn clear_texture<T: TextureTrackerSetSingle>(
    dst_texture: &Arc<Texture>,
    range: TextureInitRange,
    encoder: &mut dyn hal::DynCommandEncoder,
    texture_tracker: &mut T,
    alignments: &hal::Alignments,
    zero_buffer: &dyn hal::DynBuffer,
    snatch_guard: &SnatchGuard<'_>,
) -> Result<(), ClearError> {
    let dst_raw = dst_texture.try_raw(snatch_guard)?;

    // Issue the right barrier.
    let clear_usage = match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => hal::TextureUses::COPY_DST,
        TextureClearMode::RenderPass {
            is_color: false, ..
        } => hal::TextureUses::DEPTH_STENCIL_WRITE,
        TextureClearMode::Surface { .. } | TextureClearMode::RenderPass { is_color: true, .. } => {
            hal::TextureUses::COLOR_TARGET
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(
                dst_texture.error_ident(),
            ));
        }
    };

    let selector = TextureSelector {
        mips: range.mip_range.clone(),
        layers: range.layer_range.clone(),
    };

    // If we're in a texture-init usecase, we know that the texture is already
    // tracked since whatever caused the init requirement, will have caused the
    // usage tracker to be aware of the texture. Meaning, that it is safe to
    // call call change_replace_tracked if the life_guard is already gone (i.e.
    // the user no longer holds on to this texture).
    //
    // On the other hand, when coming via command_encoder_clear_texture, the
    // life_guard is still there since in order to call it a texture object is
    // needed.
    //
    // We could in theory distinguish these two scenarios in the internal
    // clear_texture api in order to remove this check and call the cheaper
    // change_replace_tracked whenever possible.
    let dst_barrier = texture_tracker
        .set_single(dst_texture, selector, clear_usage)
        .map(|pending| pending.into_hal(dst_raw))
        .collect::<Vec<_>>();
    unsafe {
        encoder.transition_textures(&dst_barrier);
    }

    // Record actual clearing
    match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => clear_texture_via_buffer_copies(
            &dst_texture.desc,
            alignments,
            zero_buffer,
            range,
            encoder,
            dst_raw,
        ),
        TextureClearMode::Surface { .. } => {
            clear_texture_via_render_passes(dst_texture, range, true, encoder)
        }
        TextureClearMode::RenderPass { is_color, .. } => {
            clear_texture_via_render_passes(dst_texture, range, is_color, encoder)
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(
                dst_texture.error_ident(),
            ));
        }
    }
    Ok(())
}

fn clear_texture_via_buffer_copies(
    texture_desc: &wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    alignments: &hal::Alignments,
    zero_buffer: &dyn hal::DynBuffer, // Buffer of size device::ZERO_BUFFER_SIZE
    range: TextureInitRange,
    encoder: &mut dyn hal::DynCommandEncoder,
    dst_raw: &dyn hal::DynTexture,
) {
    assert!(!texture_desc.format.is_depth_stencil_format());

    if texture_desc.format == wgt::TextureFormat::NV12 {
        // TODO: Currently COPY_DST for NV12 textures is unsupported.
        return;
    }

    // Gather list of zero_buffer copies and issue a single command then to perform them
    let mut zero_buffer_copy_regions = Vec::new();
    let buffer_copy_pitch = alignments.buffer_copy_pitch.get() as u32;
    let (block_width, block_height) = texture_desc.format.block_dimensions();
    let block_size = texture_desc.format.block_copy_size(None).unwrap();

    let bytes_per_row_alignment = get_lowest_common_denom(buffer_copy_pitch, block_size);

    for mip_level in range.mip_range {
        let mut mip_size = texture_desc.mip_level_size(mip_level).unwrap();
        // Round to multiple of block size
        mip_size.width = align_to(mip_size.width, block_width);
        mip_size.height = align_to(mip_size.height, block_height);

        let bytes_per_row = align_to(
            mip_size.width / block_width * block_size,
            bytes_per_row_alignment,
        );

        let max_rows_per_copy = crate::device::ZERO_BUFFER_SIZE as u32 / bytes_per_row;
        // round down to a multiple of rows needed by the texture format
        let max_rows_per_copy = max_rows_per_copy / block_height * block_height;
        assert!(
            max_rows_per_copy > 0,
            "Zero buffer size is too small to fill a single row \
            of a texture with format {:?} and desc {:?}",
            texture_desc.format,
            texture_desc.size
        );

        let z_range = 0..(if texture_desc.dimension == wgt::TextureDimension::D3 {
            mip_size.depth_or_array_layers
        } else {
            1
        });

        for array_layer in range.layer_range.clone() {
            // TODO: Only doing one layer at a time for volume textures right now.
            for z in z_range.clone() {
                // May need multiple copies for each subresource! However, we
                // assume that we never need to split a row.
                let mut num_rows_left = mip_size.height;
                while num_rows_left > 0 {
                    let num_rows = num_rows_left.min(max_rows_per_copy);

                    zero_buffer_copy_regions.push(hal::BufferTextureCopy {
                        buffer_layout: wgt::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
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
                            aspect: hal::FormatAspects::COLOR,
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
        encoder.copy_buffer_to_texture(zero_buffer, dst_raw, &zero_buffer_copy_regions);
    }
}

fn clear_texture_via_render_passes(
    dst_texture: &Texture,
    range: TextureInitRange,
    is_color: bool,
    encoder: &mut dyn hal::DynCommandEncoder,
) {
    assert_eq!(dst_texture.desc.dimension, wgt::TextureDimension::D2);

    let extent_base = wgt::Extent3d {
        width: dst_texture.desc.size.width,
        height: dst_texture.desc.size.height,
        depth_or_array_layers: 1, // Only one layer is cleared at a time.
    };

    for mip_level in range.mip_range {
        let extent = extent_base.mip_level_size(mip_level, dst_texture.desc.dimension);
        for depth_or_layer in range.layer_range.clone() {
            let color_attachments_tmp;
            let (color_attachments, depth_stencil_attachment) = if is_color {
                color_attachments_tmp = [Some(hal::ColorAttachment {
                    target: hal::Attachment {
                        view: Texture::get_clear_view(
                            &dst_texture.clear_mode,
                            &dst_texture.desc,
                            mip_level,
                            depth_or_layer,
                        ),
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
                            view: Texture::get_clear_view(
                                &dst_texture.clear_mode,
                                &dst_texture.desc,
                                mip_level,
                                depth_or_layer,
                            ),
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
                    sample_count: dst_texture.desc.sample_count,
                    color_attachments,
                    depth_stencil_attachment,
                    multiview: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                encoder.end_render_pass();
            }
        }
    }
}
