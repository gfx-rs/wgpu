use std::{ops::Range, sync::Arc};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    api_log,
    command::CommandBuffer,
    get_lowest_common_denom,
    global::Global,
    hal_api::HalApi,
    id::{BufferId, CommandEncoderId, DeviceId, TextureId},
    identity::GlobalIdentityHandlerFactory,
    init_tracker::{MemoryInitKind, TextureInitRange},
    resource::{Resource, Texture, TextureClearMode},
    track::{TextureSelector, TextureTracker},
};

use hal::CommandEncoder as _;
use thiserror::Error;
use wgt::{math::align_to, BufferAddress, BufferUsages, ImageSubresourceRange, TextureAspect};

/// Error encountered while attempting a clear.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ClearError {
    #[error("To use clear_texture the CLEAR_TEXTURE feature needs to be enabled")]
    MissingClearTextureFeature,
    #[error("Command encoder {0:?} is invalid")]
    InvalidCommandEncoder(CommandEncoderId),
    #[error("Device {0:?} is invalid")]
    InvalidDevice(DeviceId),
    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("Texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("Texture {0:?} can not be cleared")]
    NoValidTextureClearMode(TextureId),
    #[error("Buffer clear size {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedFillSize(BufferAddress),
    #[error("Buffer offset {0:?} is not a multiple of `COPY_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(BufferAddress),
    #[error("Clear of {start_offset}..{end_offset} would end up overrunning the bounds of the buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
    },
    #[error("Destination buffer is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
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
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_clear_buffer<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(), ClearError> {
        profiling::scope!("CommandEncoder::clear_buffer");
        api_log!("CommandEncoder::clear_buffer {dst:?}");

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::ClearBuffer { dst, offset, size });
        }

        let (dst_buffer, dst_pending) = {
            let buffer_guard = hub.buffers.read();
            let dst_buffer = buffer_guard
                .get(dst)
                .map_err(|_| ClearError::InvalidBuffer(dst))?;
            cmd_buf_data
                .trackers
                .buffers
                .set_single(dst_buffer, hal::BufferUses::COPY_DST)
                .ok_or(ClearError::InvalidBuffer(dst))?
        };
        let snatch_guard = dst_buffer.device.snatchable_lock.read();
        let dst_raw = dst_buffer
            .raw
            .get(&snatch_guard)
            .ok_or(ClearError::InvalidBuffer(dst))?;
        if !dst_buffer.usage.contains(BufferUsages::COPY_DST) {
            return Err(ClearError::MissingCopyDstUsageFlag(Some(dst), None));
        }

        // Check if offset & size are valid.
        if offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(ClearError::UnalignedBufferOffset(offset));
        }
        if let Some(size) = size {
            if size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(ClearError::UnalignedFillSize(size));
            }
            let destination_end_offset = offset + size;
            if destination_end_offset > dst_buffer.size {
                return Err(ClearError::BufferOverrun {
                    start_offset: offset,
                    end_offset: destination_end_offset,
                    buffer_size: dst_buffer.size,
                });
            }
        }

        let end = match size {
            Some(size) => offset + size,
            None => dst_buffer.size,
        };
        if offset == end {
            log::trace!("Ignoring fill_buffer of size 0");
            return Ok(());
        }

        // Mark dest as initialized.
        cmd_buf_data.buffer_memory_init_actions.extend(
            dst_buffer.initialization_status.read().create_action(
                &dst_buffer,
                offset..end,
                MemoryInitKind::ImplicitlyInitialized,
            ),
        );

        // actual hal barrier & operation
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));
        let cmd_buf_raw = cmd_buf_data.encoder.open();
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
        api_log!("CommandEncoder::clear_texture {dst:?}");

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)
            .map_err(|_| ClearError::InvalidCommandEncoder(command_encoder_id))?;
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

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

        let dst_texture = hub
            .textures
            .get(dst)
            .map_err(|_| ClearError::InvalidTexture(dst))?;

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
        if !device.is_valid() {
            return Err(ClearError::InvalidDevice(cmd_buf.device.as_info().id()));
        }
        let (encoder, tracker) = cmd_buf_data.open_encoder_and_tracker();

        clear_texture(
            &dst_texture,
            TextureInitRange {
                mip_range: subresource_mip_range,
                layer_range: subresource_layer_range,
            },
            encoder,
            &mut tracker.textures,
            &device.alignments,
            device.zero_buffer.as_ref().unwrap(),
        )
    }
}

pub(crate) fn clear_texture<A: HalApi>(
    dst_texture: &Arc<Texture<A>>,
    range: TextureInitRange,
    encoder: &mut A::CommandEncoder,
    texture_tracker: &mut TextureTracker<A>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer,
) -> Result<(), ClearError> {
    let dst_inner = dst_texture.inner();
    let dst_raw = dst_inner
        .as_ref()
        .unwrap()
        .as_raw()
        .ok_or_else(|| ClearError::InvalidTexture(dst_texture.as_info().id()))?;

    // Issue the right barrier.
    let clear_usage = match *dst_texture.clear_mode.read() {
        TextureClearMode::BufferCopy => hal::TextureUses::COPY_DST,
        TextureClearMode::RenderPass {
            is_color: false, ..
        } => hal::TextureUses::DEPTH_STENCIL_WRITE,
        TextureClearMode::Surface { .. } | TextureClearMode::RenderPass { is_color: true, .. } => {
            hal::TextureUses::COLOR_TARGET
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(
                dst_texture.as_info().id(),
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
        .unwrap()
        .map(|pending| pending.into_hal(dst_inner.as_ref().unwrap()));
    unsafe {
        encoder.transition_textures(dst_barrier.into_iter());
    }

    // Record actual clearing
    match *dst_texture.clear_mode.read() {
        TextureClearMode::BufferCopy => clear_texture_via_buffer_copies::<A>(
            &dst_texture.desc,
            alignments,
            zero_buffer,
            range,
            encoder,
            dst_raw,
        ),
        TextureClearMode::Surface { .. } => {
            clear_texture_via_render_passes(dst_texture, range, true, encoder)?
        }
        TextureClearMode::RenderPass { is_color, .. } => {
            clear_texture_via_render_passes(dst_texture, range, is_color, encoder)?
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(
                dst_texture.as_info().id(),
            ));
        }
    }
    Ok(())
}

fn clear_texture_via_buffer_copies<A: HalApi>(
    texture_desc: &wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer, // Buffer of size device::ZERO_BUFFER_SIZE
    range: TextureInitRange,
    encoder: &mut A::CommandEncoder,
    dst_raw: &A::Texture,
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
        encoder.copy_buffer_to_texture(zero_buffer, dst_raw, zero_buffer_copy_regions.into_iter());
    }
}

fn clear_texture_via_render_passes<A: HalApi>(
    dst_texture: &Texture<A>,
    range: TextureInitRange,
    is_color: bool,
    encoder: &mut A::CommandEncoder,
) -> Result<(), ClearError> {
    assert_eq!(dst_texture.desc.dimension, wgt::TextureDimension::D2);

    let extent_base = wgt::Extent3d {
        width: dst_texture.desc.size.width,
        height: dst_texture.desc.size.height,
        depth_or_array_layers: 1, // Only one layer is cleared at a time.
    };
    let clear_mode = &dst_texture.clear_mode.read();

    for mip_level in range.mip_range {
        let extent = extent_base.mip_level_size(mip_level, dst_texture.desc.dimension);
        for depth_or_layer in range.layer_range.clone() {
            let color_attachments_tmp;
            let (color_attachments, depth_stencil_attachment) = if is_color {
                color_attachments_tmp = [Some(hal::ColorAttachment {
                    target: hal::Attachment {
                        view: Texture::get_clear_view(
                            clear_mode,
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
                                clear_mode,
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
    Ok(())
}
