use std::{num::NonZeroU32, ops::Range};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    align_to,
    command::CommandBuffer,
    device::Device,
    get_lowest_common_denom,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Resource, Token},
    id::{self, BufferId, CommandEncoderId, DeviceId, TextureId},
    init_tracker::MemoryInitKind,
    resource::{Texture, TextureClearMode},
    track::{ResourceTracker, TextureSelector, TextureState},
    Stored,
};

use hal::CommandEncoder as _;
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
    #[error("failed to create view for clearing texture at mip level {mip_level}, layer {layer}")]
    FailedToCreateTextureViewForClear { mip_level: u32, layer: u32 },
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
            .use_replace(&*buffer_guard, dst, (), hal::BufferUses::COPY_DST)
            .map_err(ClearError::InvalidBuffer)?;
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
            cmd_buf_raw.transition_buffers(dst_barrier);
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
        let (mut texture_guard, _) = hub.textures.write(&mut token);

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
            .get_mut(dst) // todo: take only write access if needed
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
            None => dst_texture.full_range.levels.end,
        };
        if dst_texture.full_range.levels.start > subresource_range.base_mip_level
            || dst_texture.full_range.levels.end < subresource_level_end
        {
            return Err(ClearError::InvalidTextureLevelRange {
                texture_level_range: dst_texture.full_range.levels.clone(),
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

        clear_texture(
            &Stored {
                value: id::Valid(dst),
                ref_count: dst_texture.life_guard().ref_count.as_ref().unwrap().clone(),
            },
            dst_texture,
            subresource_range.base_mip_level..subresource_level_end,
            subresource_range.base_array_layer..subresource_layer_end,
            cmd_buf.encoder.open(),
            &mut cmd_buf.trackers.textures,
            &device_guard[cmd_buf.device_id.value],
        )
    }
}

pub(crate) fn clear_texture<A: hal::Api>(
    dst_texture_id: &Stored<TextureId>,
    dst_texture: &Texture<A>,
    mip_range: Range<u32>,
    layer_range: Range<u32>,
    encoder: &mut A::CommandEncoder,
    texture_tracker: &mut ResourceTracker<TextureState>,
    device: &Device<A>,
) -> Result<(), ClearError> {
    clear_texture_no_device(
        dst_texture_id,
        dst_texture,
        mip_range,
        layer_range,
        encoder,
        texture_tracker,
        &device.alignments,
        &device.zero_buffer,
    )
}

pub(crate) fn clear_texture_no_device<A: hal::Api>(
    dst_texture_id: &Stored<TextureId>,
    dst_texture: &Texture<A>,
    mip_range: Range<u32>,
    layer_range: Range<u32>,
    encoder: &mut A::CommandEncoder,
    texture_tracker: &mut ResourceTracker<TextureState>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer,
) -> Result<(), ClearError> {
    let dst_raw = dst_texture
        .inner
        .as_raw()
        .ok_or(ClearError::InvalidTexture(dst_texture_id.value.0))?;

    // Issue the right barrier.
    let clear_usage = match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => hal::TextureUses::COPY_DST,
        TextureClearMode::RenderPass(_) => {
            if dst_texture
                .hal_usage
                .contains(hal::TextureUses::DEPTH_STENCIL_WRITE)
            {
                hal::TextureUses::DEPTH_STENCIL_WRITE
            } else {
                hal::TextureUses::COLOR_TARGET
            }
        }
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(dst_texture_id.value.0));
        }
    };

    let dst_barrier = texture_tracker
        .change_replace(
            dst_texture_id.value,
            &dst_texture_id.ref_count,
            TextureSelector {
                levels: mip_range.clone(),
                layers: layer_range.clone(),
            },
            clear_usage,
        )
        .map(|pending| pending.into_hal(dst_texture));
    unsafe {
        encoder.transition_textures(dst_barrier);
    }

    // Record actual clearing
    match dst_texture.clear_mode {
        TextureClearMode::BufferCopy => clear_texture_via_buffer_copies::<A>(
            &dst_texture.desc,
            alignments,
            zero_buffer,
            mip_range,
            layer_range,
            encoder,
            dst_raw,
        ),
        TextureClearMode::RenderPass(_) => clear_texture_via_render_passes(
            dst_texture,
            mip_range,
            layer_range,
            clear_usage,
            encoder,
        )?,
        TextureClearMode::None => {
            return Err(ClearError::NoValidTextureClearMode(dst_texture_id.value.0));
        }
    }
    Ok(())
}

fn clear_texture_via_buffer_copies<A: hal::Api>(
    texture_desc: &wgt::TextureDescriptor<()>,
    alignments: &hal::Alignments,
    zero_buffer: &A::Buffer,
    mip_range: Range<u32>,
    layer_range: Range<u32>,
    encoder: &mut A::CommandEncoder,
    dst_raw: &A::Texture,
) {
    let mut zero_buffer_copy_regions = Vec::new();
    collect_zero_buffer_copies_for_clear_texture(
        &texture_desc,
        alignments.buffer_copy_pitch.get() as u32,
        mip_range,
        layer_range,
        &mut zero_buffer_copy_regions,
    );
    if !zero_buffer_copy_regions.is_empty() {
        unsafe {
            encoder.copy_buffer_to_texture(
                zero_buffer,
                dst_raw,
                zero_buffer_copy_regions.into_iter(),
            );
        }
    }
}

pub(crate) fn collect_zero_buffer_copies_for_clear_texture(
    texture_desc: &wgt::TextureDescriptor<()>,
    buffer_copy_pitch: u32,
    mip_range: Range<u32>,
    layer_range: Range<u32>,
    out_copy_regions: &mut Vec<hal::BufferTextureCopy>, // TODO: Something better than Vec
) {
    let format_desc = texture_desc.format.describe();

    let bytes_per_row_alignment =
        get_lowest_common_denom(buffer_copy_pitch, format_desc.block_size as u32);

    for mip_level in mip_range {
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

        for array_layer in layer_range.clone() {
            // TODO: Only doing one layer at a time for volume textures right now.
            for z in z_range.clone() {
                // May need multiple copies for each subresource! However, we assume that we never need to split a row.
                let mut num_rows_left = mip_size.height;
                while num_rows_left > 0 {
                    let num_rows = num_rows_left.min(max_rows_per_copy);

                    out_copy_regions.push(hal::BufferTextureCopy {
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
}

fn clear_texture_via_render_passes<A: hal::Api>(
    dst_texture: &Texture<A>,
    mip_range: Range<u32>,
    layer_range: Range<u32>,
    clear_usage: hal::TextureUses,
    encoder: &mut A::CommandEncoder,
) -> Result<(), ClearError> {
    let extent_base = wgt::Extent3d {
        width: dst_texture.desc.size.width,
        height: dst_texture.desc.size.height,
        depth_or_array_layers: 1, // TODO: What about 3d textures? Only one slice a time, sure but how to select it?
    };
    let sample_count = dst_texture.desc.sample_count;
    let is_3d_texture = dst_texture.desc.dimension == wgt::TextureDimension::D3;
    for mip_level in mip_range {
        let extent = extent_base.mip_level_size(mip_level, is_3d_texture);
        for layer in layer_range.clone() {
            let target = hal::Attachment {
                view: dst_texture.get_clear_view(mip_level, layer),
                usage: clear_usage,
            };

            if clear_usage == hal::TextureUses::DEPTH_STENCIL_WRITE {
                unsafe {
                    encoder.begin_render_pass(&hal::RenderPassDescriptor {
                        label: Some("clear_texture clear pass"),
                        extent,
                        sample_count,
                        color_attachments: &[],
                        depth_stencil_attachment: Some(hal::DepthStencilAttachment {
                            target,
                            depth_ops: hal::AttachmentOps::STORE,
                            stencil_ops: hal::AttachmentOps::STORE,
                            clear_value: (0.0, 0),
                        }),
                        multiview: None,
                    });
                }
            } else {
                unsafe {
                    encoder.begin_render_pass(&hal::RenderPassDescriptor {
                        label: Some("clear_texture clear pass"),
                        extent,
                        sample_count,
                        color_attachments: &[hal::ColorAttachment {
                            target,
                            resolve_target: None,
                            ops: hal::AttachmentOps::STORE,
                            clear_value: wgt::Color::TRANSPARENT,
                        }],
                        depth_stencil_attachment: None,
                        multiview: None,
                    });
                }
            };
            unsafe {
                encoder.end_render_pass();
            }
        }
    }
    Ok(())
}
