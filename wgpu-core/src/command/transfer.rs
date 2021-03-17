/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    command::{CommandBuffer, CommandEncoderError},
    conv,
    device::{all_buffer_stages, all_image_stages},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id::{BufferId, CommandEncoderId, TextureId},
    resource::{BufferUse, Texture, TextureErrorDimension, TextureUse},
    span,
    track::TextureSelector,
};

use hal::command::CommandBuffer as _;
use thiserror::Error;
use wgt::{BufferAddress, BufferUsage, Extent3d, TextureUsage};

use std::iter;

pub(crate) const BITS_PER_BYTE: u32 = 8;

pub type BufferCopyView = wgt::BufferCopyView<BufferId>;
pub type TextureCopyView = wgt::TextureCopyView<TextureId>;

#[derive(Clone, Debug)]
pub enum CopySide {
    Source,
    Destination,
}

/// Error encountered while attempting a data transfer.
#[derive(Clone, Debug, Error)]
pub enum TransferError {
    #[error("buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("Source and destination cannot be the same buffer")]
    SameSourceDestinationBuffer,
    #[error("source buffer/texture is missing the `COPY_SRC` usage flag")]
    MissingCopySrcUsageFlag,
    #[error("destination buffer/texture is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
    #[error("copy of {start_offset}..{end_offset} would end up overruning the bounds of the {side:?} buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
        side: CopySide,
    },
    #[error("copy of {dimension:?} {start_offset}..{end_offset} would end up overruning the bounds of the {side:?} texture of {dimension:?} size {texture_size}")]
    TextureOverrun {
        start_offset: u32,
        end_offset: u32,
        texture_size: u32,
        dimension: TextureErrorDimension,
        side: CopySide,
    },
    #[error("buffer offset {0} is not aligned to block size or `COPY_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(BufferAddress),
    #[error("copy size {0} does not respect `COPY_BUFFER_ALIGNMENT`")]
    UnalignedCopySize(BufferAddress),
    #[error("copy width is not a multiple of block width")]
    UnalignedCopyWidth,
    #[error("copy height is not a multiple of block height")]
    UnalignedCopyHeight,
    #[error("copy origin's x component is not a multiple of block width")]
    UnalignedCopyOriginX,
    #[error("copy origin's y component is not a multiple of block height")]
    UnalignedCopyOriginY,
    #[error("bytes per row does not respect `COPY_BYTES_PER_ROW_ALIGNMENT`")]
    UnalignedBytesPerRow,
    #[error("number of rows per image is not a multiple of block height")]
    UnalignedRowsPerImage,
    #[error("number of bytes per row is less than the number of bytes in a complete row")]
    InvalidBytesPerRow,
    #[error("image is 1D and the copy height and depth are not both set to 1")]
    InvalidCopySize,
    #[error("number of rows per image is invalid")]
    InvalidRowsPerImage,
    #[error("source and destination layers have different aspects")]
    MismatchedAspects,
    #[error("copying from textures with format {0:?} is forbidden")]
    CopyFromForbiddenTextureFormat(wgt::TextureFormat),
    #[error("copying to textures with format {0:?} is forbidden")]
    CopyToForbiddenTextureFormat(wgt::TextureFormat),
}

/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
pub enum CopyError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Copy error")]
    Transfer(#[from] TransferError),
}

//TODO: we currently access each texture twice for a transfer,
// once only to get the aspect flags, which is unfortunate.
pub(crate) fn texture_copy_view_to_hal<B: hal::Backend>(
    view: &TextureCopyView,
    size: &Extent3d,
    texture_guard: &Storage<Texture<B>, TextureId>,
) -> Result<
    (
        hal::image::SubresourceLayers,
        TextureSelector,
        hal::image::Offset,
    ),
    TransferError,
> {
    let texture = texture_guard
        .get(view.texture)
        .map_err(|_| TransferError::InvalidTexture(view.texture))?;

    let level = view.mip_level as hal::image::Level;
    let (layer, layer_count, z) = match texture.dimension {
        wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => (
            view.origin.z as hal::image::Layer,
            size.depth as hal::image::Layer,
            0,
        ),
        wgt::TextureDimension::D3 => (0, 1, view.origin.z as i32),
    };

    // TODO: Can't satisfy clippy here unless we modify
    // `TextureSelector` to use `std::ops::RangeBounds`.
    #[allow(clippy::range_plus_one)]
    Ok((
        hal::image::SubresourceLayers {
            aspects: texture.aspects,
            level,
            layers: layer..layer + layer_count,
        },
        TextureSelector {
            levels: level..level + 1,
            layers: layer..layer + layer_count,
        },
        hal::image::Offset {
            x: view.origin.x as i32,
            y: view.origin.y as i32,
            z,
        },
    ))
}

/// Function copied with minor modifications from webgpu standard https://gpuweb.github.io/gpuweb/#valid-texture-copy-range
pub(crate) fn validate_linear_texture_data(
    layout: &wgt::TextureDataLayout,
    format: wgt::TextureFormat,
    buffer_size: BufferAddress,
    buffer_side: CopySide,
    bytes_per_block: BufferAddress,
    copy_size: &Extent3d,
) -> Result<(), TransferError> {
    // Convert all inputs to BufferAddress (u64) to prevent overflow issues
    let copy_width = copy_size.width as BufferAddress;
    let copy_height = copy_size.height as BufferAddress;
    let copy_depth = copy_size.depth as BufferAddress;

    let offset = layout.offset;
    let rows_per_image = layout.rows_per_image as BufferAddress;
    let bytes_per_row = layout.bytes_per_row as BufferAddress;

    let (block_width, block_height) = conv::texture_block_size(format);
    let block_width = block_width as BufferAddress;
    let block_height = block_height as BufferAddress;
    let block_size = bytes_per_block;

    if copy_width % block_width != 0 {
        return Err(TransferError::UnalignedCopyWidth);
    }
    if copy_height % block_height != 0 {
        return Err(TransferError::UnalignedCopyHeight);
    }
    if rows_per_image % block_height != 0 {
        return Err(TransferError::UnalignedRowsPerImage);
    }

    let bytes_in_a_complete_row = block_size * copy_width / block_width;
    let required_bytes_in_copy = if copy_width == 0 || copy_height == 0 || copy_depth == 0 {
        0
    } else {
        let actual_rows_per_image = if rows_per_image == 0 {
            copy_height
        } else {
            rows_per_image
        };
        let texel_block_rows_per_image = actual_rows_per_image / block_height;
        let bytes_per_image = bytes_per_row * texel_block_rows_per_image;
        let bytes_in_last_slice =
            bytes_per_row * (copy_height / block_height - 1) + bytes_in_a_complete_row;
        bytes_per_image * (copy_depth - 1) + bytes_in_last_slice
    };

    if rows_per_image != 0 && rows_per_image < copy_height {
        return Err(TransferError::InvalidRowsPerImage);
    }
    if offset + required_bytes_in_copy > buffer_size {
        return Err(TransferError::BufferOverrun {
            start_offset: offset,
            end_offset: offset + required_bytes_in_copy,
            buffer_size,
            side: buffer_side,
        });
    }
    if offset % block_size != 0 {
        return Err(TransferError::UnalignedBufferOffset(offset));
    }
    if copy_height > 1 && bytes_per_row < bytes_in_a_complete_row {
        return Err(TransferError::InvalidBytesPerRow);
    }
    if copy_depth > 1 && rows_per_image == 0 {
        return Err(TransferError::InvalidRowsPerImage);
    }
    Ok(())
}

/// Function copied with minor modifications from webgpu standard https://gpuweb.github.io/gpuweb/#valid-texture-copy-range
pub(crate) fn validate_texture_copy_range(
    texture_copy_view: &TextureCopyView,
    texture_format: wgt::TextureFormat,
    texture_dimension: hal::image::Kind,
    texture_side: CopySide,
    copy_size: &Extent3d,
) -> Result<(), TransferError> {
    let (block_width, block_height) = conv::texture_block_size(texture_format);

    let mut extent = texture_dimension.level_extent(texture_copy_view.mip_level as u8);
    match texture_dimension {
        hal::image::Kind::D1(..) => {
            if (copy_size.height, copy_size.depth) != (1, 1) {
                return Err(TransferError::InvalidCopySize);
            }
        }
        hal::image::Kind::D2(_, _, array_layers, _) => {
            extent.depth = array_layers as u32;
        }
        hal::image::Kind::D3(..) => {}
    };

    let x_copy_max = texture_copy_view.origin.x + copy_size.width;
    if x_copy_max > extent.width {
        return Err(TransferError::TextureOverrun {
            start_offset: texture_copy_view.origin.x,
            end_offset: x_copy_max,
            texture_size: extent.width,
            dimension: TextureErrorDimension::X,
            side: texture_side,
        });
    }
    let y_copy_max = texture_copy_view.origin.y + copy_size.height;
    if y_copy_max > extent.height {
        return Err(TransferError::TextureOverrun {
            start_offset: texture_copy_view.origin.y,
            end_offset: y_copy_max,
            texture_size: extent.height,
            dimension: TextureErrorDimension::Y,
            side: texture_side,
        });
    }
    let z_copy_max = texture_copy_view.origin.z + copy_size.depth;
    if z_copy_max > extent.depth {
        return Err(TransferError::TextureOverrun {
            start_offset: texture_copy_view.origin.z,
            end_offset: z_copy_max,
            texture_size: extent.depth,
            dimension: TextureErrorDimension::Z,
            side: texture_side,
        });
    }

    if texture_copy_view.origin.x % block_width != 0 {
        return Err(TransferError::UnalignedCopyOriginX);
    }
    if texture_copy_view.origin.y % block_height != 0 {
        return Err(TransferError::UnalignedCopyOriginY);
    }
    if copy_size.width % block_width != 0 {
        return Err(TransferError::UnalignedCopyWidth);
    }
    if copy_size.height % block_height != 0 {
        return Err(TransferError::UnalignedCopyHeight);
    }
    Ok(())
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_copy_buffer_to_buffer<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: BufferAddress,
    ) -> Result<(), CopyError> {
        span!(_guard, INFO, "CommandEncoder::copy_buffer_to_buffer");

        if source == destination {
            Err(TransferError::SameSourceDestinationBuffer)?
        }
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        // we can't hold both src_pending and dst_pending in scope because they
        // borrow the buffer tracker mutably...
        let mut barriers = Vec::new();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::CopyBufferToBuffer {
                src: source,
                src_offset: source_offset,
                dst: destination,
                dst_offset: destination_offset,
                size,
            });
        }

        let (src_buffer, src_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, source, (), BufferUse::COPY_SRC)
            .map_err(TransferError::InvalidBuffer)?;
        let &(ref src_raw, _) = src_buffer
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(source))?;
        if !src_buffer.usage.contains(BufferUsage::COPY_SRC) {
            Err(TransferError::MissingCopySrcUsageFlag)?
        }
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_buffer)));

        let (dst_buffer, dst_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, destination, (), BufferUse::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        let &(ref dst_raw, _) = dst_buffer
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(destination))?;
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                Some(destination),
                None,
            ))?
        }
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_buffer)));

        if size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            Err(TransferError::UnalignedCopySize(size))?
        }
        if source_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            Err(TransferError::UnalignedBufferOffset(source_offset))?
        }
        if destination_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            Err(TransferError::UnalignedBufferOffset(destination_offset))?
        }

        let source_end_offset = source_offset + size;
        let destination_end_offset = destination_offset + size;
        if source_end_offset > src_buffer.size {
            Err(TransferError::BufferOverrun {
                start_offset: source_offset,
                end_offset: source_end_offset,
                buffer_size: src_buffer.size,
                side: CopySide::Source,
            })?
        }
        if destination_end_offset > dst_buffer.size {
            Err(TransferError::BufferOverrun {
                start_offset: destination_offset,
                end_offset: destination_end_offset,
                buffer_size: dst_buffer.size,
                side: CopySide::Destination,
            })?
        }

        if size == 0 {
            tracing::trace!("Ignoring copy_buffer_to_buffer of size 0");
            return Ok(());
        }

        let region = hal::command::BufferCopy {
            src: source_offset,
            dst: destination_offset,
            size,
        };
        let cmb_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmb_raw.copy_buffer(src_raw, dst_raw, iter::once(region));
        }
        Ok(())
    }

    pub fn command_encoder_copy_buffer_to_texture<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &BufferCopyView,
        destination: &TextureCopyView,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        span!(_guard, INFO, "CommandEncoder::copy_buffer_to_texture");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (dst_layers, dst_selector, dst_offset) =
            texture_copy_view_to_hal(destination, copy_size, &*texture_guard)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::CopyBufferToTexture {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            });
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            tracing::trace!("Ignoring copy_buffer_to_texture of size 0");
            return Ok(());
        }

        let (src_buffer, src_pending) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, source.buffer, (), BufferUse::COPY_SRC)
            .map_err(TransferError::InvalidBuffer)?;
        let &(ref src_raw, _) = src_buffer
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(source.buffer))?;
        if !src_buffer.usage.contains(BufferUsage::COPY_SRC) {
            Err(TransferError::MissingCopySrcUsageFlag)?
        }
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

        let (dst_texture, dst_pending) = cmd_buf
            .trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                dst_selector,
                TextureUse::COPY_DST,
            )
            .unwrap();
        let &(ref dst_raw, _) = dst_texture
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        if !dst_texture.usage.contains(TextureUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                None,
                Some(destination.texture),
            ))?
        }
        let dst_barriers = dst_pending.map(|pending| pending.into_hal(dst_texture));

        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_block = conv::map_texture_format(dst_texture.format, cmd_buf.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let src_bytes_per_row = source.layout.bytes_per_row;
        if bytes_per_row_alignment % bytes_per_block != 0 {
            Err(TransferError::UnalignedBytesPerRow)?
        }
        if src_bytes_per_row % bytes_per_row_alignment != 0 {
            Err(TransferError::UnalignedBytesPerRow)?
        }
        validate_texture_copy_range(
            destination,
            dst_texture.format,
            dst_texture.kind,
            CopySide::Destination,
            copy_size,
        )?;
        validate_linear_texture_data(
            &source.layout,
            dst_texture.format,
            src_buffer.size,
            CopySide::Source,
            bytes_per_block as BufferAddress,
            copy_size,
        )?;

        let (block_width, _) = conv::texture_block_size(dst_texture.format);
        if !conv::is_valid_copy_dst_texture_format(dst_texture.format) {
            Err(TransferError::CopyToForbiddenTextureFormat(
                dst_texture.format,
            ))?
        }

        let buffer_width = (source.layout.bytes_per_row / bytes_per_block) * block_width;
        let region = hal::command::BufferImageCopy {
            buffer_offset: source.layout.offset,
            buffer_width,
            buffer_height: source.layout.rows_per_image,
            image_layers: dst_layers,
            image_offset: dst_offset,
            image_extent: conv::map_extent(copy_size, dst_texture.dimension),
        };
        let cmb_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages() | all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                src_barriers.chain(dst_barriers),
            );
            cmb_raw.copy_buffer_to_image(
                src_raw,
                dst_raw,
                hal::image::Layout::TransferDstOptimal,
                iter::once(region),
            );
        }
        Ok(())
    }

    pub fn command_encoder_copy_texture_to_buffer<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TextureCopyView,
        destination: &BufferCopyView,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        span!(_guard, INFO, "CommandEncoder::copy_texture_to_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder(&mut *cmd_buf_guard, command_encoder_id)?;
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (src_layers, src_selector, src_offset) =
            texture_copy_view_to_hal(source, copy_size, &*texture_guard)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::CopyTextureToBuffer {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            });
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            tracing::trace!("Ignoring copy_texture_to_buffer of size 0");
            return Ok(());
        }

        let (src_texture, src_pending) = cmd_buf
            .trackers
            .textures
            .use_replace(
                &*texture_guard,
                source.texture,
                src_selector,
                TextureUse::COPY_SRC,
            )
            .unwrap();
        let &(ref src_raw, _) = src_texture
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        if !src_texture.usage.contains(TextureUsage::COPY_SRC) {
            Err(TransferError::MissingCopySrcUsageFlag)?
        }
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_texture));

        let (dst_buffer, dst_barriers) = cmd_buf
            .trackers
            .buffers
            .use_replace(&*buffer_guard, destination.buffer, (), BufferUse::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        let &(ref dst_raw, _) = dst_buffer
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(destination.buffer))?;
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                Some(destination.buffer),
                None,
            ))?
        }
        let dst_barrier = dst_barriers.map(|pending| pending.into_hal(dst_buffer));

        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_block = conv::map_texture_format(src_texture.format, cmd_buf.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let dst_bytes_per_row = destination.layout.bytes_per_row;
        if bytes_per_row_alignment % bytes_per_block != 0 {
            Err(TransferError::UnalignedBytesPerRow)?
        }
        if dst_bytes_per_row % bytes_per_row_alignment != 0 {
            Err(TransferError::UnalignedBytesPerRow)?
        }
        validate_texture_copy_range(
            source,
            src_texture.format,
            src_texture.kind,
            CopySide::Source,
            copy_size,
        )?;
        validate_linear_texture_data(
            &destination.layout,
            src_texture.format,
            dst_buffer.size,
            CopySide::Destination,
            bytes_per_block as BufferAddress,
            copy_size,
        )?;

        let (block_width, _) = conv::texture_block_size(src_texture.format);
        if !conv::is_valid_copy_src_texture_format(src_texture.format) {
            Err(TransferError::CopyFromForbiddenTextureFormat(
                src_texture.format,
            ))?
        }

        let buffer_width = (destination.layout.bytes_per_row / bytes_per_block) * block_width;
        let region = hal::command::BufferImageCopy {
            buffer_offset: destination.layout.offset,
            buffer_width,
            buffer_height: destination.layout.rows_per_image,
            image_layers: src_layers,
            image_offset: src_offset,
            image_extent: conv::map_extent(copy_size, src_texture.dimension),
        };
        let cmb_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages() | all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                src_barriers.chain(dst_barrier),
            );
            cmb_raw.copy_image_to_buffer(
                src_raw,
                hal::image::Layout::TransferSrcOptimal,
                dst_raw,
                iter::once(region),
            );
        }
        Ok(())
    }

    pub fn command_encoder_copy_texture_to_texture<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TextureCopyView,
        destination: &TextureCopyView,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        span!(_guard, INFO, "CommandEncoder::copy_texture_to_texture");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder(&mut *cmd_buf_guard, command_encoder_id)?;
        let (_, mut token) = hub.buffers.read(&mut token); // skip token
        let (texture_guard, _) = hub.textures.read(&mut token);
        // we can't hold both src_pending and dst_pending in scope because they
        // borrow the buffer tracker mutably...
        let mut barriers = Vec::new();
        let (src_layers, src_selector, src_offset) =
            texture_copy_view_to_hal(source, copy_size, &*texture_guard)?;
        let (dst_layers, dst_selector, dst_offset) =
            texture_copy_view_to_hal(destination, copy_size, &*texture_guard)?;
        if src_layers.aspects != dst_layers.aspects {
            Err(TransferError::MismatchedAspects)?
        }

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::CopyTextureToTexture {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            });
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            tracing::trace!("Ignoring copy_texture_to_texture of size 0");
            return Ok(());
        }

        let (src_texture, src_pending) = cmd_buf
            .trackers
            .textures
            .use_replace(
                &*texture_guard,
                source.texture,
                src_selector,
                TextureUse::COPY_SRC,
            )
            .unwrap();
        let &(ref src_raw, _) = src_texture
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        if !src_texture.usage.contains(TextureUsage::COPY_SRC) {
            Err(TransferError::MissingCopySrcUsageFlag)?
        }
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_texture)));

        let (dst_texture, dst_pending) = cmd_buf
            .trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                dst_selector,
                TextureUse::COPY_DST,
            )
            .unwrap();
        let &(ref dst_raw, _) = dst_texture
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        if !dst_texture.usage.contains(TextureUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                None,
                Some(destination.texture),
            ))?
        }
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_texture)));

        validate_texture_copy_range(
            source,
            src_texture.format,
            src_texture.kind,
            CopySide::Source,
            copy_size,
        )?;
        validate_texture_copy_range(
            destination,
            dst_texture.format,
            dst_texture.kind,
            CopySide::Destination,
            copy_size,
        )?;

        let region = hal::command::ImageCopy {
            src_subresource: src_layers,
            src_offset,
            dst_subresource: dst_layers,
            dst_offset,
            extent: conv::map_extent(copy_size, src_texture.dimension),
        };
        let cmb_raw = cmd_buf.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmb_raw.copy_image(
                src_raw,
                hal::image::Layout::TransferSrcOptimal,
                dst_raw,
                hal::image::Layout::TransferDstOptimal,
                iter::once(region),
            );
        }
        Ok(())
    }
}
