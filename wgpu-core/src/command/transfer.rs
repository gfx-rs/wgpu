/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    conv,
    device::{all_buffer_stages, all_image_stages},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id::{BufferId, CommandEncoderId, TextureId},
    resource::{BufferUse, Texture, TextureUse},
    span,
};

use hal::command::CommandBuffer as _;
use thiserror::Error;
use wgt::{BufferAddress, BufferUsage, Extent3d, TextureDataLayout, TextureUsage};

use std::iter;

type Result = std::result::Result<(), TransferError>;

pub(crate) const BITS_PER_BYTE: u32 = 8;

pub type BufferCopyView = wgt::BufferCopyView<BufferId>;

pub type TextureCopyView = wgt::TextureCopyView<TextureId>;

/// Error encountered while attempting a data transfer.
#[derive(Copy, Clone, Debug, Error, Eq, PartialEq)]
pub enum TransferError {
    #[error("source buffer/texture is missing the `COPY_SRC` usage flag")]
    MissingCopySrcUsageFlag,
    #[error("destination buffer/texture is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag,
    #[error("copy would end up overruning the bounds of the destination buffer/texture")]
    BufferOverrun,
    #[error("buffer offset is not aligned to block size")]
    UnalignedBufferOffset,
    #[error("copy size is not a multiple of block size")]
    UnalignedCopySize,
    #[error("copy width is not a multiple of block size")]
    UnalignedCopyWidth,
    #[error("copy height is not a multiple of block size")]
    UnalignedCopyHeight,
    #[error("bytes per row is not a multiple of the required alignment")]
    UnalignedBytesPerRow,
    #[error("number of rows per image is not a multiple of the required alignment")]
    UnalignedRowsPerImage,
    #[error("number of bytes per row is less than the number of bytes in a complete row")]
    InvalidBytesPerRow,
    #[error("image is 1D and the copy height and depth are not both set to 1")]
    InvalidCopySize,
    #[error("number of rows per image is invalid")]
    InvalidRowsPerImage,
    #[error("source and destination layers have different aspects")]
    MismatchedAspects,
}

//TODO: we currently access each texture twice for a transfer,
// once only to get the aspect flags, which is unfortunate.
pub(crate) fn texture_copy_view_to_hal<B: hal::Backend>(
    view: &TextureCopyView,
    texture_guard: &Storage<Texture<B>, TextureId>,
) -> (
    hal::image::SubresourceLayers,
    hal::image::SubresourceRange,
    hal::image::Offset,
) {
    let texture = &texture_guard[view.texture];
    let aspects = texture.full_range.aspects;
    let level = view.mip_level as hal::image::Level;
    let (layer, z) = match texture.dimension {
        wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
            (view.origin.z as hal::image::Layer, 0)
        }
        wgt::TextureDimension::D3 => (0, view.origin.z as i32),
    };

    // TODO: Can't satisfy clippy here unless we modify
    // `hal::image::SubresourceRange` in gfx to use `std::ops::RangeBounds`.
    #[allow(clippy::range_plus_one)]
    (
        hal::image::SubresourceLayers {
            aspects,
            level: view.mip_level as hal::image::Level,
            layers: layer..layer + 1,
        },
        hal::image::SubresourceRange {
            aspects,
            levels: level..level + 1,
            layers: layer..layer + 1,
        },
        hal::image::Offset {
            x: view.origin.x as i32,
            y: view.origin.y as i32,
            z,
        },
    )
}

/// Function copied with minor modifications from webgpu standard https://gpuweb.github.io/gpuweb/#valid-texture-copy-range
pub(crate) fn validate_linear_texture_data(
    layout: &TextureDataLayout,
    buffer_size: BufferAddress,
    bytes_per_texel: BufferAddress,
    copy_size: &Extent3d,
) -> Result {
    // Convert all inputs to BufferAddress (u64) to prevent overflow issues
    let copy_width = copy_size.width as BufferAddress;
    let copy_height = copy_size.height as BufferAddress;
    let copy_depth = copy_size.depth as BufferAddress;

    let offset = layout.offset;
    let rows_per_image = layout.rows_per_image as BufferAddress;
    let bytes_per_row = layout.bytes_per_row as BufferAddress;

    // TODO: Once compressed textures are supported, these needs to be fixed
    let block_width: BufferAddress = 1;
    let block_height: BufferAddress = 1;
    let block_size = bytes_per_texel;

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
        return Err(TransferError::BufferOverrun);
    }
    if offset % block_size != 0 {
        return Err(TransferError::UnalignedBufferOffset);
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
    texture_dimension: hal::image::Kind,
    copy_size: &Extent3d,
) -> Result {
    // TODO: Once compressed textures are supported, these needs to be fixed
    let block_width: u32 = 1;
    let block_height: u32 = 1;

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
        return Err(TransferError::BufferOverrun);
    }
    let y_copy_max = texture_copy_view.origin.y + copy_size.height;
    if y_copy_max > extent.height {
        return Err(TransferError::BufferOverrun);
    }
    let z_copy_max = texture_copy_view.origin.z + copy_size.depth;
    if z_copy_max > extent.depth {
        return Err(TransferError::BufferOverrun);
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
    ) -> Result {
        span!(_guard, INFO, "CommandEncoder::copy_buffer_to_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        // we can't hold both src_pending and dst_pending in scope because they
        // borrow the buffer tracker mutably...
        let mut barriers = Vec::new();

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => list.push(TraceCommand::CopyBufferToBuffer {
                src: source,
                src_offset: source_offset,
                dst: destination,
                dst_offset: destination_offset,
                size,
            }),
            None => (),
        }

        if size == 0 {
            log::trace!("Ignoring copy_buffer_to_buffer of size 0");
            return Ok(());
        }

        let (src_buffer, src_pending) =
            cmb.trackers
                .buffers
                .use_replace(&*buffer_guard, source, (), BufferUse::COPY_SRC);
        if !src_buffer.usage.contains(BufferUsage::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag);
        }
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_buffer)));

        let (dst_buffer, dst_pending) =
            cmb.trackers
                .buffers
                .use_replace(&*buffer_guard, destination, (), BufferUse::COPY_DST);
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag);
        }
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_buffer)));

        if size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize);
        }
        if source_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset);
        }
        if destination_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset);
        }

        let source_end_offset = source_offset + size;
        let destination_end_offset = destination_offset + size;
        if source_end_offset > src_buffer.size {
            return Err(TransferError::BufferOverrun);
        }
        if destination_end_offset > dst_buffer.size {
            return Err(TransferError::BufferOverrun);
        }

        let region = hal::command::BufferCopy {
            src: source_offset,
            dst: destination_offset,
            size,
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmb_raw.copy_buffer(&src_buffer.raw, &dst_buffer.raw, iter::once(region));
        }
        Ok(())
    }

    pub fn command_encoder_copy_buffer_to_texture<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &BufferCopyView,
        destination: &TextureCopyView,
        copy_size: &Extent3d,
    ) -> Result {
        span!(_guard, INFO, "CommandEncoder::copy_buffer_to_texture");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (dst_layers, dst_range, dst_offset) =
            texture_copy_view_to_hal(destination, &*texture_guard);

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => list.push(TraceCommand::CopyBufferToTexture {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            }),
            None => (),
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            log::trace!("Ignoring copy_buffer_to_texture of size 0");
            return Ok(());
        }

        let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            source.buffer,
            (),
            BufferUse::COPY_SRC,
        );
        if !src_buffer.usage.contains(BufferUsage::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag);
        }
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            dst_range,
            TextureUse::COPY_DST,
        );
        if !dst_texture.usage.contains(TextureUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag);
        }
        let dst_barriers = dst_pending.map(|pending| pending.into_hal(dst_texture));

        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_texel = conv::map_texture_format(dst_texture.format, cmb.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let src_bytes_per_row = source.layout.bytes_per_row;
        if bytes_per_row_alignment % bytes_per_texel != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
        if src_bytes_per_row % bytes_per_row_alignment != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
        validate_texture_copy_range(destination, dst_texture.kind, copy_size)?;
        validate_linear_texture_data(
            &source.layout,
            src_buffer.size,
            bytes_per_texel as BufferAddress,
            copy_size,
        )?;

        let buffer_width = source.layout.bytes_per_row / bytes_per_texel;
        let region = hal::command::BufferImageCopy {
            buffer_offset: source.layout.offset,
            buffer_width,
            buffer_height: source.layout.rows_per_image,
            image_layers: dst_layers,
            image_offset: dst_offset,
            image_extent: conv::map_extent(copy_size, dst_texture.dimension),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages() | all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                src_barriers.chain(dst_barriers),
            );
            cmb_raw.copy_buffer_to_image(
                &src_buffer.raw,
                &dst_texture.raw,
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
    ) -> Result {
        span!(_guard, INFO, "CommandEncoder::copy_texture_to_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (src_layers, src_range, src_offset) = texture_copy_view_to_hal(source, &*texture_guard);

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => list.push(TraceCommand::CopyTextureToBuffer {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            }),
            None => (),
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            log::trace!("Ignoring copy_texture_to_buffer of size 0");
            return Ok(());
        }

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            src_range,
            TextureUse::COPY_SRC,
        );
        if !src_texture.usage.contains(TextureUsage::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag);
        }
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_texture));

        let (dst_buffer, dst_barriers) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            destination.buffer,
            (),
            BufferUse::COPY_DST,
        );
        if !dst_buffer.usage.contains(BufferUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag);
        }
        let dst_barrier = dst_barriers.map(|pending| pending.into_hal(dst_buffer));

        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_texel = conv::map_texture_format(src_texture.format, cmb.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let dst_bytes_per_row = destination.layout.bytes_per_row;
        if bytes_per_row_alignment % bytes_per_texel != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
        if dst_bytes_per_row % bytes_per_row_alignment != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
        validate_texture_copy_range(source, src_texture.kind, copy_size)?;
        validate_linear_texture_data(
            &destination.layout,
            dst_buffer.size,
            bytes_per_texel as BufferAddress,
            copy_size,
        )?;

        let buffer_width = destination.layout.bytes_per_row / bytes_per_texel;
        let region = hal::command::BufferImageCopy {
            buffer_offset: destination.layout.offset,
            buffer_width,
            buffer_height: destination.layout.rows_per_image,
            image_layers: src_layers,
            image_offset: src_offset,
            image_extent: conv::map_extent(copy_size, src_texture.dimension),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages() | all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                src_barriers.chain(dst_barrier),
            );
            cmb_raw.copy_image_to_buffer(
                &src_texture.raw,
                hal::image::Layout::TransferSrcOptimal,
                &dst_buffer.raw,
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
    ) -> Result {
        span!(_guard, INFO, "CommandEncoder::copy_texture_to_texture");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (_, mut token) = hub.buffers.read(&mut token); // skip token
        let (texture_guard, _) = hub.textures.read(&mut token);
        // we can't hold both src_pending and dst_pending in scope because they
        // borrow the buffer tracker mutably...
        let mut barriers = Vec::new();
        let (src_layers, src_range, src_offset) = texture_copy_view_to_hal(source, &*texture_guard);
        let (dst_layers, dst_range, dst_offset) =
            texture_copy_view_to_hal(destination, &*texture_guard);
        if src_layers.aspects != dst_layers.aspects {
            return Err(TransferError::MismatchedAspects);
        }

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => list.push(TraceCommand::CopyTextureToTexture {
                src: source.clone(),
                dst: destination.clone(),
                size: *copy_size,
            }),
            None => (),
        }

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.width == 0 {
            log::trace!("Ignoring copy_texture_to_texture of size 0");
            return Ok(());
        }

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            src_range,
            TextureUse::COPY_SRC,
        );
        if !src_texture.usage.contains(TextureUsage::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag);
        }
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_texture)));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            dst_range,
            TextureUse::COPY_DST,
        );
        if !dst_texture.usage.contains(TextureUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag);
        }
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_texture)));

        validate_texture_copy_range(source, src_texture.kind, copy_size)?;
        validate_texture_copy_range(destination, dst_texture.kind, copy_size)?;

        let region = hal::command::ImageCopy {
            src_subresource: src_layers,
            src_offset,
            dst_subresource: dst_layers,
            dst_offset,
            extent: conv::map_extent(copy_size, src_texture.dimension),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_image_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );
            cmb_raw.copy_image(
                &src_texture.raw,
                hal::image::Layout::TransferSrcOptimal,
                &dst_texture.raw,
                hal::image::Layout::TransferDstOptimal,
                iter::once(region),
            );
        }
        Ok(())
    }
}
