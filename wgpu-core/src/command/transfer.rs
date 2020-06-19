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
};

use hal::command::CommandBuffer as _;
use wgt::{BufferAddress, BufferUsage, Extent3d, Origin3d, TextureDataLayout, TextureUsage};

use std::convert::TryInto as _;
use std::iter;

pub(crate) const BITS_PER_BYTE: u32 = 8;

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub layout: TextureDataLayout,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct TextureCopyView {
    pub texture: TextureId,
    pub mip_level: u32,
    pub origin: Origin3d,
}

impl TextureCopyView {
    //TODO: we currently access each texture twice for a transfer,
    // once only to get the aspect flags, which is unfortunate.
    pub(crate) fn to_hal<B: hal::Backend>(
        &self,
        texture_guard: &Storage<Texture<B>, TextureId>,
    ) -> (
        hal::image::SubresourceLayers,
        hal::image::SubresourceRange,
        hal::image::Offset,
    ) {
        let texture = &texture_guard[self.texture];
        let aspects = texture.full_range.aspects;
        let level = self.mip_level as hal::image::Level;
        let (layer, z) = match texture.dimension {
            wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
                (self.origin.z as hal::image::Layer, 0)
            }
            wgt::TextureDimension::D3 => (0, self.origin.z as i32),
        };

        // TODO: Can't satisfy clippy here unless we modify
        // `hal::image::SubresourceRange` in gfx to use `std::ops::RangeBounds`.
        #[allow(clippy::range_plus_one)]
        (
            hal::image::SubresourceLayers {
                aspects,
                level: self.mip_level as hal::image::Level,
                layers: layer..layer + 1,
            },
            hal::image::SubresourceRange {
                aspects,
                levels: level..level + 1,
                layers: layer..layer + 1,
            },
            hal::image::Offset {
                x: self.origin.x as i32,
                y: self.origin.y as i32,
                z,
            },
        )
    }
}

/// Function copied with minor modifications from webgpu standard https://gpuweb.github.io/gpuweb/#valid-texture-copy-range
pub(crate) fn validate_linear_texture_data(
    layout: &TextureDataLayout,
    buffer_size: BufferAddress,
    bytes_per_texel: BufferAddress,
    copy_size: &Extent3d,
) {
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

    assert_eq!(
        copy_width % block_width,
        0,
        "Copy width {} must be a multiple of texture block width {}",
        copy_size.width,
        block_width,
    );
    assert_eq!(
        copy_height % block_height,
        0,
        "Copy height {} must be a multiple of texture block height {}",
        copy_size.height,
        block_height,
    );
    assert_eq!(
        rows_per_image % block_height,
        0,
        "Rows per image {} must be a multiple of image format block height {}",
        rows_per_image,
        block_height,
    );

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

    if rows_per_image != 0 {
        assert!(
            rows_per_image >= copy_height,
            "Rows per image {} must be greater or equal to copy_extent.height {}",
            rows_per_image,
            copy_height
        )
    }
    assert!(
        offset + required_bytes_in_copy <= buffer_size,
        "Texture copy using buffer indices {}..{} would overrun buffer of size {}",
        offset,
        offset + required_bytes_in_copy,
        buffer_size
    );
    assert_eq!(
        offset % block_size,
        0,
        "Buffer offset {} must be a multiple of image format block size {}",
        offset,
        block_size,
    );
    if copy_height > 1 {
        assert!(
            bytes_per_row >= bytes_in_a_complete_row,
            "Bytes per row {} must be at least the size of {} {}-byte texel blocks ({})",
            bytes_per_row,
            copy_width / block_width,
            block_size,
            bytes_in_a_complete_row,
        )
    }
    if copy_depth > 1 {
        assert_ne!(
            rows_per_image, 0,
            "Rows per image {} must be set to a non zero value when copy depth > 1 ({})",
            rows_per_image, copy_depth,
        )
    }
}

/// Function copied with minor modifications from webgpu standard https://gpuweb.github.io/gpuweb/#valid-texture-copy-range
pub(crate) fn validate_texture_copy_range(
    texture_copy_view: &TextureCopyView,
    texture_dimension: hal::image::Kind,
    copy_size: &Extent3d,
) {
    // TODO: Once compressed textures are supported, these needs to be fixed
    let block_width: u32 = 1;
    let block_height: u32 = 1;

    let mut extent = texture_dimension.level_extent(
        texture_copy_view
            .mip_level
            .try_into()
            .expect("Mip level must be < 256"),
    );
    match texture_dimension {
        hal::image::Kind::D1(..) => {
            assert_eq!(
                (copy_size.height, copy_size.depth),
                (1, 1),
                "Copies with 1D textures must have height and depth of 1. Currently: ({}, {})",
                copy_size.height,
                copy_size.depth,
            );
        }
        hal::image::Kind::D2(_, _, array_layers, _) => {
            extent.depth = array_layers as u32;
        }
        hal::image::Kind::D3(..) => {}
    };

    let x_copy_max = texture_copy_view.origin.x + copy_size.width;
    assert!(
        x_copy_max <= extent.width,
        "Texture copy with X range {}..{} overruns texture width {}",
        texture_copy_view.origin.x,
        x_copy_max,
        extent.width,
    );
    let y_copy_max = texture_copy_view.origin.y + copy_size.height;
    assert!(
        y_copy_max <= extent.height,
        "Texture copy with Y range {}..{} overruns texture height {}",
        texture_copy_view.origin.y,
        y_copy_max,
        extent.height,
    );
    let z_copy_max = texture_copy_view.origin.z + copy_size.depth;
    assert!(
        z_copy_max <= extent.depth,
        "Texture copy with Z range {}..{} overruns texture depth {}",
        texture_copy_view.origin.z,
        z_copy_max,
        extent.depth,
    );

    assert_eq!(
        copy_size.width % block_width,
        0,
        "Copy width {} must be a multiple of texture block width {}",
        copy_size.width,
        block_width,
    );
    assert_eq!(
        copy_size.height % block_height,
        0,
        "Copy height {} must be a multiple of texture block height {}",
        copy_size.height,
        block_height,
    );
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
    ) {
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
            return;
        }

        let (src_buffer, src_pending) =
            cmb.trackers
                .buffers
                .use_replace(&*buffer_guard, source, (), BufferUse::COPY_SRC);
        assert!(
            src_buffer.usage.contains(BufferUsage::COPY_SRC),
            "Source buffer usage {:?} must contain usage flag COPY_SRC",
            src_buffer.usage
        );
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_buffer)));

        let (dst_buffer, dst_pending) =
            cmb.trackers
                .buffers
                .use_replace(&*buffer_guard, destination, (), BufferUse::COPY_DST);
        assert!(
            dst_buffer.usage.contains(BufferUsage::COPY_DST),
            "Destination buffer usage {:?} must contain usage flag COPY_DST",
            dst_buffer.usage
        );
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_buffer)));

        assert_eq!(
            size % wgt::COPY_BUFFER_ALIGNMENT,
            0,
            "Buffer copy size {} must be a multiple of {}",
            size,
            wgt::COPY_BUFFER_ALIGNMENT,
        );
        assert_eq!(
            source_offset % wgt::COPY_BUFFER_ALIGNMENT,
            0,
            "Buffer source offset {} must be a multiple of {}",
            source_offset,
            wgt::COPY_BUFFER_ALIGNMENT,
        );
        assert_eq!(
            destination_offset % wgt::COPY_BUFFER_ALIGNMENT,
            0,
            "Buffer destination offset {} must be a multiple of {}",
            destination_offset,
            wgt::COPY_BUFFER_ALIGNMENT,
        );

        let source_start_offset = source_offset;
        let source_end_offset = source_offset + size;
        let destination_start_offset = destination_offset;
        let destination_end_offset = destination_offset + size;
        assert!(
            source_end_offset <= src_buffer.size,
            "Buffer to buffer copy with indices {}..{} overruns source buffer of size {}",
            source_start_offset,
            source_end_offset,
            src_buffer.size
        );
        assert!(
            destination_end_offset <= dst_buffer.size,
            "Buffer to buffer copy with indices {}..{} overruns destination buffer of size {}",
            destination_start_offset,
            destination_end_offset,
            dst_buffer.size
        );

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
    }

    pub fn command_encoder_copy_buffer_to_texture<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &BufferCopyView,
        destination: &TextureCopyView,
        copy_size: &Extent3d,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (dst_layers, dst_range, dst_offset) = destination.to_hal(&*texture_guard);

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
            return;
        }

        let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            source.buffer,
            (),
            BufferUse::COPY_SRC,
        );
        assert!(src_buffer.usage.contains(BufferUsage::COPY_SRC));
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            dst_range,
            TextureUse::COPY_DST,
        );
        assert!(dst_texture.usage.contains(TextureUsage::COPY_DST));
        let dst_barriers = dst_pending.map(|pending| pending.into_hal(dst_texture));

        let bytes_per_texel = conv::map_texture_format(dst_texture.format, cmb.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        assert_eq!(wgt::COPY_BYTES_PER_ROW_ALIGNMENT % bytes_per_texel, 0);
        assert_eq!(
            source.layout.bytes_per_row % wgt::COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "Source bytes per row ({}) must be a multiple of {}",
            source.layout.bytes_per_row,
            wgt::COPY_BYTES_PER_ROW_ALIGNMENT
        );
        validate_texture_copy_range(destination, dst_texture.kind, copy_size);
        validate_linear_texture_data(
            &source.layout,
            src_buffer.size,
            bytes_per_texel as BufferAddress,
            copy_size,
        );

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
    }

    pub fn command_encoder_copy_texture_to_buffer<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TextureCopyView,
        destination: &BufferCopyView,
        copy_size: &Extent3d,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (src_layers, src_range, src_offset) = source.to_hal(&*texture_guard);

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
            return;
        }

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            src_range,
            TextureUse::COPY_SRC,
        );
        assert!(
            src_texture.usage.contains(TextureUsage::COPY_SRC),
            "Source texture usage ({:?}) must contain usage flag COPY_SRC",
            src_texture.usage
        );
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_texture));

        let (dst_buffer, dst_barriers) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            destination.buffer,
            (),
            BufferUse::COPY_DST,
        );
        assert!(
            dst_buffer.usage.contains(BufferUsage::COPY_DST),
            "Destination buffer usage {:?} must contain usage flag COPY_DST",
            dst_buffer.usage
        );
        let dst_barrier = dst_barriers.map(|pending| pending.into_hal(dst_buffer));

        let bytes_per_texel = conv::map_texture_format(src_texture.format, cmb.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        assert_eq!(wgt::COPY_BYTES_PER_ROW_ALIGNMENT % bytes_per_texel, 0);
        assert_eq!(
            destination.layout.bytes_per_row % wgt::COPY_BYTES_PER_ROW_ALIGNMENT,
            0,
            "Destination bytes per row ({}) must be a multiple of {}",
            destination.layout.bytes_per_row,
            wgt::COPY_BYTES_PER_ROW_ALIGNMENT
        );
        validate_texture_copy_range(source, src_texture.kind, copy_size);
        validate_linear_texture_data(
            &destination.layout,
            dst_buffer.size,
            bytes_per_texel as BufferAddress,
            copy_size,
        );

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
    }

    pub fn command_encoder_copy_texture_to_texture<B: GfxBackend>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TextureCopyView,
        destination: &TextureCopyView,
        copy_size: &Extent3d,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (_, mut token) = hub.buffers.read(&mut token); // skip token
        let (texture_guard, _) = hub.textures.read(&mut token);
        // we can't hold both src_pending and dst_pending in scope because they
        // borrow the buffer tracker mutably...
        let mut barriers = Vec::new();
        let (src_layers, src_range, src_offset) = source.to_hal(&*texture_guard);
        let (dst_layers, dst_range, dst_offset) = destination.to_hal(&*texture_guard);
        assert_eq!(src_layers.aspects, dst_layers.aspects);

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
            return;
        }

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            src_range,
            TextureUse::COPY_SRC,
        );
        assert!(
            src_texture.usage.contains(TextureUsage::COPY_SRC),
            "Source texture usage {:?} must contain usage flag COPY_SRC",
            src_texture.usage
        );
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_texture)));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            dst_range,
            TextureUse::COPY_DST,
        );
        assert!(
            dst_texture.usage.contains(TextureUsage::COPY_DST),
            "Destination texture usage {:?} must contain usage flag COPY_DST",
            dst_texture.usage
        );
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_texture)));

        assert_eq!(src_texture.dimension, dst_texture.dimension);
        validate_texture_copy_range(source, src_texture.kind, copy_size);
        validate_texture_copy_range(destination, dst_texture.kind, copy_size);

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
    }
}
