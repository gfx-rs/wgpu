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
