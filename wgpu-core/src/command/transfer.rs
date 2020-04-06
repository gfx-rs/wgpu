/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    conv,
    device::{all_buffer_stages, all_image_stages},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id::{BufferId, CommandEncoderId, TextureId},
};

use hal::command::CommandBuffer as _;
use wgt::{BufferAddress, BufferUsage, Extent3d, Origin3d, TextureUsage};

use std::iter;

const BITS_PER_BYTE: u32 = 8;

#[repr(C)]
#[derive(Debug)]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct TextureCopyView {
    pub texture: TextureId,
    pub mip_level: u32,
    pub array_layer: u32,
    pub origin: Origin3d,
}

impl TextureCopyView {
    //TODO: we currently access each texture twice for a transfer,
    // once only to get the aspect flags, which is unfortunate.
    fn to_selector(&self, aspects: hal::format::Aspects) -> hal::image::SubresourceRange {
        let level = self.mip_level as hal::image::Level;
        let layer = self.array_layer as hal::image::Layer;

        // TODO: Can't satisfy clippy here unless we modify
        // `hal::image::SubresourceRange` in gfx to use `std::ops::RangeBounds`.
        #[allow(clippy::range_plus_one)]
        {
            hal::image::SubresourceRange {
                aspects,
                levels: level..level + 1,
                layers: layer..layer + 1,
            }
        }
    }

    fn to_sub_layers(&self, aspects: hal::format::Aspects) -> hal::image::SubresourceLayers {
        let layer = self.array_layer as hal::image::Layer;
        // TODO: Can't satisfy clippy here unless we modify
        // `hal::image::SubresourceLayers` in gfx to use
        // `std::ops::RangeBounds`.
        #[allow(clippy::range_plus_one)]
        {
            hal::image::SubresourceLayers {
                aspects,
                level: self.mip_level as hal::image::Level,
                layers: layer..layer + 1,
            }
        }
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

        let (src_buffer, src_pending) =
            cmb.trackers
                .buffers
                .use_replace(&*buffer_guard, source, (), BufferUsage::COPY_SRC);
        assert!(src_buffer.usage.contains(BufferUsage::COPY_SRC));
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_buffer)));

        let (dst_buffer, dst_pending) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            destination,
            (),
            BufferUsage::COPY_DST,
        );
        assert!(dst_buffer.usage.contains(BufferUsage::COPY_DST));
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_buffer)));

        let region = hal::command::BufferCopy {
            src: source_offset,
            dst: destination_offset,
            size,
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_buffer_stages()..all_buffer_stages(),
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
        copy_size: Extent3d,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let aspects = texture_guard[destination.texture].full_range.aspects;

        let (src_buffer, src_pending) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            source.buffer,
            (),
            BufferUsage::COPY_SRC,
        );
        assert!(src_buffer.usage.contains(BufferUsage::COPY_SRC));
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_buffer));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            destination.to_selector(aspects),
            TextureUsage::COPY_DST,
        );
        assert!(dst_texture.usage.contains(TextureUsage::COPY_DST));
        let dst_barriers = dst_pending.map(|pending| pending.into_hal(dst_texture));

        let bytes_per_texel = conv::map_texture_format(dst_texture.format, cmb.features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let buffer_width = source.bytes_per_row / bytes_per_texel;
        assert_eq!(source.bytes_per_row % bytes_per_texel, 0);
        let region = hal::command::BufferImageCopy {
            buffer_offset: source.offset,
            buffer_width,
            buffer_height: source.rows_per_image,
            image_layers: destination.to_sub_layers(aspects),
            image_offset: conv::map_origin(destination.origin),
            image_extent: conv::map_extent(copy_size),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            cmb_raw.pipeline_barrier(
                stages..stages,
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
        copy_size: Extent3d,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[command_encoder_id];
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, _) = hub.textures.read(&mut token);
        let aspects = texture_guard[source.texture].full_range.aspects;

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            source.to_selector(aspects),
            TextureUsage::COPY_SRC,
        );
        assert!(src_texture.usage.contains(TextureUsage::COPY_SRC));
        let src_barriers = src_pending.map(|pending| pending.into_hal(src_texture));

        let (dst_buffer, dst_barriers) = cmb.trackers.buffers.use_replace(
            &*buffer_guard,
            destination.buffer,
            (),
            BufferUsage::COPY_DST,
        );
        assert!(dst_buffer.usage.contains(BufferUsage::COPY_DST));
        let dst_barrier = dst_barriers.map(|pending| pending.into_hal(dst_buffer));

        let bytes_per_texel = conv::map_texture_format(src_texture.format, cmb.features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        let buffer_width = destination.bytes_per_row / bytes_per_texel;
        assert_eq!(destination.bytes_per_row % bytes_per_texel, 0);
        let region = hal::command::BufferImageCopy {
            buffer_offset: destination.offset,
            buffer_width,
            buffer_height: destination.rows_per_image,
            image_layers: source.to_sub_layers(aspects),
            image_offset: conv::map_origin(source.origin),
            image_extent: conv::map_extent(copy_size),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            cmb_raw.pipeline_barrier(
                stages..stages,
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
        copy_size: Extent3d,
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
        let aspects = texture_guard[source.texture].full_range.aspects
            & texture_guard[destination.texture].full_range.aspects;

        let (src_texture, src_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            source.texture,
            source.to_selector(aspects),
            TextureUsage::COPY_SRC,
        );
        assert!(src_texture.usage.contains(TextureUsage::COPY_SRC));
        barriers.extend(src_pending.map(|pending| pending.into_hal(src_texture)));

        let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
            &*texture_guard,
            destination.texture,
            destination.to_selector(aspects),
            TextureUsage::COPY_DST,
        );
        assert!(dst_texture.usage.contains(TextureUsage::COPY_DST));
        barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_texture)));

        let region = hal::command::ImageCopy {
            src_subresource: source.to_sub_layers(aspects),
            src_offset: conv::map_origin(source.origin),
            dst_subresource: destination.to_sub_layers(aspects),
            dst_offset: conv::map_origin(destination.origin),
            extent: conv::map_extent(copy_size),
        };
        let cmb_raw = cmb.raw.last_mut().unwrap();
        unsafe {
            cmb_raw.pipeline_barrier(
                all_image_stages()..all_image_stages(),
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
