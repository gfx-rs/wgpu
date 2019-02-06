use crate::device::{all_buffer_stages, all_image_stages};
use crate::registry::{Items, HUB};
use crate::swap_chain::SwapChainLink;
use crate::track::{TrackPermit, Tracktion};
use crate::conv;
use crate::{
    BufferId, CommandBufferId, TextureId,
    BufferUsageFlags, TextureUsageFlags,
    Extent3d, Origin3d,
};

use hal::command::RawCommandBuffer;

use std::iter;


const BITS_PER_BYTE: u32 = 8;

#[repr(C)]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub offset: u32,
    pub row_pitch: u32,
    pub image_height: u32,
}

#[repr(C)]
pub struct TextureCopyView {
    pub texture: TextureId,
    pub level: u32,
    pub slice: u32,
    pub origin: Origin3d,
    //TODO: pub aspect: TextureAspect,
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_copy_buffer_to_buffer(
    command_buffer_id: CommandBufferId,
    src:  BufferId,
    src_offset: u32,
    dst: BufferId,
    dst_offset: u32,
    size: u32,
) {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(command_buffer_id);
    let buffer_guard = HUB.buffers.read();

    let (src_buffer, src_tracktion) = cmb.buffer_tracker
        .get_with_usage(
            &*buffer_guard,
            src,
            BufferUsageFlags::TRANSFER_SRC,
            TrackPermit::REPLACE,
        )
        .unwrap();
    let src_barrier = match src_tracktion {
        Tracktion::Init |
        Tracktion::Keep => None,
        Tracktion::Extend { .. } => unreachable!(),
        Tracktion::Replace { old } => Some(hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(old) .. hal::buffer::Access::TRANSFER_READ,
            target: &src_buffer.raw,
            families: None,
            range: None .. None,
        }),
    };

    let (dst_buffer, dst_tracktion) = cmb.buffer_tracker
        .get_with_usage(
            &*buffer_guard,
            dst,
            BufferUsageFlags::TRANSFER_DST,
            TrackPermit::REPLACE,
        )
        .unwrap();
    let dst_barrier = match dst_tracktion {
        Tracktion::Init |
        Tracktion::Keep => None,
        Tracktion::Extend { .. } => unreachable!(),
        Tracktion::Replace { old } => Some(hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(old) .. hal::buffer::Access::TRANSFER_WRITE,
            target: &dst_buffer.raw,
            families: None,
            range: None .. None,
        }),
    };

    let region = hal::command::BufferCopy {
        src: src_offset as hal::buffer::Offset,
        dst: dst_offset as hal::buffer::Offset,
        size: size as hal::buffer::Offset,
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    unsafe {
        cmb_raw.pipeline_barrier(
            all_buffer_stages() .. all_buffer_stages(),
            hal::memory::Dependencies::empty(),
            src_barrier.into_iter().chain(dst_barrier),
        );
        cmb_raw.copy_buffer(
            &src_buffer.raw,
            &dst_buffer.raw,
            iter::once(region),
        );
    }
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_copy_buffer_to_texture(
    command_buffer_id: CommandBufferId,
    source: &BufferCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(command_buffer_id);
    let buffer_guard = HUB.buffers.read();
    let texture_guard = HUB.textures.read();

    let (src_buffer, src_tracktion) = cmb.buffer_tracker
        .get_with_usage(
            &*buffer_guard,
            source.buffer,
            BufferUsageFlags::TRANSFER_SRC,
            TrackPermit::REPLACE,
        )
        .unwrap();
    let src_barrier = match src_tracktion {
        Tracktion::Init |
        Tracktion::Keep => None,
        Tracktion::Extend { .. } => unreachable!(),
        Tracktion::Replace { old } => Some(hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(old) .. hal::buffer::Access::TRANSFER_READ,
            target: &src_buffer.raw,
            families: None,
            range: None .. None,
        }),
    };

    let (dst_texture, dst_tracktion) = cmb.texture_tracker
        .get_with_usage(
            &*texture_guard,
            destination.texture,
            TextureUsageFlags::TRANSFER_DST,
            TrackPermit::REPLACE,
        )
        .unwrap();
    let aspects = dst_texture.full_range.aspects;
    let dst_texture_state = conv::map_texture_state(TextureUsageFlags::TRANSFER_DST, aspects);
    let dst_barrier = match dst_tracktion {
        Tracktion::Init |
        Tracktion::Keep => None,
        Tracktion::Extend { .. } => unreachable!(),
        Tracktion::Replace { old } => Some(hal::memory::Barrier::Image {
            states: conv::map_texture_state(old, aspects) .. dst_texture_state,
            target: &dst_texture.raw,
            families: None,
            range: dst_texture.full_range.clone(),
        }),
    };

    if let Some(ref link) = dst_texture.swap_chain_link {
        cmb.swap_chain_links.push(SwapChainLink {
            swap_chain_id: link.swap_chain_id.clone(),
            epoch: *link.epoch.lock(),
            image_index: link.image_index,
        });
    }

    let bytes_per_texel = conv::map_texture_format(dst_texture.format)
        .surface_desc().bits as u32 / BITS_PER_BYTE;
    let buffer_width = source.row_pitch / bytes_per_texel;
    assert_eq!(source.row_pitch % bytes_per_texel, 0);
    let region = hal::command::BufferImageCopy {
        buffer_offset: source.offset as hal::buffer::Offset,
        buffer_width,
        buffer_height: source.image_height,
        image_layers: hal::image::SubresourceLayers {
            aspects, //TODO
            level: destination.level as hal::image::Level,
            layers: destination.slice as u16 .. destination.slice as u16 + 1,
        },
        image_offset: conv::map_origin(destination.origin),
        image_extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    unsafe {
        cmb_raw.pipeline_barrier(
            all_buffer_stages() .. all_image_stages(),
            hal::memory::Dependencies::empty(),
            src_barrier.into_iter().chain(dst_barrier),
        );
        cmb_raw.copy_buffer_to_image(
            &src_buffer.raw,
            &dst_texture.raw,
            dst_texture_state.1,
            iter::once(region),
        );
    }
}
