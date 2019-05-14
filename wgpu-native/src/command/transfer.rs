use crate::{
    conv,
    device::{all_buffer_stages, all_image_stages},
    hub::HUB,
    resource::TexturePlacement,
    swap_chain::SwapChainLink,
    BufferAddress,
    BufferId,
    BufferUsage,
    CommandBufferId,
    Extent3d,
    Origin3d,
    TextureId,
    TextureUsage,
};

use copyless::VecHelper as _;
use hal::command::RawCommandBuffer;

use std::iter;

const BITS_PER_BYTE: u32 = 8;

#[repr(C)]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub row_pitch: u32,
    pub image_height: u32,
}

#[repr(C)]
pub struct TextureCopyView {
    pub texture: TextureId,
    pub mip_level: u32,
    pub array_layer: u32,
    pub origin: Origin3d,
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_copy_buffer_to_buffer(
    command_buffer_id: CommandBufferId,
    src: BufferId,
    src_offset: BufferAddress,
    dst: BufferId,
    dst_offset: BufferAddress,
    size: BufferAddress,
) {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[command_buffer_id];
    let buffer_guard = HUB.buffers.read();

    let (src_buffer, src_usage) = cmb
        .trackers
        .buffers
        .get_with_replaced_usage(&*buffer_guard, src, BufferUsage::TRANSFER_SRC)
        .unwrap();
    let src_barrier = src_usage.map(|old| hal::memory::Barrier::Buffer {
        states: conv::map_buffer_state(old)..hal::buffer::Access::TRANSFER_READ,
        target: &src_buffer.raw,
        families: None,
        range: None..None,
    });

    let (dst_buffer, dst_usage) = cmb
        .trackers
        .buffers
        .get_with_replaced_usage(&*buffer_guard, dst, BufferUsage::TRANSFER_DST)
        .unwrap();
    let dst_barrier = dst_usage.map(|old| hal::memory::Barrier::Buffer {
        states: conv::map_buffer_state(old)..hal::buffer::Access::TRANSFER_WRITE,
        target: &dst_buffer.raw,
        families: None,
        range: None..None,
    });

    let region = hal::command::BufferCopy {
        src: src_offset,
        dst: dst_offset,
        size,
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    unsafe {
        cmb_raw.pipeline_barrier(
            all_buffer_stages()..all_buffer_stages(),
            hal::memory::Dependencies::empty(),
            src_barrier.into_iter().chain(dst_barrier),
        );
        cmb_raw.copy_buffer(&src_buffer.raw, &dst_buffer.raw, iter::once(region));
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
    let cmb = &mut cmb_guard[command_buffer_id];
    let buffer_guard = HUB.buffers.read();
    let texture_guard = HUB.textures.read();

    let (src_buffer, src_usage) = cmb
        .trackers
        .buffers
        .get_with_replaced_usage(
            &*buffer_guard,
            source.buffer,
            BufferUsage::TRANSFER_SRC,
        )
        .unwrap();
    let src_barrier = src_usage.map(|old| hal::memory::Barrier::Buffer {
        states: conv::map_buffer_state(old)..hal::buffer::Access::TRANSFER_READ,
        target: &src_buffer.raw,
        families: None,
        range: None..None,
    });

    let (dst_texture, dst_usage) = cmb
        .trackers
        .textures
        .get_with_replaced_usage(
            &*texture_guard,
            destination.texture,
            TextureUsage::TRANSFER_DST,
        )
        .unwrap();
    let aspects = dst_texture.full_range.aspects;
    let dst_texture_state = conv::map_texture_state(TextureUsage::TRANSFER_DST, aspects);
    let dst_barrier = dst_usage.map(|old| hal::memory::Barrier::Image {
        states: conv::map_texture_state(old, aspects)..dst_texture_state,
        target: &dst_texture.raw,
        families: None,
        range: dst_texture.full_range.clone(),
    });

    if let TexturePlacement::SwapChain(ref link) = dst_texture.placement {
        cmb.swap_chain_links.alloc().init(SwapChainLink {
            swap_chain_id: link.swap_chain_id.clone(),
            epoch: *link.epoch.lock(),
            image_index: link.image_index,
        });
    }

    let bytes_per_texel = conv::map_texture_format(dst_texture.format)
        .surface_desc()
        .bits as u32
        / BITS_PER_BYTE;
    let buffer_width = source.row_pitch / bytes_per_texel;
    assert_eq!(source.row_pitch % bytes_per_texel, 0);
    let region = hal::command::BufferImageCopy {
        buffer_offset: source.offset,
        buffer_width,
        buffer_height: source.image_height,
        image_layers: hal::image::SubresourceLayers {
            aspects, //TODO
            level: destination.mip_level as hal::image::Level,
            layers: destination.array_layer as u16 .. destination.array_layer as u16 + 1,
        },
        image_offset: conv::map_origin(destination.origin),
        image_extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    let stages = all_buffer_stages() | all_image_stages();
    unsafe {
        cmb_raw.pipeline_barrier(
            stages..stages,
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

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_copy_texture_to_buffer(
    command_buffer_id: CommandBufferId,
    source: &TextureCopyView,
    destination: &BufferCopyView,
    copy_size: Extent3d,
) {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[command_buffer_id];
    let buffer_guard = HUB.buffers.read();
    let texture_guard = HUB.textures.read();

    let (src_texture, src_usage) = cmb
        .trackers
        .textures
        .get_with_replaced_usage(
            &*texture_guard,
            source.texture,
            TextureUsage::TRANSFER_SRC,
        )
        .unwrap();
    let aspects = src_texture.full_range.aspects;
    let src_texture_state = conv::map_texture_state(TextureUsage::TRANSFER_SRC, aspects);
    let src_barrier = src_usage.map(|old| hal::memory::Barrier::Image {
        states: conv::map_texture_state(old, aspects)..src_texture_state,
        target: &src_texture.raw,
        families: None,
        range: src_texture.full_range.clone(),
    });
    match src_texture.placement {
        TexturePlacement::SwapChain(_) => unimplemented!(),
        TexturePlacement::Void => unreachable!(),
        TexturePlacement::Memory(_) => (),
    }

    let (dst_buffer, dst_usage) = cmb
        .trackers
        .buffers
        .get_with_replaced_usage(
            &*buffer_guard,
            destination.buffer,
            BufferUsage::TRANSFER_DST,
        )
        .unwrap();
    let dst_barrier = dst_usage.map(|old| hal::memory::Barrier::Buffer {
        states: conv::map_buffer_state(old)..hal::buffer::Access::TRANSFER_WRITE,
        target: &dst_buffer.raw,
        families: None,
        range: None..None,
    });

    let bytes_per_texel = conv::map_texture_format(src_texture.format)
        .surface_desc()
        .bits as u32
        / BITS_PER_BYTE;
    let buffer_width = destination.row_pitch / bytes_per_texel;
    assert_eq!(destination.row_pitch % bytes_per_texel, 0);
    let region = hal::command::BufferImageCopy {
        buffer_offset: destination.offset,
        buffer_width,
        buffer_height: destination.image_height,
        image_layers: hal::image::SubresourceLayers {
            aspects, //TODO
            level: source.mip_level as hal::image::Level,
            layers: source.array_layer as u16 .. source.array_layer as u16 + 1,
        },
        image_offset: conv::map_origin(source.origin),
        image_extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    let stages = all_buffer_stages() | all_image_stages();
    unsafe {
        cmb_raw.pipeline_barrier(
            stages..stages,
            hal::memory::Dependencies::empty(),
            src_barrier.into_iter().chain(dst_barrier),
        );
        cmb_raw.copy_image_to_buffer(
            &src_texture.raw,
            src_texture_state.1,
            &dst_buffer.raw,
            iter::once(region),
        );
    }
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_copy_texture_to_texture(
    command_buffer_id: CommandBufferId,
    source: &TextureCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[command_buffer_id];
    let texture_guard = HUB.textures.read();

    let (src_texture, src_usage) = cmb
        .trackers
        .textures
        .get_with_replaced_usage(
            &*texture_guard,
            source.texture,
            TextureUsage::TRANSFER_SRC,
        )
        .unwrap();
    let (dst_texture, dst_usage) = cmb
        .trackers
        .textures
        .get_with_replaced_usage(
            &*texture_guard,
            destination.texture,
            TextureUsage::TRANSFER_DST,
        )
        .unwrap();

    let aspects = src_texture.full_range.aspects & dst_texture.full_range.aspects;
    let src_texture_state = conv::map_texture_state(TextureUsage::TRANSFER_SRC, aspects);
    let dst_texture_state = conv::map_texture_state(TextureUsage::TRANSFER_DST, aspects);

    let src_barrier = src_usage.map(|old| hal::memory::Barrier::Image {
        states: conv::map_texture_state(old, aspects)..src_texture_state,
        target: &src_texture.raw,
        families: None,
        range: src_texture.full_range.clone(),
    });
    let dst_barrier = dst_usage.map(|old| hal::memory::Barrier::Image {
        states: conv::map_texture_state(old, aspects)..dst_texture_state,
        target: &dst_texture.raw,
        families: None,
        range: dst_texture.full_range.clone(),
    });

    if let TexturePlacement::SwapChain(ref link) = dst_texture.placement {
        cmb.swap_chain_links.alloc().init(SwapChainLink {
            swap_chain_id: link.swap_chain_id.clone(),
            epoch: *link.epoch.lock(),
            image_index: link.image_index,
        });
    }

    let region = hal::command::ImageCopy {
        src_subresource: hal::image::SubresourceLayers {
            aspects,
            level: source.mip_level as hal::image::Level,
            layers: source.array_layer as u16 .. source.array_layer as u16 + 1,
        },
        src_offset: conv::map_origin(source.origin),
        dst_subresource: hal::image::SubresourceLayers {
            aspects,
            level: destination.mip_level as hal::image::Level,
            layers: destination.array_layer as u16 .. destination.array_layer as u16 + 1,
        },
        dst_offset: conv::map_origin(destination.origin),
        extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    unsafe {
        cmb_raw.pipeline_barrier(
            all_image_stages()..all_image_stages(),
            hal::memory::Dependencies::empty(),
            src_barrier.into_iter().chain(dst_barrier),
        );
        cmb_raw.copy_image(
            &src_texture.raw,
            src_texture_state.1,
            &dst_texture.raw,
            dst_texture_state.1,
            iter::once(region),
        );
    }
}
