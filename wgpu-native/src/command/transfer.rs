use crate::{
    conv,
    device::{all_buffer_stages, all_image_stages},
    gfx_select,
    hub::{GfxBackend, Token},
    BufferAddress,
    BufferId,
    BufferUsage,
    CommandEncoderId,
    Extent3d,
    Origin3d,
    TextureId,
    TextureUsage,
};

use hal::command::CommandBuffer as _;

use std::iter;

const BITS_PER_BYTE: u32 = 8;

#[repr(C)]
#[derive(Debug)]
pub struct BufferCopyView {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub row_pitch: u32,
    pub image_height: u32,
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
        hal::image::SubresourceRange {
            aspects,
            levels: level .. level + 1,
            layers: layer .. layer + 1,
        }
    }

    fn to_sub_layers(&self, aspects: hal::format::Aspects) -> hal::image::SubresourceLayers {
        let layer = self.array_layer as hal::image::Layer;
        hal::image::SubresourceLayers {
            aspects,
            level: self.mip_level as hal::image::Level,
            layers: layer .. layer + 1,
        }
    }
}

pub fn command_encoder_copy_buffer_to_buffer<B: GfxBackend>(
    command_encoder_id: CommandEncoderId,
    source: BufferId,
    source_offset: BufferAddress,
    destination: BufferId,
    destination_offset: BufferAddress,
    size: BufferAddress,
) {
    let hub = B::hub();
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

    barriers.extend(src_pending.map(|pending| hal::memory::Barrier::Buffer {
        states: pending.to_states(),
        target: &src_buffer.raw,
        families: None,
        range: None .. None,
    }));

    let (dst_buffer, dst_pending) =
        cmb.trackers
            .buffers
            .use_replace(&*buffer_guard, destination, (), BufferUsage::COPY_DST);
    assert!(dst_buffer.usage.contains(BufferUsage::COPY_DST));

    barriers.extend(dst_pending.map(|pending| hal::memory::Barrier::Buffer {
        states: pending.to_states(),
        target: &dst_buffer.raw,
        families: None,
        range: None .. None,
    }));

    let region = hal::command::BufferCopy {
        src: source_offset,
        dst: destination_offset,
        size,
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    unsafe {
        cmb_raw.pipeline_barrier(
            all_buffer_stages() .. all_buffer_stages(),
            hal::memory::Dependencies::empty(),
            barriers,
        );
        cmb_raw.copy_buffer(&src_buffer.raw, &dst_buffer.raw, iter::once(region));
    }
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_buffer_to_buffer(
    command_encoder_id: CommandEncoderId,
    source: BufferId,
    source_offset: BufferAddress,
    destination: BufferId,
    destination_offset: BufferAddress,
    size: BufferAddress,
) {
    gfx_select!(command_encoder_id => command_encoder_copy_buffer_to_buffer(
        command_encoder_id,
        source, source_offset,
        destination,
        destination_offset,
        size))
}

pub fn command_encoder_copy_buffer_to_texture<B: GfxBackend>(
    command_encoder_id: CommandEncoderId,
    source: &BufferCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    let hub = B::hub();
    let mut token = Token::root();

    let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
    let cmb = &mut cmb_guard[command_encoder_id];
    let (buffer_guard, mut token) = hub.buffers.read(&mut token);
    let (texture_guard, _) = hub.textures.read(&mut token);
    let aspects = texture_guard[destination.texture].full_range.aspects;

    let (src_buffer, src_pending) =
        cmb.trackers
            .buffers
            .use_replace(&*buffer_guard, source.buffer, (), BufferUsage::COPY_SRC);
    assert!(src_buffer.usage.contains(BufferUsage::COPY_SRC));

    let src_barriers = src_pending.map(|pending| hal::memory::Barrier::Buffer {
        states: pending.to_states(),
        target: &src_buffer.raw,
        families: None,
        range: None .. None,
    });

    let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
        &*texture_guard,
        destination.texture,
        destination.to_selector(aspects),
        TextureUsage::COPY_DST,
    );
    assert!(dst_texture.usage.contains(TextureUsage::COPY_DST));

    let dst_barriers = dst_pending.map(|pending| hal::memory::Barrier::Image {
        states: pending.to_states(),
        target: &dst_texture.raw,
        families: None,
        range: pending.selector,
    });

    let aspects = dst_texture.full_range.aspects;
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
        image_layers: destination.to_sub_layers(aspects),
        image_offset: conv::map_origin(destination.origin),
        image_extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    let stages = all_buffer_stages() | all_image_stages();
    unsafe {
        cmb_raw.pipeline_barrier(
            stages .. stages,
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

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_buffer_to_texture(
    command_encoder_id: CommandEncoderId,
    source: &BufferCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    gfx_select!(command_encoder_id => command_encoder_copy_buffer_to_texture(
        command_encoder_id,
        source,
        destination,
        copy_size))
}

pub fn command_encoder_copy_texture_to_buffer<B: GfxBackend>(
    command_encoder_id: CommandEncoderId,
    source: &TextureCopyView,
    destination: &BufferCopyView,
    copy_size: Extent3d,
) {
    let hub = B::hub();
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

    let src_barriers = src_pending.map(|pending| hal::memory::Barrier::Image {
        states: pending.to_states(),
        target: &src_texture.raw,
        families: None,
        range: pending.selector,
    });

    let (dst_buffer, dst_barriers) = cmb.trackers.buffers.use_replace(
        &*buffer_guard,
        destination.buffer,
        (),
        BufferUsage::COPY_DST,
    );
    assert!(dst_buffer.usage.contains(BufferUsage::COPY_DST));

    let dst_barrier = dst_barriers.map(|pending| hal::memory::Barrier::Buffer {
        states: pending.to_states(),
        target: &dst_buffer.raw,
        families: None,
        range: None .. None,
    });

    let aspects = src_texture.full_range.aspects;
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
        image_layers: source.to_sub_layers(aspects),
        image_offset: conv::map_origin(source.origin),
        image_extent: conv::map_extent(copy_size),
    };
    let cmb_raw = cmb.raw.last_mut().unwrap();
    let stages = all_buffer_stages() | all_image_stages();
    unsafe {
        cmb_raw.pipeline_barrier(
            stages .. stages,
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

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_texture_to_buffer(
    command_encoder_id: CommandEncoderId,
    source: &TextureCopyView,
    destination: &BufferCopyView,
    copy_size: Extent3d,
) {
    gfx_select!(command_encoder_id => command_encoder_copy_texture_to_buffer(
        command_encoder_id,
        source,
        destination,
        copy_size))
}

pub fn command_encoder_copy_texture_to_texture<B: GfxBackend>(
    command_encoder_id: CommandEncoderId,
    source: &TextureCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    let hub = B::hub();
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

    barriers.extend(src_pending.map(|pending| hal::memory::Barrier::Image {
        states: pending.to_states(),
        target: &src_texture.raw,
        families: None,
        range: pending.selector,
    }));

    let (dst_texture, dst_pending) = cmb.trackers.textures.use_replace(
        &*texture_guard,
        destination.texture,
        destination.to_selector(aspects),
        TextureUsage::COPY_DST,
    );
    assert!(dst_texture.usage.contains(TextureUsage::COPY_DST));

    barriers.extend(dst_pending.map(|pending| hal::memory::Barrier::Image {
        states: pending.to_states(),
        target: &dst_texture.raw,
        families: None,
        range: pending.selector,
    }));

    let aspects = src_texture.full_range.aspects & dst_texture.full_range.aspects;
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
            all_image_stages() .. all_image_stages(),
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

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_copy_texture_to_texture(
    command_encoder_id: CommandEncoderId,
    source: &TextureCopyView,
    destination: &TextureCopyView,
    copy_size: Extent3d,
) {
    gfx_select!(command_encoder_id => command_encoder_copy_texture_to_texture(
        command_encoder_id,
        source,
        destination,
        copy_size))
}
