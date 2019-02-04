mod allocator;
mod bind;
mod compute;
mod render;

pub(crate) use self::allocator::CommandAllocator;
pub use self::compute::*;
pub use self::render::*;

use crate::device::{
    FramebufferKey, RenderPassKey,
    all_buffer_stages, all_image_stages,
};
use crate::registry::{Items, HUB};
use crate::swap_chain::{SwapChainLink, SwapImageEpoch};
use crate::track::{BufferTracker, TextureTracker, TrackPermit, Tracktion};
use crate::{conv, resource};
use crate::{
    BufferId, CommandBufferId, ComputePassId, DeviceId,
    RenderPassId, TextureId, TextureViewId,
    BufferUsageFlags, TextureUsageFlags, Color,
    Extent3d, Origin3d,
    LifeGuard, Stored, WeaklyStored,
    B,
};

use hal::command::RawCommandBuffer;
use hal::{Device as _Device};
use log::trace;

use std::collections::hash_map::Entry;
use std::ops::Range;
use std::{iter, slice};
use std::thread::ThreadId;


#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum LoadOp {
    Clear = 0,
    Load = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum StoreOp {
    Store = 0,
}

#[repr(C)]
pub struct RenderPassColorAttachmentDescriptor<T> {
    pub attachment: T,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
pub struct RenderPassDepthStencilAttachmentDescriptor<T> {
    pub attachment: T,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub clear_depth: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_stencil: u32,
}

#[repr(C)]
pub struct RenderPassDescriptor {
    pub color_attachments: *const RenderPassColorAttachmentDescriptor<TextureViewId>,
    pub color_attachments_length: usize,
    pub depth_stencil_attachment: *const RenderPassDepthStencilAttachmentDescriptor<TextureViewId>,
}

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

pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    recorded_thread_id: ThreadId,
    device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) buffer_tracker: BufferTracker,
    pub(crate) texture_tracker: TextureTracker,
    pub(crate) swap_chain_links: Vec<SwapChainLink<SwapImageEpoch>>,
}

impl CommandBuffer<B> {
    pub(crate) fn insert_barriers<I, J, Gb, Gt>(
        raw: &mut <B as hal::Backend>::CommandBuffer,
        buffer_iter: I,
        texture_iter: J,
        buffer_guard: &Gb,
        texture_guard: &Gt,
    ) where
        I: Iterator<Item = (BufferId, Range<BufferUsageFlags>)>,
        J: Iterator<Item = (TextureId, Range<TextureUsageFlags>)>,
        Gb: Items<resource::Buffer<B>>,
        Gt: Items<resource::Texture<B>>,
    {

        let buffer_barriers = buffer_iter.map(|(id, transit)| {
            let b = buffer_guard.get(id);
            trace!("transit {:?} {:?}", id, transit);
            hal::memory::Barrier::Buffer {
                states: conv::map_buffer_state(transit.start)..conv::map_buffer_state(transit.end),
                target: &b.raw,
                range: Range {
                    start: None,
                    end: None,
                },
                families: None,
            }
        });
        let texture_barriers = texture_iter.map(|(id, transit)| {
            let t = texture_guard.get(id);
            trace!("transit {:?} {:?}", id, transit);
            let aspects = t.full_range.aspects;
            hal::memory::Barrier::Image {
                states: conv::map_texture_state(transit.start, aspects)
                    ..conv::map_texture_state(transit.end, aspects),
                target: &t.raw,
                range: t.full_range.clone(), //TODO?
                families: None,
            }
        });

        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            raw.pipeline_barrier(
                stages .. stages,
                hal::memory::Dependencies::empty(),
                buffer_barriers.chain(texture_barriers),
            );
        }
    }
}

#[repr(C)]
pub struct CommandBufferDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    pub todo: u32,
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_render_pass(
    command_buffer_id: CommandBufferId,
    desc: RenderPassDescriptor,
) -> RenderPassId {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(command_buffer_id);
    let device_guard = HUB.devices.read();
    let device = device_guard.get(cmb.device_id.value);
    let view_guard = HUB.texture_views.read();

    let mut current_comb = device.com_allocator.extend(cmb);
    unsafe {
        current_comb.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
    }
    let mut extent = None;

    let color_attachments = unsafe {
        slice::from_raw_parts(
            desc.color_attachments,
            desc.color_attachments_length,
        )
    };
    let depth_stencil_attachment = unsafe {
        desc.depth_stencil_attachment.as_ref()
    };

    let rp_key = {
        let tracker = &mut cmb.texture_tracker;
        let swap_chain_links = &mut cmb.swap_chain_links;

        let depth_stencil_key = depth_stencil_attachment.map(|at| {
            let view = view_guard.get(at.attachment);
            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }
            let query = tracker.query(&view.texture_id, TextureUsageFlags::empty());
            let (_, layout) = conv::map_texture_state(
                query.usage,
                hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL,
            );
            hal::pass::Attachment {
                format: Some(conv::map_texture_format(view.format)),
                samples: view.samples,
                ops: conv::map_load_store_ops(at.depth_load_op, at.depth_store_op),
                stencil_ops: conv::map_load_store_ops(at.stencil_load_op, at.stencil_store_op),
                layouts: layout..layout,
            }
        });

        let color_keys = color_attachments.iter().map(|at| {
            let view = view_guard.get(at.attachment);

            if view.is_owned_by_swap_chain {
                let link = match HUB.textures
                    .read()
                    .get(view.texture_id.value)
                    .swap_chain_link
                {
                    Some(ref link) => SwapChainLink {
                        swap_chain_id: link.swap_chain_id.clone(),
                        epoch: *link.epoch.lock(),
                        image_index: link.image_index,
                    },
                    None => unreachable!()
                };
                swap_chain_links.push(link);
            }

            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }
            let query = tracker.query(&view.texture_id, TextureUsageFlags::empty());
            let (_, layout) = conv::map_texture_state(query.usage, hal::format::Aspects::COLOR);
            hal::pass::Attachment {
                format: Some(conv::map_texture_format(view.format)),
                samples: view.samples,
                ops: conv::map_load_store_ops(at.load_op, at.store_op),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: layout..layout,
            }
        });

        RenderPassKey {
            attachments: color_keys.chain(depth_stencil_key).collect(),
        }
    };

    let mut render_pass_cache = device.render_passes.lock();
    let render_pass = match render_pass_cache.entry(rp_key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let color_ids = [
                (0, hal::image::Layout::ColorAttachmentOptimal),
                (1, hal::image::Layout::ColorAttachmentOptimal),
                (2, hal::image::Layout::ColorAttachmentOptimal),
                (3, hal::image::Layout::ColorAttachmentOptimal),
            ];
            let depth_id = (
                color_attachments.len(),
                hal::image::Layout::DepthStencilAttachmentOptimal,
            );

            let subpass = hal::pass::SubpassDesc {
                colors: &color_ids[..color_attachments.len()],
                depth_stencil: depth_stencil_attachment.map(|_| &depth_id),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let pass = unsafe {
                device
                    .raw
                    .create_render_pass(&e.key().attachments, &[subpass], &[])
            }
            .unwrap();
            e.insert(pass)
        }
    };

    let mut framebuffer_cache = device.framebuffers.lock();
    let fb_key = FramebufferKey {
        attachments: color_attachments
            .iter()
            .map(|at| WeaklyStored(at.attachment))
            .chain(
                depth_stencil_attachment.map(|at| WeaklyStored(at.attachment)),
            )
            .collect(),
    };
    let framebuffer = match framebuffer_cache.entry(fb_key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let fb = {
                let attachments = e
                    .key()
                    .attachments
                    .iter()
                    .map(|&WeaklyStored(id)| &view_guard.get(id).raw);

                unsafe {
                    device
                        .raw
                        .create_framebuffer(&render_pass, attachments, extent.unwrap())
                }
                .unwrap()
            };
            e.insert(fb)
        }
    };

    let rect = {
        let ex = extent.unwrap();
        hal::pso::Rect {
            x: 0,
            y: 0,
            w: ex.width as _,
            h: ex.height as _,
        }
    };

    let clear_values = color_attachments
        .iter()
        .map(|at| {
            //TODO: integer types?
            let value = hal::command::ClearColor::Float(conv::map_color(at.clear_color));
            hal::command::ClearValueRaw::from(hal::command::ClearValue::Color(value))
        })
        .chain(depth_stencil_attachment.map(|at| {
            let value = hal::command::ClearDepthStencil(at.clear_depth, at.clear_stencil);
            hal::command::ClearValueRaw::from(hal::command::ClearValue::DepthStencil(value))
        }));

    unsafe {
        current_comb.begin_render_pass(
            render_pass,
            framebuffer,
            rect,
            clear_values,
            hal::command::SubpassContents::Inline,
        );
        current_comb.set_scissors(0, iter::once(&rect));
        current_comb.set_viewports(0, iter::once(hal::pso::Viewport {
            rect,
            depth: 0.0 .. 1.0,
        }));
    }

    HUB.render_passes.write().register(RenderPass::new(
        current_comb,
        Stored {
            value: command_buffer_id,
            ref_count: cmb.life_guard.ref_count.clone(),
        },
    ))
}

#[no_mangle]
pub extern "C" fn wgpu_command_buffer_begin_compute_pass(
    command_buffer_id: CommandBufferId,
) -> ComputePassId {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(command_buffer_id);

    let raw = cmb.raw.pop().unwrap();
    let stored = Stored {
        value: command_buffer_id,
        ref_count: cmb.life_guard.ref_count.clone(),
    };

    HUB.compute_passes
        .write()
        .register(ComputePass::new(raw, stored))
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
            states: conv::map_buffer_state(old) .. hal::buffer::Access::TRANSFER_WRITE,
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

    let bytes_per_texel = conv::map_texture_format(dst_texture.format).surface_desc().bits as u32 / 8;
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
