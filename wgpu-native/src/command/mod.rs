mod allocator;
mod bind;
mod compute;
mod render;
mod transfer;

pub(crate) use self::allocator::CommandAllocator;
pub use self::compute::*;
pub use self::render::*;
pub use self::transfer::*;

use crate::{
    conv,
    device::{
        all_buffer_stages,
        all_image_stages,
        FramebufferKey,
        RenderPassContext,
        RenderPassKey,
    },
    hub::{Storage, HUB},
    pipeline::IndexFormat,
    resource::TexturePlacement,
    swap_chain::{SwapChainLink, SwapImageEpoch},
    track::{DummyUsage, Stitch, TrackerSet},
    BufferHandle,
    Color,
    CommandBufferHandle,
    CommandBufferId,
    CommandEncoderId,
    DeviceId,
    LifeGuard,
    Stored,
    TextureHandle,
    TextureUsageFlags,
    TextureViewId,
};
#[cfg(feature = "local")]
use crate::{ComputePassId, RenderPassId};

use back::Backend;
use hal::{command::RawCommandBuffer, Device as _};
use log::trace;

use std::{collections::hash_map::Entry, iter, slice, thread::ThreadId};

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

pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    is_recording: bool,
    recorded_thread_id: ThreadId,
    device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) trackers: TrackerSet,
    pub(crate) swap_chain_links: Vec<SwapChainLink<SwapImageEpoch>>,
}

impl CommandBufferHandle {
    pub(crate) fn insert_barriers(
        raw: &mut <Backend as hal::Backend>::CommandBuffer,
        base: &mut TrackerSet,
        head: &TrackerSet,
        stitch: Stitch,
        buffer_guard: &Storage<BufferHandle>,
        texture_guard: &Storage<TextureHandle>,
    ) {
        let buffer_barriers =
            base.buffers
                .consume_by_replace(&head.buffers, stitch)
                .map(|(id, transit)| {
                    let b = &buffer_guard[id];
                    trace!("transit buffer {:?} {:?}", id, transit);
                    hal::memory::Barrier::Buffer {
                        states: conv::map_buffer_state(transit.start)
                            ..conv::map_buffer_state(transit.end),
                        target: &b.raw,
                        range: None..None,
                        families: None,
                    }
                });
        let texture_barriers = base
            .textures
            .consume_by_replace(&head.textures, stitch)
            .map(|(id, transit)| {
                let t = &texture_guard[id];
                trace!("transit texture {:?} {:?}", id, transit);
                let aspects = t.full_range.aspects;
                hal::memory::Barrier::Image {
                    states: conv::map_texture_state(transit.start, aspects)
                        ..conv::map_texture_state(transit.end, aspects),
                    target: &t.raw,
                    range: t.full_range.clone(), //TODO?
                    families: None,
                }
            });
        base.views.consume_by_extend(&head.views).unwrap();

        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            raw.pipeline_barrier(
                stages..stages,
                hal::memory::Dependencies::empty(),
                buffer_barriers.chain(texture_barriers),
            );
        }
    }
}

#[repr(C)]
pub struct CommandEncoderDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    pub todo: u32,
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_finish(
    command_encoder_id: CommandEncoderId,
) -> CommandBufferId {
    HUB.command_buffers.write()[command_encoder_id].is_recording = false; //TODO: check for the old value
    command_encoder_id
}

pub fn command_encoder_begin_render_pass(
    command_encoder_id: CommandEncoderId,
    desc: RenderPassDescriptor,
) -> RenderPass<Backend> {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[command_encoder_id];
    let device_guard = HUB.devices.read();
    let device = &device_guard[cmb.device_id.value];
    let view_guard = HUB.texture_views.read();

    let mut current_comb = device.com_allocator.extend(cmb);
    unsafe {
        current_comb.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
    }
    let mut extent = None;

    let color_attachments =
        unsafe { slice::from_raw_parts(desc.color_attachments, desc.color_attachments_length) };
    let depth_stencil_attachment = unsafe { desc.depth_stencil_attachment.as_ref() };

    let rp_key = {
        let trackers = &mut cmb.trackers;
        let swap_chain_links = &mut cmb.swap_chain_links;

        let depth_stencil = depth_stencil_attachment.map(|at| {
            let view = &view_guard[at.attachment];
            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }
            trackers
                .views
                .query(at.attachment, &view.life_guard.ref_count, DummyUsage);
            let query = trackers.textures.query(
                view.texture_id.value,
                &view.texture_id.ref_count,
                TextureUsageFlags::empty(),
            );
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
            let view = &view_guard[at.attachment];

            if view.is_owned_by_swap_chain {
                let link = match HUB.textures.read()[view.texture_id.value].placement {
                    TexturePlacement::SwapChain(ref link) => SwapChainLink {
                        swap_chain_id: link.swap_chain_id.clone(),
                        epoch: *link.epoch.lock(),
                        image_index: link.image_index,
                    },
                    TexturePlacement::Memory(_) | TexturePlacement::Void => unreachable!(),
                };
                swap_chain_links.push(link);
            }

            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }
            trackers
                .views
                .query(at.attachment, &view.life_guard.ref_count, DummyUsage);
            let query = trackers.textures.query(
                view.texture_id.value,
                &view.texture_id.ref_count,
                TextureUsageFlags::empty(),
            );
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
            colors: color_keys.collect(),
            depth_stencil,
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
                    .create_render_pass(e.key().all(), &[subpass], &[])
            }
            .unwrap();
            e.insert(pass)
        }
    };

    let mut framebuffer_cache = device.framebuffers.lock();
    let fb_key = FramebufferKey {
        colors: color_attachments.iter().map(|at| at.attachment).collect(),
        depth_stencil: depth_stencil_attachment.map(|at| at.attachment),
    };
    let framebuffer = match framebuffer_cache.entry(fb_key) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let fb = {
                let attachments = e.key().all().map(|&id| &view_guard[id].raw);

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
            let value = hal::command::ClearColor::Float(conv::map_color(&at.clear_color));
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
        current_comb.set_viewports(
            0,
            iter::once(hal::pso::Viewport {
                rect,
                depth: 0.0..1.0,
            }),
        );
    }

    let context = RenderPassContext {
        colors: color_attachments
            .iter()
            .map(|at| view_guard[at.attachment].format)
            .collect(),
        depth_stencil: depth_stencil_attachment.map(|at| view_guard[at.attachment].format),
    };

    let index_state = IndexState {
        bound_buffer_view: None,
        format: IndexFormat::Uint16,
    };

    RenderPass::new(
        current_comb,
        Stored {
            value: command_encoder_id,
            ref_count: cmb.life_guard.ref_count.clone(),
        },
        context,
        index_state,
    )
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_render_pass(
    command_encoder_id: CommandEncoderId,
    desc: RenderPassDescriptor,
) -> RenderPassId {
    let pass = command_encoder_begin_render_pass(command_encoder_id, desc);
    HUB.render_passes.register_local(pass)
}

pub fn command_encoder_begin_compute_pass(
    command_encoder_id: CommandEncoderId,
) -> ComputePass<Backend> {
    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[command_encoder_id];

    let raw = cmb.raw.pop().unwrap();
    let stored = Stored {
        value: command_encoder_id,
        ref_count: cmb.life_guard.ref_count.clone(),
    };

    ComputePass::new(raw, stored)
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_compute_pass(
    command_encoder_id: CommandEncoderId,
) -> ComputePassId {
    let pass = command_encoder_begin_compute_pass(command_encoder_id);
    HUB.compute_passes.register_local(pass)
}
