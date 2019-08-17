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
    hub::{HUB, Storage, Token},
    resource::TexturePlacement,
    swap_chain::{SwapChainLink, SwapImageEpoch},
    track::{Stitch, TrackerSet},
    BufferHandle,
    BufferId,
    Color,
    CommandBufferHandle,
    CommandBufferId,
    CommandEncoderId,
    DeviceId,
    LifeGuard,
    Stored,
    TextureHandle,
    TextureId,
    TextureUsage,
    TextureViewId,
};
#[cfg(feature = "local")]
use crate::{ComputePassId, RenderPassId};

use arrayvec::ArrayVec;
use back::Backend;
use hal::{
    adapter::PhysicalDevice,
    command::RawCommandBuffer,
    Device as _
};
use log::trace;

use std::{collections::hash_map::Entry, iter, mem, slice, ptr, thread::ThreadId};

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum LoadOp {
    Clear = 0,
    Load = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum StoreOp {
    Clear = 0,
    Store = 1,
}

#[repr(C)]
#[derive(Debug)]
pub struct RenderPassColorAttachmentDescriptor {
    pub attachment: TextureViewId,
    pub resolve_target: *const TextureViewId,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
#[derive(Debug)]
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
#[derive(Debug)]
pub struct RenderPassDescriptor {
    pub color_attachments: *const RenderPassColorAttachmentDescriptor,
    pub color_attachments_length: usize,
    pub depth_stencil_attachment: *const RenderPassDepthStencilAttachmentDescriptor<TextureViewId>,
}

#[derive(Debug)]
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
        buffer_guard: &Storage<BufferHandle, BufferId>,
        texture_guard: &Storage<TextureHandle, TextureId>,
    ) {
        let buffer_barriers =
            base.buffers
                .merge_replace(&head.buffers, stitch)
                .map(|pending| {
                    trace!("transit buffer {:?}", pending);
                    hal::memory::Barrier::Buffer {
                        states: pending.to_states(),
                        target: &buffer_guard[pending.id].raw,
                        range: None .. None,
                        families: None,
                    }
                });
        let texture_barriers = base
            .textures
            .merge_replace(&head.textures, stitch)
            .map(|pending| {
                trace!("transit texture {:?}", pending);
                hal::memory::Barrier::Image {
                    states: pending.to_states(),
                    target: &texture_guard[pending.id].raw,
                    range: pending.selector,
                    families: None,
                }
            });
        base.views.merge_extend(&head.views).unwrap();
        base.bind_groups.merge_extend(&head.bind_groups).unwrap();

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
#[derive(Debug)]
pub struct CommandEncoderDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    pub todo: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct CommandBufferDescriptor {
    pub todo: u32,
}

#[no_mangle]
pub extern "C" fn wgpu_command_encoder_finish(
    command_encoder_id: CommandEncoderId,
    _desc: Option<&CommandBufferDescriptor>,
) -> CommandBufferId {
    let mut token = Token::root();
    //TODO: actually close the last recorded command buffer
    let (mut comb_guard, _) = HUB.command_buffers.write(&mut token);
    comb_guard[command_encoder_id].is_recording = false; //TODO: check for the old value
    command_encoder_id
}

pub fn command_encoder_begin_render_pass(
    command_encoder_id: CommandEncoderId,
    desc: &RenderPassDescriptor,
) -> RenderPass<Backend> {
    let mut token = Token::root();
    let (adapter_guard, mut token) = HUB.adapters.read(&mut token);
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (mut cmb_guard, mut token) = HUB.command_buffers.write(&mut token);
    let cmb = &mut cmb_guard[command_encoder_id];
    let device = &device_guard[cmb.device_id.value];

    let (_, mut token) = HUB.buffers.read(&mut token); //skip token
    let (texture_guard, mut token) = HUB.textures.read(&mut token);
    let (view_guard, _) = HUB.texture_views.read(&mut token);

    let mut current_comb = device.com_allocator.extend(cmb);
    unsafe {
        current_comb.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
    }
    let mut extent = None;
    let mut barriers = Vec::new();

    let limits = adapter_guard[device.adapter_id].physical_device.limits();
    let samples_count_limit = limits.framebuffer_color_sample_counts;

    let color_attachments =
        unsafe { slice::from_raw_parts(desc.color_attachments, desc.color_attachments_length) };
    let depth_stencil_attachment = unsafe { desc.depth_stencil_attachment.as_ref() };

    let sample_count = color_attachments.get(0).map(|at| view_guard[at.attachment].samples).unwrap_or(1);
    assert!(sample_count & samples_count_limit != 0, "Attachment sample_count must be supported by physical device limits");
    for at in color_attachments.iter() {
        let sample_count_check = view_guard[at.attachment].samples;
        assert_eq!(sample_count_check, sample_count, "All attachments must have the same sample_count");

        if let Some(resolve) = unsafe { at.resolve_target.as_ref() } {
            assert_eq!(view_guard[*resolve].samples, 1, "All target_resolves must have a sample_count of 1");
        }
    }

    let rp_key = {
        let trackers = &mut cmb.trackers;
        let swap_chain_links = &mut cmb.swap_chain_links;

        let depth_stencil = depth_stencil_attachment.map(|at| {
            let view = trackers.views
                .use_extend(&*view_guard, at.attachment, (), ())
                .unwrap();
            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }
            let old_layout = match trackers.textures.query(
                view.texture_id.value,
                view.range.clone(),
            ) {
                Some(usage) => {
                    conv::map_texture_state(
                        usage,
                        hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL,
                    ).1
                }
                None => {
                    // Required sub-resources have inconsistent states, we need to
                    // issue individual barriers instead of relying on the render pass.
                    let (texture, pending) = trackers.textures.use_replace(
                        &*texture_guard,
                        view.texture_id.value,
                        view.range.clone(),
                        TextureUsage::OUTPUT_ATTACHMENT,
                    );
                    barriers.extend(pending.map(|pending| hal::memory::Barrier::Image {
                        states: pending.to_states(),
                        target: &texture.raw,
                        families: None,
                        range: pending.selector,
                    }));
                    hal::image::Layout::DepthStencilAttachmentOptimal
                }
            };
            hal::pass::Attachment {
                format: Some(conv::map_texture_format(view.format)),
                samples: view.samples,
                ops: conv::map_load_store_ops(at.depth_load_op, at.depth_store_op),
                stencil_ops: conv::map_load_store_ops(at.stencil_load_op, at.stencil_store_op),
                layouts: old_layout .. hal::image::Layout::DepthStencilAttachmentOptimal,
            }
        });

        let mut colors = ArrayVec::new();
        let mut resolves = ArrayVec::new();

        for at in color_attachments {
            let view = trackers.views
                .use_extend(&*view_guard, at.attachment, (), ())
                .unwrap();
            if let Some(ex) = extent {
                assert_eq!(ex, view.extent);
            } else {
                extent = Some(view.extent);
            }

            if view.is_owned_by_swap_chain {
                let link = match texture_guard[view.texture_id.value].placement {
                    TexturePlacement::SwapChain(ref link) => SwapChainLink {
                        swap_chain_id: link.swap_chain_id.clone(),
                        epoch: *link.epoch.lock(),
                        image_index: link.image_index,
                    },
                    TexturePlacement::Memory(_) | TexturePlacement::Void => unreachable!(),
                };
                swap_chain_links.push(link);
            }

            let old_layout = match trackers.textures.query(
                view.texture_id.value,
                view.range.clone(),
            ) {
                Some(usage) => {
                    conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                }
                None => {
                    // Required sub-resources have inconsistent states, we need to
                    // issue individual barriers instead of relying on the render pass.
                    let (texture, pending) = trackers.textures.use_replace(
                        &*texture_guard,
                        view.texture_id.value,
                        view.range.clone(),
                        TextureUsage::OUTPUT_ATTACHMENT,
                    );
                    barriers.extend(pending.map(|pending| hal::memory::Barrier::Image {
                        states: pending.to_states(),
                        target: &texture.raw,
                        families: None,
                        range: pending.selector,
                    }));
                    hal::image::Layout::ColorAttachmentOptimal
                }
            };

            colors.push(hal::pass::Attachment {
                format: Some(conv::map_texture_format(view.format)),
                samples: view.samples,
                ops: conv::map_load_store_ops(at.load_op, at.store_op),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: old_layout .. hal::image::Layout::ColorAttachmentOptimal,
            });

            if let Some(resolve_target) = unsafe { at.resolve_target.as_ref() } {
                let view = trackers.views
                    .use_extend(&*view_guard, *resolve_target, (), ())
                    .unwrap();
                if let Some(ex) = extent {
                    assert_eq!(ex, view.extent);
                } else {
                    extent = Some(view.extent);
                }

                if view.is_owned_by_swap_chain {
                    let link = match texture_guard[view.texture_id.value].placement {
                        TexturePlacement::SwapChain(ref link) => SwapChainLink {
                            swap_chain_id: link.swap_chain_id.clone(),
                            epoch: *link.epoch.lock(),
                            image_index: link.image_index,
                        },
                        TexturePlacement::Memory(_) | TexturePlacement::Void => unreachable!(),
                    };
                    swap_chain_links.push(link);
                }

                let old_layout = match trackers.textures.query(
                    view.texture_id.value,
                    view.range.clone(),
                ) {
                    Some(usage) => {
                        conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                    }
                    None => {
                        // Required sub-resources have inconsistent states, we need to
                        // issue individual barriers instead of relying on the render pass.
                        let (texture, pending) = trackers.textures.use_replace(
                            &*texture_guard,
                            view.texture_id.value,
                            view.range.clone(),
                            TextureUsage::OUTPUT_ATTACHMENT,
                        );
                        barriers.extend(pending.map(|pending| hal::memory::Barrier::Image {
                            states: pending.to_states(),
                            target: &texture.raw,
                            families: None,
                            range: pending.selector,
                        }));
                        hal::image::Layout::ColorAttachmentOptimal
                    }
                };

                resolves.push(hal::pass::Attachment {
                    format: Some(conv::map_texture_format(view.format)),
                    samples: view.samples,
                    ops: hal::pass::AttachmentOps::new(hal::pass::AttachmentLoadOp::DontCare, hal::pass::AttachmentStoreOp::Store),
                    stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                    layouts: old_layout .. hal::image::Layout::ColorAttachmentOptimal,
                });
            }
        }

        RenderPassKey {
            colors,
            resolves,
            depth_stencil,
        }
    };

    if !barriers.is_empty() {
        unsafe {
            current_comb.pipeline_barrier(
                all_image_stages() .. all_image_stages(),
                hal::memory::Dependencies::empty(),
                barriers,
            );
        }
    }

    let mut render_pass_cache = device.render_passes.lock();
    let render_pass = match render_pass_cache.entry(rp_key.clone()) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let color_ids = [
                (0, hal::image::Layout::ColorAttachmentOptimal),
                (1, hal::image::Layout::ColorAttachmentOptimal),
                (2, hal::image::Layout::ColorAttachmentOptimal),
                (3, hal::image::Layout::ColorAttachmentOptimal),
            ];

            let mut resolve_ids = ArrayVec::<[_; crate::device::MAX_COLOR_TARGETS]>::new();
            let mut attachment_index = color_attachments.len();
            if color_attachments.iter().any(|at| at.resolve_target != ptr::null()) {
                for (i, at) in color_attachments.iter().enumerate() {
                    if at.resolve_target == ptr::null() {
                        resolve_ids.push((hal::pass::ATTACHMENT_UNUSED, hal::image::Layout::ColorAttachmentOptimal));
                    } else {
                        let sample_count_check = view_guard[color_attachments[i].attachment].samples;
                        assert!(sample_count_check > 1, "RenderPassColorAttachmentDescriptor with a resolve_target must have an attachment with sample_count > 1");
                        resolve_ids.push((attachment_index, hal::image::Layout::ColorAttachmentOptimal));
                        attachment_index += 1;
                    }
                }
            }

            let depth_id = (
                attachment_index,
                hal::image::Layout::DepthStencilAttachmentOptimal,
            );

            let subpass = hal::pass::SubpassDesc {
                colors: &color_ids[.. color_attachments.len()],
                resolves: &resolve_ids,
                depth_stencil: depth_stencil_attachment.map(|_| &depth_id),
                inputs: &[],
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
        resolves: color_attachments.iter().filter_map(|at|
            unsafe { at.resolve_target.as_ref() }.cloned()
        ).collect(),
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
        .zip(&rp_key.colors)
        .flat_map(|(at, key)| {
            match at.load_op {
                LoadOp::Load => None,
                LoadOp::Clear => {
                    use hal::format::ChannelType;
                    //TODO: validate sign/unsign and normalized ranges of the color values
                    let value = match key.format.unwrap().base_format().1 {
                        ChannelType::Unorm |
                        ChannelType::Snorm |
                        ChannelType::Ufloat |
                        ChannelType::Sfloat |
                        ChannelType::Uscaled |
                        ChannelType::Sscaled |
                        ChannelType::Srgb => {
                            hal::command::ClearColor::Sfloat(conv::map_color_f32(&at.clear_color))
                        }
                        ChannelType::Sint => {
                            hal::command::ClearColor::Sint(conv::map_color_i32(&at.clear_color))
                        }
                        ChannelType::Uint => {
                            hal::command::ClearColor::Uint(conv::map_color_u32(&at.clear_color))
                        }
                    };
                    Some(hal::command::ClearValueRaw::from(hal::command::ClearValue::Color(value)))
                }
            }
        })
        .chain(depth_stencil_attachment.and_then(|at| {
            match (at.depth_load_op, at.stencil_load_op) {
                (LoadOp::Load, LoadOp::Load) => None,
                (LoadOp::Clear, _) | (_, LoadOp::Clear) => {
                    let value = hal::command::ClearDepthStencil(at.clear_depth, at.clear_stencil);
                    Some(hal::command::ClearValueRaw::from(hal::command::ClearValue::DepthStencil(value)))
                }
            }
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
                depth: 0.0 .. 1.0,
            }),
        );
    }

    let context = RenderPassContext {
        colors: color_attachments
            .iter()
            .map(|at| view_guard[at.attachment].format)
            .collect(),
        resolves: color_attachments
            .iter()
            .filter_map(|at| unsafe { at.resolve_target.as_ref() })
            .map(|resolve| view_guard[*resolve].format)
            .collect(),
        depth_stencil: depth_stencil_attachment.map(|at| view_guard[at.attachment].format),
    };

    RenderPass::new(
        current_comb,
        Stored {
            value: command_encoder_id,
            ref_count: cmb.life_guard.ref_count.clone(),
        },
        context,
        sample_count,
    )
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_render_pass(
    command_encoder_id: CommandEncoderId,
    desc: &RenderPassDescriptor,
) -> RenderPassId {
    let pass = command_encoder_begin_render_pass(command_encoder_id, desc);
    HUB.render_passes.register_local(pass, &mut Token::root())
}

pub fn command_encoder_begin_compute_pass(
    command_encoder_id: CommandEncoderId,
) -> ComputePass<Backend> {
    let mut token = Token::root();
    let (mut cmb_guard, _) = HUB.command_buffers.write(&mut token);
    let cmb = &mut cmb_guard[command_encoder_id];

    let raw = cmb.raw.pop().unwrap();
    let trackers = mem::replace(&mut cmb.trackers, TrackerSet::new());
    let stored = Stored {
        value: command_encoder_id,
        ref_count: cmb.life_guard.ref_count.clone(),
    };

    ComputePass::new(raw, stored, trackers)
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_command_encoder_begin_compute_pass(
    command_encoder_id: CommandEncoderId,
) -> ComputePassId {
    let pass = command_encoder_begin_compute_pass(command_encoder_id);
    HUB.compute_passes.register_local(pass, &mut Token::root())
}
