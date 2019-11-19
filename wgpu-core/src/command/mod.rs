/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

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
    hub::{GfxBackend, Global, IdentityFilter, Storage, Token},
    id::{
        BufferId,
        CommandBufferId,
        CommandEncoderId,
        ComputePassId,
        DeviceId,
        RenderPassId,
        TextureId,
        TextureViewId,
    },
    resource::{Buffer, Texture, TextureUsage, TextureViewInner},
    track::{Stitch, TrackerSet},
    Color,
    Features,
    LifeGuard,
    Stored,
};

use arrayvec::ArrayVec;
use hal::{adapter::PhysicalDevice as _, command::CommandBuffer as _, device::Device as _};

use std::{borrow::Borrow, collections::hash_map::Entry, iter, mem, ptr, slice, thread::ThreadId};


pub struct RenderBundle<B: hal::Backend> {
    _raw: B::CommandBuffer,
}

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

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor {
    pub todo: u32,
}

#[derive(Debug)]
pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    is_recording: bool,
    recorded_thread_id: ThreadId,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) trackers: TrackerSet,
    pub(crate) used_swap_chain: Option<(Stored<TextureViewId>, B::Framebuffer)>,
    pub(crate) features: Features,
}

impl<B: GfxBackend> CommandBuffer<B> {
    pub(crate) fn insert_barriers(
        raw: &mut B::CommandBuffer,
        base: &mut TrackerSet,
        head: &TrackerSet,
        stitch: Stitch,
        buffer_guard: &Storage<Buffer<B>, BufferId>,
        texture_guard: &Storage<Texture<B>, TextureId>,
    ) {
        log::trace!("\tstitch {:?}", stitch);
        debug_assert_eq!(B::VARIANT, base.backend());
        debug_assert_eq!(B::VARIANT, head.backend());

        let buffer_barriers = base
            .buffers
            .merge_replace(&head.buffers, stitch)
            .map(|pending| {
                log::trace!("\tbuffer -> {:?}", pending);
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
                log::trace!("\ttexture -> {:?}", pending);
                hal::memory::Barrier::Image {
                    states: pending.to_states(),
                    target: &texture_guard[pending.id].raw,
                    range: pending.selector,
                    families: None,
                }
            });
        base.views.merge_extend(&head.views).unwrap();
        base.bind_groups.merge_extend(&head.bind_groups).unwrap();
        base.samplers.merge_extend(&head.samplers).unwrap();

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
#[derive(Clone, Debug, Default)]
pub struct CommandEncoderDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    pub todo: u32,
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct CommandBufferDescriptor {
    pub todo: u32,
}

impl<F> Global<F> {
    pub fn command_encoder_finish<B: GfxBackend>(
        &self,
        encoder_id: CommandEncoderId,
        _desc: &CommandBufferDescriptor,
    ) -> CommandBufferId {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: actually close the last recorded command buffer
        let (mut comb_guard, _) = hub.command_buffers.write(&mut token);
        let comb = &mut comb_guard[encoder_id];
        assert!(comb.is_recording);
        comb.is_recording = false;
        // stop tracking the swapchain image, if used
        if let Some((ref view_id, _)) = comb.used_swap_chain {
            comb.trackers.views.remove(view_id.value);
        }
        encoder_id
    }
}

impl<F: IdentityFilter<RenderPassId>> Global<F> {
    pub fn command_encoder_begin_render_pass<B: GfxBackend>(
        &self,
        encoder_id: CommandEncoderId,
        desc: &RenderPassDescriptor,
        id_in: F::Input,
    ) -> RenderPassId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let device = &device_guard[cmb.device_id.value];

        let limits = adapter_guard[device.adapter_id]
            .raw
            .physical_device
            .limits();
        let samples_count_limit = limits.framebuffer_color_sample_counts;

        let mut current_comb = device.com_allocator.extend(cmb);
        unsafe {
            current_comb.begin(
                hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
                hal::command::CommandBufferInheritanceInfo::default(),
            );
        }

        let pass = {
            let (_, mut token) = hub.buffers.read(&mut token); //skip token
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (view_guard, _) = hub.texture_views.read(&mut token);

            let mut extent = None;
            let mut barriers = Vec::new();
            let mut used_swap_chain_image = None::<Stored<TextureViewId>>;

            let color_attachments = unsafe {
                slice::from_raw_parts(desc.color_attachments, desc.color_attachments_length)
            };
            let depth_stencil_attachment = unsafe { desc.depth_stencil_attachment.as_ref() };

            let sample_count = color_attachments
                .get(0)
                .map(|at| view_guard[at.attachment].samples)
                .unwrap_or(1);
            assert!(
                sample_count & samples_count_limit != 0,
                "Attachment sample_count must be supported by physical device limits"
            );

            log::trace!(
                "Encoding render pass begin in command buffer {:?}",
                encoder_id
            );
            let rp_key = {
                let trackers = &mut cmb.trackers;

                let depth_stencil = depth_stencil_attachment.map(|at| {
                    let view = trackers
                        .views
                        .use_extend(&*view_guard, at.attachment, (), ())
                        .unwrap();
                    if let Some(ex) = extent {
                        assert_eq!(ex, view.extent);
                    } else {
                        extent = Some(view.extent);
                    }
                    let texture_id = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => source_id.value,
                        TextureViewInner::SwapChain { .. } => {
                            panic!("Unexpected depth/stencil use of swapchain image!")
                        }
                    };

                    let texture = &texture_guard[texture_id];
                    assert!(texture.usage.contains(TextureUsage::OUTPUT_ATTACHMENT));

                    let old_layout = match trackers.textures.query(texture_id, view.range.clone()) {
                        Some(usage) => {
                            conv::map_texture_state(
                                usage,
                                hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL,
                            )
                            .1
                        }
                        None => {
                            // Required sub-resources have inconsistent states, we need to
                            // issue individual barriers instead of relying on the render pass.
                            let pending = trackers.textures.change_replace(
                                texture_id,
                                &texture.life_guard.ref_count,
                                view.range.clone(),
                                TextureUsage::OUTPUT_ATTACHMENT,
                                &texture.full_range,
                            );

                            barriers.extend(pending.map(|pending| {
                                log::trace!("\tdepth-stencil {:?}", pending);
                                hal::memory::Barrier::Image {
                                    states: pending.to_states(),
                                    target: &texture.raw,
                                    families: None,
                                    range: pending.selector,
                                }
                            }));
                            hal::image::Layout::DepthStencilAttachmentOptimal
                        }
                    };
                    hal::pass::Attachment {
                        format: Some(conv::map_texture_format(view.format, device.features)),
                        samples: view.samples,
                        ops: conv::map_load_store_ops(at.depth_load_op, at.depth_store_op),
                        stencil_ops: conv::map_load_store_ops(
                            at.stencil_load_op,
                            at.stencil_store_op,
                        ),
                        layouts: old_layout .. hal::image::Layout::DepthStencilAttachmentOptimal,
                    }
                });

                let mut colors = ArrayVec::new();
                let mut resolves = ArrayVec::new();

                for at in color_attachments {
                    let view = &view_guard[at.attachment];
                    if let Some(ex) = extent {
                        assert_eq!(ex, view.extent);
                    } else {
                        extent = Some(view.extent);
                    }
                    assert_eq!(
                        view.samples, sample_count,
                        "All attachments must have the same sample_count"
                    );
                    let first_use = trackers.views.init(
                        at.attachment,
                        view.life_guard.ref_count.clone(),
                        (),
                        (),
                    );

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let texture = &texture_guard[source_id.value];
                            assert!(texture.usage.contains(TextureUsage::OUTPUT_ATTACHMENT));

                            let old_layout = match trackers
                                .textures
                                .query(source_id.value, view.range.clone())
                            {
                                Some(usage) => {
                                    conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                                }
                                None => {
                                    // Required sub-resources have inconsistent states, we need to
                                    // issue individual barriers instead of relying on the render pass.
                                    let pending = trackers.textures.change_replace(
                                        source_id.value,
                                        &texture.life_guard.ref_count,
                                        view.range.clone(),
                                        TextureUsage::OUTPUT_ATTACHMENT,
                                        &texture.full_range,
                                    );
                                    barriers.extend(pending.map(|pending| {
                                        log::trace!("\tcolor {:?}", pending);
                                        hal::memory::Barrier::Image {
                                            states: pending.to_states(),
                                            target: &texture.raw,
                                            families: None,
                                            range: pending.selector,
                                        }
                                    }));
                                    hal::image::Layout::ColorAttachmentOptimal
                                }
                            };
                            old_layout .. hal::image::Layout::ColorAttachmentOptimal
                        }
                        TextureViewInner::SwapChain { .. } => {
                            if let Some((ref view_id, _)) = cmb.used_swap_chain {
                                assert_eq!(view_id.value, at.attachment);
                            } else {
                                assert!(used_swap_chain_image.is_none());
                                used_swap_chain_image = Some(Stored {
                                    value: at.attachment,
                                    ref_count: view.life_guard.ref_count.clone(),
                                });
                            }

                            let end = hal::image::Layout::Present;
                            let start = if first_use {
                                hal::image::Layout::Undefined
                            } else {
                                end
                            };
                            start .. end
                        }
                    };

                    colors.push(hal::pass::Attachment {
                        format: Some(conv::map_texture_format(view.format, device.features)),
                        samples: view.samples,
                        ops: conv::map_load_store_ops(at.load_op, at.store_op),
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        layouts,
                    });
                }

                for &resolve_target in color_attachments
                    .iter()
                    .flat_map(|at| unsafe { at.resolve_target.as_ref() })
                {
                    let view = &view_guard[resolve_target];
                    assert_eq!(extent, Some(view.extent));
                    assert_eq!(
                        view.samples, 1,
                        "All resolve_targets must have a sample_count of 1"
                    );
                    let first_use = trackers.views.init(
                        resolve_target,
                        view.life_guard.ref_count.clone(),
                        (),
                        (),
                    );

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let texture = &texture_guard[source_id.value];
                            assert!(texture.usage.contains(TextureUsage::OUTPUT_ATTACHMENT));

                            let old_layout = match trackers
                                .textures
                                .query(source_id.value, view.range.clone())
                            {
                                Some(usage) => {
                                    conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                                }
                                None => {
                                    // Required sub-resources have inconsistent states, we need to
                                    // issue individual barriers instead of relying on the render pass.
                                    let pending = trackers.textures.change_replace(
                                        source_id.value,
                                        &texture.life_guard.ref_count,
                                        view.range.clone(),
                                        TextureUsage::OUTPUT_ATTACHMENT,
                                        &texture.full_range,
                                    );
                                    barriers.extend(pending.map(|pending| {
                                        log::trace!("\tresolve {:?}", pending);
                                        hal::memory::Barrier::Image {
                                            states: pending.to_states(),
                                            target: &texture.raw,
                                            families: None,
                                            range: pending.selector,
                                        }
                                    }));
                                    hal::image::Layout::ColorAttachmentOptimal
                                }
                            };
                            old_layout .. hal::image::Layout::ColorAttachmentOptimal
                        }
                        TextureViewInner::SwapChain { .. } => {
                            if let Some((ref view_id, _)) = cmb.used_swap_chain {
                                assert_eq!(view_id.value, resolve_target);
                            } else {
                                assert!(used_swap_chain_image.is_none());
                                used_swap_chain_image = Some(Stored {
                                    value: resolve_target,
                                    ref_count: view.life_guard.ref_count.clone(),
                                });
                            }

                            let end = hal::image::Layout::Present;
                            let start = if first_use {
                                hal::image::Layout::Undefined
                            } else {
                                end
                            };
                            start .. end
                        }
                    };

                    resolves.push(hal::pass::Attachment {
                        format: Some(conv::map_texture_format(view.format, device.features)),
                        samples: view.samples,
                        ops: hal::pass::AttachmentOps::new(
                            hal::pass::AttachmentLoadOp::DontCare,
                            hal::pass::AttachmentStoreOp::Store,
                        ),
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        layouts,
                    });
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
                    if color_attachments
                        .iter()
                        .any(|at| at.resolve_target != ptr::null())
                    {
                        for (i, at) in color_attachments.iter().enumerate() {
                            if at.resolve_target == ptr::null() {
                                resolve_ids.push((
                                    hal::pass::ATTACHMENT_UNUSED,
                                    hal::image::Layout::ColorAttachmentOptimal,
                                ));
                            } else {
                                let sample_count_check =
                                    view_guard[color_attachments[i].attachment].samples;
                                assert!(sample_count_check > 1, "RenderPassColorAttachmentDescriptor with a resolve_target must have an attachment with sample_count > 1");
                                resolve_ids.push((
                                    attachment_index,
                                    hal::image::Layout::ColorAttachmentOptimal,
                                ));
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

            let mut framebuffer_cache;
            let fb_key = FramebufferKey {
                colors: color_attachments.iter().map(|at| at.attachment).collect(),
                resolves: color_attachments
                    .iter()
                    .filter_map(|at| unsafe { at.resolve_target.as_ref() }.cloned())
                    .collect(),
                depth_stencil: depth_stencil_attachment.map(|at| at.attachment),
            };

            let framebuffer = match used_swap_chain_image.take() {
                Some(view_id) => {
                    assert!(cmb.used_swap_chain.is_none());
                    // Always create a new framebuffer and delete it after presentation.
                    let attachments = fb_key.all().map(|&id| match view_guard[id].inner {
                        TextureViewInner::Native { ref raw, .. } => raw,
                        TextureViewInner::SwapChain { ref image, .. } => Borrow::borrow(image),
                    });
                    let framebuffer = unsafe {
                        device
                            .raw
                            .create_framebuffer(&render_pass, attachments, extent.unwrap())
                            .unwrap()
                    };
                    cmb.used_swap_chain = Some((view_id, framebuffer));
                    &mut cmb.used_swap_chain.as_mut().unwrap().1
                }
                None => {
                    // Cache framebuffers by the device.
                    framebuffer_cache = device.framebuffers.lock();
                    match framebuffer_cache.entry(fb_key) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            let fb = {
                                let attachments =
                                    e.key().all().map(|&id| match view_guard[id].inner {
                                        TextureViewInner::Native { ref raw, .. } => raw,
                                        TextureViewInner::SwapChain { ref image, .. } => {
                                            Borrow::borrow(image)
                                        }
                                    });
                                unsafe {
                                    device.raw.create_framebuffer(
                                        &render_pass,
                                        attachments,
                                        extent.unwrap(),
                                    )
                                }
                                .unwrap()
                            };
                            e.insert(fb)
                        }
                    }
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
                                ChannelType::Unorm
                                | ChannelType::Snorm
                                | ChannelType::Ufloat
                                | ChannelType::Sfloat
                                | ChannelType::Uscaled
                                | ChannelType::Sscaled
                                | ChannelType::Srgb => hal::command::ClearColor {
                                    float32: conv::map_color_f32(&at.clear_color),
                                },
                                ChannelType::Sint => hal::command::ClearColor {
                                    sint32: conv::map_color_i32(&at.clear_color),
                                },
                                ChannelType::Uint => hal::command::ClearColor {
                                    uint32: conv::map_color_u32(&at.clear_color),
                                },
                            };
                            Some(hal::command::ClearValue { color: value })
                        }
                    }
                })
                .chain(depth_stencil_attachment.and_then(|at| {
                    match (at.depth_load_op, at.stencil_load_op) {
                        (LoadOp::Load, LoadOp::Load) => None,
                        (LoadOp::Clear, _) | (_, LoadOp::Clear) => {
                            let value = hal::command::ClearDepthStencil {
                                depth: at.clear_depth,
                                stencil: at.clear_stencil,
                            };
                            Some(hal::command::ClearValue {
                                depth_stencil: value,
                            })
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
                    value: encoder_id,
                    ref_count: cmb.life_guard.ref_count.clone(),
                },
                context,
                sample_count,
                cmb.features.max_bind_groups,
            )
        };
        hub.render_passes.register_identity(id_in, pass, &mut token)
    }
}

impl<F: IdentityFilter<ComputePassId>> Global<F> {
    pub fn command_encoder_begin_compute_pass<B: GfxBackend>(
        &self,
        encoder_id: CommandEncoderId,
        _desc: &ComputePassDescriptor,
        id_in: F::Input,
    ) -> ComputePassId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];

        let raw = cmb.raw.pop().unwrap();
        let trackers = mem::replace(&mut cmb.trackers, TrackerSet::new(encoder_id.backend()));
        let stored = Stored {
            value: encoder_id,
            ref_count: cmb.life_guard.ref_count.clone(),
        };

        let pass = ComputePass::new(raw, stored, trackers, cmb.features.max_bind_groups);
        hub.compute_passes
            .register_identity(id_in, pass, &mut token)
    }
}
