/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{
        bind::{Binder, LayoutChange},
        PhantomSlice,
        RawRenderTargets,
    },
    conv,
    device::{
        FramebufferKey,
        RenderPassContext,
        RenderPassKey,
        BIND_BUFFER_ALIGNMENT,
        MAX_VERTEX_BUFFERS,
        MAX_COLOR_TARGETS,
    },
    hub::{GfxBackend, Global, Token},
    id,
    pipeline::{IndexFormat, InputStepMode, PipelineFlags},
    resource::{BufferUsage, TextureUsage, TextureViewInner},
    track::TrackerSet,
    BufferAddress,
    Color,
    DynamicOffset,
    Stored,
};

use arrayvec::ArrayVec;
use hal::command::CommandBuffer as _;
use peek_poke::{Peek, PeekCopy, Poke};

use std::{
    borrow::Borrow,
    collections::hash_map::Entry,
    iter,
    marker::PhantomData,
    ops::Range,
};


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
pub struct RenderPassColorAttachmentDescriptorBase<T, R> {
    pub attachment: T,
    pub resolve_target: R,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_color: Color,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct RenderPassDepthStencilAttachmentDescriptorBase<T> {
    pub attachment: T,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub clear_depth: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_stencil: u32,
}

//Note: this could look better if `cbindgen` wasn't confused by &T used in place of
// a generic paramter, it's not able to mange
pub type OptionRef<'a, T> = Option<&'a T>;
pub type RenderPassColorAttachmentDescriptor<'a> =
    RenderPassColorAttachmentDescriptorBase<id::TextureViewId, OptionRef<'a, id::TextureViewId>>;
pub type RenderPassDepthStencilAttachmentDescriptor =
    RenderPassDepthStencilAttachmentDescriptorBase<id::TextureViewId>;

#[repr(C)]
#[derive(Debug)]
pub struct RenderPassDescriptor<'a> {
    pub color_attachments: *const RenderPassColorAttachmentDescriptor<'a>,
    pub color_attachments_length: usize,
    pub depth_stencil_attachment: Option<&'a RenderPassDepthStencilAttachmentDescriptor>,
}

#[derive(Clone, Copy, Debug, PeekCopy, Poke)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}

#[derive(Clone, Copy, Debug, PeekCopy, Poke)]
enum RenderCommand {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
        phantom_offsets: PhantomSlice<DynamicOffset>,
    },
    SetPipeline(id::RenderPipelineId),
    SetIndexBuffer {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    SetVertexBuffers {
        start_index: u8,
        count: u8,
        phantom_buffer_ids: PhantomSlice<id::BufferId>,
        phantom_offsets: PhantomSlice<BufferAddress>,
    },
    SetBlendColor(Color),
    SetStencilReference(u32),
    SetViewport {
        rect: Rect<f32>,
        //TODO: use half-float to reduce the size?
        depth_min: f32,
        depth_max: f32,
    },
    SetScissor(Rect<u32>),
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    DrawIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    DrawIndexedIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
    },
    End,
}

impl super::RawPass {
    pub fn new_render(parent_id: id::CommandEncoderId) -> Self {
        Self::from_vec(Vec::<RenderCommand>::with_capacity(1), parent_id)
    }
}

impl super::RawRenderPass {
    pub unsafe fn finish_render(mut self) -> (Vec<u8>, id::CommandEncoderId, RawRenderTargets) {
        self.raw.finish(RenderCommand::End);
        let (vec, parent_id) = self.raw.into_vec();
        (vec, parent_id, self.targets)
    }
}

#[derive(Debug, PartialEq)]
enum OptionalState {
    Unused,
    Required,
    Set,
}

impl OptionalState {
    fn require(&mut self, require: bool) {
        if require && *self == OptionalState::Unused {
            *self = OptionalState::Required;
        }
    }
}

#[derive(Debug, PartialEq)]
enum DrawError {
    MissingBlendColor,
    MissingStencilReference,
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
}

#[derive(Debug)]
pub struct IndexState {
    bound_buffer_view: Option<(id::BufferId, Range<BufferAddress>)>,
    format: IndexFormat,
    limit: u32,
}

impl IndexState {
    fn update_limit(&mut self) {
        self.limit = match self.bound_buffer_view {
            Some((_, ref range)) => {
                let shift = match self.format {
                    IndexFormat::Uint16 => 1,
                    IndexFormat::Uint32 => 2,
                };
                ((range.end - range.start) >> shift) as u32
            }
            None => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBufferState {
    total_size: BufferAddress,
    stride: BufferAddress,
    rate: InputStepMode,
}

impl VertexBufferState {
    const EMPTY: Self = VertexBufferState {
        total_size: 0,
        stride: 0,
        rate: InputStepMode::Vertex,
    };
}

#[derive(Debug)]
pub struct VertexState {
    inputs: [VertexBufferState; MAX_VERTEX_BUFFERS],
    vertex_limit: u32,
    instance_limit: u32,
}

impl VertexState {
    fn update_limits(&mut self) {
        self.vertex_limit = !0;
        self.instance_limit = !0;
        for vbs in &self.inputs {
            if vbs.stride == 0 {
                continue;
            }
            let limit = (vbs.total_size / vbs.stride) as u32;
            match vbs.rate {
                InputStepMode::Vertex => self.vertex_limit = self.vertex_limit.min(limit),
                InputStepMode::Instance => self.instance_limit = self.instance_limit.min(limit),
            }
        }
    }
}

#[derive(Debug)]
struct State {
    binder: Binder,
    blend_color: OptionalState,
    stencil_reference: OptionalState,
    index: IndexState,
    vertex: VertexState,
}

impl State {
    fn is_ready(&self) -> Result<(), DrawError> {
        //TODO: vertex buffers
        let bind_mask = self.binder.invalid_mask();
        if bind_mask != 0 {
            //let (expected, provided) = self.binder.entries[index as usize].info();
            return Err(DrawError::IncompatibleBindGroup {
                index: bind_mask.trailing_zeros() as u32,
            });
        }
        if self.blend_color == OptionalState::Required {
            return Err(DrawError::MissingBlendColor);
        }
        if self.stencil_reference == OptionalState::Required {
            return Err(DrawError::MissingStencilReference);
        }
        Ok(())
    }
}

// Common routines between render/compute

impl<F> Global<F> {
    pub fn command_encoder_run_render_pass<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        color_attachments: &[RenderPassColorAttachmentDescriptor],
        depth_stencil_attachment: Option<&RenderPassDepthStencilAttachmentDescriptor>,
        raw_data: &[u8],
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);

        let mut trackers = TrackerSet::new(B::VARIANT);
        let cmb = &mut cmb_guard[encoder_id];
        let device = &device_guard[cmb.device_id.value];
        let mut raw = device.com_allocator.extend(cmb);

        unsafe {
            raw.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        }

        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.render_pipelines.read(&mut token);
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let (view_guard, _) = hub.texture_views.read(&mut token);

        let (context, sample_count) = {
            use hal::{adapter::PhysicalDevice as _, device::Device as _};

            let limits = adapter_guard[device.adapter_id]
                .raw
                .physical_device
                .limits();
            let samples_count_limit = limits.framebuffer_color_sample_counts;
            let base_trackers = &cmb.trackers;

            let mut extent = None;
            let mut used_swap_chain = None::<Stored<id::SwapChainId>>;

            let sample_count = color_attachments
                .get(0)
                .map(|at| view_guard[at.attachment].samples)
                .unwrap_or(1);
            assert!(
                sample_count & samples_count_limit != 0,
                "Attachment sample_count must be supported by physical device limits"
            );

            const MAX_TOTAL_ATTACHMENTS: usize = 10;
            type OutputAttachment<'a> = (
                &'a Stored<id::TextureId>,
                &'a hal::image::SubresourceRange,
                Option<TextureUsage>,
            );
            let mut output_attachments = ArrayVec::<[OutputAttachment; MAX_TOTAL_ATTACHMENTS]>::new();

            log::trace!(
                "Encoding render pass begin in command buffer {:?}",
                encoder_id
            );
            let rp_key = {
                let depth_stencil = match depth_stencil_attachment {
                    Some(at) => {
                        let view = trackers
                            .views
                            .use_extend(&*view_guard, at.attachment, (), ())
                            .unwrap();
                        if let Some(ex) = extent {
                            assert_eq!(ex, view.extent);
                        } else {
                            extent = Some(view.extent);
                        }
                        let source_id = match view.inner {
                            TextureViewInner::Native { ref source_id, .. } => source_id,
                            TextureViewInner::SwapChain { .. } => {
                                panic!("Unexpected depth/stencil use of swapchain image!")
                            }
                        };

                        // Using render pass for transition.
                        let consistent_usage = base_trackers.textures.query(
                            source_id.value,
                            view.range.clone(),
                        );
                        output_attachments.push((source_id, &view.range, consistent_usage));

                        let old_layout = match consistent_usage {
                            Some(usage) => conv::map_texture_state(
                                usage,
                                hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL,
                            ).1,
                            None => hal::image::Layout::DepthStencilAttachmentOptimal,
                        };

                        Some(hal::pass::Attachment {
                            format: Some(conv::map_texture_format(view.format, device.features)),
                            samples: view.samples,
                            ops: conv::map_load_store_ops(at.depth_load_op, at.depth_store_op),
                            stencil_ops: conv::map_load_store_ops(
                                at.stencil_load_op,
                                at.stencil_store_op,
                            ),
                            layouts: old_layout .. hal::image::Layout::DepthStencilAttachmentOptimal,
                        })
                    }
                    None => None,
                };

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
                        view.life_guard.add_ref(),
                        PhantomData,
                    ).is_ok();

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let consistent_usage = base_trackers.textures.query(
                                source_id.value,
                                view.range.clone(),
                            );
                            output_attachments.push((source_id, &view.range, consistent_usage));

                            let old_layout = match consistent_usage {
                                Some(usage) => conv::map_texture_state(usage, hal::format::Aspects::COLOR).1,
                                None => hal::image::Layout::ColorAttachmentOptimal,
                            };
                            old_layout .. hal::image::Layout::ColorAttachmentOptimal
                        }
                        TextureViewInner::SwapChain { ref source_id, .. } => {
                            if let Some((ref sc_id, _)) = cmb.used_swap_chain {
                                assert_eq!(source_id.value, sc_id.value);
                            } else {
                                assert!(used_swap_chain.is_none());
                                used_swap_chain = Some(source_id.clone());
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
                    .flat_map(|at| at.resolve_target)
                {
                    let view = &view_guard[resolve_target];
                    assert_eq!(extent, Some(view.extent));
                    assert_eq!(
                        view.samples, 1,
                        "All resolve_targets must have a sample_count of 1"
                    );
                    let first_use = trackers.views.init(
                        resolve_target,
                        view.life_guard.add_ref(),
                        PhantomData,
                    ).is_ok();

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let consistent_usage = base_trackers.textures.query(
                                source_id.value,
                                view.range.clone(),
                            );
                            output_attachments.push((source_id, &view.range, consistent_usage));

                            let old_layout = match consistent_usage {
                                Some(usage) => conv::map_texture_state(usage, hal::format::Aspects::COLOR).1,
                                None => hal::image::Layout::ColorAttachmentOptimal,
                            };
                            old_layout .. hal::image::Layout::ColorAttachmentOptimal
                        }
                        TextureViewInner::SwapChain { ref source_id, .. } => {
                            if let Some((ref sc_id, _)) = cmb.used_swap_chain {
                                assert_eq!(source_id.value, sc_id.value);
                            } else {
                                assert!(used_swap_chain.is_none());
                                used_swap_chain = Some(source_id.clone());
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

            for (source_id, view_range, consistent_usage) in output_attachments {
                let texture = &texture_guard[source_id.value];
                assert!(texture.usage.contains(TextureUsage::OUTPUT_ATTACHMENT));

                let usage = consistent_usage.unwrap_or(TextureUsage::OUTPUT_ATTACHMENT);
                // this is important to record the `first` state.
                let _ = trackers.textures.change_replace(
                    source_id.value,
                    &source_id.ref_count,
                    view_range.clone(),
                    usage,
                );
                if consistent_usage.is_some() {
                    // If we expect the texture to be transited to a new state by the
                    // render pass configuration, make the tracker aware of that.
                    let _ = trackers.textures.change_replace(
                        source_id.value,
                        &source_id.ref_count,
                        view_range.clone(),
                        TextureUsage::OUTPUT_ATTACHMENT,
                    );
                };
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

                    let mut resolve_ids = ArrayVec::<[_; MAX_COLOR_TARGETS]>::new();
                    let mut attachment_index = color_attachments.len();
                    if color_attachments
                        .iter()
                        .any(|at| at.resolve_target.is_some())
                    {
                        for (i, at) in color_attachments.iter().enumerate() {
                            if at.resolve_target.is_none() {
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
                    .filter_map(|at| at.resolve_target)
                    .cloned()
                    .collect(),
                depth_stencil: depth_stencil_attachment.map(|at| at.attachment),
            };

            let framebuffer = match used_swap_chain.take() {
                Some(sc_id) => {
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
                    cmb.used_swap_chain = Some((sc_id, framebuffer));
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
                raw.begin_render_pass(
                    render_pass,
                    framebuffer,
                    rect,
                    clear_values,
                    hal::command::SubpassContents::Inline,
                );
                raw.set_scissors(0, iter::once(&rect));
                raw.set_viewports(
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
                    .filter_map(|at| at.resolve_target)
                    .map(|resolve| view_guard[*resolve].format)
                    .collect(),
                depth_stencil: depth_stencil_attachment.map(|at| view_guard[at.attachment].format),
            };
            (context, sample_count)
        };

        let mut state = State {
            binder: Binder::new(cmb.features.max_bind_groups),
            blend_color: OptionalState::Unused,
            stencil_reference: OptionalState::Unused,
            index: IndexState {
                bound_buffer_view: None,
                format: IndexFormat::Uint16,
                limit: 0,
            },
            vertex: VertexState {
                inputs: [VertexBufferState::EMPTY; MAX_VERTEX_BUFFERS],
                vertex_limit: 0,
                instance_limit: 0,
            },
        };

        let mut peeker = raw_data.as_ptr();
        let raw_data_end = unsafe {
            raw_data.as_ptr().add(raw_data.len())
        };
        let mut command = RenderCommand::Draw {
            vertex_count: 0,
            instance_count: 0,
            first_vertex: 0,
            first_instance: 0,
        };
        loop {
            assert!(unsafe { peeker.add(RenderCommand::max_size()) } <= raw_data_end);
            peeker = unsafe { command.peek_from(peeker) };
            match command {
                RenderCommand::SetBindGroup { index, num_dynamic_offsets, bind_group_id, phantom_offsets } => {
                    let (new_peeker, offsets) = unsafe {
                        phantom_offsets.decode_unaligned(peeker, num_dynamic_offsets as usize, raw_data_end)
                    };
                    peeker = new_peeker;

                    if cfg!(debug_assertions) {
                        for off in offsets {
                            assert_eq!(
                                *off as BufferAddress % BIND_BUFFER_ALIGNMENT,
                                0,
                                "Misaligned dynamic buffer offset: {} does not align with {}",
                                off,
                                BIND_BUFFER_ALIGNMENT
                            );
                        }
                    }

                    let bind_group = trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .unwrap();
                    assert_eq!(bind_group.dynamic_count, offsets.len());

                    trackers.merge_extend(&bind_group.used);

                    if let Some((pipeline_layout_id, follow_ups)) = state.binder
                        .provide_entry(index as usize, bind_group_id, bind_group, offsets)
                    {
                        let bind_groups = iter::once(bind_group.raw.raw())
                            .chain(follow_ups.clone().map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()));
                        unsafe {
                            raw.bind_graphics_descriptor_sets(
                                &&pipeline_layout_guard[pipeline_layout_id].raw,
                                index as usize,
                                bind_groups,
                                offsets
                                    .iter()
                                    .chain(follow_ups.flat_map(|(_, offsets)| offsets))
                                    .map(|&off| off as hal::command::DescriptorSetOffset),
                            );
                        }
                    };
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let pipeline = &pipeline_guard[pipeline_id];

                    assert!(
                        context.compatible(&pipeline.pass_context),
                        "The render pipeline is not compatible with the pass!"
                    );
                    assert_eq!(
                        pipeline.sample_count, sample_count,
                        "The render pipeline and renderpass have mismatching sample_count"
                    );

                    state.blend_color
                        .require(pipeline.flags.contains(PipelineFlags::BLEND_COLOR));
                    state.stencil_reference
                        .require(pipeline.flags.contains(PipelineFlags::STENCIL_REFERENCE));

                    unsafe {
                        raw.bind_graphics_pipeline(&pipeline.raw);
                    }

                    // Rebind resource
                    if state.binder.pipeline_layout_id != Some(pipeline.layout_id) {
                        let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id];
                        state.binder.pipeline_layout_id = Some(pipeline.layout_id);
                        state.binder
                            .reset_expectations(pipeline_layout.bind_group_layout_ids.len());
                        let mut is_compatible = true;

                        for (index, (entry, &bgl_id)) in state
                            .binder
                            .entries
                            .iter_mut()
                            .zip(&pipeline_layout.bind_group_layout_ids)
                            .enumerate()
                        {
                            match entry.expect_layout(bgl_id) {
                                LayoutChange::Match(bg_id, offsets) if is_compatible => {
                                    let desc_set = bind_group_guard[bg_id].raw.raw();
                                    unsafe {
                                        raw.bind_graphics_descriptor_sets(
                                            &pipeline_layout.raw,
                                            index,
                                            iter::once(desc_set),
                                            offsets.iter().map(|offset| *offset as u32),
                                        );
                                    }
                                }
                                LayoutChange::Match(..) | LayoutChange::Unchanged => {}
                                LayoutChange::Mismatch => {
                                    is_compatible = false;
                                }
                            }
                        }
                    }

                    // Rebind index buffer if the index format has changed with the pipeline switch
                    if state.index.format != pipeline.index_format {
                        state.index.format = pipeline.index_format;
                        state.index.update_limit();

                        if let Some((buffer_id, ref range)) = state.index.bound_buffer_view {
                            let buffer = trackers
                                .buffers
                                .use_extend(&*buffer_guard, buffer_id, (), BufferUsage::INDEX)
                                .unwrap();

                            let view = hal::buffer::IndexBufferView {
                                buffer: &buffer.raw,
                                offset: range.start,
                                index_type: conv::map_index_format(state.index.format),
                            };

                            unsafe {
                                raw.bind_index_buffer(view);
                            }
                        }
                    }
                    // Update vertex buffer limits
                    for (vbs, &(stride, rate)) in state
                        .vertex
                        .inputs
                        .iter_mut()
                        .zip(&pipeline.vertex_strides)
                    {
                        vbs.stride = stride;
                        vbs.rate = rate;
                    }
                    for vbs in state.vertex.inputs[pipeline.vertex_strides.len() ..].iter_mut() {
                        vbs.stride = 0;
                        vbs.rate = InputStepMode::Vertex;
                    }
                    state.vertex.update_limits();
                }
                RenderCommand::SetIndexBuffer { buffer_id, offset } => {
                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUsage::INDEX)
                        .unwrap();
                    assert!(buffer.usage.contains(BufferUsage::INDEX));

                    let range = offset .. buffer.size;
                    state.index.bound_buffer_view = Some((buffer_id, range));
                    state.index.update_limit();

                    let view = hal::buffer::IndexBufferView {
                        buffer: &buffer.raw,
                        offset,
                        index_type: conv::map_index_format(state.index.format),
                    };

                    unsafe {
                        raw.bind_index_buffer(view);
                    }
                }
                RenderCommand::SetVertexBuffers { start_index, count, phantom_buffer_ids, phantom_offsets } => {
                    let (new_peeker, buffer_ids) = unsafe {
                        phantom_buffer_ids.decode_unaligned(peeker, count as usize, raw_data_end)
                    };
                    let (new_peeker, offsets) = unsafe {
                        phantom_offsets.decode_unaligned(new_peeker, count as usize, raw_data_end)
                    };
                    peeker = new_peeker;

                    let pairs = state.vertex.inputs[start_index as usize ..]
                        .iter_mut()
                        .zip(buffer_ids.iter().zip(offsets))
                        .map(|(vbs, (&id, &offset))| {
                            let buffer = trackers
                                .buffers
                                .use_extend(&*buffer_guard, id, (), BufferUsage::VERTEX)
                                .unwrap();
                            assert!(buffer.usage.contains(BufferUsage::VERTEX));

                            vbs.total_size = buffer.size - offset;
                            (&buffer.raw, offset)
                        });

                    unsafe {
                        raw.bind_vertex_buffers(start_index as u32, pairs);
                    }
                    state.vertex.update_limits();
                }
                RenderCommand::SetBlendColor(ref color) => {
                    state.blend_color = OptionalState::Set;
                    unsafe {
                        raw.set_blend_constants(conv::map_color_f32(color));
                    }
                }
                RenderCommand::SetStencilReference(value) => {
                    state.stencil_reference = OptionalState::Set;
                    unsafe {
                        raw.set_stencil_reference(hal::pso::Face::all(), value);
                    }
                }
                RenderCommand::SetViewport { ref rect, depth_min, depth_max } => {
                    use std::{convert::TryFrom, i16};
                    let r = hal::pso::Rect {
                        x: i16::try_from(rect.x.round() as i64).unwrap_or(0),
                        y: i16::try_from(rect.y.round() as i64).unwrap_or(0),
                        w: i16::try_from(rect.w.round() as i64).unwrap_or(i16::MAX),
                        h: i16::try_from(rect.h.round() as i64).unwrap_or(i16::MAX),
                    };
                    unsafe {
                        raw.set_viewports(
                            0,
                            iter::once(hal::pso::Viewport {
                                rect: r,
                                depth: depth_min .. depth_max,
                            }),
                        );
                    }
                }
                RenderCommand::SetScissor(ref rect) => {
                    use std::{convert::TryFrom, i16};
                    let r = hal::pso::Rect {
                        x: i16::try_from(rect.x).unwrap_or(0),
                        y: i16::try_from(rect.y).unwrap_or(0),
                        w: i16::try_from(rect.w).unwrap_or(i16::MAX),
                        h: i16::try_from(rect.h).unwrap_or(i16::MAX),
                    };
                    unsafe {
                        raw.set_scissors(
                            0,
                            iter::once(r),
                        );
                    }
                }
                RenderCommand::Draw { vertex_count, instance_count, first_vertex, first_instance } => {
                    state.is_ready().unwrap();
                    assert!(
                        first_vertex + vertex_count <= state.vertex.vertex_limit,
                        "Vertex out of range!"
                    );
                    assert!(
                        first_instance + instance_count <= state.vertex.instance_limit,
                        "Instance out of range!"
                    );

                    unsafe {
                        raw.draw(
                            first_vertex .. first_vertex + vertex_count,
                            first_instance .. first_instance + instance_count,
                        );
                    }
                }
                RenderCommand::DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance } => {
                    state.is_ready().unwrap();

                    //TODO: validate that base_vertex + max_index() is within the provided range
                    assert!(
                        first_index + index_count <= state.index.limit,
                        "Index out of range!"
                    );
                    assert!(
                        first_instance + instance_count <= state.vertex.instance_limit,
                        "Instance out of range!"
                    );

                    unsafe {
                        raw.draw_indexed(
                            first_index .. first_index + index_count,
                            base_vertex,
                            first_instance .. first_instance + instance_count,
                        );
                    }
                }
                RenderCommand::DrawIndirect { buffer_id, offset } => {
                    state.is_ready().unwrap();

                    let buffer = trackers
                        .buffers
                        .use_extend(
                            &*buffer_guard,
                            buffer_id,
                            (),
                            BufferUsage::INDIRECT,
                        )
                        .unwrap();
                    assert!(buffer.usage.contains(BufferUsage::INDIRECT));

                    unsafe {
                        raw.draw_indirect(&buffer.raw, offset, 1, 0);
                    }
                }
                RenderCommand::DrawIndexedIndirect { buffer_id, offset } => {
                    state.is_ready().unwrap();

                    let buffer = trackers
                        .buffers
                        .use_extend(
                            &*buffer_guard,
                            buffer_id,
                            (),
                            BufferUsage::INDIRECT,
                        )
                        .unwrap();
                    assert!(buffer.usage.contains(BufferUsage::INDIRECT));

                    unsafe {
                        raw.draw_indexed_indirect(&buffer.raw, offset, 1, 0);
                    }
                }
                RenderCommand::End => break,
            }
        }

        super::CommandBuffer::insert_barriers(
            cmb.raw.last_mut().unwrap(),
            &mut cmb.trackers,
            &trackers,
            &*buffer_guard,
            &*texture_guard,
        );
        unsafe {
            cmb.raw.last_mut().unwrap().finish();
            raw.end_render_pass();
        }
        cmb.raw.push(raw);
    }
}

pub mod render_ffi {
    use super::{
        RenderCommand,
        super::{PhantomSlice, RawRenderPass, Rect},
    };
    use crate::{
        id,
        BufferAddress,
        Color,
        DynamicOffset,
        RawString,
    };
    use std::{convert::TryInto, slice};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_bind_group(
        pass: &mut RawRenderPass,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        pass.raw.encode(&RenderCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
            phantom_offsets: PhantomSlice::new(),
        });
        pass.raw.encode_slice(
            slice::from_raw_parts(offsets, offset_length),
        );
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_pipeline(
        pass: &mut RawRenderPass,
        pipeline_id: id::RenderPipelineId,
    ) {
        pass.raw.encode(&RenderCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_index_buffer(
        pass: &mut RawRenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        pass.raw.encode(&RenderCommand::SetIndexBuffer {
            buffer_id,
            offset,
        });
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointers
    /// (`buffer_ids` and `offsets`) are valid for `length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_vertex_buffers(
        pass: &mut RawRenderPass,
        start_slot: u32,
        buffer_ids: *const id::BufferId,
        offsets: *const BufferAddress,
        length: usize,
    ) {
        pass.raw.encode(&RenderCommand::SetVertexBuffers {
            start_index: start_slot.try_into().unwrap(),
            count: length.try_into().unwrap(),
            phantom_buffer_ids: PhantomSlice::new(),
            phantom_offsets: PhantomSlice::new(),
        });
        pass.raw.encode_slice(
            slice::from_raw_parts(buffer_ids, length),
        );
        pass.raw.encode_slice(
            slice::from_raw_parts(offsets, length),
        );
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_blend_color(
        pass: &mut RawRenderPass,
        color: &Color,
    ) {
        pass.raw.encode(&RenderCommand::SetBlendColor(*color));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_stencil_reference(
        pass: &mut RawRenderPass,
        value: u32,
    ) {
        pass.raw.encode(&RenderCommand::SetStencilReference(value));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_viewport(
        pass: &mut RawRenderPass,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) {
        pass.raw.encode(&RenderCommand::SetViewport {
            rect: Rect { x, y, w, h },
            depth_min,
            depth_max,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_scissor_rect(
        pass: &mut RawRenderPass,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) {
        pass.raw.encode(&RenderCommand::SetScissor(Rect { x, y, w, h }));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_draw(
        pass: &mut RawRenderPass,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        pass.raw.encode(&RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_draw_indexed(
        pass: &mut RawRenderPass,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        pass.raw.encode(&RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_draw_indirect(
        pass: &mut RawRenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        pass.raw.encode(&RenderCommand::DrawIndirect {
            buffer_id,
            offset,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_draw_indexed_indirect(
        pass: &mut RawRenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        pass.raw.encode(&RenderCommand::DrawIndexedIndirect {
            buffer_id,
            offset,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_execute_bundles(
        _pass: &mut RawRenderPass,
        _bundles: *const id::RenderBundleId,
        _bundles_length: usize,
    ) {
        unimplemented!()
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_push_debug_group(
        _pass: &mut RawRenderPass,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_pop_debug_group(
        _pass: &mut RawRenderPass,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_insert_debug_marker(
        _pass: &mut RawRenderPass,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_finish(
        pass: &mut RawRenderPass,
        length: &mut usize,
    ) -> *const u8 {
        //TODO: put target information into the byte stream
        pass.raw.finish(RenderCommand::End);
        *length = pass.raw.size();
        pass.raw.base
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_destroy(pass: *mut RawRenderPass) {
        let _ = Box::from_raw(pass).raw.into_vec();
    }
}
