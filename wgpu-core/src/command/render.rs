/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::BindError,
    command::{
        bind::{Binder, LayoutChange},
        check_buffer_usage, check_texture_usage, BasePass, BasePassRef, RenderCommandError,
    },
    conv,
    device::{
        AttachmentData, FramebufferKey, RenderPassContext, RenderPassKey, MAX_COLOR_TARGETS,
        MAX_VERTEX_BUFFERS,
    },
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
    pipeline::PipelineFlags,
    resource::{BufferUse, TextureUse, TextureViewInner},
    span,
    track::TrackerSet,
    MissingBufferUsageError, MissingTextureUsageError, Stored,
};

use arrayvec::ArrayVec;
use hal::command::CommandBuffer as _;
use wgt::{
    BufferAddress, BufferSize, BufferUsage, Color, IndexFormat, InputStepMode, TextureUsage,
};

#[cfg(any(feature = "serial-pass", feature = "replay"))]
use serde::Deserialize;
#[cfg(any(feature = "serial-pass", feature = "trace"))]
use serde::Serialize;

use std::{borrow::Borrow, collections::hash_map::Entry, fmt, iter, ops::Range, str};

/// Operation to perform to the output attachment at the start of a renderpass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub enum LoadOp {
    /// Clear the output attachment with the clear color. Clearing is faster than loading.
    Clear = 0,
    /// Do not clear output attachment.
    Load = 1,
}

/// Operation to perform to the output attachment at the end of a renderpass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub enum StoreOp {
    /// Clear the render target. If you don't care about the contents of the target, this can be faster.
    Clear = 0,
    /// Store the result of the renderpass.
    Store = 1,
}

/// Describes an individual channel within a render pass, such as color, depth, or stencil.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub struct PassChannel<V> {
    /// Operation to perform to the output attachment at the start of a renderpass. This must be clear if it
    /// is the first renderpass rendering to a swap chain image.
    pub load_op: LoadOp,
    /// Operation to perform to the output attachment at the end of a renderpass.
    pub store_op: StoreOp,
    /// If load_op is [`LoadOp::Clear`], the attachement will be cleared to this color.
    pub clear_value: V,
    /// If true, the relevant channel is not changed by a renderpass, and the corresponding attachment
    /// can be used inside the pass by other read-only usages.
    pub read_only: bool,
}

/// Describes a color attachment to a render pass.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub struct ColorAttachmentDescriptor {
    /// The view to use as an attachment.
    pub attachment: id::TextureViewId,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<id::TextureViewId>,
    /// What operations will be performed on this color attachment.
    pub channel: PassChannel<Color>,
}

/// Describes a depth/stencil attachment to a render pass.
#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub struct DepthStencilAttachmentDescriptor {
    /// The view to use as an attachment.
    pub attachment: id::TextureViewId,
    /// What operations will be performed on the depth part of the attachment.
    pub depth: PassChannel<f32>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil: PassChannel<u32>,
}

fn is_depth_stencil_read_only(
    desc: &DepthStencilAttachmentDescriptor,
    aspects: hal::format::Aspects,
) -> Result<bool, RenderPassError> {
    if aspects.contains(hal::format::Aspects::DEPTH) && !desc.depth.read_only {
        return Ok(false);
    }
    if (desc.depth.load_op, desc.depth.store_op) != (LoadOp::Load, StoreOp::Store) {
        return Err(RenderPassError::InvalidDepthOps);
    }
    if aspects.contains(hal::format::Aspects::STENCIL) && !desc.stencil.read_only {
        return Ok(false);
    }
    if (desc.stencil.load_op, desc.stencil.store_op) != (LoadOp::Load, StoreOp::Store) {
        return Err(RenderPassError::InvalidStencilOps);
    }
    Ok(true)
}

pub type RenderPassDescriptor<'a> =
    wgt::RenderPassDescriptor<'a, ColorAttachmentDescriptor, &'a DepthStencilAttachmentDescriptor>;

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(any(feature = "serial-pass", feature = "trace"), derive(Serialize))]
#[cfg_attr(any(feature = "serial-pass", feature = "replay"), derive(Deserialize))]
pub enum RenderCommand {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::RenderPipelineId),
    SetIndexBuffer {
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
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
    SetPushConstant {
        stages: wgt::ShaderStage,
        offset: u32,
        size_bytes: u32,
        /// None means there is no data and the data should be an array of zeros.
        ///
        /// Facilitates clears in renderbundles which explicitly do their clears.
        values_offset: Option<u32>,
    },
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
    MultiDrawIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
        /// Count of `None` represents a non-multi call.
        count: Option<u32>,
        indexed: bool,
    },
    MultiDrawIndirectCount {
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
        indexed: bool,
    },
    PushDebugGroup {
        color: u32,
        len: usize,
    },
    PopDebugGroup,
    InsertDebugMarker {
        color: u32,
        len: usize,
    },
    ExecuteBundle(id::RenderBundleId),
}

#[cfg_attr(feature = "serial-pass", derive(Deserialize, Serialize))]
pub struct RenderPass {
    base: BasePass<RenderCommand>,
    parent_id: id::CommandEncoderId,
    color_targets: ArrayVec<[ColorAttachmentDescriptor; MAX_COLOR_TARGETS]>,
    depth_stencil_target: Option<DepthStencilAttachmentDescriptor>,
}

impl RenderPass {
    pub fn new(parent_id: id::CommandEncoderId, desc: RenderPassDescriptor) -> Self {
        RenderPass {
            base: BasePass::new(),
            parent_id,
            color_targets: desc.color_attachments.iter().cloned().collect(),
            depth_stencil_target: desc.depth_stencil_attachment.cloned(),
        }
    }

    pub fn parent_id(&self) -> id::CommandEncoderId {
        self.parent_id
    }
}

impl fmt::Debug for RenderPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RenderPass {{ encoder_id: {:?}, color_targets: {:?}, depth_stencil_target: {:?}, data: {:?} commands, {:?} dynamic offsets, and {:?} push constant u32s }}",
            self.parent_id,
            self.color_targets,
            self.depth_stencil_target,
            self.base.commands.len(),
            self.base.dynamic_offsets.len(),
            self.base.push_constant_data.len(),
        )
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

#[derive(Clone, Debug, PartialEq)]
pub enum DrawError {
    MissingBlendColor,
    MissingStencilReference,
    MissingPipeline,
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawError::MissingBlendColor => write!(f, "blend color needs to be set"),
            DrawError::MissingStencilReference => write!(f, "stencil reference needs to be set"),
            DrawError::MissingPipeline => write!(f, "render pipeline must be set"),
            DrawError::IncompatibleBindGroup { index } => write!(f, "current render pipeline has a layout which is incompatible with a currently set bind group, first differing at entry index {}", index),
        }
    }
}

#[derive(Debug, Default)]
struct IndexState {
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

    fn reset(&mut self) {
        self.bound_buffer_view = None;
        self.limit = 0;
    }
}

#[derive(Clone, Copy, Debug)]
struct VertexBufferState {
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

#[derive(Debug, Default)]
struct VertexState {
    inputs: ArrayVec<[VertexBufferState; MAX_VERTEX_BUFFERS]>,
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

    fn reset(&mut self) {
        self.inputs.clear();
        self.vertex_limit = 0;
        self.instance_limit = 0;
    }
}

#[derive(Debug)]
struct State {
    binder: Binder,
    blend_color: OptionalState,
    stencil_reference: OptionalState,
    pipeline: OptionalState,
    index: IndexState,
    vertex: VertexState,
    debug_scope_depth: u32,
}

impl State {
    fn is_ready(&self) -> Result<(), DrawError> {
        //TODO: vertex buffers
        let bind_mask = self.binder.invalid_mask();
        if bind_mask != 0 {
            //let (expected, provided) = self.binder.entries[index as usize].info();
            return Err(DrawError::IncompatibleBindGroup {
                index: bind_mask.trailing_zeros(),
            });
        }
        if self.pipeline == OptionalState::Required {
            return Err(DrawError::MissingPipeline);
        }
        if self.blend_color == OptionalState::Required {
            return Err(DrawError::MissingBlendColor);
        }
        if self.stencil_reference == OptionalState::Required {
            return Err(DrawError::MissingStencilReference);
        }
        Ok(())
    }

    /// Reset the `RenderBundle`-related states.
    fn reset_bundle(&mut self) {
        self.binder.reset();
        self.pipeline = OptionalState::Required;
        self.index.reset();
        self.vertex.reset();
    }
}

/// Error encountered when performing a render pass.
#[derive(Clone, Debug)]
pub enum RenderPassError {
    InvalidSampleCount(u8),
    InvalidResolveSourceSampleCount,
    InvalidResolveTargetSampleCount,
    ExtentStateMismatch {
        state_extent: hal::image::Extent,
        view_extent: hal::image::Extent,
    },
    SwapChainImageAsDepthStencil,
    InvalidDepthOps,
    InvalidStencilOps,
    SampleCountMismatch {
        actual: u8,
        expected: u8,
    },
    SwapChainMismatch,
    InvalidValuesOffset,
    MissingDeviceFeatures(wgt::Features),
    IndirectBufferOverrun {
        offset: u64,
        count: Option<u32>,
        begin_offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    IndirectCountBufferOverrun {
        begin_count_offset: u64,
        end_count_offset: u64,
        count_buffer_size: u64,
    },
    InvalidPopDebugGroup,
    IncompatibleRenderBundle,
    RenderCommand(RenderCommandError),
    Draw(DrawError),
    Bind(BindError),
}

impl From<RenderCommandError> for RenderPassError {
    fn from(error: RenderCommandError) -> Self {
        Self::RenderCommand(error)
    }
}

impl From<MissingBufferUsageError> for RenderPassError {
    fn from(error: MissingBufferUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

impl From<MissingTextureUsageError> for RenderPassError {
    fn from(error: MissingTextureUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

impl From<DrawError> for RenderPassError {
    fn from(error: DrawError) -> Self {
        Self::Draw(error)
    }
}

impl From<BindError> for RenderPassError {
    fn from(error: BindError) -> Self {
        Self::Bind(error)
    }
}

impl fmt::Display for RenderPassError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidSampleCount(count) => write!(
                f,
                "attachment's sample count {} is invalid",
                count,
            ),
            Self::InvalidResolveSourceSampleCount => write!(f, "attachment with resolve target must be multi-sampled"),
            Self::InvalidResolveTargetSampleCount => write!(f, "resolve target must have a sample count of 1"),
            Self::ExtentStateMismatch { state_extent, view_extent } => write!(
                f,
                "extent state {:?} must match extent from view {:?}",
                state_extent,
                view_extent,
            ),
            Self::SwapChainImageAsDepthStencil => write!(f, "attempted to use a swap chain image as a depth/stencil attachment"),
            Self::InvalidDepthOps => write!(f, "unable to clear non-present/read-only depth"),
            Self::InvalidStencilOps => write!(f, "unable to clear non-present/read-only stencil"),
            Self::SampleCountMismatch { actual, expected } => write!(
                f,
                "all attachments must have the same sample count, found {} != {}",
                actual,
                expected,
            ),
            Self::SwapChainMismatch => write!(f, "texture view's swap chain must match swap chain in use"),
            Self::InvalidValuesOffset => write!(f, "setting `values_offset` to be `None` is only for internal use in render bundles"),
            Self::MissingDeviceFeatures(expected) => write!(
                f,
                "required device features not enabled: {:?}",
                expected,
            ),
            Self::IndirectBufferOverrun { offset, count, begin_offset, end_offset, buffer_size } => write!(
                f,
                "indirect draw with offset {}{} uses bytes {}..{} which overruns indirect buffer of size {}",
                offset,
                count.map_or_else(String::new, |v| format!(" and count {}", v)),
                begin_offset,
                end_offset,
                buffer_size,
            ),
            Self::IndirectCountBufferOverrun { begin_count_offset, end_count_offset, count_buffer_size } => write!(
                f,
                "indirect draw uses bytes {}..{} which overruns indirect buffer of size {}",
                begin_count_offset,
                end_count_offset,
                count_buffer_size,
            ),
            Self::InvalidPopDebugGroup => write!(f, "cannot pop debug group, because number of pushed debug groups is zero"),
            Self::IncompatibleRenderBundle => write!(f, "render bundle output formats do not match render pass attachment formats"),
            Self::RenderCommand(error) => write!(f, "{}", error),
            Self::Draw(error) => write!(f, "{}", error),
            Self::Bind(error) => write!(f, "{}", error),
        }
    }
}

fn check_device_features(
    actual: wgt::Features,
    expected: wgt::Features,
) -> Result<(), RenderPassError> {
    if !actual.contains(expected) {
        Err(RenderPassError::MissingDeviceFeatures(expected))
    } else {
        Ok(())
    }
}

// Common routines between render/compute

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_run_render_pass<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        pass: &RenderPass,
    ) -> Result<(), RenderPassError> {
        self.command_encoder_run_render_pass_impl::<B>(
            encoder_id,
            pass.base.as_ref(),
            &pass.color_targets,
            pass.depth_stencil_target.as_ref(),
        )
    }

    #[doc(hidden)]
    pub fn command_encoder_run_render_pass_impl<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        mut base: BasePassRef<RenderCommand>,
        color_attachments: &[ColorAttachmentDescriptor],
        depth_stencil_attachment: Option<&DepthStencilAttachmentDescriptor>,
    ) -> Result<(), RenderPassError> {
        span!(_guard, INFO, "CommandEncoder::run_render_pass");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);

        let mut trackers = TrackerSet::new(B::VARIANT);
        let cmb = &mut cmb_guard[encoder_id];
        let device = &device_guard[cmb.device_id.value];
        let mut raw = device.com_allocator.extend(cmb);

        #[cfg(feature = "trace")]
        match cmb.commands {
            Some(ref mut list) => {
                list.push(crate::device::trace::Command::RunRenderPass {
                    base: BasePass::from_ref(base),
                    target_colors: color_attachments.iter().cloned().collect(),
                    target_depth_stencil: depth_stencil_attachment.cloned(),
                });
            }
            None => {}
        }

        unsafe {
            raw.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        }

        let (bundle_guard, mut token) = hub.render_bundles.read(&mut token);
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.render_pipelines.read(&mut token);
        let (buffer_guard, mut token) = hub.buffers.read(&mut token);
        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let (view_guard, _) = hub.texture_views.read(&mut token);

        // We default to false intentionally, even if depth-stencil isn't used at all.
        // This allows us to use the primary raw pipeline in `RenderPipeline`,
        // instead of the special read-only one, which would be `None`.
        let mut is_ds_read_only = false;

        struct OutputAttachment<'a> {
            texture_id: &'a Stored<id::TextureId>,
            range: &'a hal::image::SubresourceRange,
            previous_use: Option<TextureUse>,
            new_use: TextureUse,
        }
        const MAX_TOTAL_ATTACHMENTS: usize = 2 * MAX_COLOR_TARGETS + 1;
        let mut output_attachments = ArrayVec::<[OutputAttachment; MAX_TOTAL_ATTACHMENTS]>::new();

        let context = {
            use hal::device::Device as _;

            let sample_count_limit = device.hal_limits.framebuffer_color_sample_counts;
            let base_trackers = &cmb.trackers;

            let mut extent = None;
            let mut used_swap_chain = None::<Stored<id::SwapChainId>>;

            let sample_count = color_attachments
                .get(0)
                .map(|at| view_guard[at.attachment].samples)
                .unwrap_or(1);
            if sample_count & sample_count_limit == 0 {
                return Err(RenderPassError::InvalidSampleCount(sample_count));
            }

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
                            if ex != view.extent {
                                return Err(RenderPassError::ExtentStateMismatch {
                                    state_extent: ex,
                                    view_extent: view.extent,
                                });
                            }
                        } else {
                            extent = Some(view.extent);
                        }
                        let source_id = match view.inner {
                            TextureViewInner::Native { ref source_id, .. } => source_id,
                            TextureViewInner::SwapChain { .. } => {
                                return Err(RenderPassError::SwapChainImageAsDepthStencil)
                            }
                        };

                        // Using render pass for transition.
                        let previous_use = base_trackers
                            .textures
                            .query(source_id.value, view.range.clone());
                        let new_use = if is_depth_stencil_read_only(at, view.range.aspects)? {
                            is_ds_read_only = true;
                            TextureUse::ATTACHMENT_READ
                        } else {
                            TextureUse::ATTACHMENT_WRITE
                        };
                        output_attachments.push(OutputAttachment {
                            texture_id: source_id,
                            range: &view.range,
                            previous_use,
                            new_use,
                        });

                        let new_layout = conv::map_texture_state(new_use, view.range.aspects).1;
                        let old_layout = match previous_use {
                            Some(usage) => conv::map_texture_state(usage, view.range.aspects).1,
                            None => new_layout,
                        };

                        let ds_at = hal::pass::Attachment {
                            format: Some(conv::map_texture_format(
                                view.format,
                                device.private_features,
                            )),
                            samples: view.samples,
                            ops: conv::map_load_store_ops(&at.depth),
                            stencil_ops: conv::map_load_store_ops(&at.stencil),
                            layouts: old_layout..new_layout,
                        };
                        Some((ds_at, new_layout))
                    }
                    None => None,
                };

                let mut colors = ArrayVec::new();
                let mut resolves = ArrayVec::new();

                for at in color_attachments {
                    let view = trackers
                        .views
                        .use_extend(&*view_guard, at.attachment, (), ())
                        .unwrap();
                    if let Some(ex) = extent {
                        if ex != view.extent {
                            return Err(RenderPassError::ExtentStateMismatch {
                                state_extent: ex,
                                view_extent: view.extent,
                            });
                        }
                    } else {
                        extent = Some(view.extent);
                    }
                    if view.samples != sample_count {
                        return Err(RenderPassError::SampleCountMismatch {
                            actual: view.samples,
                            expected: sample_count,
                        });
                    }

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let previous_use = base_trackers
                                .textures
                                .query(source_id.value, view.range.clone());
                            let new_use = TextureUse::ATTACHMENT_WRITE;
                            output_attachments.push(OutputAttachment {
                                texture_id: source_id,
                                range: &view.range,
                                previous_use,
                                new_use,
                            });

                            let new_layout =
                                conv::map_texture_state(new_use, hal::format::Aspects::COLOR).1;
                            let old_layout = match previous_use {
                                Some(usage) => {
                                    conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                                }
                                None => new_layout,
                            };
                            old_layout..new_layout
                        }
                        TextureViewInner::SwapChain { ref source_id, .. } => {
                            if let Some((ref sc_id, _)) = cmb.used_swap_chain {
                                if source_id.value != sc_id.value {
                                    return Err(RenderPassError::SwapChainMismatch);
                                }
                            } else {
                                assert!(used_swap_chain.is_none());
                                used_swap_chain = Some(source_id.clone());
                            }

                            let end = hal::image::Layout::Present;
                            let start = match at.channel.load_op {
                                LoadOp::Clear => hal::image::Layout::Undefined,
                                LoadOp::Load => end,
                            };
                            start..end
                        }
                    };

                    let color_at = hal::pass::Attachment {
                        format: Some(conv::map_texture_format(
                            view.format,
                            device.private_features,
                        )),
                        samples: view.samples,
                        ops: conv::map_load_store_ops(&at.channel),
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        layouts,
                    };
                    colors.push((color_at, hal::image::Layout::ColorAttachmentOptimal));
                }

                for resolve_target in color_attachments.iter().flat_map(|at| at.resolve_target) {
                    let view = trackers
                        .views
                        .use_extend(&*view_guard, resolve_target, (), ())
                        .unwrap();
                    if extent != Some(view.extent) {
                        return Err(RenderPassError::ExtentStateMismatch {
                            state_extent: extent.unwrap_or_default(),
                            view_extent: view.extent,
                        });
                    }
                    if view.samples != 1 {
                        return Err(RenderPassError::InvalidResolveTargetSampleCount);
                    }

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let previous_use = base_trackers
                                .textures
                                .query(source_id.value, view.range.clone());
                            let new_use = TextureUse::ATTACHMENT_WRITE;
                            output_attachments.push(OutputAttachment {
                                texture_id: source_id,
                                range: &view.range,
                                previous_use,
                                new_use,
                            });

                            let new_layout =
                                conv::map_texture_state(new_use, hal::format::Aspects::COLOR).1;
                            let old_layout = match previous_use {
                                Some(usage) => {
                                    conv::map_texture_state(usage, hal::format::Aspects::COLOR).1
                                }
                                None => new_layout,
                            };
                            old_layout..new_layout
                        }
                        TextureViewInner::SwapChain { ref source_id, .. } => {
                            if let Some((ref sc_id, _)) = cmb.used_swap_chain {
                                if source_id.value != sc_id.value {
                                    return Err(RenderPassError::SwapChainMismatch);
                                }
                            } else {
                                assert!(used_swap_chain.is_none());
                                used_swap_chain = Some(source_id.clone());
                            }
                            hal::image::Layout::Undefined..hal::image::Layout::Present
                        }
                    };

                    let resolve_at = hal::pass::Attachment {
                        format: Some(conv::map_texture_format(
                            view.format,
                            device.private_features,
                        )),
                        samples: view.samples,
                        ops: hal::pass::AttachmentOps::new(
                            hal::pass::AttachmentLoadOp::DontCare,
                            hal::pass::AttachmentStoreOp::Store,
                        ),
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        layouts,
                    };
                    resolves.push((resolve_at, hal::image::Layout::ColorAttachmentOptimal));
                }

                RenderPassKey {
                    colors,
                    resolves,
                    depth_stencil,
                }
            };

            let mut render_pass_cache = device.render_passes.lock();
            let render_pass = match render_pass_cache.entry(rp_key.clone()) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(entry) => {
                    let color_ids: [hal::pass::AttachmentRef; MAX_COLOR_TARGETS] = [
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
                        for ((i, at), &(_, layout)) in color_attachments
                            .iter()
                            .enumerate()
                            .zip(entry.key().resolves.iter())
                        {
                            let real_attachment_index = match at.resolve_target {
                                Some(resolve_attachment) => {
                                    let attachment_sample_count = view_guard[at.attachment].samples;
                                    if attachment_sample_count == 1 {
                                        return Err(
                                            RenderPassError::InvalidResolveSourceSampleCount,
                                        );
                                    }
                                    if view_guard[resolve_attachment].samples != 1 {
                                        return Err(
                                            RenderPassError::InvalidResolveTargetSampleCount,
                                        );
                                    }
                                    attachment_index + i
                                }
                                None => hal::pass::ATTACHMENT_UNUSED,
                            };
                            resolve_ids.push((real_attachment_index, layout));
                        }
                        attachment_index += color_attachments.len();
                    }

                    let depth_id = depth_stencil_attachment.map(|at| {
                        let aspects = view_guard[at.attachment].range.aspects;
                        let usage = if is_ds_read_only {
                            TextureUse::ATTACHMENT_READ
                        } else {
                            TextureUse::ATTACHMENT_WRITE
                        };
                        (attachment_index, conv::map_texture_state(usage, aspects).1)
                    });

                    let subpass = hal::pass::SubpassDesc {
                        colors: &color_ids[..color_attachments.len()],
                        resolves: &resolve_ids,
                        depth_stencil: depth_id.as_ref(),
                        inputs: &[],
                        preserves: &[],
                    };
                    let all = entry.key().all().map(|(at, _)| at);

                    let pass =
                        unsafe { device.raw.create_render_pass(all, iter::once(subpass), &[]) }
                            .unwrap();
                    entry.insert(pass)
                }
            };

            let mut framebuffer_cache;
            let fb_key = FramebufferKey {
                colors: color_attachments.iter().map(|at| at.attachment).collect(),
                resolves: color_attachments
                    .iter()
                    .filter_map(|at| at.resolve_target)
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
                .flat_map(|(at, (rat, _layout))| {
                    match at.channel.load_op {
                        LoadOp::Load => None,
                        LoadOp::Clear => {
                            use hal::format::ChannelType;
                            //TODO: validate sign/unsign and normalized ranges of the color values
                            let value = match rat.format.unwrap().base_format().1 {
                                ChannelType::Unorm
                                | ChannelType::Snorm
                                | ChannelType::Ufloat
                                | ChannelType::Sfloat
                                | ChannelType::Uscaled
                                | ChannelType::Sscaled
                                | ChannelType::Srgb => hal::command::ClearColor {
                                    float32: conv::map_color_f32(&at.channel.clear_value),
                                },
                                ChannelType::Sint => hal::command::ClearColor {
                                    sint32: conv::map_color_i32(&at.channel.clear_value),
                                },
                                ChannelType::Uint => hal::command::ClearColor {
                                    uint32: conv::map_color_u32(&at.channel.clear_value),
                                },
                            };
                            Some(hal::command::ClearValue { color: value })
                        }
                    }
                })
                .chain(depth_stencil_attachment.and_then(|at| {
                    match (at.depth.load_op, at.stencil.load_op) {
                        (LoadOp::Load, LoadOp::Load) => None,
                        (LoadOp::Clear, _) | (_, LoadOp::Clear) => {
                            let value = hal::command::ClearDepthStencil {
                                depth: at.depth.clear_value,
                                stencil: at.stencil.clear_value,
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
                        depth: 0.0..1.0,
                    }),
                );
            }

            RenderPassContext {
                attachments: AttachmentData {
                    colors: color_attachments
                        .iter()
                        .map(|at| view_guard[at.attachment].format)
                        .collect(),
                    resolves: color_attachments
                        .iter()
                        .filter_map(|at| at.resolve_target)
                        .map(|resolve| view_guard[resolve].format)
                        .collect(),
                    depth_stencil: depth_stencil_attachment
                        .map(|at| view_guard[at.attachment].format),
                },
                sample_count,
            }
        };

        let mut state = State {
            binder: Binder::new(cmb.limits.max_bind_groups),
            blend_color: OptionalState::Unused,
            stencil_reference: OptionalState::Unused,
            pipeline: OptionalState::Required,
            index: IndexState::default(),
            vertex: VertexState::default(),
            debug_scope_depth: 0,
        };

        for command in base.commands {
            match *command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let max_bind_groups = device.limits.max_bind_groups;
                    if (index as u32) >= max_bind_groups {
                        return Err(RenderCommandError::BindGroupIndexOutOfRange {
                            index,
                            max: max_bind_groups,
                        }
                        .into());
                    }

                    let offsets = &base.dynamic_offsets[..num_dynamic_offsets as usize];
                    base.dynamic_offsets = &base.dynamic_offsets[num_dynamic_offsets as usize..];

                    let bind_group = trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .unwrap();
                    bind_group
                        .validate_dynamic_bindings(offsets)
                        .map_err(RenderPassError::from)?;

                    trackers.merge_extend(&bind_group.used);

                    if let Some((pipeline_layout_id, follow_ups)) = state.binder.provide_entry(
                        index as usize,
                        bind_group_id,
                        bind_group,
                        offsets,
                    ) {
                        let bind_groups = iter::once(bind_group.raw.raw()).chain(
                            follow_ups
                                .clone()
                                .map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()),
                        );
                        unsafe {
                            raw.bind_graphics_descriptor_sets(
                                &pipeline_layout_guard[pipeline_layout_id].raw,
                                index as usize,
                                bind_groups,
                                offsets
                                    .iter()
                                    .chain(follow_ups.flat_map(|(_, offsets)| offsets))
                                    .cloned(),
                            );
                        }
                    };
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    state.pipeline = OptionalState::Set;
                    let pipeline = trackers
                        .render_pipes
                        .use_extend(&*pipeline_guard, pipeline_id, (), ())
                        .unwrap();

                    if !context.compatible(&pipeline.pass_context) {
                        return Err(RenderCommandError::IncompatiblePipeline.into());
                    }
                    if is_ds_read_only
                        && pipeline
                            .flags
                            .contains(PipelineFlags::DEPTH_STENCIL_READ_ONLY)
                    {
                        return Err(RenderCommandError::IncompatibleReadOnlyDepthStencil.into());
                    }

                    state
                        .blend_color
                        .require(pipeline.flags.contains(PipelineFlags::BLEND_COLOR));
                    state
                        .stencil_reference
                        .require(pipeline.flags.contains(PipelineFlags::STENCIL_REFERENCE));

                    unsafe {
                        raw.bind_graphics_pipeline(&pipeline.raw);
                    }

                    // Rebind resource
                    if state.binder.pipeline_layout_id != Some(pipeline.layout_id.value) {
                        let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id.value];
                        state.binder.pipeline_layout_id = Some(pipeline.layout_id.value);
                        state
                            .binder
                            .reset_expectations(pipeline_layout.bind_group_layout_ids.len());
                        let mut is_compatible = true;

                        for (index, (entry, bgl_id)) in state
                            .binder
                            .entries
                            .iter_mut()
                            .zip(&pipeline_layout.bind_group_layout_ids)
                            .enumerate()
                        {
                            match entry.expect_layout(bgl_id.value) {
                                LayoutChange::Match(bg_id, offsets) if is_compatible => {
                                    let desc_set = bind_group_guard[bg_id].raw.raw();
                                    unsafe {
                                        raw.bind_graphics_descriptor_sets(
                                            &pipeline_layout.raw,
                                            index,
                                            iter::once(desc_set),
                                            offsets.iter().cloned(),
                                        );
                                    }
                                }
                                LayoutChange::Match(..) | LayoutChange::Unchanged => {}
                                LayoutChange::Mismatch => {
                                    is_compatible = false;
                                }
                            }
                        }

                        // Clear push constant ranges
                        let non_overlapping = super::bind::compute_nonoverlapping_ranges(
                            &pipeline_layout.push_constant_ranges,
                        );
                        for range in non_overlapping {
                            let offset = range.range.start;
                            let size_bytes = range.range.end - offset;
                            super::push_constant_clear(
                                offset,
                                size_bytes,
                                |clear_offset, clear_data| unsafe {
                                    raw.push_graphics_constants(
                                        &pipeline_layout.raw,
                                        conv::map_shader_stage_flags(range.stages),
                                        clear_offset,
                                        clear_data,
                                    );
                                },
                            );
                        }
                    }

                    // Rebind index buffer if the index format has changed with the pipeline switch
                    if state.index.format != pipeline.index_format {
                        state.index.format = pipeline.index_format;
                        state.index.update_limit();

                        if let Some((buffer_id, ref range)) = state.index.bound_buffer_view {
                            let buffer = trackers
                                .buffers
                                .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                                .unwrap();

                            let view = hal::buffer::IndexBufferView {
                                buffer: &buffer.raw,
                                range: hal::buffer::SubRange {
                                    offset: range.start,
                                    size: Some(range.end - range.start),
                                },
                                index_type: conv::map_index_format(state.index.format),
                            };

                            unsafe {
                                raw.bind_index_buffer(view);
                            }
                        }
                    }
                    // Update vertex buffer limits
                    for (vbs, &(stride, rate)) in
                        state.vertex.inputs.iter_mut().zip(&pipeline.vertex_strides)
                    {
                        vbs.stride = stride;
                        vbs.rate = rate;
                    }
                    let vertex_strides_len = pipeline.vertex_strides.len();
                    for vbs in state.vertex.inputs.iter_mut().skip(vertex_strides_len) {
                        vbs.stride = 0;
                        vbs.rate = InputStepMode::Vertex;
                    }
                    state.vertex.update_limits();
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    offset,
                    size,
                } => {
                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                        .unwrap();
                    check_buffer_usage(buffer.usage, BufferUsage::INDEX)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    state.index.bound_buffer_view = Some((buffer_id, offset..end));
                    state.index.update_limit();

                    let view = hal::buffer::IndexBufferView {
                        buffer: &buffer.raw,
                        range: hal::buffer::SubRange {
                            offset,
                            size: Some(end - offset),
                        },
                        index_type: conv::map_index_format(state.index.format),
                    };

                    unsafe {
                        raw.bind_index_buffer(view);
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::VERTEX)
                        .unwrap();
                    check_buffer_usage(buffer.usage, BufferUsage::VERTEX)?;
                    let empty_slots = (1 + slot as usize).saturating_sub(state.vertex.inputs.len());
                    state
                        .vertex
                        .inputs
                        .extend(iter::repeat(VertexBufferState::EMPTY).take(empty_slots));
                    state.vertex.inputs[slot as usize].total_size = match size {
                        Some(s) => s.get(),
                        None => buffer.size - offset,
                    };

                    let range = hal::buffer::SubRange {
                        offset,
                        size: size.map(|s| s.get()),
                    };
                    unsafe {
                        raw.bind_vertex_buffers(slot, iter::once((&buffer.raw, range)));
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
                RenderCommand::SetViewport {
                    ref rect,
                    depth_min,
                    depth_max,
                } => {
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
                                depth: depth_min..depth_max,
                            }),
                        );
                    }
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let values_offset =
                        values_offset.ok_or(RenderPassError::InvalidValuesOffset)?;

                    let end_offset_bytes = offset + size_bytes;
                    let values_end_offset = (values_offset + size_bytes / 4) as usize;
                    let data_slice =
                        &base.push_constant_data[(values_offset as usize)..values_end_offset];

                    let pipeline_layout_id = state
                        .binder
                        .pipeline_layout_id
                        .ok_or(RenderCommandError::UnboundPipeline)?;
                    let pipeline_layout = &pipeline_layout_guard[pipeline_layout_id];

                    pipeline_layout
                        .validate_push_constant_ranges(stages, offset, end_offset_bytes)
                        .map_err(RenderCommandError::from)?;

                    unsafe {
                        raw.push_graphics_constants(
                            &pipeline_layout.raw,
                            conv::map_shader_stage_flags(stages),
                            offset,
                            data_slice,
                        )
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
                        raw.set_scissors(0, iter::once(r));
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    state.is_ready()?;
                    let last_vertex = first_vertex + vertex_count;
                    let vertex_limit = state.vertex.vertex_limit;
                    if last_vertex > vertex_limit {
                        return Err(RenderCommandError::VertexBeyondLimit {
                            last_vertex,
                            vertex_limit,
                        }
                        .into());
                    }
                    let last_instance = first_instance + instance_count;
                    let instance_limit = state.vertex.instance_limit;
                    if last_instance > instance_limit {
                        return Err(RenderCommandError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit,
                        }
                        .into());
                    }

                    unsafe {
                        raw.draw(
                            first_vertex..first_vertex + vertex_count,
                            first_instance..first_instance + instance_count,
                        );
                    }
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    state.is_ready()?;

                    //TODO: validate that base_vertex + max_index() is within the provided range
                    let last_index = first_index + index_count;
                    let index_limit = state.index.limit;
                    if last_index > index_limit {
                        return Err(RenderCommandError::IndexBeyondLimit {
                            last_index,
                            index_limit,
                        }
                        .into());
                    }
                    let last_instance = first_instance + instance_count;
                    let instance_limit = state.vertex.instance_limit;
                    if last_instance > instance_limit {
                        return Err(RenderCommandError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit,
                        }
                        .into());
                    }

                    unsafe {
                        raw.draw_indexed(
                            first_index..first_index + index_count,
                            base_vertex,
                            first_instance..first_instance + instance_count,
                        );
                    }
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count,
                    indexed,
                } => {
                    state.is_ready()?;

                    let stride = match indexed {
                        false => 16,
                        true => 20,
                    };

                    if count.is_some() {
                        check_device_features(device.features, wgt::Features::MULTI_DRAW_INDIRECT)?;
                    }

                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .unwrap();
                    check_buffer_usage(buffer.usage, BufferUsage::INDIRECT)?;

                    let actual_count = count.unwrap_or(1);

                    let begin_offset = offset;
                    let end_offset = offset + stride * actual_count as u64;
                    if end_offset > buffer.size {
                        return Err(RenderPassError::IndirectBufferOverrun {
                            offset,
                            count,
                            begin_offset,
                            end_offset,
                            buffer_size: buffer.size,
                        });
                    }

                    match indexed {
                        false => unsafe {
                            raw.draw_indirect(&buffer.raw, offset, actual_count, stride as u32);
                        },
                        true => unsafe {
                            raw.draw_indexed_indirect(
                                &buffer.raw,
                                offset,
                                actual_count,
                                stride as u32,
                            );
                        },
                    }
                }
                RenderCommand::MultiDrawIndirectCount {
                    buffer_id,
                    offset,
                    count_buffer_id,
                    count_buffer_offset,
                    max_count,
                    indexed,
                } => {
                    state.is_ready()?;

                    let stride = match indexed {
                        false => 16,
                        true => 20,
                    };

                    check_device_features(
                        device.features,
                        wgt::Features::MULTI_DRAW_INDIRECT_COUNT,
                    )?;

                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .unwrap();
                    check_buffer_usage(buffer.usage, BufferUsage::INDIRECT)?;
                    let count_buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, count_buffer_id, (), BufferUse::INDIRECT)
                        .unwrap();
                    check_buffer_usage(count_buffer.usage, BufferUsage::INDIRECT)?;

                    let begin_offset = offset;
                    let end_offset = offset + stride * max_count as u64;
                    if end_offset > buffer.size {
                        return Err(RenderPassError::IndirectBufferOverrun {
                            offset,
                            count: None,
                            begin_offset,
                            end_offset,
                            buffer_size: buffer.size,
                        });
                    }

                    let begin_count_offset = count_buffer_offset;
                    let end_count_offset = count_buffer_offset + 4;
                    if end_count_offset > count_buffer.size {
                        return Err(RenderPassError::IndirectCountBufferOverrun {
                            begin_count_offset,
                            end_count_offset,
                            count_buffer_size: count_buffer.size,
                        });
                    }

                    match indexed {
                        false => unsafe {
                            raw.draw_indirect_count(
                                &buffer.raw,
                                offset,
                                &count_buffer.raw,
                                count_buffer_offset,
                                max_count,
                                stride as u32,
                            );
                        },
                        true => unsafe {
                            raw.draw_indexed_indirect_count(
                                &buffer.raw,
                                offset,
                                &count_buffer.raw,
                                count_buffer_offset,
                                max_count,
                                stride as u32,
                            );
                        },
                    }
                }
                RenderCommand::PushDebugGroup { color, len } => {
                    state.debug_scope_depth += 1;
                    let label = str::from_utf8(&base.string_data[..len]).unwrap();
                    unsafe {
                        raw.begin_debug_marker(label, color);
                    }
                    base.string_data = &base.string_data[len..];
                }
                RenderCommand::PopDebugGroup => {
                    if state.debug_scope_depth == 0 {
                        return Err(RenderPassError::InvalidPopDebugGroup);
                    }
                    state.debug_scope_depth -= 1;
                    unsafe {
                        raw.end_debug_marker();
                    }
                }
                RenderCommand::InsertDebugMarker { color, len } => {
                    let label = str::from_utf8(&base.string_data[..len]).unwrap();
                    unsafe {
                        raw.insert_debug_marker(label, color);
                    }
                }
                RenderCommand::ExecuteBundle(bundle_id) => {
                    let bundle = trackers
                        .bundles
                        .use_extend(&*bundle_guard, bundle_id, (), ())
                        .unwrap();

                    if !context.compatible(&bundle.context) {
                        return Err(RenderPassError::IncompatibleRenderBundle);
                    }

                    unsafe {
                        bundle.execute(
                            &mut raw,
                            &*pipeline_layout_guard,
                            &*bind_group_guard,
                            &*pipeline_guard,
                            &*buffer_guard,
                        )?;
                    }

                    trackers.merge_extend(&bundle.used);
                    state.reset_bundle();
                }
            }
        }

        log::trace!("Merging {:?} with the render pass", encoder_id);
        unsafe {
            raw.end_render_pass();
        }

        for ot in output_attachments {
            let texture = &texture_guard[ot.texture_id.value];
            check_texture_usage(texture.usage, TextureUsage::OUTPUT_ATTACHMENT)?;

            // the tracker set of the pass is always in "extend" mode
            trackers
                .textures
                .change_extend(
                    ot.texture_id.value,
                    &ot.texture_id.ref_count,
                    ot.range.clone(),
                    ot.new_use,
                )
                .unwrap();

            if let Some(usage) = ot.previous_use {
                // Make the attachment tracks to be aware of the internal
                // transition done by the render pass, by registering the
                // previous usage as the initial state.
                trackers
                    .textures
                    .prepend(
                        ot.texture_id.value,
                        &ot.texture_id.ref_count,
                        ot.range.clone(),
                        usage,
                    )
                    .unwrap();
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
        }
        cmb.raw.push(raw);

        Ok(())
    }
}

pub mod render_ffi {
    use super::{super::Rect, RenderCommand, RenderPass};
    use crate::{id, span, RawString};
    use std::{convert::TryInto, ffi, slice};
    use wgt::{BufferAddress, BufferSize, Color, DynamicOffset};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_bind_group(
        pass: &mut RenderPass,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_bind_group");
        pass.base.commands.push(RenderCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
        });
        pass.base
            .dynamic_offsets
            .extend_from_slice(slice::from_raw_parts(offsets, offset_length));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_pipeline(
        pass: &mut RenderPass,
        pipeline_id: id::RenderPipelineId,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_pipeline");
        pass.base
            .commands
            .push(RenderCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_index_buffer(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_index_buffer");
        pass.base.commands.push(RenderCommand::SetIndexBuffer {
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_vertex_buffer(
        pass: &mut RenderPass,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_vertex_buffer");
        pass.base.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_blend_color(pass: &mut RenderPass, color: &Color) {
        span!(_guard, DEBUG, "RenderPass::set_blend_color");
        pass.base
            .commands
            .push(RenderCommand::SetBlendColor(*color));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_stencil_reference(pass: &mut RenderPass, value: u32) {
        span!(_guard, DEBUG, "RenderPass::set_stencil_buffer");
        pass.base
            .commands
            .push(RenderCommand::SetStencilReference(value));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_viewport(
        pass: &mut RenderPass,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_viewport");
        pass.base.commands.push(RenderCommand::SetViewport {
            rect: Rect { x, y, w, h },
            depth_min,
            depth_max,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_set_scissor_rect(
        pass: &mut RenderPass,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_scissor_rect");
        pass.base
            .commands
            .push(RenderCommand::SetScissor(Rect { x, y, w, h }));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_set_push_constants(
        pass: &mut RenderPass,
        stages: wgt::ShaderStage,
        offset: u32,
        size_bytes: u32,
        data: *const u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_push_constants");
        let data_slice = slice::from_raw_parts(data, (size_bytes / 4) as usize);
        let value_offset = pass.base.push_constant_data.len().try_into().expect(
            "Ran out of push constant space. Don't set 4gb of push constants per RenderPass.",
        );
        pass.base.push_constant_data.extend_from_slice(data_slice);
        pass.base.commands.push(RenderCommand::SetPushConstant {
            stages,
            offset,
            size_bytes,
            values_offset: Some(value_offset),
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_draw(
        pass: &mut RenderPass,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::draw");
        pass.base.commands.push(RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_draw_indexed(
        pass: &mut RenderPass,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::draw_indexed");
        pass.base.commands.push(RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_draw_indirect(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        span!(_guard, DEBUG, "RenderPass::draw_indirect");
        pass.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: None,
            indexed: false,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_draw_indexed_indirect(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        span!(_guard, DEBUG, "RenderPass::draw_indexed_indirect");
        pass.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: None,
            indexed: true,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_multi_draw_indirect(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::multi_draw_indirect");
        pass.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: Some(count),
            indexed: false,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_multi_draw_indexed_indirect(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::multi_draw_indexed_indirect");
        pass.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: Some(count),
            indexed: true,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_multi_draw_indirect_count(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::multi_draw_indirect_count");
        pass.base
            .commands
            .push(RenderCommand::MultiDrawIndirectCount {
                buffer_id,
                offset,
                count_buffer_id,
                count_buffer_offset,
                max_count,
                indexed: false,
            });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_multi_draw_indexed_indirect_count(
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) {
        span!(
            _guard,
            DEBUG,
            "RenderPass::multi_draw_indexed_indirect_count"
        );
        pass.base
            .commands
            .push(RenderCommand::MultiDrawIndirectCount {
                buffer_id,
                offset,
                count_buffer_id,
                count_buffer_offset,
                max_count,
                indexed: true,
            });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_push_debug_group(
        pass: &mut RenderPass,
        label: RawString,
        color: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::push_debug_group");
        let bytes = ffi::CStr::from_ptr(label).to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(RenderCommand::PushDebugGroup {
            color,
            len: bytes.len(),
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_pop_debug_group(pass: &mut RenderPass) {
        span!(_guard, DEBUG, "RenderPass::pop_debug_group");
        pass.base.commands.push(RenderCommand::PopDebugGroup);
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_insert_debug_marker(
        pass: &mut RenderPass,
        label: RawString,
        color: u32,
    ) {
        span!(_guard, DEBUG, "RenderPass::insert_debug_marker");
        let bytes = ffi::CStr::from_ptr(label).to_bytes();
        pass.base.string_data.extend_from_slice(bytes);

        pass.base.commands.push(RenderCommand::InsertDebugMarker {
            color,
            len: bytes.len(),
        });
    }

    #[no_mangle]
    pub unsafe fn wgpu_render_pass_execute_bundles(
        pass: &mut RenderPass,
        render_bundle_ids: *const id::RenderBundleId,
        render_bundle_ids_length: usize,
    ) {
        span!(_guard, DEBUG, "RenderPass::execute_bundles");
        for &bundle_id in slice::from_raw_parts(render_bundle_ids, render_bundle_ids_length) {
            pass.base
                .commands
                .push(RenderCommand::ExecuteBundle(bundle_id));
        }
    }
}
