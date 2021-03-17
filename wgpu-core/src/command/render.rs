/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::BindError,
    command::{
        bind::{Binder, LayoutChange},
        BasePass, BasePassRef, CommandBuffer, CommandEncoderError, DrawError, ExecutionError,
        MapPassErr, PassErrorScope, RenderCommand, RenderCommandError, StateChange,
    },
    conv,
    device::{
        AttachmentData, AttachmentDataVec, FramebufferKey, RenderPassCompatibilityError,
        RenderPassContext, RenderPassKey, MAX_COLOR_TARGETS, MAX_VERTEX_BUFFERS,
    },
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
    pipeline::PipelineFlags,
    resource::{BufferUse, TextureUse, TextureView, TextureViewInner},
    span,
    track::{TextureSelector, TrackerSet, UsageConflict},
    validation::{
        check_buffer_usage, check_texture_usage, MissingBufferUsageError, MissingTextureUsageError,
    },
    Stored, MAX_BIND_GROUPS,
};

use arrayvec::ArrayVec;
use hal::command::CommandBuffer as _;
use thiserror::Error;
use wgt::{BufferAddress, BufferUsage, Color, IndexFormat, InputStepMode, TextureUsage};

#[cfg(any(feature = "serial-pass", feature = "replay"))]
use serde::Deserialize;
#[cfg(any(feature = "serial-pass", feature = "trace"))]
use serde::Serialize;

use std::{
    borrow::{Borrow, Cow},
    collections::hash_map::Entry,
    fmt, iter,
    num::NonZeroU32,
    ops::Range,
    str,
};

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
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
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

impl DepthStencilAttachmentDescriptor {
    fn is_read_only(&self, aspects: hal::format::Aspects) -> Result<bool, RenderPassErrorInner> {
        if aspects.contains(hal::format::Aspects::DEPTH) && !self.depth.read_only {
            return Ok(false);
        }
        if (self.depth.load_op, self.depth.store_op) != (LoadOp::Load, StoreOp::Store) {
            return Err(RenderPassErrorInner::InvalidDepthOps);
        }
        if aspects.contains(hal::format::Aspects::STENCIL) && !self.stencil.read_only {
            return Ok(false);
        }
        if (self.stencil.load_op, self.stencil.store_op) != (LoadOp::Load, StoreOp::Store) {
            return Err(RenderPassErrorInner::InvalidStencilOps);
        }
        Ok(true)
    }
}

/// Describes the attachments of a render pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderPassDescriptor<'a> {
    /// The color attachments of the render pass.
    pub color_attachments: Cow<'a, [ColorAttachmentDescriptor]>,
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<&'a DepthStencilAttachmentDescriptor>,
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
        Self {
            base: BasePass::new(),
            parent_id,
            color_targets: desc.color_attachments.iter().cloned().collect(),
            depth_stencil_target: desc.depth_stencil_attachment.cloned(),
        }
    }

    pub fn parent_id(&self) -> id::CommandEncoderId {
        self.parent_id
    }

    #[cfg(feature = "trace")]
    pub fn into_command(self) -> crate::device::trace::Command {
        crate::device::trace::Command::RunRenderPass {
            base: self.base,
            target_colors: self.color_targets.into_iter().collect(),
            target_depth_stencil: self.depth_stencil_target,
        }
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

#[derive(Debug, Default)]
struct IndexState {
    bound_buffer_view: Option<(id::Valid<id::BufferId>, Range<BufferAddress>)>,
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
    pipeline_flags: PipelineFlags,
    binder: Binder,
    blend_color: OptionalState,
    stencil_reference: u32,
    pipeline: StateChange<id::RenderPipelineId>,
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
        if self.pipeline.is_unset() {
            return Err(DrawError::MissingPipeline);
        }
        if self.blend_color == OptionalState::Required {
            return Err(DrawError::MissingBlendColor);
        }
        Ok(())
    }

    /// Reset the `RenderBundle`-related states.
    fn reset_bundle(&mut self) {
        self.binder.reset();
        self.pipeline.reset();
        self.index.reset();
        self.vertex.reset();
    }
}

/// Error encountered when performing a render pass.
#[derive(Clone, Debug, Error)]
pub enum RenderPassErrorInner {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("attachment texture view {0:?} is invalid")]
    InvalidAttachment(id::TextureViewId),
    #[error("attachments have different sizes")]
    MismatchAttachments,
    #[error("attachment's sample count {0} is invalid")]
    InvalidSampleCount(u8),
    #[error("attachment with resolve target must be multi-sampled")]
    InvalidResolveSourceSampleCount,
    #[error("resolve target must have a sample count of 1")]
    InvalidResolveTargetSampleCount,
    #[error("not enough memory left")]
    OutOfMemory,
    #[error("extent state {state_extent:?} must match extent from view {view_extent:?}")]
    ExtentStateMismatch {
        state_extent: hal::image::Extent,
        view_extent: hal::image::Extent,
    },
    #[error("attempted to use a swap chain image as a depth/stencil attachment")]
    SwapChainImageAsDepthStencil,
    #[error("unable to clear non-present/read-only depth")]
    InvalidDepthOps,
    #[error("unable to clear non-present/read-only stencil")]
    InvalidStencilOps,
    #[error("all attachments must have the same sample count, found {actual} != {expected}")]
    SampleCountMismatch { actual: u8, expected: u8 },
    #[error("texture view's swap chain must match swap chain in use")]
    SwapChainMismatch,
    #[error("setting `values_offset` to be `None` is only for internal use in render bundles")]
    InvalidValuesOffset,
    #[error("required device features not enabled: {0:?}")]
    MissingDeviceFeatures(wgt::Features),
    #[error("indirect draw with offset {offset}{} uses bytes {begin_offset}..{end_offset} which overruns indirect buffer of size {buffer_size}", count.map_or_else(String::new, |v| format!(" and count {}", v)))]
    IndirectBufferOverrun {
        offset: u64,
        count: Option<NonZeroU32>,
        begin_offset: u64,
        end_offset: u64,
        buffer_size: u64,
    },
    #[error("indirect draw uses bytes {begin_count_offset}..{end_count_offset} which overruns indirect buffer of size {count_buffer_size}")]
    IndirectCountBufferOverrun {
        begin_count_offset: u64,
        end_count_offset: u64,
        count_buffer_size: u64,
    },
    #[error("cannot pop debug group, because number of pushed debug groups is zero")]
    InvalidPopDebugGroup,
    #[error(transparent)]
    ResourceUsageConflict(#[from] UsageConflict),
    #[error("render bundle is incompatible, {0}")]
    IncompatibleRenderBundle(#[from] RenderPassCompatibilityError),
    #[error(transparent)]
    RenderCommand(#[from] RenderCommandError),
    #[error(transparent)]
    Draw(#[from] DrawError),
    #[error(transparent)]
    Bind(#[from] BindError),
}

impl From<MissingBufferUsageError> for RenderPassErrorInner {
    fn from(error: MissingBufferUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

impl From<MissingTextureUsageError> for RenderPassErrorInner {
    fn from(error: MissingTextureUsageError) -> Self {
        Self::RenderCommand(error.into())
    }
}

/// Error encountered when performing a render pass.
#[derive(Clone, Debug, Error)]
#[error("Render pass error {scope}: {inner}")]
pub struct RenderPassError {
    pub scope: PassErrorScope,
    #[source]
    inner: RenderPassErrorInner,
}

impl<T, E> MapPassErr<T, RenderPassError> for Result<T, E>
where
    E: Into<RenderPassErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, RenderPassError> {
        self.map_err(|inner| RenderPassError {
            scope,
            inner: inner.into(),
        })
    }
}

fn check_device_features(
    actual: wgt::Features,
    expected: wgt::Features,
) -> Result<(), RenderPassErrorInner> {
    if !actual.contains(expected) {
        Err(RenderPassErrorInner::MissingDeviceFeatures(expected))
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
        let scope = PassErrorScope::Pass(encoder_id);

        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut cmb_guard, mut token) = hub.command_buffers.write(&mut token);

        let mut trackers = TrackerSet::new(B::VARIANT);
        let cmd_buf =
            CommandBuffer::get_encoder(&mut *cmb_guard, encoder_id).map_pass_err(scope)?;
        let device = &device_guard[cmd_buf.device_id.value];
        let mut raw = device.cmd_allocator.extend(cmd_buf);

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(crate::device::trace::Command::RunRenderPass {
                base: BasePass::from_ref(base),
                target_colors: color_attachments.iter().cloned().collect(),
                target_depth_stencil: depth_stencil_attachment.cloned(),
            });
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

        struct RenderAttachment<'a> {
            texture_id: &'a Stored<id::TextureId>,
            selector: &'a TextureSelector,
            previous_use: Option<TextureUse>,
            new_use: TextureUse,
        }
        let mut render_attachments = AttachmentDataVec::<RenderAttachment>::new();

        let mut attachment_width = None;
        let mut attachment_height = None;
        let mut valid_attachment = true;

        let context = {
            use hal::device::Device as _;

            let sample_count_limit = device.hal_limits.framebuffer_color_sample_counts;
            let base_trackers = &cmd_buf.trackers;

            let mut extent = None;
            let mut sample_count = 0;
            let mut depth_stencil_aspects = hal::format::Aspects::empty();
            let mut used_swap_chain = None::<Stored<id::SwapChainId>>;

            let mut add_view = |view: &TextureView<B>| {
                if let Some(ex) = extent {
                    if ex != view.extent {
                        return Err(RenderPassErrorInner::ExtentStateMismatch {
                            state_extent: ex,
                            view_extent: view.extent,
                        });
                    }
                } else {
                    extent = Some(view.extent);
                }
                if sample_count == 0 {
                    sample_count = view.samples;
                } else if sample_count != view.samples {
                    return Err(RenderPassErrorInner::SampleCountMismatch {
                        actual: view.samples,
                        expected: sample_count,
                    });
                }
                Ok(())
            };

            tracing::trace!(
                "Encoding render pass begin in command buffer {:?}",
                encoder_id
            );
            let rp_key = {
                let depth_stencil = match depth_stencil_attachment {
                    Some(at) => {
                        let view = trackers
                            .views
                            .use_extend(&*view_guard, at.attachment, (), ())
                            .map_err(|_| RenderPassErrorInner::InvalidAttachment(at.attachment))
                            .map_pass_err(scope)?;
                        add_view(view).map_pass_err(scope)?;
                        depth_stencil_aspects = view.aspects;

                        let source_id = match view.inner {
                            TextureViewInner::Native { ref source_id, .. } => source_id,
                            TextureViewInner::SwapChain { .. } => {
                                return Err(RenderPassErrorInner::SwapChainImageAsDepthStencil)
                                    .map_pass_err(scope)
                            }
                        };

                        // Using render pass for transition.
                        let previous_use = base_trackers
                            .textures
                            .query(source_id.value, view.selector.clone());
                        let new_use = if at.is_read_only(view.aspects).map_pass_err(scope)? {
                            is_ds_read_only = true;
                            TextureUse::ATTACHMENT_READ
                        } else {
                            TextureUse::ATTACHMENT_WRITE
                        };
                        render_attachments.push(RenderAttachment {
                            texture_id: source_id,
                            selector: &view.selector,
                            previous_use,
                            new_use,
                        });

                        let new_layout = conv::map_texture_state(new_use, view.aspects).1;
                        let old_layout = match previous_use {
                            Some(usage) => conv::map_texture_state(usage, view.aspects).1,
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
                        .map_err(|_| RenderPassErrorInner::InvalidAttachment(at.attachment))
                        .map_pass_err(scope)?;
                    add_view(view).map_pass_err(scope)?;

                    valid_attachment &= *attachment_width.get_or_insert(view.extent.width)
                        == view.extent.width
                        && *attachment_height.get_or_insert(view.extent.height)
                            == view.extent.height;

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let previous_use = base_trackers
                                .textures
                                .query(source_id.value, view.selector.clone());
                            let new_use = TextureUse::ATTACHMENT_WRITE;
                            render_attachments.push(RenderAttachment {
                                texture_id: source_id,
                                selector: &view.selector,
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
                            if let Some((ref sc_id, _)) = cmd_buf.used_swap_chain {
                                if source_id.value != sc_id.value {
                                    return Err(RenderPassErrorInner::SwapChainMismatch)
                                        .map_pass_err(scope);
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

                if !valid_attachment {
                    return Err(RenderPassErrorInner::MismatchAttachments).map_pass_err(scope);
                }

                for resolve_target in color_attachments.iter().flat_map(|at| at.resolve_target) {
                    let view = trackers
                        .views
                        .use_extend(&*view_guard, resolve_target, (), ())
                        .map_err(|_| RenderPassErrorInner::InvalidAttachment(resolve_target))
                        .map_pass_err(scope)?;
                    if extent != Some(view.extent) {
                        return Err(RenderPassErrorInner::ExtentStateMismatch {
                            state_extent: extent.unwrap_or_default(),
                            view_extent: view.extent,
                        })
                        .map_pass_err(scope);
                    }
                    if view.samples != 1 {
                        return Err(RenderPassErrorInner::InvalidResolveTargetSampleCount)
                            .map_pass_err(scope);
                    }
                    if sample_count == 1 {
                        return Err(RenderPassErrorInner::InvalidResolveSourceSampleCount)
                            .map_pass_err(scope);
                    }

                    let layouts = match view.inner {
                        TextureViewInner::Native { ref source_id, .. } => {
                            let previous_use = base_trackers
                                .textures
                                .query(source_id.value, view.selector.clone());
                            let new_use = TextureUse::ATTACHMENT_WRITE;
                            render_attachments.push(RenderAttachment {
                                texture_id: source_id,
                                selector: &view.selector,
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
                            if let Some((ref sc_id, _)) = cmd_buf.used_swap_chain {
                                if source_id.value != sc_id.value {
                                    return Err(RenderPassErrorInner::SwapChainMismatch)
                                        .map_pass_err(scope);
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

            if sample_count & sample_count_limit == 0 {
                return Err(RenderPassErrorInner::InvalidSampleCount(sample_count))
                    .map_pass_err(scope);
            }

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
                                Some(_) => attachment_index + i,
                                None => hal::pass::ATTACHMENT_UNUSED,
                            };
                            resolve_ids.push((real_attachment_index, layout));
                        }
                        attachment_index += color_attachments.len();
                    }

                    let depth_id = depth_stencil_attachment.map(|_| {
                        let usage = if is_ds_read_only {
                            TextureUse::ATTACHMENT_READ
                        } else {
                            TextureUse::ATTACHMENT_WRITE
                        };
                        (
                            attachment_index,
                            conv::map_texture_state(usage, depth_stencil_aspects).1,
                        )
                    });

                    let subpass = hal::pass::SubpassDesc {
                        colors: &color_ids[..color_attachments.len()],
                        resolves: &resolve_ids,
                        depth_stencil: depth_id.as_ref(),
                        inputs: &[],
                        preserves: &[],
                    };
                    let all = entry
                        .key()
                        .all()
                        .map(|(at, _)| at)
                        .collect::<AttachmentDataVec<_>>();

                    let pass =
                        unsafe { device.raw.create_render_pass(all, iter::once(subpass), &[]) }
                            .unwrap();
                    entry.insert(pass)
                }
            };

            let mut framebuffer_cache;
            let fb_key = FramebufferKey {
                colors: color_attachments
                    .iter()
                    .map(|at| id::Valid(at.attachment))
                    .collect(),
                resolves: color_attachments
                    .iter()
                    .filter_map(|at| at.resolve_target)
                    .map(id::Valid)
                    .collect(),
                depth_stencil: depth_stencil_attachment.map(|at| id::Valid(at.attachment)),
            };
            let context = RenderPassContext {
                attachments: AttachmentData {
                    colors: fb_key
                        .colors
                        .iter()
                        .map(|&at| view_guard[at].format)
                        .collect(),
                    resolves: fb_key
                        .resolves
                        .iter()
                        .map(|&at| view_guard[at].format)
                        .collect(),
                    depth_stencil: fb_key.depth_stencil.map(|at| view_guard[at].format),
                },
                sample_count,
            };

            let framebuffer = match used_swap_chain.take() {
                Some(sc_id) => {
                    assert!(cmd_buf.used_swap_chain.is_none());
                    // Always create a new framebuffer and delete it after presentation.
                    let attachments = fb_key
                        .all()
                        .map(|&id| match view_guard[id].inner {
                            TextureViewInner::Native { ref raw, .. } => raw,
                            TextureViewInner::SwapChain { ref image, .. } => Borrow::borrow(image),
                        })
                        .collect::<AttachmentDataVec<_>>();
                    let framebuffer = unsafe {
                        device
                            .raw
                            .create_framebuffer(&render_pass, attachments, extent.unwrap())
                            .or(Err(RenderPassErrorInner::OutOfMemory))
                            .map_pass_err(scope)?
                    };
                    cmd_buf.used_swap_chain = Some((sc_id, framebuffer));
                    &mut cmd_buf.used_swap_chain.as_mut().unwrap().1
                }
                None => {
                    // Cache framebuffers by the device.
                    framebuffer_cache = device.framebuffers.lock();
                    match framebuffer_cache.entry(fb_key) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            let fb = {
                                let attachments = e
                                    .key()
                                    .all()
                                    .map(|&id| match view_guard[id].inner {
                                        TextureViewInner::Native { ref raw, .. } => raw,
                                        TextureViewInner::SwapChain { ref image, .. } => {
                                            Borrow::borrow(image)
                                        }
                                    })
                                    .collect::<AttachmentDataVec<_>>();
                                unsafe {
                                    device
                                        .raw
                                        .create_framebuffer(
                                            &render_pass,
                                            attachments,
                                            extent.unwrap(),
                                        )
                                        .or(Err(RenderPassErrorInner::OutOfMemory))
                                        .map_pass_err(scope)?
                                }
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
                }))
                .collect::<ArrayVec<[_; MAX_COLOR_TARGETS + 1]>>();

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

            context
        };

        let mut state = State {
            pipeline_flags: PipelineFlags::empty(),
            binder: Binder::new(cmd_buf.limits.max_bind_groups),
            blend_color: OptionalState::Unused,
            stencil_reference: 0,
            pipeline: StateChange::new(),
            index: IndexState::default(),
            vertex: VertexState::default(),
            debug_scope_depth: 0,
        };
        let mut temp_offsets = Vec::new();

        for command in base.commands {
            match *command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let scope = PassErrorScope::SetBindGroup(bind_group_id);
                    let max_bind_groups = device.limits.max_bind_groups;
                    if (index as u32) >= max_bind_groups {
                        return Err(RenderCommandError::BindGroupIndexOutOfRange {
                            index,
                            max: max_bind_groups,
                        })
                        .map_pass_err(scope);
                    }

                    temp_offsets.clear();
                    temp_offsets
                        .extend_from_slice(&base.dynamic_offsets[..num_dynamic_offsets as usize]);
                    base.dynamic_offsets = &base.dynamic_offsets[num_dynamic_offsets as usize..];

                    let bind_group = trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .unwrap();
                    bind_group
                        .validate_dynamic_bindings(&temp_offsets)
                        .map_pass_err(scope)?;

                    trackers
                        .merge_extend(&bind_group.used)
                        .map_pass_err(scope)?;

                    if let Some((pipeline_layout_id, follow_ups)) = state.binder.provide_entry(
                        index as usize,
                        id::Valid(bind_group_id),
                        bind_group,
                        &temp_offsets,
                    ) {
                        let bind_groups = iter::once(bind_group.raw.raw())
                            .chain(
                                follow_ups
                                    .clone()
                                    .map(|(bg_id, _)| bind_group_guard[bg_id].raw.raw()),
                            )
                            .collect::<ArrayVec<[_; MAX_BIND_GROUPS]>>();
                        temp_offsets.extend(follow_ups.flat_map(|(_, offsets)| offsets));
                        unsafe {
                            raw.bind_graphics_descriptor_sets(
                                &pipeline_layout_guard[pipeline_layout_id].raw,
                                index as usize,
                                bind_groups,
                                &temp_offsets,
                            );
                        }
                    };
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let scope = PassErrorScope::SetPipelineRender(pipeline_id);
                    if state.pipeline.set_and_check_redundant(pipeline_id) {
                        continue;
                    }

                    let pipeline = trackers
                        .render_pipes
                        .use_extend(&*pipeline_guard, pipeline_id, (), ())
                        .map_err(|_| RenderCommandError::InvalidPipeline(pipeline_id))
                        .map_pass_err(scope)?;

                    context
                        .check_compatible(&pipeline.pass_context)
                        .map_err(RenderCommandError::IncompatiblePipeline)
                        .map_pass_err(scope)?;

                    state.pipeline_flags = pipeline.flags;

                    if pipeline.flags.contains(PipelineFlags::WRITES_DEPTH_STENCIL)
                        && is_ds_read_only
                    {
                        return Err(RenderCommandError::IncompatibleReadOnlyDepthStencil)
                            .map_pass_err(scope);
                    }

                    state
                        .blend_color
                        .require(pipeline.flags.contains(PipelineFlags::BLEND_COLOR));

                    unsafe {
                        raw.bind_graphics_pipeline(&pipeline.raw);
                    }

                    if pipeline.flags.contains(PipelineFlags::STENCIL_REFERENCE) {
                        unsafe {
                            raw.set_stencil_reference(
                                hal::pso::Face::all(),
                                state.stencil_reference,
                            );
                        }
                    }

                    // Rebind resource
                    if state.binder.pipeline_layout_id != Some(pipeline.layout_id.value) {
                        let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id.value];

                        state.binder.change_pipeline_layout(
                            &*pipeline_layout_guard,
                            pipeline.layout_id.value,
                        );

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
                            let &(ref buffer, _) = buffer_guard[buffer_id].raw.as_ref().unwrap();

                            let range = hal::buffer::SubRange {
                                offset: range.start,
                                size: Some(range.end - range.start),
                            };
                            let index_type = conv::map_index_format(state.index.format);
                            unsafe {
                                raw.bind_index_buffer(buffer, range, index_type);
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
                    let scope = PassErrorScope::SetIndexBuffer(buffer_id);
                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                        .map_err(|e| RenderCommandError::Buffer(buffer_id, e))
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, BufferUsage::INDEX).map_pass_err(scope)?;
                    let &(ref buf_raw, _) = buffer
                        .raw
                        .as_ref()
                        .ok_or(RenderCommandError::DestroyedBuffer(buffer_id))
                        .map_pass_err(scope)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    state.index.bound_buffer_view = Some((id::Valid(buffer_id), offset..end));
                    state.index.update_limit();

                    let range = hal::buffer::SubRange {
                        offset,
                        size: Some(end - offset),
                    };
                    let index_type = conv::map_index_format(state.index.format);
                    unsafe {
                        raw.bind_index_buffer(buf_raw, range, index_type);
                    }
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetVertexBuffer(buffer_id);
                    let buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::VERTEX)
                        .map_err(|e| RenderCommandError::Buffer(buffer_id, e))
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, BufferUsage::VERTEX).map_pass_err(scope)?;
                    let &(ref buf_raw, _) = buffer
                        .raw
                        .as_ref()
                        .ok_or(RenderCommandError::DestroyedBuffer(buffer_id))
                        .map_pass_err(scope)?;

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
                        raw.bind_vertex_buffers(slot, iter::once((buf_raw, range)));
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
                    state.stencil_reference = value;
                    if state
                        .pipeline_flags
                        .contains(PipelineFlags::STENCIL_REFERENCE)
                    {
                        unsafe {
                            raw.set_stencil_reference(hal::pso::Face::all(), value);
                        }
                    }
                }
                RenderCommand::SetViewport {
                    ref rect,
                    depth_min,
                    depth_max,
                } => {
                    let scope = PassErrorScope::SetViewport;
                    use std::{convert::TryFrom, i16};
                    if rect.w <= 0.0
                        || rect.h <= 0.0
                        || depth_min < 0.0
                        || depth_min > 1.0
                        || depth_max < 0.0
                        || depth_max > 1.0
                    {
                        return Err(RenderCommandError::InvalidViewport).map_pass_err(scope);
                    }
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
                    let scope = PassErrorScope::SetPushConstant;
                    let values_offset = values_offset
                        .ok_or(RenderPassErrorInner::InvalidValuesOffset)
                        .map_pass_err(scope)?;

                    let end_offset_bytes = offset + size_bytes;
                    let values_end_offset =
                        (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                    let data_slice =
                        &base.push_constant_data[(values_offset as usize)..values_end_offset];

                    let pipeline_layout_id = state
                        .binder
                        .pipeline_layout_id
                        .ok_or(DrawError::MissingPipeline)
                        .map_pass_err(scope)?;
                    let pipeline_layout = &pipeline_layout_guard[pipeline_layout_id];

                    pipeline_layout
                        .validate_push_constant_ranges(stages, offset, end_offset_bytes)
                        .map_err(RenderCommandError::from)
                        .map_pass_err(scope)?;

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
                    let scope = PassErrorScope::SetScissorRect;
                    use std::{convert::TryFrom, i16};
                    if rect.w == 0
                        || rect.h == 0
                        || rect.x + rect.w > attachment_width.unwrap()
                        || rect.y + rect.h > attachment_height.unwrap()
                    {
                        return Err(RenderCommandError::InvalidScissorRect).map_pass_err(scope);
                    }
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
                    let scope = PassErrorScope::Draw;
                    state.is_ready().map_pass_err(scope)?;
                    let last_vertex = first_vertex + vertex_count;
                    let vertex_limit = state.vertex.vertex_limit;
                    if last_vertex > vertex_limit {
                        return Err(DrawError::VertexBeyondLimit {
                            last_vertex,
                            vertex_limit,
                        })
                        .map_pass_err(scope);
                    }
                    let last_instance = first_instance + instance_count;
                    let instance_limit = state.vertex.instance_limit;
                    if last_instance > instance_limit {
                        return Err(DrawError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit,
                        })
                        .map_pass_err(scope);
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
                    let scope = PassErrorScope::DrawIndexed;
                    state.is_ready().map_pass_err(scope)?;

                    //TODO: validate that base_vertex + max_index() is within the provided range
                    let last_index = first_index + index_count;
                    let index_limit = state.index.limit;
                    if last_index > index_limit {
                        return Err(DrawError::IndexBeyondLimit {
                            last_index,
                            index_limit,
                        })
                        .map_pass_err(scope);
                    }
                    let last_instance = first_instance + instance_count;
                    let instance_limit = state.vertex.instance_limit;
                    if last_instance > instance_limit {
                        return Err(DrawError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit,
                        })
                        .map_pass_err(scope);
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
                    let scope = if indexed {
                        PassErrorScope::DrawIndexedIndirect
                    } else {
                        PassErrorScope::DrawIndirect
                    };
                    state.is_ready().map_pass_err(scope)?;

                    let stride = match indexed {
                        false => 16,
                        true => 20,
                    };

                    if count.is_some() {
                        check_device_features(device.features, wgt::Features::MULTI_DRAW_INDIRECT)
                            .map_pass_err(scope)?;
                    }

                    let indirect_buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .map_err(|e| RenderCommandError::Buffer(buffer_id, e))
                        .map_pass_err(scope)?;
                    check_buffer_usage(indirect_buffer.usage, BufferUsage::INDIRECT)
                        .map_pass_err(scope)?;
                    let &(ref indirect_raw, _) = indirect_buffer
                        .raw
                        .as_ref()
                        .ok_or(RenderCommandError::DestroyedBuffer(buffer_id))
                        .map_pass_err(scope)?;

                    let actual_count = count.map_or(1, |c| c.get());

                    let begin_offset = offset;
                    let end_offset = offset + stride * actual_count as u64;
                    if end_offset > indirect_buffer.size {
                        return Err(RenderPassErrorInner::IndirectBufferOverrun {
                            offset,
                            count,
                            begin_offset,
                            end_offset,
                            buffer_size: indirect_buffer.size,
                        })
                        .map_pass_err(scope);
                    }

                    match indexed {
                        false => unsafe {
                            raw.draw_indirect(indirect_raw, offset, actual_count, stride as u32);
                        },
                        true => unsafe {
                            raw.draw_indexed_indirect(
                                indirect_raw,
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
                    let scope = if indexed {
                        PassErrorScope::DrawIndexedIndirect
                    } else {
                        PassErrorScope::DrawIndirect
                    };
                    state.is_ready().map_pass_err(scope)?;

                    let stride = match indexed {
                        false => 16,
                        true => 20,
                    };

                    check_device_features(
                        device.features,
                        wgt::Features::MULTI_DRAW_INDIRECT_COUNT,
                    )
                    .map_pass_err(scope)?;

                    let indirect_buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .map_err(|e| RenderCommandError::Buffer(buffer_id, e))
                        .map_pass_err(scope)?;
                    check_buffer_usage(indirect_buffer.usage, BufferUsage::INDIRECT)
                        .map_pass_err(scope)?;
                    let &(ref indirect_raw, _) = indirect_buffer
                        .raw
                        .as_ref()
                        .ok_or(RenderCommandError::DestroyedBuffer(buffer_id))
                        .map_pass_err(scope)?;

                    let count_buffer = trackers
                        .buffers
                        .use_extend(&*buffer_guard, count_buffer_id, (), BufferUse::INDIRECT)
                        .map_err(|e| RenderCommandError::Buffer(count_buffer_id, e))
                        .map_pass_err(scope)?;
                    check_buffer_usage(count_buffer.usage, BufferUsage::INDIRECT)
                        .map_pass_err(scope)?;
                    let &(ref count_raw, _) = count_buffer
                        .raw
                        .as_ref()
                        .ok_or(RenderCommandError::DestroyedBuffer(count_buffer_id))
                        .map_pass_err(scope)?;

                    let begin_offset = offset;
                    let end_offset = offset + stride * max_count as u64;
                    if end_offset > indirect_buffer.size {
                        return Err(RenderPassErrorInner::IndirectBufferOverrun {
                            offset,
                            count: None,
                            begin_offset,
                            end_offset,
                            buffer_size: indirect_buffer.size,
                        })
                        .map_pass_err(scope);
                    }

                    let begin_count_offset = count_buffer_offset;
                    let end_count_offset = count_buffer_offset + 4;
                    if end_count_offset > count_buffer.size {
                        return Err(RenderPassErrorInner::IndirectCountBufferOverrun {
                            begin_count_offset,
                            end_count_offset,
                            count_buffer_size: count_buffer.size,
                        })
                        .map_pass_err(scope);
                    }

                    match indexed {
                        false => unsafe {
                            raw.draw_indirect_count(
                                indirect_raw,
                                offset,
                                count_raw,
                                count_buffer_offset,
                                max_count,
                                stride as u32,
                            );
                        },
                        true => unsafe {
                            raw.draw_indexed_indirect_count(
                                indirect_raw,
                                offset,
                                count_raw,
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
                    let scope = PassErrorScope::PopDebugGroup;
                    if state.debug_scope_depth == 0 {
                        return Err(RenderPassErrorInner::InvalidPopDebugGroup).map_pass_err(scope);
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
                    base.string_data = &base.string_data[len..];
                }
                RenderCommand::ExecuteBundle(bundle_id) => {
                    let scope = PassErrorScope::ExecuteBundle;
                    let bundle = trackers
                        .bundles
                        .use_extend(&*bundle_guard, bundle_id, (), ())
                        .unwrap();

                    context
                        .check_compatible(&bundle.context)
                        .map_err(RenderPassErrorInner::IncompatibleRenderBundle)
                        .map_pass_err(scope)?;

                    unsafe {
                        bundle.execute(
                            &mut raw,
                            &*pipeline_layout_guard,
                            &*bind_group_guard,
                            &*pipeline_guard,
                            &*buffer_guard,
                        )
                    }
                    .map_err(|e| match e {
                        ExecutionError::DestroyedBuffer(id) => {
                            RenderCommandError::DestroyedBuffer(id)
                        }
                    })
                    .map_pass_err(scope)?;

                    trackers.merge_extend(&bundle.used).map_pass_err(scope)?;
                    state.reset_bundle();
                }
            }
        }

        tracing::trace!("Merging {:?} with the render pass", encoder_id);
        unsafe {
            raw.end_render_pass();
        }

        for ra in render_attachments {
            let texture = &texture_guard[ra.texture_id.value];
            check_texture_usage(texture.usage, TextureUsage::RENDER_ATTACHMENT)
                .map_pass_err(scope)?;

            // the tracker set of the pass is always in "extend" mode
            trackers
                .textures
                .change_extend(
                    ra.texture_id.value,
                    &ra.texture_id.ref_count,
                    ra.selector.clone(),
                    ra.new_use,
                )
                .unwrap();

            if let Some(usage) = ra.previous_use {
                // Make the attachment tracks to be aware of the internal
                // transition done by the render pass, by registering the
                // previous usage as the initial state.
                trackers
                    .textures
                    .prepend(
                        ra.texture_id.value,
                        &ra.texture_id.ref_count,
                        ra.selector.clone(),
                        usage,
                    )
                    .unwrap();
            }
        }

        super::CommandBuffer::insert_barriers(
            cmd_buf.raw.last_mut().unwrap(),
            &mut cmd_buf.trackers,
            &trackers,
            &*buffer_guard,
            &*texture_guard,
        );
        unsafe {
            cmd_buf.raw.last_mut().unwrap().finish();
        }
        cmd_buf.raw.push(raw);

        Ok(())
    }
}

pub mod render_ffi {
    use super::{
        super::{Rect, RenderCommand},
        RenderPass,
    };
    use crate::{id, span, RawString};
    use std::{convert::TryInto, ffi, num::NonZeroU32, slice};
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
        data: *const u8,
    ) {
        span!(_guard, DEBUG, "RenderPass::set_push_constants");
        assert_eq!(
            offset & (wgt::PUSH_CONSTANT_ALIGNMENT - 1),
            0,
            "Push constant offset must be aligned to 4 bytes."
        );
        assert_eq!(
            size_bytes & (wgt::PUSH_CONSTANT_ALIGNMENT - 1),
            0,
            "Push constant size must be aligned to 4 bytes."
        );
        let data_slice = slice::from_raw_parts(data, size_bytes as usize);
        let value_offset = pass.base.push_constant_data.len().try_into().expect(
            "Ran out of push constant space. Don't set 4gb of push constants per RenderPass.",
        );

        pass.base.push_constant_data.extend(
            data_slice
                .chunks_exact(wgt::PUSH_CONSTANT_ALIGNMENT as usize)
                .map(|arr| u32::from_ne_bytes([arr[0], arr[1], arr[2], arr[3]])),
        );

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
            count: NonZeroU32::new(count),
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
            count: NonZeroU32::new(count),
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
