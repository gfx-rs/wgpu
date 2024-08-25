/*! Render Bundles

A render bundle is a prerecorded sequence of commands that can be replayed on a
command encoder with a single call. A single bundle can replayed any number of
times, on different encoders. Constructing a render bundle lets `wgpu` validate
and analyze its commands up front, so that replaying a bundle can be more
efficient than simply re-recording its commands each time.

Not all commands are available in bundles; for example, a render bundle may not
contain a [`RenderCommand::SetViewport`] command.

Most of `wgpu`'s backend graphics APIs have something like bundles. For example,
Vulkan calls them "secondary command buffers", and Metal calls them "indirect
command buffers". Although we plan to take advantage of these platform features
at some point in the future, for now `wgpu`'s implementation of render bundles
does not use them: at the hal level, `wgpu` render bundles just replay the
commands.

## Render Bundle Isolation

One important property of render bundles is that the draw calls in a render
bundle depend solely on the pipeline and state established within the render
bundle itself. A draw call in a bundle will never use a vertex buffer, say, that
was set in the `RenderPass` before executing the bundle. We call this property
'isolation', in that a render bundle is somewhat isolated from the passes that
use it.

Render passes are also isolated from the effects of bundles. After executing a
render bundle, a render pass's pipeline, bind groups, and vertex and index
buffers are are unset, so the bundle cannot affect later draw calls in the pass.

A render pass is not fully isolated from a bundle's effects on push constant
values. Draw calls following a bundle's execution will see whatever values the
bundle writes to push constant storage. Setting a pipeline initializes any push
constant storage it could access to zero, and this initialization may also be
visible after bundle execution.

## Render Bundle Lifecycle

To create a render bundle:

1) Create a [`RenderBundleEncoder`] by calling
   [`Global::device_create_render_bundle_encoder`][Gdcrbe].

2) Record commands in the `RenderBundleEncoder` using functions from the
   [`bundle_ffi`] module.

3) Call [`Global::render_bundle_encoder_finish`][Grbef], which analyzes and cleans up
   the command stream and returns a `RenderBundleId`.

4) Then, any number of times, call [`render_pass_execute_bundles`][wrpeb] to
   execute the bundle as part of some render pass.

## Implementation

The most complex part of render bundles is the "finish" step, mostly implemented
in [`RenderBundleEncoder::finish`]. This consumes the commands stored in the
encoder's [`BasePass`], while validating everything, tracking the state,
dropping redundant or unnecessary commands, and presenting the results as a new
[`RenderBundle`]. It doesn't actually execute any commands.

This step also enforces the 'isolation' property mentioned above: every draw
call is checked to ensure that the resources it uses on were established since
the last time the pipeline was set. This means the bundle can be executed
verbatim without any state tracking.

### Execution

When the bundle is used in an actual render pass, `RenderBundle::execute` is
called. It goes through the commands and issues them into the native command
buffer. Thanks to isolation, it doesn't track any bind group invalidations or
index format changes.

[Gdcrbe]: crate::global::Global::device_create_render_bundle_encoder
[Grbef]: crate::global::Global::render_bundle_encoder_finish
[wrpeb]: crate::global::Global::render_pass_execute_bundles
!*/

#![allow(clippy::reversed_empty_ranges)]

use crate::{
    binding_model::{BindError, BindGroup, PipelineLayout},
    command::{
        BasePass, BindGroupStateChange, ColorAttachmentError, DrawError, MapPassErr,
        PassErrorScope, RenderCommandError, StateChange,
    },
    device::{
        AttachmentData, Device, DeviceError, MissingDownlevelFlags, RenderPassContext,
        SHADER_STAGE_COUNT,
    },
    hub::Hub,
    id,
    init_tracker::{BufferInitTrackerAction, MemoryInitKind, TextureInitTrackerAction},
    pipeline::{PipelineFlags, RenderPipeline, VertexStep},
    resource::{Buffer, DestroyedResourceError, Labeled, ParentDevice, TrackingData},
    resource_log,
    snatch::SnatchGuard,
    track::RenderBundleScope,
    Label, LabelHelpers,
};
use arrayvec::ArrayVec;

use std::{borrow::Cow, mem::size_of, num::NonZeroU32, ops::Range, sync::Arc};
use thiserror::Error;

use super::{
    render_command::{ArcRenderCommand, RenderCommand},
    DrawKind,
};

/// <https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-draw>
fn validate_draw(
    vertex: &[Option<VertexState>],
    step: &[VertexStep],
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), DrawError> {
    let vertices_end = first_vertex as u64 + vertex_count as u64;
    let instances_end = first_instance as u64 + instance_count as u64;

    for (idx, (vbs, step)) in vertex.iter().zip(step).enumerate() {
        let Some(vbs) = vbs else {
            continue;
        };

        let stride_count = match step.mode {
            wgt::VertexStepMode::Vertex => vertices_end,
            wgt::VertexStepMode::Instance => instances_end,
        };

        if stride_count == 0 {
            continue;
        }

        let offset = (stride_count - 1) * step.stride + step.last_stride;
        let limit = vbs.range.end - vbs.range.start;
        if offset > limit {
            return Err(DrawError::VertexOutOfBounds {
                step_mode: step.mode,
                offset,
                limit,
                slot: idx as u32,
            });
        }
    }

    Ok(())
}

// See https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-drawindexed
fn validate_indexed_draw(
    vertex: &[Option<VertexState>],
    step: &[VertexStep],
    index_state: &IndexState,
    first_index: u32,
    index_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), DrawError> {
    let last_index = first_index as u64 + index_count as u64;
    let index_limit = index_state.limit();
    if last_index > index_limit {
        return Err(DrawError::IndexBeyondLimit {
            last_index,
            index_limit,
        });
    }

    let stride_count = first_instance as u64 + instance_count as u64;
    for (idx, (vbs, step)) in vertex.iter().zip(step).enumerate() {
        let Some(vbs) = vbs else {
            continue;
        };

        if stride_count == 0 || step.mode != wgt::VertexStepMode::Instance {
            continue;
        }

        let offset = (stride_count - 1) * step.stride + step.last_stride;
        let limit = vbs.range.end - vbs.range.start;
        if offset > limit {
            return Err(DrawError::VertexOutOfBounds {
                step_mode: step.mode,
                offset,
                limit,
                slot: idx as u32,
            });
        }
    }

    Ok(())
}

/// Describes a [`RenderBundleEncoder`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderBundleEncoderDescriptor<'a> {
    /// Debug label of the render bundle encoder.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The formats of the color attachments that this render bundle is capable
    /// to rendering to.
    ///
    /// This must match the formats of the color attachments in the
    /// renderpass this render bundle is executed in.
    pub color_formats: Cow<'a, [Option<wgt::TextureFormat>]>,
    /// Information about the depth attachment that this render bundle is
    /// capable to rendering to.
    ///
    /// The format must match the format of the depth attachments in the
    /// renderpass this render bundle is executed in.
    pub depth_stencil: Option<wgt::RenderBundleDepthStencil>,
    /// Sample count this render bundle is capable of rendering to.
    ///
    /// This must match the pipelines and the renderpasses it is used in.
    pub sample_count: u32,
    /// If this render bundle will rendering to multiple array layers in the
    /// attachments at the same time.
    pub multiview: Option<NonZeroU32>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct RenderBundleEncoder {
    base: BasePass<RenderCommand>,
    parent_id: id::DeviceId,
    pub(crate) context: RenderPassContext,
    pub(crate) is_depth_read_only: bool,
    pub(crate) is_stencil_read_only: bool,

    // Resource binding dedupe state.
    #[cfg_attr(feature = "serde", serde(skip))]
    current_bind_groups: BindGroupStateChange,
    #[cfg_attr(feature = "serde", serde(skip))]
    current_pipeline: StateChange<id::RenderPipelineId>,
}

impl RenderBundleEncoder {
    pub fn new(
        desc: &RenderBundleEncoderDescriptor,
        parent_id: id::DeviceId,
        base: Option<BasePass<RenderCommand>>,
    ) -> Result<Self, CreateRenderBundleError> {
        let (is_depth_read_only, is_stencil_read_only) = match desc.depth_stencil {
            Some(ds) => {
                let aspects = hal::FormatAspects::from(ds.format);
                (
                    !aspects.contains(hal::FormatAspects::DEPTH) || ds.depth_read_only,
                    !aspects.contains(hal::FormatAspects::STENCIL) || ds.stencil_read_only,
                )
            }
            // There's no depth/stencil attachment, so these values just don't
            // matter.  Choose the most accommodating value, to simplify
            // validation.
            None => (true, true),
        };

        // TODO: should be device.limits.max_color_attachments
        let max_color_attachments = hal::MAX_COLOR_ATTACHMENTS;

        //TODO: validate that attachment formats are renderable,
        // have expected aspects, support multisampling.
        Ok(Self {
            base: base.unwrap_or_else(|| BasePass::new(&desc.label)),
            parent_id,
            context: RenderPassContext {
                attachments: AttachmentData {
                    colors: if desc.color_formats.len() > max_color_attachments {
                        return Err(CreateRenderBundleError::ColorAttachment(
                            ColorAttachmentError::TooMany {
                                given: desc.color_formats.len(),
                                limit: max_color_attachments,
                            },
                        ));
                    } else {
                        desc.color_formats.iter().cloned().collect()
                    },
                    resolves: ArrayVec::new(),
                    depth_stencil: desc.depth_stencil.map(|ds| ds.format),
                },
                sample_count: {
                    let sc = desc.sample_count;
                    if sc == 0 || sc > 32 || !sc.is_power_of_two() {
                        return Err(CreateRenderBundleError::InvalidSampleCount(sc));
                    }
                    sc
                },
                multiview: desc.multiview,
            },

            is_depth_read_only,
            is_stencil_read_only,
            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        })
    }

    pub fn dummy(parent_id: id::DeviceId) -> Self {
        Self {
            base: BasePass::new(&None),
            parent_id,
            context: RenderPassContext {
                attachments: AttachmentData {
                    colors: ArrayVec::new(),
                    resolves: ArrayVec::new(),
                    depth_stencil: None,
                },
                sample_count: 0,
                multiview: None,
            },
            is_depth_read_only: false,
            is_stencil_read_only: false,

            current_bind_groups: BindGroupStateChange::new(),
            current_pipeline: StateChange::new(),
        }
    }

    #[cfg(feature = "trace")]
    pub(crate) fn to_base_pass(&self) -> BasePass<RenderCommand> {
        self.base.clone()
    }

    pub fn parent(&self) -> id::DeviceId {
        self.parent_id
    }

    /// Convert this encoder's commands into a [`RenderBundle`].
    ///
    /// We want executing a [`RenderBundle`] to be quick, so we take
    /// this opportunity to clean up the [`RenderBundleEncoder`]'s
    /// command stream and gather metadata about it that will help
    /// keep [`ExecuteBundle`] simple and fast. We remove redundant
    /// commands (along with their side data), note resource usage,
    /// and accumulate buffer and texture initialization actions.
    ///
    /// [`ExecuteBundle`]: RenderCommand::ExecuteBundle
    pub(crate) fn finish(
        self,
        desc: &RenderBundleDescriptor,
        device: &Arc<Device>,
        hub: &Hub,
    ) -> Result<Arc<RenderBundle>, RenderBundleError> {
        let scope = PassErrorScope::Bundle;

        device.check_is_valid().map_pass_err(scope)?;

        let bind_group_guard = hub.bind_groups.read();
        let pipeline_guard = hub.render_pipelines.read();
        let buffer_guard = hub.buffers.read();

        let mut state = State {
            trackers: RenderBundleScope::new(),
            pipeline: None,
            bind: (0..hal::MAX_BIND_GROUPS).map(|_| None).collect(),
            vertex: (0..hal::MAX_VERTEX_BUFFERS).map(|_| None).collect(),
            index: None,
            flat_dynamic_offsets: Vec::new(),
            device: device.clone(),
            commands: Vec::new(),
            buffer_memory_init_actions: Vec::new(),
            texture_memory_init_actions: Vec::new(),
            next_dynamic_offset: 0,
        };

        let indices = &state.device.tracker_indices;
        state.trackers.buffers.set_size(indices.buffers.size());
        state.trackers.textures.set_size(indices.textures.size());

        let base = &self.base;

        for &command in &base.commands {
            match command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let scope = PassErrorScope::SetBindGroup;
                    set_bind_group(
                        &mut state,
                        &bind_group_guard,
                        &base.dynamic_offsets,
                        index,
                        num_dynamic_offsets,
                        bind_group_id,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let scope = PassErrorScope::SetPipelineRender;
                    set_pipeline(
                        &mut state,
                        &pipeline_guard,
                        &self.context,
                        self.is_depth_read_only,
                        self.is_stencil_read_only,
                        pipeline_id,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    index_format,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetIndexBuffer;
                    set_index_buffer(
                        &mut state,
                        &buffer_guard,
                        buffer_id,
                        index_format,
                        offset,
                        size,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetVertexBuffer;
                    set_vertex_buffer(&mut state, &buffer_guard, slot, buffer_id, offset, size)
                        .map_pass_err(scope)?;
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let scope = PassErrorScope::SetPushConstant;
                    set_push_constant(&mut state, stages, offset, size_bytes, values_offset)
                        .map_pass_err(scope)?;
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    let scope = PassErrorScope::Draw {
                        kind: DrawKind::Draw,
                        indexed: false,
                    };
                    draw(
                        &mut state,
                        &base.dynamic_offsets,
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    let scope = PassErrorScope::Draw {
                        kind: DrawKind::Draw,
                        indexed: true,
                    };
                    draw_indexed(
                        &mut state,
                        &base.dynamic_offsets,
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed,
                } => {
                    let scope = PassErrorScope::Draw {
                        kind: DrawKind::DrawIndirect,
                        indexed,
                    };
                    multi_draw_indirect(
                        &mut state,
                        &base.dynamic_offsets,
                        &buffer_guard,
                        buffer_id,
                        offset,
                        indexed,
                    )
                    .map_pass_err(scope)?;
                }
                RenderCommand::MultiDrawIndirect { .. }
                | RenderCommand::MultiDrawIndirectCount { .. } => unimplemented!(),
                RenderCommand::PushDebugGroup { color: _, len: _ } => unimplemented!(),
                RenderCommand::InsertDebugMarker { color: _, len: _ } => unimplemented!(),
                RenderCommand::PopDebugGroup => unimplemented!(),
                // Must check the TIMESTAMP_QUERY_INSIDE_PASSES feature
                RenderCommand::WriteTimestamp { .. }
                | RenderCommand::BeginOcclusionQuery { .. }
                | RenderCommand::EndOcclusionQuery
                | RenderCommand::BeginPipelineStatisticsQuery { .. }
                | RenderCommand::EndPipelineStatisticsQuery => unimplemented!(),
                RenderCommand::ExecuteBundle(_)
                | RenderCommand::SetBlendConstant(_)
                | RenderCommand::SetStencilReference(_)
                | RenderCommand::SetViewport { .. }
                | RenderCommand::SetScissor(_) => unreachable!("not supported by a render bundle"),
            }
        }

        let State {
            trackers,
            flat_dynamic_offsets,
            device,
            commands,
            buffer_memory_init_actions,
            texture_memory_init_actions,
            ..
        } = state;

        let tracker_indices = device.tracker_indices.bundles.clone();
        let discard_hal_labels = device
            .instance_flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS);

        let render_bundle = RenderBundle {
            base: BasePass {
                label: desc.label.as_ref().map(|cow| cow.to_string()),
                commands,
                dynamic_offsets: flat_dynamic_offsets,
                string_data: Vec::new(),
                push_constant_data: Vec::new(),
            },
            is_depth_read_only: self.is_depth_read_only,
            is_stencil_read_only: self.is_stencil_read_only,
            device: device.clone(),
            used: trackers,
            buffer_memory_init_actions,
            texture_memory_init_actions,
            context: self.context,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(tracker_indices),
            discard_hal_labels,
        };

        let render_bundle = Arc::new(render_bundle);

        Ok(render_bundle)
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        self.base.commands.push(RenderCommand::SetIndexBuffer {
            buffer_id,
            index_format,
            offset,
            size,
        });
    }
}

fn set_bind_group(
    state: &mut State,
    bind_group_guard: &crate::lock::RwLockReadGuard<crate::storage::Storage<BindGroup>>,
    dynamic_offsets: &[u32],
    index: u32,
    num_dynamic_offsets: usize,
    bind_group_id: id::Id<id::markers::BindGroup>,
) -> Result<(), RenderBundleErrorInner> {
    let bind_group = bind_group_guard
        .get_owned(bind_group_id)
        .map_err(|_| RenderCommandError::InvalidBindGroupId(bind_group_id))?;

    bind_group.same_device(&state.device)?;

    let max_bind_groups = state.device.limits.max_bind_groups;
    if index >= max_bind_groups {
        return Err(RenderCommandError::BindGroupIndexOutOfRange {
            index,
            max: max_bind_groups,
        }
        .into());
    }

    // Identify the next `num_dynamic_offsets` entries from `dynamic_offsets`.
    let offsets_range = state.next_dynamic_offset..state.next_dynamic_offset + num_dynamic_offsets;
    state.next_dynamic_offset = offsets_range.end;
    let offsets = &dynamic_offsets[offsets_range.clone()];

    bind_group.validate_dynamic_bindings(index, offsets)?;

    state
        .buffer_memory_init_actions
        .extend_from_slice(&bind_group.used_buffer_ranges);
    state
        .texture_memory_init_actions
        .extend_from_slice(&bind_group.used_texture_ranges);

    state.set_bind_group(index, &bind_group, offsets_range);
    unsafe { state.trackers.merge_bind_group(&bind_group.used)? };
    state.trackers.bind_groups.insert_single(bind_group);
    // Note: stateless trackers are not merged: the lifetime reference
    // is held to the bind group itself.
    Ok(())
}

fn set_pipeline(
    state: &mut State,
    pipeline_guard: &crate::lock::RwLockReadGuard<crate::storage::Storage<RenderPipeline>>,
    context: &RenderPassContext,
    is_depth_read_only: bool,
    is_stencil_read_only: bool,
    pipeline_id: id::Id<id::markers::RenderPipeline>,
) -> Result<(), RenderBundleErrorInner> {
    let pipeline = pipeline_guard
        .get_owned(pipeline_id)
        .map_err(|_| RenderCommandError::InvalidPipelineId(pipeline_id))?;

    pipeline.same_device(&state.device)?;

    context
        .check_compatible(&pipeline.pass_context, pipeline.as_ref())
        .map_err(RenderCommandError::IncompatiblePipelineTargets)?;

    if pipeline.flags.contains(PipelineFlags::WRITES_DEPTH) && is_depth_read_only {
        return Err(RenderCommandError::IncompatibleDepthAccess(pipeline.error_ident()).into());
    }
    if pipeline.flags.contains(PipelineFlags::WRITES_STENCIL) && is_stencil_read_only {
        return Err(RenderCommandError::IncompatibleStencilAccess(pipeline.error_ident()).into());
    }

    let pipeline_state = PipelineState::new(&pipeline);

    state
        .commands
        .push(ArcRenderCommand::SetPipeline(pipeline.clone()));

    // If this pipeline uses push constants, zero out their values.
    if let Some(iter) = pipeline_state.zero_push_constants() {
        state.commands.extend(iter)
    }

    state.invalidate_bind_groups(&pipeline_state, &pipeline.layout);
    state.pipeline = Some(pipeline_state);

    state.trackers.render_pipelines.insert_single(pipeline);
    Ok(())
}

fn set_index_buffer(
    state: &mut State,
    buffer_guard: &crate::lock::RwLockReadGuard<crate::storage::Storage<Buffer>>,
    buffer_id: id::Id<id::markers::Buffer>,
    index_format: wgt::IndexFormat,
    offset: u64,
    size: Option<std::num::NonZeroU64>,
) -> Result<(), RenderBundleErrorInner> {
    let buffer = buffer_guard
        .get_owned(buffer_id)
        .map_err(|_| RenderCommandError::InvalidBufferId(buffer_id))?;

    state
        .trackers
        .buffers
        .merge_single(&buffer, hal::BufferUses::INDEX)?;

    buffer.same_device(&state.device)?;
    buffer.check_usage(wgt::BufferUsages::INDEX)?;

    let end = match size {
        Some(s) => offset + s.get(),
        None => buffer.size,
    };
    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..end,
            MemoryInitKind::NeedsInitializedMemory,
        ));
    state.set_index_buffer(buffer, index_format, offset..end);
    Ok(())
}

fn set_vertex_buffer(
    state: &mut State,
    buffer_guard: &crate::lock::RwLockReadGuard<crate::storage::Storage<Buffer>>,
    slot: u32,
    buffer_id: id::Id<id::markers::Buffer>,
    offset: u64,
    size: Option<std::num::NonZeroU64>,
) -> Result<(), RenderBundleErrorInner> {
    let max_vertex_buffers = state.device.limits.max_vertex_buffers;
    if slot >= max_vertex_buffers {
        return Err(RenderCommandError::VertexBufferIndexOutOfRange {
            index: slot,
            max: max_vertex_buffers,
        }
        .into());
    }

    let buffer = buffer_guard
        .get_owned(buffer_id)
        .map_err(|_| RenderCommandError::InvalidBufferId(buffer_id))?;

    state
        .trackers
        .buffers
        .merge_single(&buffer, hal::BufferUses::VERTEX)?;

    buffer.same_device(&state.device)?;
    buffer.check_usage(wgt::BufferUsages::VERTEX)?;

    let end = match size {
        Some(s) => offset + s.get(),
        None => buffer.size,
    };
    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..end,
            MemoryInitKind::NeedsInitializedMemory,
        ));
    state.vertex[slot as usize] = Some(VertexState::new(buffer, offset..end));
    Ok(())
}

fn set_push_constant(
    state: &mut State,
    stages: wgt::ShaderStages,
    offset: u32,
    size_bytes: u32,
    values_offset: Option<u32>,
) -> Result<(), RenderBundleErrorInner> {
    let end_offset = offset + size_bytes;

    let pipeline_state = state.pipeline()?;

    pipeline_state
        .pipeline
        .layout
        .validate_push_constant_ranges(stages, offset, end_offset)?;

    state.commands.push(ArcRenderCommand::SetPushConstant {
        stages,
        offset,
        size_bytes,
        values_offset,
    });
    Ok(())
}

fn draw(
    state: &mut State,
    dynamic_offsets: &[u32],
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) -> Result<(), RenderBundleErrorInner> {
    let pipeline = state.pipeline()?;
    let used_bind_groups = pipeline.used_bind_groups;

    validate_draw(
        &state.vertex[..],
        &pipeline.steps,
        first_vertex,
        vertex_count,
        first_instance,
        instance_count,
    )?;

    if instance_count > 0 && vertex_count > 0 {
        state.flush_vertices();
        state.flush_binds(used_bind_groups, dynamic_offsets);
        state.commands.push(ArcRenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }
    Ok(())
}

fn draw_indexed(
    state: &mut State,
    dynamic_offsets: &[u32],
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) -> Result<(), RenderBundleErrorInner> {
    let pipeline = state.pipeline()?;
    let used_bind_groups = pipeline.used_bind_groups;
    let index = match state.index {
        Some(ref index) => index,
        None => return Err(DrawError::MissingIndexBuffer.into()),
    };

    validate_indexed_draw(
        &state.vertex[..],
        &pipeline.steps,
        index,
        first_index,
        index_count,
        first_instance,
        instance_count,
    )?;

    if instance_count > 0 && index_count > 0 {
        state.flush_index();
        state.flush_vertices();
        state.flush_binds(used_bind_groups, dynamic_offsets);
        state.commands.push(ArcRenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }
    Ok(())
}

fn multi_draw_indirect(
    state: &mut State,
    dynamic_offsets: &[u32],
    buffer_guard: &crate::lock::RwLockReadGuard<crate::storage::Storage<Buffer>>,
    buffer_id: id::Id<id::markers::Buffer>,
    offset: u64,
    indexed: bool,
) -> Result<(), RenderBundleErrorInner> {
    state
        .device
        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;

    let pipeline = state.pipeline()?;
    let used_bind_groups = pipeline.used_bind_groups;

    let buffer = buffer_guard
        .get_owned(buffer_id)
        .map_err(|_| RenderCommandError::InvalidBufferId(buffer_id))?;

    state
        .trackers
        .buffers
        .merge_single(&buffer, hal::BufferUses::INDIRECT)?;

    buffer.same_device(&state.device)?;
    buffer.check_usage(wgt::BufferUsages::INDIRECT)?;

    state
        .buffer_memory_init_actions
        .extend(buffer.initialization_status.read().create_action(
            &buffer,
            offset..(offset + size_of::<wgt::DrawIndirectArgs>() as u64),
            MemoryInitKind::NeedsInitializedMemory,
        ));

    if indexed {
        let index = match state.index {
            Some(ref mut index) => index,
            None => return Err(DrawError::MissingIndexBuffer.into()),
        };
        state.commands.extend(index.flush());
    }

    state.flush_vertices();
    state.flush_binds(used_bind_groups, dynamic_offsets);
    state.commands.push(ArcRenderCommand::MultiDrawIndirect {
        buffer,
        offset,
        count: None,
        indexed,
    });
    Ok(())
}

/// Error type returned from `RenderBundleEncoder::new` if the sample count is invalid.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateRenderBundleError {
    #[error(transparent)]
    ColorAttachment(#[from] ColorAttachmentError),
    #[error("Invalid number of samples {0}")]
    InvalidSampleCount(u32),
}

/// Error type returned from `RenderBundleEncoder::new` if the sample count is invalid.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ExecutionError {
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Using {0} in a render bundle is not implemented")]
    Unimplemented(&'static str),
}

pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;

//Note: here, `RenderBundle` is just wrapping a raw stream of render commands.
// The plan is to back it by an actual Vulkan secondary buffer, D3D12 Bundle,
// or Metal indirect command buffer.
#[derive(Debug)]
pub struct RenderBundle {
    // Normalized command stream. It can be executed verbatim,
    // without re-binding anything on the pipeline change.
    base: BasePass<ArcRenderCommand>,
    pub(super) is_depth_read_only: bool,
    pub(super) is_stencil_read_only: bool,
    pub(crate) device: Arc<Device>,
    pub(crate) used: RenderBundleScope,
    pub(super) buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    pub(super) texture_memory_init_actions: Vec<TextureInitTrackerAction>,
    pub(super) context: RenderPassContext,
    /// The `label` from the descriptor used to create the resource.
    label: String,
    pub(crate) tracking_data: TrackingData,
    discard_hal_labels: bool,
}

impl Drop for RenderBundle {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());
    }
}

#[cfg(send_sync)]
unsafe impl Send for RenderBundle {}
#[cfg(send_sync)]
unsafe impl Sync for RenderBundle {}

impl RenderBundle {
    /// Actually encode the contents into a native command buffer.
    ///
    /// This is partially duplicating the logic of `render_pass_end`.
    /// However the point of this function is to be lighter, since we already had
    /// a chance to go through the commands in `render_bundle_encoder_finish`.
    ///
    /// Note that the function isn't expected to fail, generally.
    /// All the validation has already been done by this point.
    /// The only failure condition is if some of the used buffers are destroyed.
    pub(super) unsafe fn execute(
        &self,
        raw: &mut dyn hal::DynCommandEncoder,
        snatch_guard: &SnatchGuard,
    ) -> Result<(), ExecutionError> {
        let mut offsets = self.base.dynamic_offsets.as_slice();
        let mut pipeline_layout = None::<Arc<PipelineLayout>>;
        if !self.discard_hal_labels {
            if let Some(ref label) = self.base.label {
                unsafe { raw.begin_debug_marker(label) };
            }
        }

        use ArcRenderCommand as Cmd;
        for command in self.base.commands.iter() {
            match command {
                Cmd::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group,
                } => {
                    let raw_bg = bind_group.try_raw(snatch_guard)?;
                    unsafe {
                        raw.set_bind_group(
                            pipeline_layout.as_ref().unwrap().raw(),
                            *index,
                            raw_bg,
                            &offsets[..*num_dynamic_offsets],
                        )
                    };
                    offsets = &offsets[*num_dynamic_offsets..];
                }
                Cmd::SetPipeline(pipeline) => {
                    unsafe { raw.set_render_pipeline(pipeline.raw()) };

                    pipeline_layout = Some(pipeline.layout.clone());
                }
                Cmd::SetIndexBuffer {
                    buffer,
                    index_format,
                    offset,
                    size,
                } => {
                    let buffer = buffer.try_raw(snatch_guard)?;
                    let bb = hal::BufferBinding {
                        buffer,
                        offset: *offset,
                        size: *size,
                    };
                    unsafe { raw.set_index_buffer(bb, *index_format) };
                }
                Cmd::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                    size,
                } => {
                    let buffer = buffer.try_raw(snatch_guard)?;
                    let bb = hal::BufferBinding {
                        buffer,
                        offset: *offset,
                        size: *size,
                    };
                    unsafe { raw.set_vertex_buffer(*slot, bb) };
                }
                Cmd::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let pipeline_layout = pipeline_layout.as_ref().unwrap();

                    if let Some(values_offset) = *values_offset {
                        let values_end_offset =
                            (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                        let data_slice = &self.base.push_constant_data
                            [(values_offset as usize)..values_end_offset];

                        unsafe {
                            raw.set_push_constants(
                                pipeline_layout.raw(),
                                *stages,
                                *offset,
                                data_slice,
                            )
                        }
                    } else {
                        super::push_constant_clear(
                            *offset,
                            *size_bytes,
                            |clear_offset, clear_data| {
                                unsafe {
                                    raw.set_push_constants(
                                        pipeline_layout.raw(),
                                        *stages,
                                        clear_offset,
                                        clear_data,
                                    )
                                };
                            },
                        );
                    }
                }
                Cmd::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    unsafe {
                        raw.draw(
                            *first_vertex,
                            *vertex_count,
                            *first_instance,
                            *instance_count,
                        )
                    };
                }
                Cmd::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    unsafe {
                        raw.draw_indexed(
                            *first_index,
                            *index_count,
                            *base_vertex,
                            *first_instance,
                            *instance_count,
                        )
                    };
                }
                Cmd::MultiDrawIndirect {
                    buffer,
                    offset,
                    count: None,
                    indexed: false,
                } => {
                    let buffer = buffer.try_raw(snatch_guard)?;
                    unsafe { raw.draw_indirect(buffer, *offset, 1) };
                }
                Cmd::MultiDrawIndirect {
                    buffer,
                    offset,
                    count: None,
                    indexed: true,
                } => {
                    let buffer = buffer.try_raw(snatch_guard)?;
                    unsafe { raw.draw_indexed_indirect(buffer, *offset, 1) };
                }
                Cmd::MultiDrawIndirect { .. } | Cmd::MultiDrawIndirectCount { .. } => {
                    return Err(ExecutionError::Unimplemented("multi-draw-indirect"))
                }
                Cmd::PushDebugGroup { .. } | Cmd::InsertDebugMarker { .. } | Cmd::PopDebugGroup => {
                    return Err(ExecutionError::Unimplemented("debug-markers"))
                }
                Cmd::WriteTimestamp { .. }
                | Cmd::BeginOcclusionQuery { .. }
                | Cmd::EndOcclusionQuery
                | Cmd::BeginPipelineStatisticsQuery { .. }
                | Cmd::EndPipelineStatisticsQuery => {
                    return Err(ExecutionError::Unimplemented("queries"))
                }
                Cmd::ExecuteBundle(_)
                | Cmd::SetBlendConstant(_)
                | Cmd::SetStencilReference(_)
                | Cmd::SetViewport { .. }
                | Cmd::SetScissor(_) => unreachable!(),
            }
        }

        if !self.discard_hal_labels {
            if let Some(_) = self.base.label {
                unsafe { raw.end_debug_marker() };
            }
        }

        Ok(())
    }
}

crate::impl_resource_type!(RenderBundle);
crate::impl_labeled!(RenderBundle);
crate::impl_parent_device!(RenderBundle);
crate::impl_storage_item!(RenderBundle);
crate::impl_trackable!(RenderBundle);

/// A render bundle's current index buffer state.
///
/// [`RenderBundleEncoder::finish`] records the currently set index buffer here,
/// and calls [`State::flush_index`] before any indexed draw command to produce
/// a `SetIndexBuffer` command if one is necessary.
#[derive(Debug)]
struct IndexState {
    buffer: Arc<Buffer>,
    format: wgt::IndexFormat,
    range: Range<wgt::BufferAddress>,
    is_dirty: bool,
}

impl IndexState {
    /// Return the number of entries in the current index buffer.
    ///
    /// Panic if no index buffer has been set.
    fn limit(&self) -> u64 {
        let bytes_per_index = match self.format {
            wgt::IndexFormat::Uint16 => 2,
            wgt::IndexFormat::Uint32 => 4,
        };

        (self.range.end - self.range.start) / bytes_per_index
    }

    /// Generate a `SetIndexBuffer` command to prepare for an indexed draw
    /// command, if needed.
    fn flush(&mut self) -> Option<ArcRenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(ArcRenderCommand::SetIndexBuffer {
                buffer: self.buffer.clone(),
                index_format: self.format,
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }
}

/// The state of a single vertex buffer slot during render bundle encoding.
///
/// [`RenderBundleEncoder::finish`] uses this to drop redundant
/// `SetVertexBuffer` commands from the final [`RenderBundle`]. It
/// records one vertex buffer slot's state changes here, and then
/// calls this type's [`flush`] method just before any draw command to
/// produce a `SetVertexBuffer` commands if one is necessary.
///
/// [`flush`]: IndexState::flush
#[derive(Debug)]
struct VertexState {
    buffer: Arc<Buffer>,
    range: Range<wgt::BufferAddress>,
    is_dirty: bool,
}

impl VertexState {
    fn new(buffer: Arc<Buffer>, range: Range<wgt::BufferAddress>) -> Self {
        Self {
            buffer,
            range,
            is_dirty: true,
        }
    }

    /// Generate a `SetVertexBuffer` command for this slot, if necessary.
    ///
    /// `slot` is the index of the vertex buffer slot that `self` tracks.
    fn flush(&mut self, slot: u32) -> Option<ArcRenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(ArcRenderCommand::SetVertexBuffer {
                slot,
                buffer: self.buffer.clone(),
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }
}

/// A bind group that has been set at a particular index during render bundle encoding.
#[derive(Debug)]
struct BindState {
    /// The id of the bind group set at this index.
    bind_group: Arc<BindGroup>,

    /// The range of dynamic offsets for this bind group, in the original
    /// command stream's `BassPass::dynamic_offsets` array.
    dynamic_offsets: Range<usize>,

    /// True if this index's contents have been changed since the last time we
    /// generated a `SetBindGroup` command.
    is_dirty: bool,
}

/// The bundle's current pipeline, and some cached information needed for validation.
struct PipelineState {
    /// The pipeline
    pipeline: Arc<RenderPipeline>,

    /// How this pipeline's vertex shader traverses each vertex buffer, indexed
    /// by vertex buffer slot number.
    steps: Vec<VertexStep>,

    /// Ranges of push constants this pipeline uses, copied from the pipeline
    /// layout.
    push_constant_ranges: ArrayVec<wgt::PushConstantRange, { SHADER_STAGE_COUNT }>,

    /// The number of bind groups this pipeline uses.
    used_bind_groups: usize,
}

impl PipelineState {
    fn new(pipeline: &Arc<RenderPipeline>) -> Self {
        Self {
            pipeline: pipeline.clone(),
            steps: pipeline.vertex_steps.to_vec(),
            push_constant_ranges: pipeline
                .layout
                .push_constant_ranges
                .iter()
                .cloned()
                .collect(),
            used_bind_groups: pipeline.layout.bind_group_layouts.len(),
        }
    }

    /// Return a sequence of commands to zero the push constant ranges this
    /// pipeline uses. If no initialization is necessary, return `None`.
    fn zero_push_constants(&self) -> Option<impl Iterator<Item = ArcRenderCommand>> {
        if !self.push_constant_ranges.is_empty() {
            let nonoverlapping_ranges =
                super::bind::compute_nonoverlapping_ranges(&self.push_constant_ranges);

            Some(
                nonoverlapping_ranges
                    .into_iter()
                    .map(|range| ArcRenderCommand::SetPushConstant {
                        stages: range.stages,
                        offset: range.range.start,
                        size_bytes: range.range.end - range.range.start,
                        values_offset: None, // write zeros
                    }),
            )
        } else {
            None
        }
    }
}

/// State for analyzing and cleaning up bundle command streams.
///
/// To minimize state updates, [`RenderBundleEncoder::finish`]
/// actually just applies commands like [`SetBindGroup`] and
/// [`SetIndexBuffer`] to the simulated state stored here, and then
/// calls the `flush_foo` methods before draw calls to produce the
/// update commands we actually need.
///
/// [`SetBindGroup`]: RenderCommand::SetBindGroup
/// [`SetIndexBuffer`]: RenderCommand::SetIndexBuffer
struct State {
    /// Resources used by this bundle. This will become [`RenderBundle::used`].
    trackers: RenderBundleScope,

    /// The currently set pipeline, if any.
    pipeline: Option<PipelineState>,

    /// The bind group set at each index, if any.
    bind: ArrayVec<Option<BindState>, { hal::MAX_BIND_GROUPS }>,

    /// The state of each vertex buffer slot.
    vertex: ArrayVec<Option<VertexState>, { hal::MAX_VERTEX_BUFFERS }>,

    /// The current index buffer, if one has been set. We flush this state
    /// before indexed draw commands.
    index: Option<IndexState>,

    /// Dynamic offset values used by the cleaned-up command sequence.
    ///
    /// This becomes the final [`RenderBundle`]'s [`BasePass`]'s
    /// [`dynamic_offsets`] list.
    ///
    /// [`dynamic_offsets`]: BasePass::dynamic_offsets
    flat_dynamic_offsets: Vec<wgt::DynamicOffset>,

    device: Arc<Device>,
    commands: Vec<ArcRenderCommand>,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_init_actions: Vec<TextureInitTrackerAction>,
    next_dynamic_offset: usize,
}

impl State {
    /// Return the current pipeline state. Return an error if none is set.
    fn pipeline(&self) -> Result<&PipelineState, RenderBundleErrorInner> {
        self.pipeline
            .as_ref()
            .ok_or(DrawError::MissingPipeline.into())
    }

    /// Mark all non-empty bind group table entries from `index` onwards as dirty.
    fn invalidate_bind_group_from(&mut self, index: usize) {
        for contents in self.bind[index..].iter_mut().flatten() {
            contents.is_dirty = true;
        }
    }

    fn set_bind_group(
        &mut self,
        slot: u32,
        bind_group: &Arc<BindGroup>,
        dynamic_offsets: Range<usize>,
    ) {
        // If this call wouldn't actually change this index's state, we can
        // return early.  (If there are dynamic offsets, the range will always
        // be different.)
        if dynamic_offsets.is_empty() {
            if let Some(ref contents) = self.bind[slot as usize] {
                if contents.bind_group.is_equal(bind_group) {
                    return;
                }
            }
        }

        // Record the index's new state.
        self.bind[slot as usize] = Some(BindState {
            bind_group: bind_group.clone(),
            dynamic_offsets,
            is_dirty: true,
        });

        // Once we've changed the bind group at a particular index, all
        // subsequent indices need to be rewritten.
        self.invalidate_bind_group_from(slot as usize + 1);
    }

    /// Determine which bind group slots need to be re-set after a pipeline change.
    ///
    /// Given that we are switching from the current pipeline state to `new`,
    /// whose layout is `layout`, mark all the bind group slots that we need to
    /// emit new `SetBindGroup` commands for as dirty.
    ///
    /// According to `wgpu_hal`'s rules:
    ///
    /// - If the layout of any bind group slot changes, then that slot and
    ///   all following slots must have their bind groups re-established.
    ///
    /// - Changing the push constant ranges at all requires re-establishing
    ///   all bind groups.
    fn invalidate_bind_groups(&mut self, new: &PipelineState, layout: &PipelineLayout) {
        match self.pipeline {
            None => {
                // Establishing entirely new pipeline state.
                self.invalidate_bind_group_from(0);
            }
            Some(ref old) => {
                if old.pipeline.is_equal(&new.pipeline) {
                    // Everything is derived from the pipeline, so if the id has
                    // not changed, there's no need to consider anything else.
                    return;
                }

                // Any push constant change invalidates all groups.
                if old.push_constant_ranges != new.push_constant_ranges {
                    self.invalidate_bind_group_from(0);
                } else {
                    let first_changed = self.bind.iter().zip(&layout.bind_group_layouts).position(
                        |(entry, layout)| match *entry {
                            Some(ref contents) => !contents.bind_group.layout.is_equal(layout),
                            None => false,
                        },
                    );
                    if let Some(slot) = first_changed {
                        self.invalidate_bind_group_from(slot);
                    }
                }
            }
        }
    }

    /// Set the bundle's current index buffer and its associated parameters.
    fn set_index_buffer(
        &mut self,
        buffer: Arc<Buffer>,
        format: wgt::IndexFormat,
        range: Range<wgt::BufferAddress>,
    ) {
        match self.index {
            Some(ref current)
                if current.buffer.is_equal(&buffer)
                    && current.format == format
                    && current.range == range =>
            {
                return
            }
            _ => (),
        }

        self.index = Some(IndexState {
            buffer,
            format,
            range,
            is_dirty: true,
        });
    }

    /// Generate a `SetIndexBuffer` command to prepare for an indexed draw
    /// command, if needed.
    fn flush_index(&mut self) {
        let commands = self.index.as_mut().and_then(|index| index.flush());
        self.commands.extend(commands);
    }

    fn flush_vertices(&mut self) {
        let commands = self
            .vertex
            .iter_mut()
            .enumerate()
            .flat_map(|(i, vs)| vs.as_mut().and_then(|vs| vs.flush(i as u32)));
        self.commands.extend(commands);
    }

    /// Generate `SetBindGroup` commands for any bind groups that need to be updated.
    fn flush_binds(&mut self, used_bind_groups: usize, dynamic_offsets: &[wgt::DynamicOffset]) {
        // Append each dirty bind group's dynamic offsets to `flat_dynamic_offsets`.
        for contents in self.bind[..used_bind_groups].iter().flatten() {
            if contents.is_dirty {
                self.flat_dynamic_offsets
                    .extend_from_slice(&dynamic_offsets[contents.dynamic_offsets.clone()]);
            }
        }

        // Then, generate `SetBindGroup` commands to update the dirty bind
        // groups. After this, all bind groups are clean.
        let commands = self.bind[..used_bind_groups]
            .iter_mut()
            .enumerate()
            .flat_map(|(i, entry)| {
                if let Some(ref mut contents) = *entry {
                    if contents.is_dirty {
                        contents.is_dirty = false;
                        let offsets = &contents.dynamic_offsets;
                        return Some(ArcRenderCommand::SetBindGroup {
                            index: i.try_into().unwrap(),
                            bind_group: contents.bind_group.clone(),
                            num_dynamic_offsets: offsets.end - offsets.start,
                        });
                    }
                }
                None
            });

        self.commands.extend(commands);
    }
}

/// Error encountered when finishing recording a render bundle.
#[derive(Clone, Debug, Error)]
pub(super) enum RenderBundleErrorInner {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    RenderCommand(RenderCommandError),
    #[error(transparent)]
    Draw(#[from] DrawError),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error(transparent)]
    Bind(#[from] BindError),
}

impl<T> From<T> for RenderBundleErrorInner
where
    T: Into<RenderCommandError>,
{
    fn from(t: T) -> Self {
        Self::RenderCommand(t.into())
    }
}

/// Error encountered when finishing recording a render bundle.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct RenderBundleError {
    pub scope: PassErrorScope,
    #[source]
    inner: RenderBundleErrorInner,
}

impl RenderBundleError {
    pub fn from_device_error(e: DeviceError) -> Self {
        Self {
            scope: PassErrorScope::Bundle,
            inner: e.into(),
        }
    }
}

impl<T, E> MapPassErr<T, RenderBundleError> for Result<T, E>
where
    E: Into<RenderBundleErrorInner>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, RenderBundleError> {
        self.map_err(|inner| RenderBundleError {
            scope,
            inner: inner.into(),
        })
    }
}

pub mod bundle_ffi {
    use super::{RenderBundleEncoder, RenderCommand};
    use crate::{id, RawString};
    use std::{convert::TryInto, slice};
    use wgt::{BufferAddress, BufferSize, DynamicOffset, IndexFormat};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_bind_group(
        bundle: &mut RenderBundleEncoder,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        let offsets = unsafe { slice::from_raw_parts(offsets, offset_length) };

        let redundant = bundle.current_bind_groups.set_and_check_redundant(
            bind_group_id,
            index,
            &mut bundle.base.dynamic_offsets,
            offsets,
        );

        if redundant {
            return;
        }

        bundle.base.commands.push(RenderCommand::SetBindGroup {
            index,
            num_dynamic_offsets: offset_length,
            bind_group_id,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_pipeline(
        bundle: &mut RenderBundleEncoder,
        pipeline_id: id::RenderPipelineId,
    ) {
        if bundle.current_pipeline.set_and_check_redundant(pipeline_id) {
            return;
        }

        bundle
            .base
            .commands
            .push(RenderCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_vertex_buffer(
        bundle: &mut RenderBundleEncoder,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        bundle.base.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_index_buffer(
        encoder: &mut RenderBundleEncoder,
        buffer: id::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        encoder.set_index_buffer(buffer, index_format, offset, size);
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `data` elements.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_push_constants(
        pass: &mut RenderBundleEncoder,
        stages: wgt::ShaderStages,
        offset: u32,
        size_bytes: u32,
        data: *const u8,
    ) {
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
        let data_slice = unsafe { slice::from_raw_parts(data, size_bytes as usize) };
        let value_offset = pass.base.push_constant_data.len().try_into().expect(
            "Ran out of push constant space. Don't set 4gb of push constants per RenderBundle.",
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
    pub extern "C" fn wgpu_render_bundle_draw(
        bundle: &mut RenderBundleEncoder,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        bundle.base.commands.push(RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw_indexed(
        bundle: &mut RenderBundleEncoder,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        bundle.base.commands.push(RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw_indirect(
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        bundle.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: None,
            indexed: false,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw_indexed_indirect(
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        bundle.base.commands.push(RenderCommand::MultiDrawIndirect {
            buffer_id,
            offset,
            count: None,
            indexed: true,
        });
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given `label`
    /// is a valid null-terminated string.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_push_debug_group(
        _bundle: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_pop_debug_group(_bundle: &mut RenderBundleEncoder) {
        //TODO
    }

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given `label`
    /// is a valid null-terminated string.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_insert_debug_marker(
        _bundle: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        //TODO
    }
}
