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

4) Then, any number of times, call [`wgpu_render_pass_execute_bundles`][wrpeb] to
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
[wrpeb]: crate::command::render_ffi::wgpu_render_pass_execute_bundles
!*/

#![allow(clippy::reversed_empty_ranges)]

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{buffer_binding_type_alignment, BindGroup, BindGroupLayout, PipelineLayout},
    command::{
        BasePass, BindGroupStateChange, ColorAttachmentError, DrawError, MapPassErr,
        PassErrorScope, RenderCommand, RenderCommandError, StateChange,
    },
    conv,
    device::{
        AttachmentData, Device, DeviceError, MissingDownlevelFlags,
        RenderPassCompatibilityCheckType, RenderPassContext, SHADER_STAGE_COUNT,
    },
    error::{ErrorFormatter, PrettyError},
    hal_api::HalApi,
    hub::Hub,
    id,
    init_tracker::{BufferInitTrackerAction, MemoryInitKind, TextureInitTrackerAction},
    pipeline::{PipelineFlags, RenderPipeline, VertexStep},
    resource::{Resource, ResourceInfo, ResourceType},
    resource_log,
    track::RenderBundleScope,
    validation::check_buffer_usage,
    Label, LabelHelpers,
};
use arrayvec::ArrayVec;

use std::{borrow::Cow, mem, num::NonZeroU32, ops::Range, sync::Arc};
use thiserror::Error;

use hal::CommandEncoder as _;

/// https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-draw
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
    if last_index <= index_limit {
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

        //TODO: validate that attachment formats are renderable,
        // have expected aspects, support multisampling.
        Ok(Self {
            base: base.unwrap_or_else(|| BasePass::new(&desc.label)),
            parent_id,
            context: RenderPassContext {
                attachments: AttachmentData {
                    colors: if desc.color_formats.len() > hal::MAX_COLOR_ATTACHMENTS {
                        return Err(CreateRenderBundleError::ColorAttachment(
                            ColorAttachmentError::TooMany {
                                given: desc.color_formats.len(),
                                limit: hal::MAX_COLOR_ATTACHMENTS,
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
                    if sc == 0 || sc > 32 || !conv::is_power_of_two_u32(sc) {
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
        BasePass::from_ref(self.base.as_ref())
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
    pub(crate) fn finish<A: HalApi>(
        self,
        desc: &RenderBundleDescriptor,
        device: &Arc<Device<A>>,
        hub: &Hub<A>,
    ) -> Result<RenderBundle<A>, RenderBundleError> {
        let bind_group_guard = hub.bind_groups.read();
        let pipeline_guard = hub.render_pipelines.read();
        let query_set_guard = hub.query_sets.read();
        let buffer_guard = hub.buffers.read();
        let texture_guard = hub.textures.read();

        let mut state = State {
            trackers: RenderBundleScope::new(
                &*buffer_guard,
                &*texture_guard,
                &*bind_group_guard,
                &*pipeline_guard,
                &*query_set_guard,
            ),
            pipeline: None,
            bind: (0..hal::MAX_BIND_GROUPS).map(|_| None).collect(),
            vertex: (0..hal::MAX_VERTEX_BUFFERS).map(|_| None).collect(),
            index: None,
            flat_dynamic_offsets: Vec::new(),
        };
        let mut commands = Vec::new();
        let mut buffer_memory_init_actions = Vec::new();
        let mut texture_memory_init_actions = Vec::new();

        let base = self.base.as_ref();
        let mut next_dynamic_offset = 0;

        for &command in base.commands {
            match command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let scope = PassErrorScope::SetBindGroup(bind_group_id);

                    let bind_group = state
                        .trackers
                        .bind_groups
                        .write()
                        .add_single(&*bind_group_guard, bind_group_id)
                        .ok_or(RenderCommandError::InvalidBindGroup(bind_group_id))
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(bind_group.device.info.id())
                        .map_pass_err(scope)?;

                    let max_bind_groups = device.limits.max_bind_groups;
                    if index >= max_bind_groups {
                        return Err(RenderCommandError::BindGroupIndexOutOfRange {
                            index,
                            max: max_bind_groups,
                        })
                        .map_pass_err(scope);
                    }

                    // Identify the next `num_dynamic_offsets` entries from `base.dynamic_offsets`.
                    let num_dynamic_offsets = num_dynamic_offsets;
                    let offsets_range =
                        next_dynamic_offset..next_dynamic_offset + num_dynamic_offsets;
                    next_dynamic_offset = offsets_range.end;
                    let offsets = &base.dynamic_offsets[offsets_range.clone()];

                    if bind_group.dynamic_binding_info.len() != offsets.len() {
                        return Err(RenderCommandError::InvalidDynamicOffsetCount {
                            actual: offsets.len(),
                            expected: bind_group.dynamic_binding_info.len(),
                        })
                        .map_pass_err(scope);
                    }

                    // Check for misaligned offsets.
                    for (offset, info) in offsets
                        .iter()
                        .map(|offset| *offset as wgt::BufferAddress)
                        .zip(bind_group.dynamic_binding_info.iter())
                    {
                        let (alignment, limit_name) =
                            buffer_binding_type_alignment(&device.limits, info.binding_type);
                        if offset % alignment as u64 != 0 {
                            return Err(RenderCommandError::UnalignedBufferOffset(
                                offset, limit_name, alignment,
                            ))
                            .map_pass_err(scope);
                        }
                    }

                    buffer_memory_init_actions.extend_from_slice(&bind_group.used_buffer_ranges);
                    texture_memory_init_actions.extend_from_slice(&bind_group.used_texture_ranges);

                    state.set_bind_group(index, bind_group_guard.get(bind_group_id).as_ref().unwrap(), &bind_group.layout, offsets_range);
                    unsafe {
                        state
                            .trackers
                            .merge_bind_group(&bind_group.used)
                            .map_pass_err(scope)?
                    };
                    //Note: stateless trackers are not merged: the lifetime reference
                    // is held to the bind group itself.
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let scope = PassErrorScope::SetPipelineRender(pipeline_id);

                    let pipeline = state
                        .trackers
                        .render_pipelines
                        .write()
                        .add_single(&*pipeline_guard, pipeline_id)
                        .ok_or(RenderCommandError::InvalidPipeline(pipeline_id))
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(pipeline.device.info.id())
                        .map_pass_err(scope)?;

                    self.context
                        .check_compatible(&pipeline.pass_context, RenderPassCompatibilityCheckType::RenderPipeline)
                        .map_err(RenderCommandError::IncompatiblePipelineTargets)
                        .map_pass_err(scope)?;

                    if (pipeline.flags.contains(PipelineFlags::WRITES_DEPTH)
                        && self.is_depth_read_only)
                        || (pipeline.flags.contains(PipelineFlags::WRITES_STENCIL)
                            && self.is_stencil_read_only)
                    {
                        return Err(RenderCommandError::IncompatiblePipelineRods)
                            .map_pass_err(scope);
                    }

                    let pipeline_state = PipelineState::new(pipeline);

                    commands.push(command);

                    // If this pipeline uses push constants, zero out their values.
                    if let Some(iter) = pipeline_state.zero_push_constants() {
                        commands.extend(iter)
                    }

                    state.invalidate_bind_groups(&pipeline_state, &pipeline.layout);
                    state.pipeline = Some(pipeline_state);
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    index_format,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetIndexBuffer(buffer_id);
                    let buffer = state
                        .trackers
                        .buffers
                        .write()
                        .merge_single(&*buffer_guard, buffer_id, hal::BufferUses::INDEX)
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(buffer.device.info.id())
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, wgt::BufferUsages::INDEX)
                        .map_pass_err(scope)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    buffer_memory_init_actions.extend(buffer.initialization_status.read().create_action(
                        buffer,
                        offset..end,
                        MemoryInitKind::NeedsInitializedMemory,
                    ));
                    state.set_index_buffer(buffer_id, index_format, offset..end);
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetVertexBuffer(buffer_id);

                    let max_vertex_buffers = device.limits.max_vertex_buffers;
                    if slot >= max_vertex_buffers {
                        return Err(RenderCommandError::VertexBufferIndexOutOfRange {
                            index: slot,
                            max: max_vertex_buffers,
                        })
                        .map_pass_err(scope);
                    }

                    let buffer = state
                        .trackers
                        .buffers
                        .write()
                        .merge_single(&*buffer_guard, buffer_id, hal::BufferUses::VERTEX)
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(buffer.device.info.id())
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, wgt::BufferUsages::VERTEX)
                        .map_pass_err(scope)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    buffer_memory_init_actions.extend(buffer.initialization_status.read().create_action(
                        buffer,
                        offset..end,
                        MemoryInitKind::NeedsInitializedMemory,
                    ));
                    state.vertex[slot as usize] = Some(VertexState::new(buffer_id, offset..end));
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset: _,
                } => {
                    let scope = PassErrorScope::SetPushConstant;
                    let end_offset = offset + size_bytes;

                    let pipeline_state = state.pipeline(scope)?;

                    pipeline_state.pipeline.layout
                        .validate_push_constant_ranges(stages, offset, end_offset)
                        .map_pass_err(scope)?;

                    commands.push(command);
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    let scope = PassErrorScope::Draw {
                        indexed: false,
                        indirect: false,
                        pipeline: state.pipeline_id(),
                    };
                    let pipeline = state.pipeline(scope)?;
                    let used_bind_groups = pipeline.used_bind_groups;

                    validate_draw(
                        &state.vertex[..],
                        &pipeline.steps,
                        first_vertex,
                        vertex_count,
                        first_instance,
                        instance_count,
                    ).map_pass_err(scope)?;

                    if instance_count > 0 && vertex_count > 0 {
                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds(used_bind_groups, base.dynamic_offsets));
                        commands.push(command);
                    }
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex: _,
                    first_instance,
                } => {
                    let scope = PassErrorScope::Draw {
                        indexed: true,
                        indirect: false,
                        pipeline: state.pipeline_id(),
                    };
                    let pipeline = state.pipeline(scope)?;
                    let used_bind_groups = pipeline.used_bind_groups;
                    let index = match state.index {
                        Some(ref index) => index,
                        None => return Err(DrawError::MissingIndexBuffer).map_pass_err(scope),
                    };

                    validate_indexed_draw(
                        &state.vertex[..],
                        &pipeline.steps,
                        index,
                        first_index,
                        index_count,
                        first_instance,
                        instance_count,
                    ).map_pass_err(scope)?;

                    if instance_count > 0 && index_count > 0 {
                        commands.extend(state.flush_index());
                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds(used_bind_groups, base.dynamic_offsets));
                        commands.push(command);
                    }
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: false,
                } => {
                    let scope = PassErrorScope::Draw {
                        indexed: false,
                        indirect: true,
                        pipeline: state.pipeline_id(),
                    };
                    device
                        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)
                        .map_pass_err(scope)?;

                    let pipeline = state.pipeline(scope)?;
                    let used_bind_groups = pipeline.used_bind_groups;

                    let buffer = state
                        .trackers
                        .buffers
                        .write()
                        .merge_single(&*buffer_guard, buffer_id, hal::BufferUses::INDIRECT)
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(buffer.device.info.id())
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, wgt::BufferUsages::INDIRECT)
                        .map_pass_err(scope)?;

                    buffer_memory_init_actions.extend(buffer.initialization_status.read().create_action(
                        buffer,
                        offset..(offset + mem::size_of::<wgt::DrawIndirectArgs>() as u64),
                        MemoryInitKind::NeedsInitializedMemory,
                    ));

                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds(used_bind_groups, base.dynamic_offsets));
                    commands.push(command);
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: true,
                } => {
                    let scope = PassErrorScope::Draw {
                        indexed: true,
                        indirect: true,
                        pipeline: state.pipeline_id(),
                    };
                    device
                        .require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)
                        .map_pass_err(scope)?;

                    let pipeline = state.pipeline(scope)?;
                    let used_bind_groups = pipeline.used_bind_groups;

                    let buffer = state
                        .trackers
                        .buffers
                        .write()
                        .merge_single(&*buffer_guard, buffer_id, hal::BufferUses::INDIRECT)
                        .map_pass_err(scope)?;
                    self.check_valid_to_use(buffer.device.info.id())
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, wgt::BufferUsages::INDIRECT)
                        .map_pass_err(scope)?;

                    buffer_memory_init_actions.extend(buffer.initialization_status.read().create_action(
                        buffer,
                        offset..(offset + mem::size_of::<wgt::DrawIndirectArgs>() as u64),
                        MemoryInitKind::NeedsInitializedMemory,
                    ));

                    let index = match state.index {
                        Some(ref mut index) => index,
                        None => return Err(DrawError::MissingIndexBuffer).map_pass_err(scope),
                    };

                    commands.extend(index.flush());
                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds(used_bind_groups, base.dynamic_offsets));
                    commands.push(command);
                }
                RenderCommand::MultiDrawIndirect { .. }
                | RenderCommand::MultiDrawIndirectCount { .. } => unimplemented!(),
                RenderCommand::PushDebugGroup { color: _, len: _ } => unimplemented!(),
                RenderCommand::InsertDebugMarker { color: _, len: _ } => unimplemented!(),
                RenderCommand::PopDebugGroup => unimplemented!(),
                RenderCommand::WriteTimestamp { .. } // Must check the TIMESTAMP_QUERY_INSIDE_PASSES feature
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

        Ok(RenderBundle {
            base: BasePass {
                label: desc.label.as_ref().map(|cow| cow.to_string()),
                commands,
                dynamic_offsets: state.flat_dynamic_offsets,
                string_data: Vec::new(),
                push_constant_data: Vec::new(),
            },
            is_depth_read_only: self.is_depth_read_only,
            is_stencil_read_only: self.is_stencil_read_only,
            device: device.clone(),
            used: state.trackers,
            buffer_memory_init_actions,
            texture_memory_init_actions,
            context: self.context,
            info: ResourceInfo::new(desc.label.borrow_or_default()),
            discard_hal_labels: device
                .instance_flags
                .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS),
        })
    }

    fn check_valid_to_use(&self, device_id: id::DeviceId) -> Result<(), RenderBundleErrorInner> {
        if device_id != self.parent_id {
            return Err(RenderBundleErrorInner::NotValidToUse);
        }

        Ok(())
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
    #[error("Buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("BindGroup {0:?} is invalid")]
    InvalidBindGroup(id::BindGroupId),
    #[error("Using {0} in a render bundle is not implemented")]
    Unimplemented(&'static str),
}
impl PrettyError for ExecutionError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::DestroyedBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::InvalidBindGroup(id) => {
                fmt.bind_group_label(&id);
            }
            Self::Unimplemented(_reason) => {}
        };
    }
}

pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;

//Note: here, `RenderBundle` is just wrapping a raw stream of render commands.
// The plan is to back it by an actual Vulkan secondary buffer, D3D12 Bundle,
// or Metal indirect command buffer.
#[derive(Debug)]
pub struct RenderBundle<A: HalApi> {
    // Normalized command stream. It can be executed verbatim,
    // without re-binding anything on the pipeline change.
    base: BasePass<RenderCommand>,
    pub(super) is_depth_read_only: bool,
    pub(super) is_stencil_read_only: bool,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) used: RenderBundleScope<A>,
    pub(super) buffer_memory_init_actions: Vec<BufferInitTrackerAction<A>>,
    pub(super) texture_memory_init_actions: Vec<TextureInitTrackerAction<A>>,
    pub(super) context: RenderPassContext,
    pub(crate) info: ResourceInfo<RenderBundle<A>>,
    discard_hal_labels: bool,
}

impl<A: HalApi> Drop for RenderBundle<A> {
    fn drop(&mut self) {
        resource_log!("Destroy raw RenderBundle {:?}", self.info.label());

        #[cfg(feature = "trace")]
        if let Some(t) = self.device.trace.lock().as_mut() {
            t.add(trace::Action::DestroyRenderBundle(self.info.id()));
        }
    }
}

#[cfg(send_sync)]
unsafe impl<A: HalApi> Send for RenderBundle<A> {}
#[cfg(send_sync)]
unsafe impl<A: HalApi> Sync for RenderBundle<A> {}

impl<A: HalApi> RenderBundle<A> {
    /// Actually encode the contents into a native command buffer.
    ///
    /// This is partially duplicating the logic of `command_encoder_run_render_pass`.
    /// However the point of this function is to be lighter, since we already had
    /// a chance to go through the commands in `render_bundle_encoder_finish`.
    ///
    /// Note that the function isn't expected to fail, generally.
    /// All the validation has already been done by this point.
    /// The only failure condition is if some of the used buffers are destroyed.
    pub(super) unsafe fn execute(&self, raw: &mut A::CommandEncoder) -> Result<(), ExecutionError> {
        let trackers = &self.used;
        let mut offsets = self.base.dynamic_offsets.as_slice();
        let mut pipeline_layout = None::<Arc<PipelineLayout<A>>>;
        if !self.discard_hal_labels {
            if let Some(ref label) = self.base.label {
                unsafe { raw.begin_debug_marker(label) };
            }
        }

        let snatch_guard = self.device.snatchable_lock.read();

        for command in self.base.commands.iter() {
            match *command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let bind_groups = trackers.bind_groups.read();
                    let bind_group = bind_groups.get(bind_group_id).unwrap();
                    let raw_bg = bind_group
                        .raw(&snatch_guard)
                        .ok_or(ExecutionError::InvalidBindGroup(bind_group_id))?;
                    unsafe {
                        raw.set_bind_group(
                            pipeline_layout.as_ref().unwrap().raw(),
                            index,
                            raw_bg,
                            &offsets[..num_dynamic_offsets],
                        )
                    };
                    offsets = &offsets[num_dynamic_offsets..];
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let render_pipelines = trackers.render_pipelines.read();
                    let pipeline = render_pipelines.get(pipeline_id).unwrap();
                    unsafe { raw.set_render_pipeline(pipeline.raw()) };

                    pipeline_layout = Some(pipeline.layout.clone());
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    index_format,
                    offset,
                    size,
                } => {
                    let buffers = trackers.buffers.read();
                    let buffer: &A::Buffer = buffers
                        .get(buffer_id)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?
                        .raw(&snatch_guard)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    let bb = hal::BufferBinding {
                        buffer,
                        offset,
                        size,
                    };
                    unsafe { raw.set_index_buffer(bb, index_format) };
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let buffers = trackers.buffers.read();
                    let buffer = buffers
                        .get(buffer_id)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?
                        .raw(&snatch_guard)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    let bb = hal::BufferBinding {
                        buffer,
                        offset,
                        size,
                    };
                    unsafe { raw.set_vertex_buffer(slot, bb) };
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let pipeline_layout = pipeline_layout.as_ref().unwrap();

                    if let Some(values_offset) = values_offset {
                        let values_end_offset =
                            (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                        let data_slice = &self.base.push_constant_data
                            [(values_offset as usize)..values_end_offset];

                        unsafe {
                            raw.set_push_constants(
                                pipeline_layout.raw(),
                                stages,
                                offset,
                                data_slice,
                            )
                        }
                    } else {
                        super::push_constant_clear(
                            offset,
                            size_bytes,
                            |clear_offset, clear_data| {
                                unsafe {
                                    raw.set_push_constants(
                                        pipeline_layout.raw(),
                                        stages,
                                        clear_offset,
                                        clear_data,
                                    )
                                };
                            },
                        );
                    }
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    unsafe { raw.draw(first_vertex, vertex_count, first_instance, instance_count) };
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    unsafe {
                        raw.draw_indexed(
                            first_index,
                            index_count,
                            base_vertex,
                            first_instance,
                            instance_count,
                        )
                    };
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: false,
                } => {
                    let buffers = trackers.buffers.read();
                    let buffer = buffers
                        .get(buffer_id)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?
                        .raw(&snatch_guard)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    unsafe { raw.draw_indirect(buffer, offset, 1) };
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: true,
                } => {
                    let buffers = trackers.buffers.read();
                    let buffer = buffers
                        .get(buffer_id)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?
                        .raw(&snatch_guard)
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    unsafe { raw.draw_indexed_indirect(buffer, offset, 1) };
                }
                RenderCommand::MultiDrawIndirect { .. }
                | RenderCommand::MultiDrawIndirectCount { .. } => {
                    return Err(ExecutionError::Unimplemented("multi-draw-indirect"))
                }
                RenderCommand::PushDebugGroup { .. }
                | RenderCommand::InsertDebugMarker { .. }
                | RenderCommand::PopDebugGroup => {
                    return Err(ExecutionError::Unimplemented("debug-markers"))
                }
                RenderCommand::WriteTimestamp { .. }
                | RenderCommand::BeginOcclusionQuery { .. }
                | RenderCommand::EndOcclusionQuery
                | RenderCommand::BeginPipelineStatisticsQuery { .. }
                | RenderCommand::EndPipelineStatisticsQuery => {
                    return Err(ExecutionError::Unimplemented("queries"))
                }
                RenderCommand::ExecuteBundle(_)
                | RenderCommand::SetBlendConstant(_)
                | RenderCommand::SetStencilReference(_)
                | RenderCommand::SetViewport { .. }
                | RenderCommand::SetScissor(_) => unreachable!(),
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

impl<A: HalApi> Resource for RenderBundle<A> {
    const TYPE: ResourceType = "RenderBundle";

    type Marker = crate::id::markers::RenderBundle;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

/// A render bundle's current index buffer state.
///
/// [`RenderBundleEncoder::finish`] records the currently set index buffer here,
/// and calls [`State::flush_index`] before any indexed draw command to produce
/// a `SetIndexBuffer` command if one is necessary.
#[derive(Debug)]
struct IndexState {
    buffer: id::BufferId,
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
    fn flush(&mut self) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetIndexBuffer {
                buffer_id: self.buffer,
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
    buffer: id::BufferId,
    range: Range<wgt::BufferAddress>,
    is_dirty: bool,
}

impl VertexState {
    fn new(buffer: id::BufferId, range: Range<wgt::BufferAddress>) -> Self {
        Self {
            buffer,
            range,
            is_dirty: true,
        }
    }

    /// Generate a `SetVertexBuffer` command for this slot, if necessary.
    ///
    /// `slot` is the index of the vertex buffer slot that `self` tracks.
    fn flush(&mut self, slot: u32) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetVertexBuffer {
                slot,
                buffer_id: self.buffer,
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
struct BindState<A: HalApi> {
    /// The id of the bind group set at this index.
    bind_group: Arc<BindGroup<A>>,

    /// The layout of `group`.
    layout: Arc<BindGroupLayout<A>>,

    /// The range of dynamic offsets for this bind group, in the original
    /// command stream's `BassPass::dynamic_offsets` array.
    dynamic_offsets: Range<usize>,

    /// True if this index's contents have been changed since the last time we
    /// generated a `SetBindGroup` command.
    is_dirty: bool,
}

/// The bundle's current pipeline, and some cached information needed for validation.
struct PipelineState<A: HalApi> {
    /// The pipeline
    pipeline: Arc<RenderPipeline<A>>,

    /// How this pipeline's vertex shader traverses each vertex buffer, indexed
    /// by vertex buffer slot number.
    steps: Vec<VertexStep>,

    /// Ranges of push constants this pipeline uses, copied from the pipeline
    /// layout.
    push_constant_ranges: ArrayVec<wgt::PushConstantRange, { SHADER_STAGE_COUNT }>,

    /// The number of bind groups this pipeline uses.
    used_bind_groups: usize,
}

impl<A: HalApi> PipelineState<A> {
    fn new(pipeline: &Arc<RenderPipeline<A>>) -> Self {
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
    fn zero_push_constants(&self) -> Option<impl Iterator<Item = RenderCommand>> {
        if !self.push_constant_ranges.is_empty() {
            let nonoverlapping_ranges =
                super::bind::compute_nonoverlapping_ranges(&self.push_constant_ranges);

            Some(
                nonoverlapping_ranges
                    .into_iter()
                    .map(|range| RenderCommand::SetPushConstant {
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
struct State<A: HalApi> {
    /// Resources used by this bundle. This will become [`RenderBundle::used`].
    trackers: RenderBundleScope<A>,

    /// The currently set pipeline, if any.
    pipeline: Option<PipelineState<A>>,

    /// The bind group set at each index, if any.
    bind: ArrayVec<Option<BindState<A>>, { hal::MAX_BIND_GROUPS }>,

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
}

impl<A: HalApi> State<A> {
    /// Return the id of the current pipeline, if any.
    fn pipeline_id(&self) -> Option<id::RenderPipelineId> {
        self.pipeline.as_ref().map(|p| p.pipeline.as_info().id())
    }

    /// Return the current pipeline state. Return an error if none is set.
    fn pipeline(&self, scope: PassErrorScope) -> Result<&PipelineState<A>, RenderBundleError> {
        self.pipeline
            .as_ref()
            .ok_or(DrawError::MissingPipeline)
            .map_pass_err(scope)
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
        bind_group: &Arc<BindGroup<A>>,
        layout: &Arc<BindGroupLayout<A>>,
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
            layout: layout.clone(),
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
    fn invalidate_bind_groups(&mut self, new: &PipelineState<A>, layout: &PipelineLayout<A>) {
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
                            Some(ref contents) => !contents.layout.is_equal(layout),
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
        buffer: id::BufferId,
        format: wgt::IndexFormat,
        range: Range<wgt::BufferAddress>,
    ) {
        match self.index {
            Some(ref current)
                if current.buffer == buffer
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
    fn flush_index(&mut self) -> Option<RenderCommand> {
        self.index.as_mut().and_then(|index| index.flush())
    }

    fn flush_vertices(&mut self) -> impl Iterator<Item = RenderCommand> + '_ {
        self.vertex
            .iter_mut()
            .enumerate()
            .flat_map(|(i, vs)| vs.as_mut().and_then(|vs| vs.flush(i as u32)))
    }

    /// Generate `SetBindGroup` commands for any bind groups that need to be updated.
    fn flush_binds(
        &mut self,
        used_bind_groups: usize,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) -> impl Iterator<Item = RenderCommand> + '_ {
        // Append each dirty bind group's dynamic offsets to `flat_dynamic_offsets`.
        for contents in self.bind[..used_bind_groups].iter().flatten() {
            if contents.is_dirty {
                self.flat_dynamic_offsets
                    .extend_from_slice(&dynamic_offsets[contents.dynamic_offsets.clone()]);
            }
        }

        // Then, generate `SetBindGroup` commands to update the dirty bind
        // groups. After this, all bind groups are clean.
        self.bind[..used_bind_groups]
            .iter_mut()
            .enumerate()
            .flat_map(|(i, entry)| {
                if let Some(ref mut contents) = *entry {
                    if contents.is_dirty {
                        contents.is_dirty = false;
                        let offsets = &contents.dynamic_offsets;
                        return Some(RenderCommand::SetBindGroup {
                            index: i.try_into().unwrap(),
                            bind_group_id: contents.bind_group.as_info().id(),
                            num_dynamic_offsets: offsets.end - offsets.start,
                        });
                    }
                }
                None
            })
    }
}

/// Error encountered when finishing recording a render bundle.
#[derive(Clone, Debug, Error)]
pub(super) enum RenderBundleErrorInner {
    #[error("Resource is not valid to use with this render bundle because the resource and the bundle come from different devices")]
    NotValidToUse,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    RenderCommand(RenderCommandError),
    #[error(transparent)]
    Draw(#[from] DrawError),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
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
    pub(crate) const INVALID_DEVICE: Self = RenderBundleError {
        scope: PassErrorScope::Bundle,
        inner: RenderBundleErrorInner::Device(DeviceError::Invalid),
    };
}
impl PrettyError for RenderBundleError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        fmt.error(self);
        self.scope.fmt_pretty(fmt);
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
        let redundant = unsafe {
            bundle.current_bind_groups.set_and_check_redundant(
                bind_group_id,
                index,
                &mut bundle.base.dynamic_offsets,
                offsets,
                offset_length,
            )
        };

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
