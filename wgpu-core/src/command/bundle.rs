/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! Render Bundles

    ## Software implementation

    The path from nothing to using a render bundle consists of 3 phases.

    ### Initial command encoding

    User creates a `RenderBundleEncoder` and populates it by issuing commands
    from `bundle_ffi` module, just like with `RenderPass`, except that the
    set of available commands is reduced. Everything is written into a `RawPass`.

    ### Bundle baking

    Once the commands are encoded, user calls `render_bundle_encoder_finish`.
    This is perhaps the most complex part of the logic. It consumes the
    commands stored in `RawPass`, while validating everything, tracking the state,
    and re-recording the commands into a separate `Vec<RenderCommand>`. It
    doesn't actually execute any commands.

    What's more important, is that the produced vector of commands is "normalized",
    which means it can be executed verbatim without any state tracking. More
    formally, "normalized" command stream guarantees that any state required by
    a draw call is set explicitly by one of the commands between the draw call
    and the last changing of the pipeline.

    ### Execution

    When the bundle is used in an actual render pass, `RenderBundle::execute` is
    called. It goes through the commands and issues them into the native command
    buffer. Thanks to the "normalized" property, it doesn't track any bind group
    invalidations or index format changes.
!*/
#![allow(clippy::reversed_empty_ranges)]

use crate::{
    command::{
        BasePass, DrawError, MapPassErr, PassErrorScope, RenderCommand, RenderCommandError,
        StateChange,
    },
    conv,
    device::{
        AttachmentData, Device, DeviceError, RenderPassContext, MAX_VERTEX_BUFFERS,
        SHADER_STAGE_COUNT,
    },
    hub::{GfxBackend, GlobalIdentityHandlerFactory, Hub, Resource, Storage, Token},
    id,
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::BufferUse,
    track::{TrackerSet, UsageConflict},
    validation::check_buffer_usage,
    Label, LabelHelpers, LifeGuard, Stored, MAX_BIND_GROUPS,
};
use arrayvec::ArrayVec;
use std::{borrow::Cow, iter, mem, ops::Range};
use thiserror::Error;

/// Describes a [`RenderBundleEncoder`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct RenderBundleEncoderDescriptor<'a> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The formats of the color attachments that this render bundle is capable to rendering to. This
    /// must match the formats of the color attachments in the renderpass this render bundle is executed in.
    pub color_formats: Cow<'a, [wgt::TextureFormat]>,
    /// The formats of the depth attachment that this render bundle is capable to rendering to. This
    /// must match the formats of the depth attachments in the renderpass this render bundle is executed in.
    pub depth_stencil_format: Option<wgt::TextureFormat>,
    /// Sample count this render bundle is capable of rendering to. This must match the pipelines and
    /// the renderpasses it is used in.
    pub sample_count: u32,
}

#[derive(Debug)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub struct RenderBundleEncoder {
    base: BasePass<RenderCommand>,
    parent_id: id::DeviceId,
    pub(crate) context: RenderPassContext,
}

impl RenderBundleEncoder {
    pub fn new(
        desc: &RenderBundleEncoderDescriptor,
        parent_id: id::DeviceId,
        base: Option<BasePass<RenderCommand>>,
    ) -> Result<Self, CreateRenderBundleError> {
        profiling::scope!("RenderBundleEncoder::new");
        Ok(Self {
            base: base.unwrap_or_else(|| BasePass::new(&desc.label)),
            parent_id,
            context: RenderPassContext {
                attachments: AttachmentData {
                    colors: desc.color_formats.iter().cloned().collect(),
                    resolves: ArrayVec::new(),
                    depth_stencil: desc.depth_stencil_format,
                },
                sample_count: {
                    let sc = desc.sample_count;
                    if sc == 0 || sc > 32 || !conv::is_power_of_two(sc) {
                        return Err(CreateRenderBundleError::InvalidSampleCount(sc));
                    }
                    sc as u8
                },
            },
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
            },
        }
    }

    #[cfg(feature = "trace")]
    pub(crate) fn to_base_pass(&self) -> BasePass<RenderCommand> {
        BasePass::from_ref(self.base.as_ref())
    }

    pub fn parent(&self) -> id::DeviceId {
        self.parent_id
    }

    pub(crate) fn finish<B: hal::Backend, G: GlobalIdentityHandlerFactory>(
        self,
        desc: &RenderBundleDescriptor,
        device: &Device<B>,
        hub: &Hub<B, G>,
        token: &mut Token<Device<B>>,
    ) -> Result<RenderBundle, RenderBundleError> {
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(token);
        let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
        let (pipeline_guard, mut token) = hub.render_pipelines.read(&mut token);
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        let mut state = State {
            trackers: TrackerSet::new(self.parent_id.backend()),
            index: IndexState::new(),
            vertex: (0..MAX_VERTEX_BUFFERS)
                .map(|_| VertexState::new())
                .collect(),
            bind: (0..MAX_BIND_GROUPS).map(|_| BindState::new()).collect(),
            push_constant_ranges: PushConstantState::new(),
            raw_dynamic_offsets: Vec::new(),
            flat_dynamic_offsets: Vec::new(),
            used_bind_groups: 0,
            pipeline: StateChange::new(),
        };
        let mut commands = Vec::new();
        let mut base = self.base.as_ref();
        let mut pipeline_layout_id = None::<id::Valid<id::PipelineLayoutId>>;
        let mut buffer_memory_init_actions = Vec::new();

        for &command in base.commands {
            match command {
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

                    let offsets = &base.dynamic_offsets[..num_dynamic_offsets as usize];
                    base.dynamic_offsets = &base.dynamic_offsets[num_dynamic_offsets as usize..];
                    // Check for misaligned offsets.
                    if let Some(offset) = offsets
                        .iter()
                        .map(|offset| *offset as wgt::BufferAddress)
                        .find(|offset| offset % wgt::BIND_BUFFER_ALIGNMENT != 0)
                    {
                        return Err(RenderCommandError::UnalignedBufferOffset(offset))
                            .map_pass_err(scope);
                    }

                    let bind_group = state
                        .trackers
                        .bind_groups
                        .use_extend(&*bind_group_guard, bind_group_id, (), ())
                        .map_err(|_| RenderCommandError::InvalidBindGroup(bind_group_id))
                        .map_pass_err(scope)?;
                    if bind_group.dynamic_binding_info.len() != offsets.len() {
                        return Err(RenderCommandError::InvalidDynamicOffsetCount {
                            actual: offsets.len(),
                            expected: bind_group.dynamic_binding_info.len(),
                        })
                        .map_pass_err(scope);
                    }

                    buffer_memory_init_actions.extend_from_slice(&bind_group.used_buffer_ranges);

                    state.set_bind_group(index, bind_group_id, bind_group.layout_id, offsets);
                    state
                        .trackers
                        .merge_extend(&bind_group.used)
                        .map_pass_err(scope)?;
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let scope = PassErrorScope::SetPipelineRender(pipeline_id);
                    if state.pipeline.set_and_check_redundant(pipeline_id) {
                        continue;
                    }

                    let pipeline = state
                        .trackers
                        .render_pipes
                        .use_extend(&*pipeline_guard, pipeline_id, (), ())
                        .unwrap();

                    self.context
                        .check_compatible(&pipeline.pass_context)
                        .map_err(RenderCommandError::IncompatiblePipeline)
                        .map_pass_err(scope)?;

                    //TODO: check read-only depth

                    let layout = &pipeline_layout_guard[pipeline.layout_id.value];
                    pipeline_layout_id = Some(pipeline.layout_id.value);

                    state.set_pipeline(
                        pipeline.strip_index_format,
                        &pipeline.vertex_strides,
                        &layout.bind_group_layout_ids,
                        &layout.push_constant_ranges,
                    );
                    commands.push(command);
                    if let Some(iter) = state.flush_push_constants() {
                        commands.extend(iter)
                    }
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
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                        .unwrap();
                    check_buffer_usage(buffer.usage, wgt::BufferUsage::INDEX)
                        .map_pass_err(scope)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    buffer_memory_init_actions.push(MemoryInitTrackerAction {
                        id: buffer_id,
                        range: offset..end,
                        kind: MemoryInitKind::NeedsInitializedMemory,
                    });
                    state.index.set_format(index_format);
                    state.index.set_buffer(buffer_id, offset..end);
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let scope = PassErrorScope::SetVertexBuffer(buffer_id);
                    let buffer = state
                        .trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::VERTEX)
                        .unwrap();
                    check_buffer_usage(buffer.usage, wgt::BufferUsage::VERTEX)
                        .map_pass_err(scope)?;

                    let end = match size {
                        Some(s) => offset + s.get(),
                        None => buffer.size,
                    };
                    buffer_memory_init_actions.push(MemoryInitTrackerAction {
                        id: buffer_id,
                        range: offset..end,
                        kind: MemoryInitKind::NeedsInitializedMemory,
                    });
                    state.vertex[slot as usize].set_buffer(buffer_id, offset..end);
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset: _,
                } => {
                    let scope = PassErrorScope::SetPushConstant;
                    let end_offset = offset + size_bytes;

                    let pipeline_layout_id = pipeline_layout_id
                        .ok_or(DrawError::MissingPipeline)
                        .map_pass_err(scope)?;
                    let pipeline_layout = &pipeline_layout_guard[pipeline_layout_id];

                    pipeline_layout
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
                        pipeline: state.pipeline.last_state,
                    };
                    let vertex_limits = state.vertex_limits();
                    let last_vertex = first_vertex + vertex_count;
                    if last_vertex > vertex_limits.vertex_limit {
                        return Err(DrawError::VertexBeyondLimit {
                            last_vertex,
                            vertex_limit: vertex_limits.vertex_limit,
                            slot: vertex_limits.vertex_limit_slot,
                        })
                        .map_pass_err(scope);
                    }
                    let last_instance = first_instance + instance_count;
                    if last_instance > vertex_limits.instance_limit {
                        return Err(DrawError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit: vertex_limits.instance_limit,
                            slot: vertex_limits.instance_limit_slot,
                        })
                        .map_pass_err(scope);
                    }
                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds());
                    commands.push(command);
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
                        pipeline: state.pipeline.last_state,
                    };
                    //TODO: validate that base_vertex + max_index() is within the provided range
                    let vertex_limits = state.vertex_limits();
                    let index_limit = state.index.limit();
                    let last_index = first_index + index_count;
                    if last_index > index_limit {
                        return Err(DrawError::IndexBeyondLimit {
                            last_index,
                            index_limit,
                        })
                        .map_pass_err(scope);
                    }
                    let last_instance = first_instance + instance_count;
                    if last_instance > vertex_limits.instance_limit {
                        return Err(DrawError::InstanceBeyondLimit {
                            last_instance,
                            instance_limit: vertex_limits.instance_limit,
                            slot: vertex_limits.instance_limit_slot,
                        })
                        .map_pass_err(scope);
                    }
                    commands.extend(state.index.flush());
                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds());
                    commands.push(command);
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
                        pipeline: state.pipeline.last_state,
                    };
                    let buffer = state
                        .trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .unwrap();
                    check_buffer_usage(buffer.usage, wgt::BufferUsage::INDIRECT)
                        .map_pass_err(scope)?;

                    buffer_memory_init_actions.extend(
                        buffer
                            .initialization_status
                            .check(
                                offset..(offset + mem::size_of::<wgt::DrawIndirectArgs>() as u64),
                            )
                            .map(|range| MemoryInitTrackerAction {
                                id: buffer_id,
                                range,
                                kind: MemoryInitKind::NeedsInitializedMemory,
                            }),
                    );

                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds());
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
                        pipeline: state.pipeline.last_state,
                    };
                    let buffer = state
                        .trackers
                        .buffers
                        .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                        .map_err(|err| RenderCommandError::Buffer(buffer_id, err))
                        .map_pass_err(scope)?;
                    check_buffer_usage(buffer.usage, wgt::BufferUsage::INDIRECT)
                        .map_pass_err(scope)?;

                    buffer_memory_init_actions.extend(
                        buffer
                            .initialization_status
                            .check(
                                offset..(offset + mem::size_of::<wgt::DrawIndirectArgs>() as u64),
                            )
                            .map(|range| MemoryInitTrackerAction {
                                id: buffer_id,
                                range,
                                kind: MemoryInitKind::NeedsInitializedMemory,
                            }),
                    );

                    commands.extend(state.index.flush());
                    commands.extend(state.flush_vertices());
                    commands.extend(state.flush_binds());
                    commands.push(command);
                }
                RenderCommand::MultiDrawIndirect { .. }
                | RenderCommand::MultiDrawIndirectCount { .. } => unimplemented!(),
                RenderCommand::PushDebugGroup { color: _, len: _ } => unimplemented!(),
                RenderCommand::InsertDebugMarker { color: _, len: _ } => unimplemented!(),
                RenderCommand::PopDebugGroup => unimplemented!(),
                RenderCommand::WriteTimestamp { .. }
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
            device_id: Stored {
                value: id::Valid(self.parent_id),
                ref_count: device.life_guard.add_ref(),
            },
            used: state.trackers,
            buffer_memory_init_actions,
            context: self.context,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    pub fn set_index_buffer(
        &mut self,
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        profiling::scope!("RenderBundle::set_index_buffer");
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
pub enum CreateRenderBundleError {
    #[error("invalid number of samples {0}")]
    InvalidSampleCount(u32),
}

/// Error type returned from `RenderBundleEncoder::new` if the sample count is invalid.
#[derive(Clone, Debug, Error)]
pub enum ExecutionError {
    #[error("buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("using {0} in a render bundle is not implemented")]
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
    base: BasePass<RenderCommand>,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) used: TrackerSet,
    pub(crate) buffer_memory_init_actions: Vec<MemoryInitTrackerAction<id::BufferId>>,
    pub(crate) context: RenderPassContext,
    pub(crate) life_guard: LifeGuard,
}

unsafe impl Send for RenderBundle {}
unsafe impl Sync for RenderBundle {}

impl RenderBundle {
    /// Actually encode the contents into a native command buffer.
    ///
    /// This is partially duplicating the logic of `command_encoder_run_render_pass`.
    /// However the point of this function is to be lighter, since we already had
    /// a chance to go through the commands in `render_bundle_encoder_finish`.
    ///
    /// Note that the function isn't expected to fail, generally.
    /// All the validation has already been done by this point.
    /// The only failure condition is if some of the used buffers are destroyed.
    pub(crate) unsafe fn execute<B: GfxBackend>(
        &self,
        cmd_buf: &mut B::CommandBuffer,
        pipeline_layout_guard: &Storage<
            crate::binding_model::PipelineLayout<B>,
            id::PipelineLayoutId,
        >,
        bind_group_guard: &Storage<crate::binding_model::BindGroup<B>, id::BindGroupId>,
        pipeline_guard: &Storage<crate::pipeline::RenderPipeline<B>, id::RenderPipelineId>,
        buffer_guard: &Storage<crate::resource::Buffer<B>, id::BufferId>,
    ) -> Result<(), ExecutionError> {
        use hal::command::CommandBuffer as _;

        let mut offsets = self.base.dynamic_offsets.as_slice();
        let mut pipeline_layout_id = None::<id::Valid<id::PipelineLayoutId>>;
        if let Some(ref label) = self.base.label {
            cmd_buf.begin_debug_marker(label, 0);
        }

        for command in self.base.commands.iter() {
            match *command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let bind_group = bind_group_guard.get(bind_group_id).unwrap();
                    cmd_buf.bind_graphics_descriptor_sets(
                        &pipeline_layout_guard[pipeline_layout_id.unwrap()].raw,
                        index as usize,
                        iter::once(bind_group.raw.raw()),
                        offsets.iter().take(num_dynamic_offsets as usize).cloned(),
                    );
                    offsets = &offsets[num_dynamic_offsets as usize..];
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let pipeline = pipeline_guard.get(pipeline_id).unwrap();
                    cmd_buf.bind_graphics_pipeline(&pipeline.raw);

                    pipeline_layout_id = Some(pipeline.layout_id.value);
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    index_format,
                    offset,
                    size,
                } => {
                    let index_type = conv::map_index_format(index_format);

                    let &(ref buffer, _) = buffer_guard
                        .get(buffer_id)
                        .unwrap()
                        .raw
                        .as_ref()
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    let range = hal::buffer::SubRange {
                        offset,
                        size: size.map(|s| s.get()),
                    };
                    cmd_buf.bind_index_buffer(buffer, range, index_type);
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let &(ref buffer, _) = buffer_guard
                        .get(buffer_id)
                        .unwrap()
                        .raw
                        .as_ref()
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    let range = hal::buffer::SubRange {
                        offset,
                        size: size.map(|s| s.get()),
                    };
                    cmd_buf.bind_vertex_buffers(slot, iter::once((buffer, range)));
                }
                RenderCommand::SetPushConstant {
                    stages,
                    offset,
                    size_bytes,
                    values_offset,
                } => {
                    let pipeline_layout_id = pipeline_layout_id.unwrap();
                    let pipeline_layout = &pipeline_layout_guard[pipeline_layout_id];

                    if let Some(values_offset) = values_offset {
                        let values_end_offset =
                            (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
                        let data_slice = &self.base.push_constant_data
                            [(values_offset as usize)..values_end_offset];

                        cmd_buf.push_graphics_constants(
                            &pipeline_layout.raw,
                            conv::map_shader_stage_flags(stages),
                            offset,
                            &data_slice,
                        )
                    } else {
                        super::push_constant_clear(
                            offset,
                            size_bytes,
                            |clear_offset, clear_data| {
                                cmd_buf.push_graphics_constants(
                                    &pipeline_layout.raw,
                                    conv::map_shader_stage_flags(stages),
                                    clear_offset,
                                    clear_data,
                                );
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
                    cmd_buf.draw(
                        first_vertex..first_vertex + vertex_count,
                        first_instance..first_instance + instance_count,
                    );
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    cmd_buf.draw_indexed(
                        first_index..first_index + index_count,
                        base_vertex,
                        first_instance..first_instance + instance_count,
                    );
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: false,
                } => {
                    let &(ref buffer, _) = buffer_guard
                        .get(buffer_id)
                        .unwrap()
                        .raw
                        .as_ref()
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    cmd_buf.draw_indirect(buffer, offset, 1, 0);
                }
                RenderCommand::MultiDrawIndirect {
                    buffer_id,
                    offset,
                    count: None,
                    indexed: true,
                } => {
                    let &(ref buffer, _) = buffer_guard
                        .get(buffer_id)
                        .unwrap()
                        .raw
                        .as_ref()
                        .ok_or(ExecutionError::DestroyedBuffer(buffer_id))?;
                    cmd_buf.draw_indexed_indirect(buffer, offset, 1, 0);
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

        if let Some(_) = self.base.label {
            cmd_buf.end_debug_marker();
        }

        Ok(())
    }
}

impl Resource for RenderBundle {
    const TYPE: &'static str = "RenderBundle";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Debug)]
struct IndexState {
    buffer: Option<id::BufferId>,
    format: wgt::IndexFormat,
    pipeline_format: Option<wgt::IndexFormat>,
    range: Range<wgt::BufferAddress>,
    is_dirty: bool,
}

impl IndexState {
    fn new() -> Self {
        Self {
            buffer: None,
            format: wgt::IndexFormat::default(),
            pipeline_format: None,
            range: 0..0,
            is_dirty: false,
        }
    }

    fn limit(&self) -> u32 {
        assert!(self.buffer.is_some());
        let bytes_per_index = match self.format {
            wgt::IndexFormat::Uint16 => 2,
            wgt::IndexFormat::Uint32 => 4,
        };
        ((self.range.end - self.range.start) / bytes_per_index) as u32
    }

    fn flush(&mut self) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetIndexBuffer {
                buffer_id: self.buffer.unwrap(),
                index_format: self.format,
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }

    fn set_format(&mut self, format: wgt::IndexFormat) {
        if self.format != format {
            self.format = format;
            self.is_dirty = true;
        }
    }

    fn set_buffer(&mut self, id: id::BufferId, range: Range<wgt::BufferAddress>) {
        self.buffer = Some(id);
        self.range = range;
        self.is_dirty = true;
    }
}

#[derive(Debug)]
struct VertexState {
    buffer: Option<id::BufferId>,
    range: Range<wgt::BufferAddress>,
    stride: wgt::BufferAddress,
    rate: wgt::InputStepMode,
    is_dirty: bool,
}

impl VertexState {
    fn new() -> Self {
        Self {
            buffer: None,
            range: 0..0,
            stride: 0,
            rate: wgt::InputStepMode::Vertex,
            is_dirty: false,
        }
    }

    fn set_buffer(&mut self, buffer_id: id::BufferId, range: Range<wgt::BufferAddress>) {
        self.buffer = Some(buffer_id);
        self.range = range;
        self.is_dirty = true;
    }

    fn flush(&mut self, slot: u32) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetVertexBuffer {
                slot,
                buffer_id: self.buffer.unwrap(),
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct BindState {
    bind_group: Option<(id::BindGroupId, id::BindGroupLayoutId)>,
    dynamic_offsets: Range<usize>,
    is_dirty: bool,
}

impl BindState {
    fn new() -> Self {
        Self {
            bind_group: None,
            dynamic_offsets: 0..0,
            is_dirty: false,
        }
    }

    fn set_group(
        &mut self,
        bind_group_id: id::BindGroupId,
        layout_id: id::BindGroupLayoutId,
        dyn_offset: usize,
        dyn_count: usize,
    ) -> bool {
        match self.bind_group {
            Some((bg_id, _)) if bg_id == bind_group_id && dyn_count == 0 => false,
            _ => {
                self.bind_group = Some((bind_group_id, layout_id));
                self.dynamic_offsets = dyn_offset..dyn_offset + dyn_count;
                self.is_dirty = true;
                true
            }
        }
    }
}

#[derive(Debug)]
struct PushConstantState {
    ranges: ArrayVec<[wgt::PushConstantRange; SHADER_STAGE_COUNT]>,
    is_dirty: bool,
}
impl PushConstantState {
    fn new() -> Self {
        Self {
            ranges: ArrayVec::new(),
            is_dirty: false,
        }
    }

    fn set_push_constants(&mut self, new_ranges: &[wgt::PushConstantRange]) -> bool {
        if &*self.ranges != new_ranges {
            self.ranges = new_ranges.iter().cloned().collect();
            self.is_dirty = true;
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct VertexLimitState {
    /// Length of the shortest vertex rate vertex buffer
    vertex_limit: u32,
    /// Buffer slot which the shortest vertex rate vertex buffer is bound to
    vertex_limit_slot: u32,
    /// Length of the shortest instance rate vertex buffer
    instance_limit: u32,
    /// Buffer slot which the shortest instance rate vertex buffer is bound to
    instance_limit_slot: u32,
}

#[derive(Debug)]
struct State {
    trackers: TrackerSet,
    index: IndexState,
    vertex: ArrayVec<[VertexState; MAX_VERTEX_BUFFERS]>,
    bind: ArrayVec<[BindState; MAX_BIND_GROUPS]>,
    push_constant_ranges: PushConstantState,
    raw_dynamic_offsets: Vec<wgt::DynamicOffset>,
    flat_dynamic_offsets: Vec<wgt::DynamicOffset>,
    used_bind_groups: usize,
    pipeline: StateChange<id::RenderPipelineId>,
}

impl State {
    fn vertex_limits(&self) -> VertexLimitState {
        let mut vert_state = VertexLimitState {
            vertex_limit: u32::MAX,
            vertex_limit_slot: 0,
            instance_limit: u32::MAX,
            instance_limit_slot: 0,
        };
        for (idx, vbs) in self.vertex.iter().enumerate() {
            if vbs.stride == 0 {
                continue;
            }
            let limit = ((vbs.range.end - vbs.range.start) / vbs.stride) as u32;
            match vbs.rate {
                wgt::InputStepMode::Vertex => {
                    if limit < vert_state.vertex_limit {
                        vert_state.vertex_limit = limit;
                        vert_state.vertex_limit_slot = idx as _;
                    }
                }
                wgt::InputStepMode::Instance => {
                    if limit < vert_state.instance_limit {
                        vert_state.instance_limit = limit;
                        vert_state.instance_limit_slot = idx as _;
                    }
                }
            }
        }
        vert_state
    }

    fn invalidate_group_from(&mut self, slot: usize) {
        for bind in self.bind[slot..].iter_mut() {
            if bind.bind_group.is_some() {
                bind.is_dirty = true;
            }
        }
    }

    fn set_bind_group(
        &mut self,
        slot: u8,
        bind_group_id: id::BindGroupId,
        layout_id: id::Valid<id::BindGroupLayoutId>,
        offsets: &[wgt::DynamicOffset],
    ) {
        if self.bind[slot as usize].set_group(
            bind_group_id,
            layout_id.0,
            self.raw_dynamic_offsets.len(),
            offsets.len(),
        ) {
            self.invalidate_group_from(slot as usize + 1);
        }
        self.raw_dynamic_offsets.extend(offsets);
    }

    fn set_pipeline(
        &mut self,
        index_format: Option<wgt::IndexFormat>,
        vertex_strides: &[(wgt::BufferAddress, wgt::InputStepMode)],
        layout_ids: &[id::Valid<id::BindGroupLayoutId>],
        push_constant_layouts: &[wgt::PushConstantRange],
    ) {
        self.index.pipeline_format = index_format;

        for (vs, &(stride, step_mode)) in self.vertex.iter_mut().zip(vertex_strides) {
            if vs.stride != stride || vs.rate != step_mode {
                vs.stride = stride;
                vs.rate = step_mode;
                vs.is_dirty = true;
            }
        }

        let push_constants_changed = self
            .push_constant_ranges
            .set_push_constants(push_constant_layouts);

        self.used_bind_groups = layout_ids.len();
        let invalid_from = if push_constants_changed {
            Some(0)
        } else {
            self.bind
                .iter()
                .zip(layout_ids)
                .position(|(bs, layout_id)| match bs.bind_group {
                    Some((_, bgl_id)) => bgl_id != layout_id.0,
                    None => false,
                })
        };
        if let Some(slot) = invalid_from {
            self.invalidate_group_from(slot);
        }
    }

    fn flush_push_constants(&mut self) -> Option<impl Iterator<Item = RenderCommand>> {
        let is_dirty = self.push_constant_ranges.is_dirty;

        if is_dirty {
            let nonoverlapping_ranges =
                super::bind::compute_nonoverlapping_ranges(&self.push_constant_ranges.ranges);

            Some(
                nonoverlapping_ranges
                    .into_iter()
                    .map(|range| RenderCommand::SetPushConstant {
                        stages: range.stages,
                        offset: range.range.start,
                        size_bytes: range.range.end - range.range.start,
                        values_offset: None,
                    }),
            )
        } else {
            None
        }
    }

    fn flush_vertices(&mut self) -> impl Iterator<Item = RenderCommand> + '_ {
        self.vertex
            .iter_mut()
            .enumerate()
            .flat_map(|(i, vs)| vs.flush(i as u32))
    }

    fn flush_binds(&mut self) -> impl Iterator<Item = RenderCommand> + '_ {
        for bs in self.bind[..self.used_bind_groups].iter() {
            if bs.is_dirty {
                self.flat_dynamic_offsets
                    .extend_from_slice(&self.raw_dynamic_offsets[bs.dynamic_offsets.clone()]);
            }
        }
        self.bind
            .iter_mut()
            .take(self.used_bind_groups)
            .enumerate()
            .flat_map(|(i, bs)| {
                if bs.is_dirty {
                    bs.is_dirty = false;
                    Some(RenderCommand::SetBindGroup {
                        index: i as u8,
                        bind_group_id: bs.bind_group.unwrap().0,
                        num_dynamic_offsets: (bs.dynamic_offsets.end - bs.dynamic_offsets.start)
                            as u8,
                    })
                } else {
                    None
                }
            })
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
    ResourceUsageConflict(#[from] UsageConflict),
    #[error(transparent)]
    Draw(#[from] DrawError),
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
    use wgt::{BufferAddress, BufferSize, DynamicOffset};

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
        bundle.base.commands.push(RenderCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
        });
        if offset_length != 0 {
            bundle
                .base
                .dynamic_offsets
                .extend_from_slice(slice::from_raw_parts(offsets, offset_length));
        }
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_pipeline(
        bundle: &mut RenderBundleEncoder,
        pipeline_id: id::RenderPipelineId,
    ) {
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

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `data` elements.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_push_constants(
        pass: &mut RenderBundleEncoder,
        stages: wgt::ShaderStage,
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
        let data_slice = slice::from_raw_parts(data, size_bytes as usize);
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
    pub extern "C" fn wgpu_render_pass_bundle_indexed_indirect(
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
