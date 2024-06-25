use crate::{
    binding_model::BindGroup,
    hal_api::HalApi,
    id,
    pipeline::RenderPipeline,
    resource::{Buffer, QuerySet},
};
use wgt::{BufferAddress, BufferSize, Color};

use std::{num::NonZeroU32, sync::Arc};

use super::{
    DrawKind, PassErrorScope, Rect, RenderBundle, RenderCommandError, RenderPassError,
    RenderPassErrorInner,
};

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::RenderPipelineId),
    SetIndexBuffer {
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetBlendConstant(Color),
    SetStencilReference(u32),
    SetViewport {
        rect: Rect<f32>,
        //TODO: use half-float to reduce the size?
        depth_min: f32,
        depth_max: f32,
    },
    SetScissor(Rect<u32>),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    ///
    /// See [`wgpu::RenderPass::set_push_constants`] for a detailed explanation
    /// of the restrictions these commands must satisfy.
    SetPushConstant {
        /// Which stages we are setting push constant values for.
        stages: wgt::ShaderStages,

        /// The byte offset within the push constant storage to write to.  This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        ///
        /// `None` means zeros should be written to the destination range, and
        /// there is no corresponding data in `push_constant_data`. This is used
        /// by render bundles, which explicitly clear out any state that
        /// post-bundle code might see.
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
        count: Option<NonZeroU32>,
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
    WriteTimestamp {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    BeginOcclusionQuery {
        query_index: u32,
    },
    EndOcclusionQuery,
    BeginPipelineStatisticsQuery {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(id::RenderBundleId),
}

impl RenderCommand {
    /// Resolves all ids in a list of commands into the corresponding resource Arc.
    //
    // TODO: Once resolving is done on-the-fly during recording, this function should be only needed with the replay feature:
    // #[cfg(feature = "replay")]
    pub fn resolve_render_command_ids<A: HalApi>(
        hub: &crate::hub::Hub<A>,
        commands: &[RenderCommand],
    ) -> Result<Vec<ArcRenderCommand<A>>, RenderPassError> {
        let buffers_guard = hub.buffers.read();
        let bind_group_guard = hub.bind_groups.read();
        let query_set_guard = hub.query_sets.read();
        let pipelines_guard = hub.render_pipelines.read();

        let resolved_commands: Vec<ArcRenderCommand<A>> = commands
            .iter()
            .map(|c| -> Result<ArcRenderCommand<A>, RenderPassError> {
                Ok(match *c {
                    RenderCommand::SetBindGroup {
                        index,
                        num_dynamic_offsets,
                        bind_group_id,
                    } => ArcRenderCommand::SetBindGroup {
                        index,
                        num_dynamic_offsets,
                        bind_group: bind_group_guard.get_owned(bind_group_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::SetBindGroup(bind_group_id),
                                inner: RenderPassErrorInner::InvalidBindGroup(index),
                            }
                        })?,
                    },

                    RenderCommand::SetPipeline(pipeline_id) => ArcRenderCommand::SetPipeline(
                        pipelines_guard
                            .get_owned(pipeline_id)
                            .map_err(|_| RenderPassError {
                                scope: PassErrorScope::SetPipelineRender(pipeline_id),
                                inner: RenderCommandError::InvalidPipeline(pipeline_id).into(),
                            })?,
                    ),

                    RenderCommand::SetPushConstant {
                        offset,
                        size_bytes,
                        values_offset,
                        stages,
                    } => ArcRenderCommand::SetPushConstant {
                        offset,
                        size_bytes,
                        values_offset,
                        stages,
                    },

                    RenderCommand::PushDebugGroup { color, len } => {
                        ArcRenderCommand::PushDebugGroup { color, len }
                    }

                    RenderCommand::PopDebugGroup => ArcRenderCommand::PopDebugGroup,

                    RenderCommand::InsertDebugMarker { color, len } => {
                        ArcRenderCommand::InsertDebugMarker { color, len }
                    }

                    RenderCommand::WriteTimestamp {
                        query_set_id,
                        query_index,
                    } => ArcRenderCommand::WriteTimestamp {
                        query_set: query_set_guard.get_owned(query_set_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::WriteTimestamp,
                                inner: RenderPassErrorInner::InvalidQuerySet(query_set_id),
                            }
                        })?,
                        query_index,
                    },

                    RenderCommand::BeginPipelineStatisticsQuery {
                        query_set_id,
                        query_index,
                    } => ArcRenderCommand::BeginPipelineStatisticsQuery {
                        query_set: query_set_guard.get_owned(query_set_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::BeginPipelineStatisticsQuery,
                                inner: RenderPassErrorInner::InvalidQuerySet(query_set_id),
                            }
                        })?,
                        query_index,
                    },

                    RenderCommand::EndPipelineStatisticsQuery => {
                        ArcRenderCommand::EndPipelineStatisticsQuery
                    }

                    RenderCommand::SetIndexBuffer {
                        buffer_id,
                        index_format,
                        offset,
                        size,
                    } => ArcRenderCommand::SetIndexBuffer {
                        buffer: buffers_guard.get_owned(buffer_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::SetIndexBuffer(buffer_id),
                                inner: RenderCommandError::InvalidBufferId(buffer_id).into(),
                            }
                        })?,
                        index_format,
                        offset,
                        size,
                    },

                    RenderCommand::SetVertexBuffer {
                        slot,
                        buffer_id,
                        offset,
                        size,
                    } => ArcRenderCommand::SetVertexBuffer {
                        slot,
                        buffer: buffers_guard.get_owned(buffer_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::SetVertexBuffer(buffer_id),
                                inner: RenderCommandError::InvalidBufferId(buffer_id).into(),
                            }
                        })?,
                        offset,
                        size,
                    },

                    RenderCommand::SetBlendConstant(color) => {
                        ArcRenderCommand::SetBlendConstant(color)
                    }

                    RenderCommand::SetStencilReference(reference) => {
                        ArcRenderCommand::SetStencilReference(reference)
                    }

                    RenderCommand::SetViewport {
                        rect,
                        depth_min,
                        depth_max,
                    } => ArcRenderCommand::SetViewport {
                        rect,
                        depth_min,
                        depth_max,
                    },

                    RenderCommand::SetScissor(scissor) => ArcRenderCommand::SetScissor(scissor),

                    RenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => ArcRenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    },

                    RenderCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    } => ArcRenderCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex,
                        first_instance,
                    },

                    RenderCommand::MultiDrawIndirect {
                        buffer_id,
                        offset,
                        count,
                        indexed,
                    } => ArcRenderCommand::MultiDrawIndirect {
                        buffer: buffers_guard.get_owned(buffer_id).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::Draw {
                                    kind: if count.is_some() {
                                        DrawKind::MultiDrawIndirect
                                    } else {
                                        DrawKind::DrawIndirect
                                    },
                                    indexed,
                                    pipeline: None,
                                },
                                inner: RenderCommandError::InvalidBufferId(buffer_id).into(),
                            }
                        })?,
                        offset,
                        count,
                        indexed,
                    },

                    RenderCommand::MultiDrawIndirectCount {
                        buffer_id,
                        offset,
                        count_buffer_id,
                        count_buffer_offset,
                        max_count,
                        indexed,
                    } => {
                        let scope = PassErrorScope::Draw {
                            kind: DrawKind::MultiDrawIndirectCount,
                            indexed,
                            pipeline: None,
                        };
                        ArcRenderCommand::MultiDrawIndirectCount {
                            buffer: buffers_guard.get_owned(buffer_id).map_err(|_| {
                                RenderPassError {
                                    scope,
                                    inner: RenderCommandError::InvalidBufferId(buffer_id).into(),
                                }
                            })?,
                            offset,
                            count_buffer: buffers_guard.get_owned(count_buffer_id).map_err(
                                |_| RenderPassError {
                                    scope,
                                    inner: RenderCommandError::InvalidBufferId(count_buffer_id)
                                        .into(),
                                },
                            )?,
                            count_buffer_offset,
                            max_count,
                            indexed,
                        }
                    }

                    RenderCommand::BeginOcclusionQuery { query_index } => {
                        ArcRenderCommand::BeginOcclusionQuery { query_index }
                    }

                    RenderCommand::EndOcclusionQuery => ArcRenderCommand::EndOcclusionQuery,

                    RenderCommand::ExecuteBundle(bundle) => ArcRenderCommand::ExecuteBundle(
                        hub.render_bundles.read().get_owned(bundle).map_err(|_| {
                            RenderPassError {
                                scope: PassErrorScope::ExecuteBundle,
                                inner: RenderCommandError::InvalidRenderBundle(bundle).into(),
                            }
                        })?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, RenderPassError>>()?;
        Ok(resolved_commands)
    }
}

/// Equivalent to `RenderCommand` with the Ids resolved into resource Arcs.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum ArcRenderCommand<A: HalApi> {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Arc<BindGroup<A>>,
    },
    SetPipeline(Arc<RenderPipeline<A>>),
    SetIndexBuffer {
        buffer: Arc<Buffer<A>>,
        index_format: wgt::IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: Arc<Buffer<A>>,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetBlendConstant(Color),
    SetStencilReference(u32),
    SetViewport {
        rect: Rect<f32>,
        depth_min: f32,
        depth_max: f32,
    },
    SetScissor(Rect<u32>),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    ///
    /// See [`wgpu::RenderPass::set_push_constants`] for a detailed explanation
    /// of the restrictions these commands must satisfy.
    SetPushConstant {
        /// Which stages we are setting push constant values for.
        stages: wgt::ShaderStages,

        /// The byte offset within the push constant storage to write to.  This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        ///
        /// `None` means zeros should be written to the destination range, and
        /// there is no corresponding data in `push_constant_data`. This is used
        /// by render bundles, which explicitly clear out any state that
        /// post-bundle code might see.
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
        buffer: Arc<Buffer<A>>,
        offset: BufferAddress,
        /// Count of `None` represents a non-multi call.
        count: Option<NonZeroU32>,
        indexed: bool,
    },
    MultiDrawIndirectCount {
        buffer: Arc<Buffer<A>>,
        offset: BufferAddress,
        count_buffer: Arc<Buffer<A>>,
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
    WriteTimestamp {
        query_set: Arc<QuerySet<A>>,
        query_index: u32,
    },
    BeginOcclusionQuery {
        query_index: u32,
    },
    EndOcclusionQuery,
    BeginPipelineStatisticsQuery {
        query_set: Arc<QuerySet<A>>,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(Arc<RenderBundle<A>>),
}

#[cfg(feature = "trace")]
impl<A: HalApi> From<&ArcRenderCommand<A>> for RenderCommand {
    fn from(value: &ArcRenderCommand<A>) -> Self {
        use crate::resource::Resource as _;

        match value {
            ArcRenderCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => RenderCommand::SetBindGroup {
                index: *index,
                num_dynamic_offsets: *num_dynamic_offsets,
                bind_group_id: bind_group.as_info().id(),
            },

            ArcRenderCommand::SetPipeline(pipeline) => {
                RenderCommand::SetPipeline(pipeline.as_info().id())
            }

            ArcRenderCommand::SetPushConstant {
                offset,
                size_bytes,
                values_offset,
                stages,
            } => RenderCommand::SetPushConstant {
                offset: *offset,
                size_bytes: *size_bytes,
                values_offset: *values_offset,
                stages: *stages,
            },

            ArcRenderCommand::PushDebugGroup { color, len } => RenderCommand::PushDebugGroup {
                color: *color,
                len: *len,
            },

            ArcRenderCommand::PopDebugGroup => RenderCommand::PopDebugGroup,

            ArcRenderCommand::InsertDebugMarker { color, len } => {
                RenderCommand::InsertDebugMarker {
                    color: *color,
                    len: *len,
                }
            }

            ArcRenderCommand::WriteTimestamp {
                query_set,
                query_index,
            } => RenderCommand::WriteTimestamp {
                query_set_id: query_set.as_info().id(),
                query_index: *query_index,
            },

            ArcRenderCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => RenderCommand::BeginPipelineStatisticsQuery {
                query_set_id: query_set.as_info().id(),
                query_index: *query_index,
            },

            ArcRenderCommand::EndPipelineStatisticsQuery => {
                RenderCommand::EndPipelineStatisticsQuery
            }
            ArcRenderCommand::SetIndexBuffer {
                buffer,
                index_format,
                offset,
                size,
            } => RenderCommand::SetIndexBuffer {
                buffer_id: buffer.as_info().id(),
                index_format: *index_format,
                offset: *offset,
                size: *size,
            },

            ArcRenderCommand::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => RenderCommand::SetVertexBuffer {
                slot: *slot,
                buffer_id: buffer.as_info().id(),
                offset: *offset,
                size: *size,
            },

            ArcRenderCommand::SetBlendConstant(color) => RenderCommand::SetBlendConstant(*color),

            ArcRenderCommand::SetStencilReference(reference) => {
                RenderCommand::SetStencilReference(*reference)
            }

            ArcRenderCommand::SetViewport {
                rect,
                depth_min,
                depth_max,
            } => RenderCommand::SetViewport {
                rect: *rect,
                depth_min: *depth_min,
                depth_max: *depth_max,
            },

            ArcRenderCommand::SetScissor(scissor) => RenderCommand::SetScissor(*scissor),

            ArcRenderCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => RenderCommand::Draw {
                vertex_count: *vertex_count,
                instance_count: *instance_count,
                first_vertex: *first_vertex,
                first_instance: *first_instance,
            },

            ArcRenderCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => RenderCommand::DrawIndexed {
                index_count: *index_count,
                instance_count: *instance_count,
                first_index: *first_index,
                base_vertex: *base_vertex,
                first_instance: *first_instance,
            },

            ArcRenderCommand::MultiDrawIndirect {
                buffer,
                offset,
                count,
                indexed,
            } => RenderCommand::MultiDrawIndirect {
                buffer_id: buffer.as_info().id(),
                offset: *offset,
                count: *count,
                indexed: *indexed,
            },

            ArcRenderCommand::MultiDrawIndirectCount {
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_count,
                indexed,
            } => RenderCommand::MultiDrawIndirectCount {
                buffer_id: buffer.as_info().id(),
                offset: *offset,
                count_buffer_id: count_buffer.as_info().id(),
                count_buffer_offset: *count_buffer_offset,
                max_count: *max_count,
                indexed: *indexed,
            },

            ArcRenderCommand::BeginOcclusionQuery { query_index } => {
                RenderCommand::BeginOcclusionQuery {
                    query_index: *query_index,
                }
            }

            ArcRenderCommand::EndOcclusionQuery => RenderCommand::EndOcclusionQuery,

            ArcRenderCommand::ExecuteBundle(bundle) => {
                RenderCommand::ExecuteBundle(bundle.as_info().id())
            }
        }
    }
}
