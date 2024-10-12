use crate::{
    binding_model::BindGroup,
    id,
    pipeline::RenderPipeline,
    resource::{Buffer, QuerySet},
};
use wgt::{BufferAddress, BufferSize, Color};

use std::{num::NonZeroU32, sync::Arc};

use super::{Rect, RenderBundle};

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group_id: Option<id::BindGroupId>,
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
    #[cfg(any(feature = "serde", feature = "replay"))]
    pub fn resolve_render_command_ids(
        hub: &crate::hub::Hub,
        commands: &[RenderCommand],
    ) -> Result<Vec<ArcRenderCommand>, super::RenderPassError> {
        use super::{DrawKind, PassErrorScope, RenderPassError};

        let buffers_guard = hub.buffers.read();
        let bind_group_guard = hub.bind_groups.read();
        let query_set_guard = hub.query_sets.read();
        let pipelines_guard = hub.render_pipelines.read();
        let render_bundles_guard = hub.render_bundles.read();

        let resolved_commands: Vec<ArcRenderCommand> =
            commands
                .iter()
                .map(|c| -> Result<ArcRenderCommand, RenderPassError> {
                    Ok(match *c {
                        RenderCommand::SetBindGroup {
                            index,
                            num_dynamic_offsets,
                            bind_group_id,
                        } => {
                            if bind_group_id.is_none() {
                                return Ok(ArcRenderCommand::SetBindGroup {
                                    index,
                                    num_dynamic_offsets,
                                    bind_group: None,
                                });
                            }

                            let bind_group_id = bind_group_id.unwrap();
                            let bg = bind_group_guard.get(bind_group_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::SetBindGroup,
                                    inner: e.into(),
                                }
                            })?;

                            ArcRenderCommand::SetBindGroup {
                                index,
                                num_dynamic_offsets,
                                bind_group: Some(bg),
                            }
                        }

                        RenderCommand::SetPipeline(pipeline_id) => ArcRenderCommand::SetPipeline(
                            pipelines_guard.get(pipeline_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::SetPipelineRender,
                                    inner: e.into(),
                                }
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
                            query_set: query_set_guard.get(query_set_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::WriteTimestamp,
                                    inner: e.into(),
                                }
                            })?,
                            query_index,
                        },

                        RenderCommand::BeginPipelineStatisticsQuery {
                            query_set_id,
                            query_index,
                        } => ArcRenderCommand::BeginPipelineStatisticsQuery {
                            query_set: query_set_guard.get(query_set_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::BeginPipelineStatisticsQuery,
                                    inner: e.into(),
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
                            buffer: buffers_guard.get(buffer_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::SetIndexBuffer,
                                    inner: e.into(),
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
                            buffer: buffers_guard.get(buffer_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::SetVertexBuffer,
                                    inner: e.into(),
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
                            buffer: buffers_guard.get(buffer_id).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::Draw {
                                        kind: if count.is_some() {
                                            DrawKind::MultiDrawIndirect
                                        } else {
                                            DrawKind::DrawIndirect
                                        },
                                        indexed,
                                    },
                                    inner: e.into(),
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
                            };
                            ArcRenderCommand::MultiDrawIndirectCount {
                                buffer: buffers_guard.get(buffer_id).get().map_err(|e| {
                                    RenderPassError {
                                        scope,
                                        inner: e.into(),
                                    }
                                })?,
                                offset,
                                count_buffer: buffers_guard.get(count_buffer_id).get().map_err(
                                    |e| RenderPassError {
                                        scope,
                                        inner: e.into(),
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
                            render_bundles_guard.get(bundle).get().map_err(|e| {
                                RenderPassError {
                                    scope: PassErrorScope::ExecuteBundle,
                                    inner: e.into(),
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
pub enum ArcRenderCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Option<Arc<BindGroup>>,
    },
    SetPipeline(Arc<RenderPipeline>),
    SetIndexBuffer {
        buffer: Arc<Buffer>,
        index_format: wgt::IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: Arc<Buffer>,
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
        buffer: Arc<Buffer>,
        offset: BufferAddress,
        /// Count of `None` represents a non-multi call.
        count: Option<NonZeroU32>,
        indexed: bool,
    },
    MultiDrawIndirectCount {
        buffer: Arc<Buffer>,
        offset: BufferAddress,
        count_buffer: Arc<Buffer>,
        count_buffer_offset: BufferAddress,
        max_count: u32,
        indexed: bool,
    },
    PushDebugGroup {
        #[cfg_attr(target_os = "emscripten", allow(dead_code))]
        color: u32,
        len: usize,
    },
    PopDebugGroup,
    InsertDebugMarker {
        #[cfg_attr(target_os = "emscripten", allow(dead_code))]
        color: u32,
        len: usize,
    },
    WriteTimestamp {
        query_set: Arc<QuerySet>,
        query_index: u32,
    },
    BeginOcclusionQuery {
        query_index: u32,
    },
    EndOcclusionQuery,
    BeginPipelineStatisticsQuery {
        query_set: Arc<QuerySet>,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(Arc<RenderBundle>),
}
