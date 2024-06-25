use std::sync::Arc;

use crate::{
    binding_model::BindGroup,
    hal_api::HalApi,
    id,
    pipeline::ComputePipeline,
    resource::{Buffer, QuerySet},
};

use super::{ComputePassError, ComputePassErrorInner, PassErrorScope};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group_id: id::BindGroupId,
    },

    SetPipeline(id::ComputePipelineId),

    /// Set a range of push constants to values stored in `push_constant_data`.
    SetPushConstant {
        /// The byte offset within the push constant storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in `push_constant_data` of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        values_offset: u32,
    },

    Dispatch([u32; 3]),

    DispatchIndirect {
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
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

    BeginPipelineStatisticsQuery {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },

    EndPipelineStatisticsQuery,
}

impl ComputeCommand {
    /// Resolves all ids in a list of commands into the corresponding resource Arc.
    //
    // TODO: Once resolving is done on-the-fly during recording, this function should be only needed with the replay feature:
    // #[cfg(feature = "replay")]
    pub fn resolve_compute_command_ids<A: HalApi>(
        hub: &crate::hub::Hub<A>,
        commands: &[ComputeCommand],
    ) -> Result<Vec<ArcComputeCommand<A>>, ComputePassError> {
        let buffers_guard = hub.buffers.read();
        let bind_group_guard = hub.bind_groups.read();
        let query_set_guard = hub.query_sets.read();
        let pipelines_guard = hub.compute_pipelines.read();

        let resolved_commands: Vec<ArcComputeCommand<A>> = commands
            .iter()
            .map(|c| -> Result<ArcComputeCommand<A>, ComputePassError> {
                Ok(match *c {
                    ComputeCommand::SetBindGroup {
                        index,
                        num_dynamic_offsets,
                        bind_group_id,
                    } => ArcComputeCommand::SetBindGroup {
                        index,
                        num_dynamic_offsets,
                        bind_group: bind_group_guard.get_owned(bind_group_id).map_err(|_| {
                            ComputePassError {
                                scope: PassErrorScope::SetBindGroup(bind_group_id),
                                inner: ComputePassErrorInner::InvalidBindGroupId(bind_group_id),
                            }
                        })?,
                    },

                    ComputeCommand::SetPipeline(pipeline_id) => ArcComputeCommand::SetPipeline(
                        pipelines_guard
                            .get_owned(pipeline_id)
                            .map_err(|_| ComputePassError {
                                scope: PassErrorScope::SetPipelineCompute(pipeline_id),
                                inner: ComputePassErrorInner::InvalidPipeline(pipeline_id),
                            })?,
                    ),

                    ComputeCommand::SetPushConstant {
                        offset,
                        size_bytes,
                        values_offset,
                    } => ArcComputeCommand::SetPushConstant {
                        offset,
                        size_bytes,
                        values_offset,
                    },

                    ComputeCommand::Dispatch(dim) => ArcComputeCommand::Dispatch(dim),

                    ComputeCommand::DispatchIndirect { buffer_id, offset } => {
                        ArcComputeCommand::DispatchIndirect {
                            buffer: buffers_guard.get_owned(buffer_id).map_err(|_| {
                                ComputePassError {
                                    scope: PassErrorScope::Dispatch {
                                        indirect: true,
                                        pipeline: None, // TODO: not used right now, but once we do the resolve during recording we can use this again.
                                    },
                                    inner: ComputePassErrorInner::InvalidBufferId(buffer_id),
                                }
                            })?,
                            offset,
                        }
                    }

                    ComputeCommand::PushDebugGroup { color, len } => {
                        ArcComputeCommand::PushDebugGroup { color, len }
                    }

                    ComputeCommand::PopDebugGroup => ArcComputeCommand::PopDebugGroup,

                    ComputeCommand::InsertDebugMarker { color, len } => {
                        ArcComputeCommand::InsertDebugMarker { color, len }
                    }

                    ComputeCommand::WriteTimestamp {
                        query_set_id,
                        query_index,
                    } => ArcComputeCommand::WriteTimestamp {
                        query_set: query_set_guard.get_owned(query_set_id).map_err(|_| {
                            ComputePassError {
                                scope: PassErrorScope::WriteTimestamp,
                                inner: ComputePassErrorInner::InvalidQuerySet(query_set_id),
                            }
                        })?,
                        query_index,
                    },

                    ComputeCommand::BeginPipelineStatisticsQuery {
                        query_set_id,
                        query_index,
                    } => ArcComputeCommand::BeginPipelineStatisticsQuery {
                        query_set: query_set_guard.get_owned(query_set_id).map_err(|_| {
                            ComputePassError {
                                scope: PassErrorScope::BeginPipelineStatisticsQuery,
                                inner: ComputePassErrorInner::InvalidQuerySet(query_set_id),
                            }
                        })?,
                        query_index,
                    },

                    ComputeCommand::EndPipelineStatisticsQuery => {
                        ArcComputeCommand::EndPipelineStatisticsQuery
                    }
                })
            })
            .collect::<Result<Vec<_>, ComputePassError>>()?;
        Ok(resolved_commands)
    }
}

/// Equivalent to `ComputeCommand` but the Ids resolved into resource Arcs.
#[derive(Clone, Debug)]
pub enum ArcComputeCommand<A: HalApi> {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Arc<BindGroup<A>>,
    },

    SetPipeline(Arc<ComputePipeline<A>>),

    /// Set a range of push constants to values stored in `push_constant_data`.
    SetPushConstant {
        /// The byte offset within the push constant storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in `push_constant_data` of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        values_offset: u32,
    },

    Dispatch([u32; 3]),

    DispatchIndirect {
        buffer: Arc<Buffer<A>>,
        offset: wgt::BufferAddress,
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

    BeginPipelineStatisticsQuery {
        query_set: Arc<QuerySet<A>>,
        query_index: u32,
    },

    EndPipelineStatisticsQuery,
}

#[cfg(feature = "trace")]
impl<A: HalApi> From<&ArcComputeCommand<A>> for ComputeCommand {
    fn from(value: &ArcComputeCommand<A>) -> Self {
        use crate::resource::Resource as _;

        match value {
            ArcComputeCommand::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => ComputeCommand::SetBindGroup {
                index: *index,
                num_dynamic_offsets: *num_dynamic_offsets,
                bind_group_id: bind_group.as_info().id(),
            },

            ArcComputeCommand::SetPipeline(pipeline) => {
                ComputeCommand::SetPipeline(pipeline.as_info().id())
            }

            ArcComputeCommand::SetPushConstant {
                offset,
                size_bytes,
                values_offset,
            } => ComputeCommand::SetPushConstant {
                offset: *offset,
                size_bytes: *size_bytes,
                values_offset: *values_offset,
            },

            ArcComputeCommand::Dispatch(dim) => ComputeCommand::Dispatch(*dim),

            ArcComputeCommand::DispatchIndirect { buffer, offset } => {
                ComputeCommand::DispatchIndirect {
                    buffer_id: buffer.as_info().id(),
                    offset: *offset,
                }
            }

            ArcComputeCommand::PushDebugGroup { color, len } => ComputeCommand::PushDebugGroup {
                color: *color,
                len: *len,
            },

            ArcComputeCommand::PopDebugGroup => ComputeCommand::PopDebugGroup,

            ArcComputeCommand::InsertDebugMarker { color, len } => {
                ComputeCommand::InsertDebugMarker {
                    color: *color,
                    len: *len,
                }
            }

            ArcComputeCommand::WriteTimestamp {
                query_set,
                query_index,
            } => ComputeCommand::WriteTimestamp {
                query_set_id: query_set.as_info().id(),
                query_index: *query_index,
            },

            ArcComputeCommand::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => ComputeCommand::BeginPipelineStatisticsQuery {
                query_set_id: query_set.as_info().id(),
                query_index: *query_index,
            },

            ArcComputeCommand::EndPipelineStatisticsQuery => {
                ComputeCommand::EndPipelineStatisticsQuery
            }
        }
    }
}
