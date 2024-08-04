use std::sync::Arc;

use crate::{
    binding_model::BindGroup,
    id,
    pipeline::ComputePipeline,
    resource::{Buffer, QuerySet},
};

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
    #[cfg(any(feature = "serde", feature = "replay"))]
    pub fn resolve_compute_command_ids(
        hub: &crate::hub::Hub,
        commands: &[ComputeCommand],
    ) -> Result<Vec<ArcComputeCommand>, super::ComputePassError> {
        use super::{ComputePassError, ComputePassErrorInner, PassErrorScope};

        let buffers_guard = hub.buffers.read();
        let bind_group_guard = hub.bind_groups.read();
        let query_set_guard = hub.query_sets.read();
        let pipelines_guard = hub.compute_pipelines.read();

        let resolved_commands: Vec<ArcComputeCommand> = commands
            .iter()
            .map(|c| -> Result<ArcComputeCommand, ComputePassError> {
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
                                scope: PassErrorScope::SetBindGroup,
                                inner: ComputePassErrorInner::InvalidBindGroupId(bind_group_id),
                            }
                        })?,
                    },

                    ComputeCommand::SetPipeline(pipeline_id) => ArcComputeCommand::SetPipeline(
                        pipelines_guard
                            .get_owned(pipeline_id)
                            .map_err(|_| ComputePassError {
                                scope: PassErrorScope::SetPipelineCompute,
                                inner: ComputePassErrorInner::InvalidPipelineId(pipeline_id),
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
                                    scope: PassErrorScope::Dispatch { indirect: true },
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
pub enum ArcComputeCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Arc<BindGroup>,
    },

    SetPipeline(Arc<ComputePipeline>),

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
        buffer: Arc<Buffer>,
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
        query_set: Arc<QuerySet>,
        query_index: u32,
    },

    BeginPipelineStatisticsQuery {
        query_set: Arc<QuerySet>,
        query_index: u32,
    },

    EndPipelineStatisticsQuery,
}
