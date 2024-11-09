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
        bind_group_id: Option<id::BindGroupId>,
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
        use super::{ComputePassError, PassErrorScope};

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
                    } => {
                        if bind_group_id.is_none() {
                            return Ok(ArcComputeCommand::SetBindGroup {
                                index,
                                num_dynamic_offsets,
                                bind_group: None,
                            });
                        }

                        let bind_group_id = bind_group_id.unwrap();
                        let bg = bind_group_guard.get(bind_group_id).get().map_err(|e| {
                            ComputePassError {
                                scope: PassErrorScope::SetBindGroup,
                                inner: e.into(),
                            }
                        })?;

                        ArcComputeCommand::SetBindGroup {
                            index,
                            num_dynamic_offsets,
                            bind_group: Some(bg),
                        }
                    }
                    ComputeCommand::SetPipeline(pipeline_id) => ArcComputeCommand::SetPipeline(
                        pipelines_guard
                            .get(pipeline_id)
                            .get()
                            .map_err(|e| ComputePassError {
                                scope: PassErrorScope::SetPipelineCompute,
                                inner: e.into(),
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
                            buffer: buffers_guard.get(buffer_id).get().map_err(|e| {
                                ComputePassError {
                                    scope: PassErrorScope::Dispatch { indirect: true },
                                    inner: e.into(),
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
                        query_set: query_set_guard.get(query_set_id).get().map_err(|e| {
                            ComputePassError {
                                scope: PassErrorScope::WriteTimestamp,
                                inner: e.into(),
                            }
                        })?,
                        query_index,
                    },

                    ComputeCommand::BeginPipelineStatisticsQuery {
                        query_set_id,
                        query_index,
                    } => ArcComputeCommand::BeginPipelineStatisticsQuery {
                        query_set: query_set_guard.get(query_set_id).get().map_err(|e| {
                            ComputePassError {
                                scope: PassErrorScope::BeginPipelineStatisticsQuery,
                                inner: e.into(),
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
        bind_group: Option<Arc<BindGroup>>,
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

    BeginPipelineStatisticsQuery {
        query_set: Arc<QuerySet>,
        query_index: u32,
    },

    EndPipelineStatisticsQuery,
}
