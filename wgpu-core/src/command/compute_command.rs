use std::sync::Arc;

use crate::{
    binding_model::BindGroup,
    hal_api::HalApi,
    id,
    pipeline::ComputePipeline,
    resource::{Buffer, QuerySet, Resource as _},
};

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group_id: id::BindGroupId,
    },

    SetPipeline(id::ComputePipelineId),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    SetPushConstant {
        /// The byte offset within the push constant storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
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

/// Equivalent to `ComputeCommand` but the Ids resolved into resource Arcs.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum ArcComputeCommand<A: HalApi> {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Arc<BindGroup<A>>,
    },

    SetPipeline(Arc<ComputePipeline<A>>),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    SetPushConstant {
        /// The byte offset within the push constant storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
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
