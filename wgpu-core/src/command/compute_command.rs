use std::sync::Arc;

use crate::{
    binding_model::BindGroup,
    hal_api::HalApi,
    id,
    pipeline::ComputePipeline,
    resource::{Buffer, QuerySet},
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
