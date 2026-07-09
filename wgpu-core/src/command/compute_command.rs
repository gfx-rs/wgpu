use alloc::vec::Vec;

#[cfg(feature = "serde")]
use crate::command::serde_object_reference_struct;
use crate::command::{ArcReferences, ReferenceType};

#[cfg(feature = "serde")]
use macro_rules_attribute::apply;

/// cbindgen:ignore
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub enum ComputeCommand<R: ReferenceType> {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Option<R::BindGroup>,
    },

    SetPipeline(R::ComputePipeline),

    /// Bind (or, with `None`, unbind) a resource table as compute-pass encoder
    /// state (work item 0.7 of the bindless feature). The hal binding is emitted
    /// at dispatch time, using the bound pipeline layout's group count as the
    /// set index (D15).
    SetResourceTable {
        resource_table: Option<R::ResourceTable>,
    },

    /// Set a range of immediates to values stored in `immediates_data`.
    SetImmediate {
        /// The byte offset within the immediate data storage to write to. This
        /// must be a multiple of four.
        offset: u32,

        /// The immediate data to be written.
        data: Vec<u32>,
    },

    DispatchWorkgroups([u32; 3]),

    DispatchWorkgroupsIndirect {
        buffer: R::Buffer,
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
        query_set: R::QuerySet,
        query_index: u32,
    },

    BeginPipelineStatisticsQuery {
        query_set: R::QuerySet,
        query_index: u32,
    },

    EndPipelineStatisticsQuery,

    TransitionResources {
        buffer_transitions: Vec<wgt::BufferTransition<R::Buffer>>,
        texture_transitions: Vec<wgt::TextureTransition<R::TextureView>>,
    },
}

/// cbindgen:ignore
pub type ArcComputeCommand = ComputeCommand<ArcReferences>;
