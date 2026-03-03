#[cfg(feature = "serde")]
use crate::command::serde_object_reference_struct;
use crate::command::{ArcReferences, ReferenceType};
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use macro_rules_attribute::apply;

/// cbindgen:ignore
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", apply(serde_object_reference_struct))]
pub enum RayTracingCommand<R: ReferenceType> {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: usize,
        bind_group: Option<R::BindGroup>,
    },

    SetPipeline(R::RayTracingPipeline),

    /// Set a range of immediates to values stored in `immediates_data`.
    SetImmediate {
        /// The byte offset within the immediate data storage to write to.  This
        /// must be a multiple of four.
        offset: u32,

        /// The immediate data to be written.
        data: Vec<u32>,
    },

    TraceRays([u32; 3]),

    PushDebugGroup {
        color: u32,
        len: usize,
    },

    PopDebugGroup,

    InsertDebugMarker {
        color: u32,
        len: usize,
    },
}

/// cbindgen:ignore
pub type ArcRayTracingCommand = RayTracingCommand<ArcReferences>;
