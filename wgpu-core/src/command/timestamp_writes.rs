use std::sync::Arc;

use crate::id;

/// Describes the writing of timestamp values in a render or compute pass.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PassTimestampWrites {
    /// The query set to write the timestamps to.
    pub query_set: id::QuerySetId,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}

/// Describes the writing of timestamp values in a render or compute pass with the query set resolved.
pub struct ArcPassTimestampWrites {
    /// The query set to write the timestamps to.
    pub query_set: Arc<crate::resource::QuerySet>,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}
