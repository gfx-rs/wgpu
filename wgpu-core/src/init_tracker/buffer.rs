use super::{InitTracker, MemoryInitKind};
use crate::id::BufferId;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) struct BufferInitTrackerAction {
    pub(crate) id: BufferId,
    pub(crate) range: Range<wgt::BufferAddress>,
    pub(crate) kind: MemoryInitKind,
}

pub(crate) type BufferInitTracker = InitTracker<wgt::BufferAddress>;
