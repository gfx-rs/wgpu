use super::{InitTracker, MemoryInitKind};
use crate::id::BufferId;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) struct BufferInitTrackerAction {
    pub id: BufferId,
    pub range: Range<wgt::BufferAddress>,
    pub kind: MemoryInitKind,
}

pub(crate) type BufferInitTracker = InitTracker<wgt::BufferAddress>;

impl BufferInitTracker {
    /// Checks if an action has/requires any effect on the initialization status
    /// and shrinks its range if possible.
    pub(crate) fn check_action(
        &self,
        action: &BufferInitTrackerAction,
    ) -> Option<BufferInitTrackerAction> {
        self.create_action(action.id, action.range.clone(), action.kind)
    }

    /// Creates an action if it would have any effect on the initialization
    /// status and shrinks the range if possible.
    pub(crate) fn create_action(
        &self,
        id: BufferId,
        query_range: Range<wgt::BufferAddress>,
        kind: MemoryInitKind,
    ) -> Option<BufferInitTrackerAction> {
        self.check(query_range)
            .map(|range| BufferInitTrackerAction { id, range, kind })
    }
}
