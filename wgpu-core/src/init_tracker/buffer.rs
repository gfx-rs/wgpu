use super::{InitTracker, MemoryInitKind};
use crate::{hal_api::HalApi, resource::Buffer};
use std::{ops::Range, sync::Arc};

#[derive(Debug, Clone)]
pub(crate) struct BufferInitTrackerAction<A: HalApi> {
    pub buffer: Arc<Buffer<A>>,
    pub range: Range<wgt::BufferAddress>,
    pub kind: MemoryInitKind,
}

pub(crate) type BufferInitTracker = InitTracker<wgt::BufferAddress>;

impl BufferInitTracker {
    /// Checks if an action has/requires any effect on the initialization status
    /// and shrinks its range if possible.
    pub(crate) fn check_action<A: HalApi>(
        &self,
        action: &BufferInitTrackerAction<A>,
    ) -> Option<BufferInitTrackerAction<A>> {
        self.create_action(&action.buffer, action.range.clone(), action.kind)
    }

    /// Creates an action if it would have any effect on the initialization
    /// status and shrinks the range if possible.
    pub(crate) fn create_action<A: HalApi>(
        &self,
        buffer: &Arc<Buffer<A>>,
        query_range: Range<wgt::BufferAddress>,
        kind: MemoryInitKind,
    ) -> Option<BufferInitTrackerAction<A>> {
        self.check(query_range)
            .map(|range| BufferInitTrackerAction {
                buffer: buffer.clone(),
                range,
                kind,
            })
    }
}
