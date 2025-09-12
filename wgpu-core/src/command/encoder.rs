use alloc::{sync::Arc, vec::Vec};

use crate::{
    command::memory_init::CommandBufferTextureMemoryActions, device::Device,
    init_tracker::BufferInitTrackerAction, ray_tracing::AsAction, snatch::SnatchGuard,
    track::Tracker,
};

/// State applicable when encoding commands onto a compute pass, or onto a
/// render pass, or directly with a command encoder.
pub(crate) struct EncodingState<'snatch_guard, 'cmd_enc, 'raw_encoder> {
    pub(crate) device: &'cmd_enc Arc<Device>,

    pub(crate) raw_encoder: &'raw_encoder mut dyn hal::DynCommandEncoder,

    pub(crate) tracker: &'cmd_enc mut Tracker,
    pub(crate) buffer_memory_init_actions: &'cmd_enc mut Vec<BufferInitTrackerAction>,
    pub(crate) texture_memory_actions: &'cmd_enc mut CommandBufferTextureMemoryActions,
    pub(crate) as_actions: &'cmd_enc mut Vec<AsAction>,
    pub(crate) indirect_draw_validation_resources:
        &'cmd_enc mut crate::indirect_validation::DrawResources,

    pub(crate) snatch_guard: &'snatch_guard SnatchGuard<'snatch_guard>,

    /// Current debug scope nesting depth.
    ///
    /// When encoding a compute or render pass, this is the depth of debug
    /// scopes in the pass, not the depth of debug scopes in the parent encoder.
    pub(crate) debug_scope_depth: &'cmd_enc mut u32,
}
