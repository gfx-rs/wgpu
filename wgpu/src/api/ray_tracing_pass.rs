use wgt::DynamicOffset;

use crate::{BindGroup, Label, RayTracingPipeline, api::SharedDeferredCommandBufferActions, dispatch};

/// In-progress recording of a ray tracing pass.
///
/// It can be created with [`CommandEncoder::begin_ray_tracing_pass`].
#[derive(Debug)]
pub struct RayTracingPass<'encoder> {
    pub(crate) inner: dispatch::DispatchRayTracingPass,

    /// Shared with CommandEncoder to enqueue deferred actions from within a pass.
    pub(crate) actions: SharedDeferredCommandBufferActions,

    /// This lifetime is used to protect the [`CommandEncoder`] from being used
    /// while the pass is alive. This needs to be PhantomDrop to prevent the lifetime
    /// from being shortened.
    pub(crate) _encoder_guard: crate::api::PhantomDrop<&'encoder ()>,
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPass<'_>: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(RayTracingPass<'_> => .inner);

impl RayTracingPass<'_> {
    /// Drops the lifetime relationship to the parent command encoder, making usage of
    /// the encoder while this pass is recorded a run-time error instead.
    ///
    /// Attention: As long as the ray tracing pass has not been ended, any mutating operation on the parent
    /// command encoder will cause a run-time error and invalidate it!
    /// By default, the lifetime constraint prevents this, but it can be useful
    /// to handle this at run time, such as when storing the pass and encoder in the same
    /// data structure.
    ///
    /// This operation has no effect on pass recording.
    /// It's a safe operation, since [`CommandEncoder`] is in a locked state as long as the pass is active
    /// regardless of the lifetime constraint or its absence.
    pub fn forget_lifetime(self) -> RayTracingPass<'static> {
        RayTracingPass {
            inner: self.inner,
            actions: self.actions,
            _encoder_guard: crate::api::PhantomDrop::default(),
        }
    }
    
    /// Sets the active ray tracing pipeline.
    pub fn set_pipeline(&mut self, pipeline: &RayTracingPipeline) {
        self.inner.set_pipeline(&pipeline.inner);
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `trace_rays()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub fn set_bind_group<'a, BG>(&mut self, index: u32, bind_group: BG, offsets: &[DynamicOffset])
    where
        Option<&'a BindGroup>: From<BG>,
    {
        let bg: Option<&BindGroup> = bind_group.into();
        let bg = bg.map(|bg| &bg.inner);
        self.inner.set_bind_group(index, bg, offsets);
    }
}


/// Describes the attachments of a ray tracing pass.
///
/// For use with [`CommandEncoder::begin_ray_tracing_pass`].
#[derive(Clone, Default, Debug)]
pub struct RayTracingPassDescriptor<'a> {
    /// Debug label of the ray tracing pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RayTracingPassDescriptor<'_>: Send, Sync);