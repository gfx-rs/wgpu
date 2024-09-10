use std::{marker::PhantomData, sync::Arc, thread};

use crate::context::DynContext;
use crate::*;

/// In-progress recording of a compute pass.
///
/// It can be created with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassEncoder`](
/// https://gpuweb.github.io/gpuweb/#compute-pass-encoder).
#[derive(Debug)]
pub struct ComputePass<'encoder> {
    /// The inner data of the compute pass, separated out so it's easy to replace the lifetime with 'static if desired.
    pub(crate) inner: ComputePassInner,

    /// This lifetime is used to protect the [`CommandEncoder`] from being used
    /// while the pass is alive.
    pub(crate) encoder_guard: PhantomData<&'encoder ()>,
}

impl<'encoder> ComputePass<'encoder> {
    /// Drops the lifetime relationship to the parent command encoder, making usage of
    /// the encoder while this pass is recorded a run-time error instead.
    ///
    /// Attention: As long as the compute pass has not been ended, any mutating operation on the parent
    /// command encoder will cause a run-time error and invalidate it!
    /// By default, the lifetime constraint prevents this, but it can be useful
    /// to handle this at run time, such as when storing the pass and encoder in the same
    /// data structure.
    ///
    /// This operation has no effect on pass recording.
    /// It's a safe operation, since [`CommandEncoder`] is in a locked state as long as the pass is active
    /// regardless of the lifetime constraint or its absence.
    pub fn forget_lifetime(self) -> ComputePass<'static> {
        ComputePass {
            inner: self.inner,
            encoder_guard: PhantomData,
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `dispatch()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&BindGroup>,
        offsets: &[DynamicOffset],
    ) {
        let bg = bind_group.map(|x| x.data.as_ref());
        DynContext::compute_pass_set_bind_group(
            &*self.inner.context,
            self.inner.data.as_mut(),
            index,
            bg,
            offsets,
        );
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        DynContext::compute_pass_set_pipeline(
            &*self.inner.context,
            self.inner.data.as_mut(),
            pipeline.data.as_ref(),
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::compute_pass_insert_debug_marker(
            &*self.inner.context,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::compute_pass_push_debug_group(
            &*self.inner.context,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::compute_pass_pop_debug_group(&*self.inner.context, self.inner.data.as_mut());
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        DynContext::compute_pass_dispatch_workgroups(
            &*self.inner.context,
            self.inner.data.as_mut(),
            x,
            y,
            z,
        );
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DispatchIndirectArgs`](crate::util::DispatchIndirectArgs).
    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::compute_pass_dispatch_workgroups_indirect(
            &*self.inner.context,
            self.inner.data.as_mut(),
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
    /// Set push constant data for subsequent dispatch calls.
    ///
    /// Write the bytes in `data` at offset `offset` within push constant
    /// storage.  Both `offset` and the length of `data` must be
    /// multiples of [`PUSH_CONSTANT_ALIGNMENT`], which is always 4.
    ///
    /// For example, if `offset` is `4` and `data` is eight bytes long, this
    /// call will write `data` to bytes `4..12` of push constant storage.
    pub fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
        DynContext::compute_pass_set_push_constants(
            &*self.inner.context,
            self.inner.data.as_mut(),
            offset,
            data,
        );
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
    /// Issue a timestamp command at this point in the queue. The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_write_timestamp(
            &*self.inner.context,
            self.inner.data.as_mut(),
            query_set.data.as_ref(),
            query_index,
        )
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
    /// Start a pipeline statistics query on this compute pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_begin_pipeline_statistics_query(
            &*self.inner.context,
            self.inner.data.as_mut(),
            query_set.data.as_ref(),
            query_index,
        );
    }

    /// End the pipeline statistics query on this compute pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        DynContext::compute_pass_end_pipeline_statistics_query(
            &*self.inner.context,
            self.inner.data.as_mut(),
        );
    }
}

#[derive(Debug)]
pub(crate) struct ComputePassInner {
    pub(crate) data: Box<Data>,
    pub(crate) context: Arc<C>,
}

impl Drop for ComputePassInner {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.compute_pass_end(self.data.as_mut());
        }
    }
}

/// Describes the timestamp writes of a compute pass.
///
/// For use with [`ComputePassDescriptor`].
/// At least one of `beginning_of_pass_write_index` and `end_of_pass_write_index` must be `Some`.
///
/// Corresponds to [WebGPU `GPUComputePassTimestampWrites`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepasstimestampwrites).
#[derive(Clone, Debug)]
pub struct ComputePassTimestampWrites<'a> {
    /// The query set to write to.
    pub query_set: &'a QuerySet,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePassTimestampWrites<'_>: Send, Sync);

/// Describes the attachments of a compute pass.
///
/// For use with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepassdescriptor).
#[derive(Clone, Default, Debug)]
pub struct ComputePassDescriptor<'a> {
    /// Debug label of the compute pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Defines which timestamp values will be written for this pass, and where to write them to.
    ///
    /// Requires [`Features::TIMESTAMP_QUERY`] to be enabled.
    pub timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePassDescriptor<'_>: Send, Sync);
