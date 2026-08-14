use alloc::sync::Arc;

use alloc::borrow::Cow;

use parking_lot::Mutex;
use wgpu_core::command::{
    CommandEncoderError, ComputePassDescriptor, EncoderStateError, PassStateError,
    PassTimestampWrites,
};
use wgt::{BufferAddress, DynamicOffset};

use crate::global::Global;
use crate::id;

impl Global {
    /// Creates a compute pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_begin_compute_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_, PassTimestampWrites<id::QuerySetId>>,
        id_in: Option<id::ComputePassEncoderId>,
    ) -> (id::ComputePassEncoderId, Option<CommandEncoderError>) {
        let fid = self.hub.compute_passes.prepare(id_in);

        let cmd_enc = self.hub.command_encoders.get(encoder_id);

        let desc = ComputePassDescriptor {
            label: desc.label.as_deref().map(Cow::Borrowed),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: self.hub.query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                }),
        };

        let (pass, err) = cmd_enc.begin_compute_pass(&desc);

        // no lock rank here because only one thread should be using compute pass
        // and it's only used by id variants of compute pass methods on global
        // so no deadlock (or concurrent lock) should happen in practise
        let id = fid.assign(Arc::new(Mutex::new(pass)));

        (id, err)
    }

    pub fn compute_pass_end(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), EncoderStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.end()
    }

    pub fn compute_pass_drop(&self, pass_id: id::ComputePassEncoderId) {
        self.hub.compute_passes.remove(pass_id);
    }
}

// Recording a compute pass.
//
// The only error that should be returned from these methods is
// `EncoderStateError::Ended`, when the pass has already ended and an immediate
// validation error is raised.
//
// All other errors should be stored in the pass for later reporting when
// `CommandEncoder.finish()` is called.
impl Global {
    pub fn compute_pass_set_bind_group(
        &self,
        pass_id: id::ComputePassEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.set_bind_group(
            index,
            bind_group_id.map(|bind_group_id| self.hub.bind_groups.get(bind_group_id)),
            offsets,
        )
    }

    pub fn compute_pass_set_pipeline(
        &self,
        pass_id: id::ComputePassEncoderId,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        let pipeline = self.hub.compute_pipelines.get(pipeline_id);
        pass.set_pipeline(pipeline)
    }

    pub fn compute_pass_set_immediates(
        &self,
        pass_id: id::ComputePassEncoderId,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.set_immediates(offset, data)
    }

    pub fn compute_pass_dispatch_workgroups(
        &self,
        pass_id: id::ComputePassEncoderId,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.dispatch_workgroups(groups_x, groups_y, groups_z)
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_id: id::ComputePassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.dispatch_workgroups_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.push_debug_group(label, color)
    }

    pub fn compute_pass_pop_debug_group(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.pop_debug_group()
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.insert_debug_marker(label, color)
    }

    pub fn compute_pass_write_timestamp(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        let query_set = self.hub.query_sets.get(query_set_id);
        pass.write_timestamp(query_set, query_index)
    }

    pub fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        let query_set = self.hub.query_sets.get(query_set_id);
        pass.begin_pipeline_statistics_query(query_set, query_index)
    }

    pub fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        pass.end_pipeline_statistics_query()
    }
}
