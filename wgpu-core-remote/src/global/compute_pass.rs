use alloc::sync::Arc;

use alloc::borrow::Cow;

use parking_lot::Mutex;
use wgpu_core::command::{
    CommandEncoderError, ComputePass, ComputePassDescriptor, EncoderStateError, PassStateError,
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
    ) -> (ComputePass, Option<CommandEncoderError>) {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);

        let desc = ComputePassDescriptor {
            label: desc.label.as_deref().map(Cow::Borrowed),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: hub.query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                }),
        };

        cmd_enc.begin_compute_pass(&desc)
    }

    pub fn command_encoder_begin_compute_pass_with_id(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'_, PassTimestampWrites<id::QuerySetId>>,
        id_in: Option<id::ComputePassEncoderId>,
    ) -> (id::ComputePassEncoderId, Option<CommandEncoderError>) {
        let fid = self.hub.compute_passes.prepare(id_in);

        let (pass, err) = self.command_encoder_begin_compute_pass(encoder_id, desc);

        // no lock rank here because only one thread should be using compute pass
        // and it's only used by id variants of compute pass methods on global
        // so no deadlock (or concurrent lock) should happen in practise
        let id = fid.assign(Arc::new(Mutex::new(pass)));

        (id, err)
    }

    pub fn compute_pass_end(&self, pass: &mut ComputePass) -> Result<(), EncoderStateError> {
        pass.end()
    }

    pub fn compute_pass_end_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), EncoderStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_end(&mut pass)
    }

    pub fn compute_pass_drop(&self, pass_id: id::ComputePassEncoderId) {
        self.hub.compute_passes.remove(pass_id);
    }
}

impl Global {
    pub fn compute_pass_set_bind_group(
        &self,
        pass: &mut ComputePass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        pass.set_bind_group(
            index,
            bind_group_id.map(|bind_group_id| self.hub.bind_groups.get(bind_group_id)),
            offsets,
        )
    }

    pub fn compute_pass_set_bind_group_with_id(
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
        self.compute_pass_set_bind_group(&mut pass, index, bind_group_id, offsets)
    }

    pub fn compute_pass_set_pipeline(
        &self,
        pass: &mut ComputePass,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), PassStateError> {
        let pipeline = self.hub.compute_pipelines.get(pipeline_id);
        pass.set_pipeline(pipeline)
    }

    pub fn compute_pass_set_pipeline_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_set_pipeline(&mut pass, pipeline_id)
    }

    pub fn compute_pass_set_immediates(
        &self,
        pass: &mut ComputePass,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        pass.set_immediates(offset, data)
    }

    pub fn compute_pass_set_immediates_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_set_immediates(&mut pass, offset, data)
    }

    pub fn compute_pass_dispatch_workgroups(
        &self,
        pass: &mut ComputePass,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) -> Result<(), PassStateError> {
        pass.dispatch_workgroups(groups_x, groups_y, groups_z)
    }

    pub fn compute_pass_dispatch_workgroups_with_id(
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
        self.compute_pass_dispatch_workgroups(&mut pass, groups_x, groups_y, groups_z)
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass: &mut ComputePass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        pass.dispatch_workgroups_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn compute_pass_dispatch_workgroups_indirect_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_dispatch_workgroups_indirect(&mut pass, buffer_id, offset)
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        pass.push_debug_group(label, color)
    }

    pub fn compute_pass_push_debug_group_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_push_debug_group(&mut pass, label, color)
    }

    pub fn compute_pass_pop_debug_group(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), PassStateError> {
        pass.pop_debug_group()
    }

    pub fn compute_pass_pop_debug_group_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_pop_debug_group(&mut pass)
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass: &mut ComputePass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        pass.insert_debug_marker(label, color)
    }

    pub fn compute_pass_insert_debug_marker_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_insert_debug_marker(&mut pass, label, color)
    }

    pub fn compute_pass_write_timestamp(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let query_set = self.hub.query_sets.get(query_set_id);
        pass.write_timestamp(query_set, query_index)
    }

    pub fn compute_pass_write_timestamp_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_write_timestamp(&mut pass, query_set_id, query_index)
    }

    pub fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let query_set = self.hub.query_sets.get(query_set_id);
        pass.begin_pipeline_statistics_query(query_set, query_index)
    }

    pub fn compute_pass_begin_pipeline_statistics_query_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_begin_pipeline_statistics_query(&mut pass, query_set_id, query_index)
    }

    pub fn compute_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut ComputePass,
    ) -> Result<(), PassStateError> {
        pass.end_pipeline_statistics_query()
    }

    pub fn compute_pass_end_pipeline_statistics_query_with_id(
        &self,
        pass_id: id::ComputePassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.compute_passes.get(pass_id);
        let mut pass = pass
            .try_lock()
            .expect("ComputePasses should not be accessed concurrently");
        self.compute_pass_end_pipeline_statistics_query(&mut pass)
    }

    pub fn compute_pass_transition_resources(
        &self,
        pass: &mut ComputePass,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<id::BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<id::TextureViewId>>,
    ) -> Result<(), PassStateError> {
        pass.transition_resources(
            buffer_transitions.map(|bt| wgt::BufferTransition {
                buffer: self.hub.buffers.get(bt.buffer),
                state: bt.state,
            }),
            texture_transitions.map(|tt| wgt::TextureTransition {
                texture: self.hub.texture_views.get(tt.texture),
                selector: tt.selector,
                state: tt.state,
            }),
        )
    }
}
