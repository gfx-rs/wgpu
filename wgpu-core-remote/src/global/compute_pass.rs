use alloc::borrow::Cow;

use wgpu_core::command::{ComputePass, PassTimestampWrites};
use wgpu_core_remote_types::encoders::{
    BindingCommand, ComputePassDescriptor, ComputePassEncoderCommand, DebugCommand,
};
use wgt::{BufferAddress, DynamicOffset};

use crate::global::Global;
use crate::hub::Hub;
use crate::id;

impl Global {
    /// Creates a compute pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the `Locked` state.
    pub fn command_encoder_begin_compute_pass<'a>(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &ComputePassDescriptor<'a>,
        id_in: id::ComputePassEncoderId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            command_encoders,
            compute_passes,
            query_sets,
            ..
        } = &mut *hub;

        let cmd_enc = command_encoders.get(encoder_id);

        let desc = wgpu_core::command::ComputePassDescriptor {
            label: desc.label.as_deref().map(Cow::Borrowed),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index.to_std(),
                    end_of_pass_write_index: tw.end_of_pass_write_index.to_std(),
                }),
        };

        let pass = cmd_enc.begin_compute_pass(&desc);

        compute_passes.assign(id_in, pass);
    }

    pub fn compute_pass_end(&self, pass_id: id::ComputePassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.end()
    }

    pub fn compute_pass_remove(&self, pass_id: id::ComputePassEncoderId) -> ComputePass {
        let mut hub = self.hub.borrow_mut();
        hub.compute_passes.remove(pass_id)
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
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_passes,
            bind_groups,
            ..
        } = &mut *hub;
        let pass = compute_passes.get_mut(pass_id);
        pass.set_bind_group(
            index,
            bind_group_id.map(|bind_group_id| bind_groups.get(bind_group_id)),
            offsets,
        )
    }

    pub fn compute_pass_set_pipeline(
        &self,
        pass_id: id::ComputePassEncoderId,
        pipeline_id: id::ComputePipelineId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_passes,
            compute_pipelines,
            ..
        } = &mut *hub;
        let pass = compute_passes.get_mut(pass_id);
        let pipeline = compute_pipelines.get(pipeline_id);
        pass.set_pipeline(pipeline)
    }

    pub fn compute_pass_set_immediates(
        &self,
        pass_id: id::ComputePassEncoderId,
        offset: u32,
        data: &[u8],
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.set_immediates(offset, data)
    }

    pub fn compute_pass_dispatch_workgroups(
        &self,
        pass_id: id::ComputePassEncoderId,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.dispatch_workgroups(groups_x, groups_y, groups_z)
    }

    pub fn compute_pass_dispatch_workgroups_indirect(
        &self,
        pass_id: id::ComputePassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_passes,
            buffers,
            ..
        } = &mut *hub;
        let pass = compute_passes.get_mut(pass_id);
        pass.dispatch_workgroups_indirect(buffers.get(buffer_id), offset)
    }

    pub fn compute_pass_push_debug_group(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.push_debug_group(label, color)
    }

    pub fn compute_pass_pop_debug_group(&self, pass_id: id::ComputePassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.pop_debug_group()
    }

    pub fn compute_pass_insert_debug_marker(
        &self,
        pass_id: id::ComputePassEncoderId,
        label: &str,
        color: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.insert_debug_marker(label, color)
    }

    pub fn compute_pass_write_timestamp(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_passes,
            query_sets,
            ..
        } = &mut *hub;
        let pass = compute_passes.get_mut(pass_id);
        let query_set = query_sets.get(query_set_id);
        pass.write_timestamp(query_set, query_index)
    }

    pub fn compute_pass_begin_pipeline_statistics_query(
        &self,
        pass_id: id::ComputePassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            compute_passes,
            query_sets,
            ..
        } = &mut *hub;
        let pass = compute_passes.get_mut(pass_id);
        let query_set = query_sets.get(query_set_id);
        pass.begin_pipeline_statistics_query(query_set, query_index)
    }

    pub fn compute_pass_end_pipeline_statistics_query(&self, pass_id: id::ComputePassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.compute_passes.get_mut(pass_id);
        pass.end_pipeline_statistics_query()
    }
}

impl Global {
    pub fn handle_compute_pass_command(
        &self,
        pass_id: id::ComputePassEncoderId,
        command: ComputePassEncoderCommand,
    ) {
        match command {
            ComputePassEncoderCommand::BindingCommand(command) => match command {
                BindingCommand::SetBindGroup {
                    index,
                    bind_group,
                    dynamic_offsets,
                } => self.compute_pass_set_bind_group(pass_id, index, bind_group, &dynamic_offsets),
                BindingCommand::SetImmediates { range_offset, data } => {
                    self.compute_pass_set_immediates(pass_id, range_offset, &data)
                }
            },
            ComputePassEncoderCommand::SetPipeline(pipeline_id) => {
                self.compute_pass_set_pipeline(pass_id, pipeline_id)
            }
            ComputePassEncoderCommand::DispatchWorkgroups {
                workgroup_count_x,
                workgroup_count_y,
                workgroup_count_z,
            } => self.compute_pass_dispatch_workgroups(
                pass_id,
                workgroup_count_x,
                workgroup_count_y,
                workgroup_count_z,
            ),
            ComputePassEncoderCommand::DispatchWorkgroupsIndirect {
                indirect_buffer,
                indirect_offset,
            } => self.compute_pass_dispatch_workgroups_indirect(
                pass_id,
                indirect_buffer,
                indirect_offset,
            ),
            ComputePassEncoderCommand::DebugCommand(debug_command) => match debug_command {
                DebugCommand::PushDebugGroup(label) => {
                    self.compute_pass_push_debug_group(pass_id, &label, 0)
                }
                DebugCommand::PopDebugGroup => self.compute_pass_pop_debug_group(pass_id),
                DebugCommand::InsertDebugMarker(label) => {
                    self.compute_pass_insert_debug_marker(pass_id, &label, 0)
                }
            },
            ComputePassEncoderCommand::End => self.compute_pass_end(pass_id),
        }
    }
}
