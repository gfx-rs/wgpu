use wgt::WasmNotSendSync;

use crate::{global, hal_api::HalApi, id};

use super::{ComputePass, ComputePassError};

/// Trait for type erasing ComputePass.
// TODO(#5124): wgpu-core's ComputePass trait should not be hal type dependent.
// Practically speaking this allows us merge gfx_select with type erasure:
// The alternative would be to introduce ComputePassId which then first needs to be looked up and then dispatch via gfx_select.
pub trait DynComputePass: std::fmt::Debug + WasmNotSendSync {
    fn run(&mut self, context: &global::Global) -> Result<(), ComputePassError>;
    fn set_bind_group(
        &mut self,
        context: &global::Global,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), ComputePassError>;
    fn set_pipeline(
        &mut self,
        context: &global::Global,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), ComputePassError>;
    fn set_push_constant(&mut self, context: &global::Global, offset: u32, data: &[u8]);
    fn dispatch_workgroups(
        &mut self,
        context: &global::Global,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    );
    fn dispatch_workgroups_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), ComputePassError>;
    fn push_debug_group(&mut self, context: &global::Global, label: &str, color: u32);
    fn pop_debug_group(&mut self, context: &global::Global);
    fn insert_debug_marker(&mut self, context: &global::Global, label: &str, color: u32);
    fn write_timestamp(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError>;
    fn begin_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError>;
    fn end_pipeline_statistics_query(&mut self, context: &global::Global);
}

impl<A: HalApi> DynComputePass for ComputePass<A> {
    fn run(&mut self, context: &global::Global) -> Result<(), ComputePassError> {
        context.command_encoder_run_compute_pass(self)
    }

    fn set_bind_group(
        &mut self,
        context: &global::Global,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), ComputePassError> {
        context.compute_pass_set_bind_group(self, index, bind_group_id, offsets)
    }

    fn set_pipeline(
        &mut self,
        context: &global::Global,
        pipeline_id: id::ComputePipelineId,
    ) -> Result<(), ComputePassError> {
        context.compute_pass_set_pipeline(self, pipeline_id)
    }

    fn set_push_constant(&mut self, context: &global::Global, offset: u32, data: &[u8]) {
        context.compute_pass_set_push_constant(self, offset, data)
    }

    fn dispatch_workgroups(
        &mut self,
        context: &global::Global,
        groups_x: u32,
        groups_y: u32,
        groups_z: u32,
    ) {
        context.compute_pass_dispatch_workgroups(self, groups_x, groups_y, groups_z)
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), ComputePassError> {
        context.compute_pass_dispatch_workgroups_indirect(self, buffer_id, offset)
    }

    fn push_debug_group(&mut self, context: &global::Global, label: &str, color: u32) {
        context.compute_pass_push_debug_group(self, label, color)
    }

    fn pop_debug_group(&mut self, context: &global::Global) {
        context.compute_pass_pop_debug_group(self)
    }

    fn insert_debug_marker(&mut self, context: &global::Global, label: &str, color: u32) {
        context.compute_pass_insert_debug_marker(self, label, color)
    }

    fn write_timestamp(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        context.compute_pass_write_timestamp(self, query_set_id, query_index)
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), ComputePassError> {
        context.compute_pass_begin_pipeline_statistics_query(self, query_set_id, query_index)
    }

    fn end_pipeline_statistics_query(&mut self, context: &global::Global) {
        context.compute_pass_end_pipeline_statistics_query(self)
    }
}
