use wgt::WasmNotSendSync;

use crate::{global, hal_api::HalApi, id};

use super::{RenderPass, RenderPassError};

/// Trait for type erasing RenderPass.
// TODO(#5124): wgpu-core's RenderPass trait should not be hal type dependent.
// Practically speaking this allows us merge gfx_select with type erasure:
// The alternative would be to introduce RenderPassId which then first needs to be looked up and then dispatch via gfx_select.
pub trait DynRenderPass: std::fmt::Debug + WasmNotSendSync {
    fn set_bind_group(
        &mut self,
        context: &global::Global,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), RenderPassError>;
    fn set_index_buffer(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), RenderPassError>;
    fn set_vertex_buffer(
        &mut self,
        context: &global::Global,
        slot: u32,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), RenderPassError>;
    fn set_pipeline(
        &mut self,
        context: &global::Global,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), RenderPassError>;
    fn set_push_constants(
        &mut self,
        context: &global::Global,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> Result<(), RenderPassError>;
    fn draw(
        &mut self,
        context: &global::Global,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), RenderPassError>;
    fn draw_indexed(
        &mut self,
        context: &global::Global,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), RenderPassError>;
    fn draw_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), RenderPassError>;
    fn draw_indexed_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), RenderPassError>;
    fn multi_draw_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError>;
    fn multi_draw_indexed_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError>;
    fn multi_draw_indirect_count(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError>;
    fn multi_draw_indexed_indirect_count(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError>;
    fn set_blend_constant(
        &mut self,
        context: &global::Global,
        color: wgt::Color,
    ) -> Result<(), RenderPassError>;
    fn set_scissor_rect(
        &mut self,
        context: &global::Global,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), RenderPassError>;
    fn set_viewport(
        &mut self,
        context: &global::Global,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), RenderPassError>;
    fn set_stencil_reference(
        &mut self,
        context: &global::Global,
        reference: u32,
    ) -> Result<(), RenderPassError>;
    fn push_debug_group(
        &mut self,
        context: &global::Global,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError>;
    fn pop_debug_group(&mut self, context: &global::Global) -> Result<(), RenderPassError>;
    fn insert_debug_marker(
        &mut self,
        context: &global::Global,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError>;
    fn write_timestamp(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError>;
    fn begin_occlusion_query(
        &mut self,
        context: &global::Global,
        query_index: u32,
    ) -> Result<(), RenderPassError>;
    fn end_occlusion_query(&mut self, context: &global::Global) -> Result<(), RenderPassError>;
    fn begin_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError>;
    fn end_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
    ) -> Result<(), RenderPassError>;
    fn execute_bundles(
        &mut self,
        context: &global::Global,
        bundles: &[id::RenderBundleId],
    ) -> Result<(), RenderPassError>;
    fn end(&mut self, context: &global::Global) -> Result<(), RenderPassError>;

    fn label(&self) -> Option<&str>;
}

impl<A: HalApi> DynRenderPass for RenderPass<A> {
    fn set_index_buffer(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_index_buffer(self, buffer_id, index_format, offset, size)
    }

    fn set_vertex_buffer(
        &mut self,
        context: &global::Global,
        slot: u32,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_vertex_buffer(self, slot, buffer_id, offset, size)
    }

    fn set_bind_group(
        &mut self,
        context: &global::Global,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_bind_group(self, index, bind_group_id, offsets)
    }

    fn set_pipeline(
        &mut self,
        context: &global::Global,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_pipeline(self, pipeline_id)
    }

    fn set_push_constants(
        &mut self,
        context: &global::Global,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_push_constants(self, stages, offset, data)
    }

    fn draw(
        &mut self,
        context: &global::Global,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_draw(
            self,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        )
    }

    fn draw_indexed(
        &mut self,
        context: &global::Global,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_draw_indexed(
            self,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    fn draw_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), RenderPassError> {
        context.render_pass_draw_indirect(self, buffer_id, offset)
    }

    fn draw_indexed_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), RenderPassError> {
        context.render_pass_draw_indexed_indirect(self, buffer_id, offset)
    }

    fn multi_draw_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_multi_draw_indirect(self, buffer_id, offset, count)
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_multi_draw_indexed_indirect(self, buffer_id, offset, count)
    }

    fn multi_draw_indirect_count(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_multi_draw_indirect_count(
            self,
            buffer_id,
            offset,
            count_buffer_id,
            count_buffer_offset,
            max_count,
        )
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        context: &global::Global,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_multi_draw_indexed_indirect_count(
            self,
            buffer_id,
            offset,
            count_buffer_id,
            count_buffer_offset,
            max_count,
        )
    }

    fn set_blend_constant(
        &mut self,
        context: &global::Global,
        color: wgt::Color,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_blend_constant(self, color)
    }

    fn set_scissor_rect(
        &mut self,
        context: &global::Global,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_scissor_rect(self, x, y, width, height)
    }

    fn set_viewport(
        &mut self,
        context: &global::Global,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_viewport(self, x, y, width, height, min_depth, max_depth)
    }

    fn set_stencil_reference(
        &mut self,
        context: &global::Global,
        reference: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_set_stencil_reference(self, reference)
    }

    fn push_debug_group(
        &mut self,
        context: &global::Global,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_push_debug_group(self, label, color)
    }

    fn pop_debug_group(&mut self, context: &global::Global) -> Result<(), RenderPassError> {
        context.render_pass_pop_debug_group(self)
    }

    fn insert_debug_marker(
        &mut self,
        context: &global::Global,
        label: &str,
        color: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_insert_debug_marker(self, label, color)
    }

    fn write_timestamp(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_write_timestamp(self, query_set_id, query_index)
    }

    fn begin_occlusion_query(
        &mut self,
        context: &global::Global,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_begin_occlusion_query(self, query_index)
    }

    fn end_occlusion_query(&mut self, context: &global::Global) -> Result<(), RenderPassError> {
        context.render_pass_end_occlusion_query(self)
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), RenderPassError> {
        context.render_pass_begin_pipeline_statistics_query(self, query_set_id, query_index)
    }

    fn end_pipeline_statistics_query(
        &mut self,
        context: &global::Global,
    ) -> Result<(), RenderPassError> {
        context.render_pass_end_pipeline_statistics_query(self)
    }

    fn execute_bundles(
        &mut self,
        context: &global::Global,
        bundles: &[id::RenderBundleId],
    ) -> Result<(), RenderPassError> {
        context.render_pass_execute_bundles(self, bundles)
    }

    fn end(&mut self, context: &global::Global) -> Result<(), RenderPassError> {
        context.render_pass_end(self)
    }

    fn label(&self) -> Option<&str> {
        self.label()
    }
}
