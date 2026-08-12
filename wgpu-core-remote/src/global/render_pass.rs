use alloc::borrow::Cow;
use alloc::sync::Arc;
use core::num::NonZeroU32;

use parking_lot::Mutex;
use wgpu_core::command::{
    CommandEncoderError, EncoderStateError, PassStateError, PassTimestampWrites, RenderPass,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassError,
    ResolvedRenderPassDescriptor,
};
use wgpu_core::Label;
use wgt::{BufferAddress, BufferSize, Color, DynamicOffset, IndexFormat};

use crate::global::Global;
use crate::id;

/// Describes the attachments of a render pass.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderPassDescriptor<'a> {
    pub label: Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments: Cow<'a, [Option<RenderPassColorAttachment<id::TextureViewId>>]>,
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<id::TextureViewId>>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PassTimestampWrites<id::QuerySetId>>,
    /// Defines where the occlusion query results will be stored for this pass.
    pub occlusion_query_set: Option<id::QuerySetId>,
    /// The multiview array layers that will be used
    pub multiview_mask: Option<NonZeroU32>,
}

impl Global {
    /// Creates a render pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the [`Locked`] state.
    ///
    /// [`Locked`]: crate::command::CommandEncoderStatus::Locked
    pub fn command_encoder_begin_render_pass(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &RenderPassDescriptor<'_>,
    ) -> (RenderPass, Option<CommandEncoderError>) {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);

        let texture_views = hub.texture_views.read();
        let query_sets = hub.query_sets.read();

        let desc = ResolvedRenderPassDescriptor {
            label: desc.label.as_deref().map(Cow::Borrowed),
            color_attachments: Cow::Owned(
                desc.color_attachments
                    .iter()
                    .map(|at| {
                        at.as_ref().map(|at| RenderPassColorAttachment {
                            view: texture_views.get(at.view),
                            depth_slice: at.depth_slice,
                            resolve_target: at
                                .resolve_target
                                .as_ref()
                                .map(|rt| texture_views.get(*rt)),
                            load_op: at.load_op,
                            store_op: at.store_op,
                        })
                    })
                    .collect(),
            ),
            depth_stencil_attachment: desc.depth_stencil_attachment.as_ref().map(|at| {
                RenderPassDepthStencilAttachment {
                    view: texture_views.get(at.view),
                    depth: at.depth.clone(),
                    stencil: at.stencil.clone(),
                }
            }),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                }),
            occlusion_query_set: desc
                .occlusion_query_set
                .as_ref()
                .map(|query_set| query_sets.get(*query_set)),
            multiview_mask: desc.multiview_mask,
        };

        drop(texture_views);
        drop(query_sets);

        cmd_enc.begin_render_pass(desc)
    }

    pub fn command_encoder_begin_render_pass_with_id(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &RenderPassDescriptor<'_>,
        id_in: Option<id::RenderPassEncoderId>,
    ) -> (id::RenderPassEncoderId, Option<CommandEncoderError>) {
        let hub = &self.hub;
        let fid = hub.render_passes.prepare(id_in);
        let (render_pass, error) = self.command_encoder_begin_render_pass(encoder_id, desc);
        // no lock rank here because only one thread should be using renderpass
        // and it's only used by id variants of render pass methods on global
        // so no deadlock (or concurrent lock) should happen in practise
        let id = fid.assign(Arc::new(Mutex::new(render_pass)));
        (id, error)
    }

    pub fn render_pass_end(&self, pass: &mut RenderPass) -> Result<(), EncoderStateError> {
        pass.end()
    }

    pub fn render_pass_end_with_id(
        &self,
        pass: id::RenderPassEncoderId,
    ) -> Result<(), EncoderStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be accessed concurrently");
        self.render_pass_end(&mut pass)
    }

    pub fn render_pass_drop(&self, pass: id::RenderPassEncoderId) {
        self.hub.render_passes.remove(pass);
    }
}

impl Global {
    pub fn render_pass_set_bind_group(
        &self,
        pass: &mut RenderPass,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        pass.set_bind_group(
            index,
            bind_group_id.map(|id| self.hub.bind_groups.get(id)),
            offsets,
        )
    }

    pub fn render_pass_set_bind_group_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_bind_group(&mut pass, index, bind_group_id, offsets)
    }

    pub fn render_pass_set_pipeline(
        &self,
        pass: &mut RenderPass,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), PassStateError> {
        let pipeline = self.resolve_render_pipeline_id(pipeline_id);
        pass.set_pipeline(pipeline)
    }

    pub fn render_pass_set_pipeline_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_pipeline(&mut pass, pipeline_id)
    }

    pub fn render_pass_set_index_buffer(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), PassStateError> {
        pass.set_index_buffer(
            self.resolve_buffer_id(buffer_id),
            index_format,
            offset,
            size,
        )
    }

    pub fn render_pass_set_index_buffer_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_index_buffer(&mut pass, buffer_id, index_format, offset, size)
    }

    pub fn render_pass_set_vertex_buffer(
        &self,
        pass: &mut RenderPass,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), PassStateError> {
        pass.set_vertex_buffer(
            slot,
            buffer_id.map(|id| self.resolve_buffer_id(id)),
            offset,
            size,
        )
    }

    pub fn render_pass_set_vertex_buffer_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_vertex_buffer(&mut pass, slot, buffer_id, offset, size)
    }

    pub fn render_pass_set_blend_constant(
        &self,
        pass: &mut RenderPass,
        color: Color,
    ) -> Result<(), PassStateError> {
        pass.set_blend_constant(color)
    }

    pub fn render_pass_set_blend_constant_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        color: Color,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_blend_constant(&mut pass, color)
    }

    pub fn render_pass_set_stencil_reference(
        &self,
        pass: &mut RenderPass,
        value: u32,
    ) -> Result<(), PassStateError> {
        pass.set_stencil_reference(value)
    }

    pub fn render_pass_set_stencil_reference_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        value: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_stencil_reference(&mut pass, value)
    }

    pub fn render_pass_set_viewport(
        &self,
        pass: &mut RenderPass,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) -> Result<(), PassStateError> {
        pass.set_viewport(x, y, w, h, depth_min, depth_max)
    }

    pub fn render_pass_set_viewport_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_viewport(&mut pass, x, y, w, h, depth_min, depth_max)
    }

    pub fn render_pass_set_scissor_rect(
        &self,
        pass: &mut RenderPass,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), PassStateError> {
        pass.set_scissor_rect(x, y, w, h)
    }

    pub fn render_pass_set_scissor_rect_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_scissor_rect(&mut pass, x, y, w, h)
    }

    pub fn render_pass_set_immediates(
        &self,
        pass: &mut RenderPass,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        pass.set_immediates(offset, data)
    }

    pub fn render_pass_set_immediates_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_set_immediates(&mut pass, offset, data)
    }

    pub fn render_pass_draw(
        &self,
        pass: &mut RenderPass,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        pass.draw(vertex_count, instance_count, first_vertex, first_instance)
    }

    pub fn render_pass_draw_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_draw(
            &mut pass,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        )
    }

    pub fn render_pass_draw_indexed(
        &self,
        pass: &mut RenderPass,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        pass.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    pub fn render_pass_draw_indexed_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_draw_indexed(
            &mut pass,
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    pub fn render_pass_draw_mesh_tasks(
        &self,
        pass: &mut RenderPass,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<(), RenderPassError> {
        pass.draw_mesh_tasks(group_count_x, group_count_y, group_count_z)
    }

    pub fn render_pass_draw_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        pass.draw_indirect(self.resolve_buffer_id(buffer_id), offset)
    }

    pub fn render_pass_draw_indirect_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_draw_indirect(&mut pass, buffer_id, offset)
    }

    pub fn render_pass_draw_indexed_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        pass.draw_indexed_indirect(self.resolve_buffer_id(buffer_id), offset)
    }

    pub fn render_pass_draw_indexed_indirect_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_draw_indexed_indirect(&mut pass, buffer_id, offset)
    }

    pub fn render_pass_draw_mesh_tasks_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) -> Result<(), RenderPassError> {
        pass.draw_mesh_tasks_indirect(self.resolve_buffer_id(buffer_id), offset)
    }

    pub fn render_pass_multi_draw_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) -> Result<(), PassStateError> {
        pass.multi_draw_indirect(self.resolve_buffer_id(buffer_id), offset, count)
    }

    pub fn render_pass_multi_draw_indexed_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) -> Result<(), PassStateError> {
        pass.multi_draw_indexed_indirect(self.resolve_buffer_id(buffer_id), offset, count)
    }

    pub fn render_pass_multi_draw_mesh_tasks_indirect(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count: u32,
    ) -> Result<(), RenderPassError> {
        pass.multi_draw_mesh_tasks_indirect(self.resolve_buffer_id(buffer_id), offset, count)
    }

    pub fn render_pass_multi_draw_indirect_count(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) -> Result<(), PassStateError> {
        pass.multi_draw_indirect_count(
            self.resolve_buffer_id(buffer_id),
            offset,
            self.resolve_buffer_id(count_buffer_id),
            count_buffer_offset,
            max_count,
        )
    }

    pub fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) -> Result<(), PassStateError> {
        pass.multi_draw_indexed_indirect_count(
            self.resolve_buffer_id(buffer_id),
            offset,
            self.resolve_buffer_id(count_buffer_id),
            count_buffer_offset,
            max_count,
        )
    }

    pub fn render_pass_multi_draw_mesh_tasks_indirect_count(
        &self,
        pass: &mut RenderPass,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    ) -> Result<(), RenderPassError> {
        pass.multi_draw_mesh_tasks_indirect_count(
            self.resolve_buffer_id(buffer_id),
            offset,
            self.resolve_buffer_id(count_buffer_id),
            count_buffer_offset,
            max_count,
        )
    }

    pub fn render_pass_push_debug_group(
        &self,
        pass: &mut RenderPass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        pass.push_debug_group(label, color)
    }

    pub fn render_pass_push_debug_group_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_push_debug_group(&mut pass, label, color)
    }

    pub fn render_pass_pop_debug_group(&self, pass: &mut RenderPass) -> Result<(), PassStateError> {
        pass.pop_debug_group()
    }

    pub fn render_pass_pop_debug_group_with_id(
        &self,
        pass: id::RenderPassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_pop_debug_group(&mut pass)
    }

    pub fn render_pass_insert_debug_marker(
        &self,
        pass: &mut RenderPass,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        pass.insert_debug_marker(label, color)
    }

    pub fn render_pass_insert_debug_marker_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        label: &str,
        color: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_insert_debug_marker(&mut pass, label, color)
    }

    pub fn render_pass_write_timestamp(
        &self,
        pass: &mut RenderPass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        pass.write_timestamp(self.resolve_query_set_id(query_set_id), query_index)
    }

    pub fn render_pass_write_timestamp_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_write_timestamp(&mut pass, query_set_id, query_index)
    }

    pub fn render_pass_begin_occlusion_query(
        &self,
        pass: &mut RenderPass,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        pass.begin_occlusion_query(query_index)
    }

    pub fn render_pass_begin_occlusion_query_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_begin_occlusion_query(&mut pass, query_index)
    }

    pub fn render_pass_end_occlusion_query(
        &self,
        pass: &mut RenderPass,
    ) -> Result<(), PassStateError> {
        pass.end_occlusion_query()
    }

    pub fn render_pass_end_occlusion_query_with_id(
        &self,
        pass: id::RenderPassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_end_occlusion_query(&mut pass)
    }

    pub fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: &mut RenderPass,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        pass.begin_pipeline_statistics_query(self.resolve_query_set_id(query_set_id), query_index)
    }

    pub fn render_pass_begin_pipeline_statistics_query_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_begin_pipeline_statistics_query(&mut pass, query_set_id, query_index)
    }

    pub fn render_pass_end_pipeline_statistics_query(
        &self,
        pass: &mut RenderPass,
    ) -> Result<(), PassStateError> {
        pass.end_pipeline_statistics_query()
    }

    pub fn render_pass_end_pipeline_statistics_query_with_id(
        &self,
        pass: id::RenderPassEncoderId,
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_end_pipeline_statistics_query(&mut pass)
    }

    pub fn render_pass_execute_bundles(
        &self,
        pass: &mut RenderPass,
        render_bundle_ids: &[id::RenderBundleId],
    ) -> Result<(), PassStateError> {
        let hub = &self.hub;
        let bundles = hub.render_bundles.read();
        let render_bundles = render_bundle_ids
            .iter()
            .map(|&id| bundles.get(id))
            .collect::<Vec<_>>();

        pass.execute_bundles(&render_bundles)
    }

    pub fn render_pass_execute_bundles_with_id(
        &self,
        pass: id::RenderPassEncoderId,
        render_bundle_ids: &[id::RenderBundleId],
    ) -> Result<(), PassStateError> {
        let pass = self.hub.render_passes.get(pass);
        let mut pass = pass
            .try_lock()
            .expect("RenderPasses should not be used concurrently");
        self.render_pass_execute_bundles(&mut pass, render_bundle_ids)
    }
}
