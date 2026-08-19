use alloc::borrow::Cow;
use core::num::NonZeroU32;

use wgpu_core::command::{
    PassTimestampWrites, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    ResolvedRenderPassDescriptor,
};
use wgpu_core::Label;
use wgt::{BufferAddress, BufferSize, Color, DynamicOffset, IndexFormat};

use crate::global::Global;
use crate::hub::Hub;
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
        id_in: id::RenderPassEncoderId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            command_encoders,
            render_passes,
            texture_views,
            query_sets,
            ..
        } = &mut *hub;

        let cmd_enc = command_encoders.get(encoder_id);

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

        let render_pass = cmd_enc.begin_render_pass(desc);

        render_passes.assign(id_in, render_pass);
    }

    pub fn render_pass_end(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.end()
    }

    pub fn render_pass_drop(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        hub.render_passes.remove(pass);
    }
}

// Recording a render pass.
//
// The only error that should be returned from these methods is
// `EncoderStateError::Ended`, when the pass has already ended and an immediate
// validation error is raised.
//
// All other errors should be stored in the pass for later reporting when
// `CommandEncoder.finish()` is called.
impl Global {
    pub fn render_pass_set_bind_group(
        &self,
        pass: id::RenderPassEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[DynamicOffset],
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            bind_groups,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.set_bind_group(index, bind_group_id.map(|id| bind_groups.get(id)), offsets)
    }

    pub fn render_pass_set_pipeline(
        &self,
        pass: id::RenderPassEncoderId,
        pipeline_id: id::RenderPipelineId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            render_pipelines,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        let pipeline = render_pipelines.get(pipeline_id);
        pass.set_pipeline(pipeline)
    }

    pub fn render_pass_set_index_buffer(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            buffers,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.set_index_buffer(buffers.get(buffer_id), index_format, offset, size)
    }

    pub fn render_pass_set_vertex_buffer(
        &self,
        pass: id::RenderPassEncoderId,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            buffers,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.set_vertex_buffer(slot, buffer_id.map(|id| buffers.get(id)), offset, size)
    }

    pub fn render_pass_set_blend_constant(&self, pass: id::RenderPassEncoderId, color: Color) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.set_blend_constant(color)
    }

    pub fn render_pass_set_stencil_reference(&self, pass: id::RenderPassEncoderId, value: u32) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.set_stencil_reference(value)
    }

    pub fn render_pass_set_viewport(
        &self,
        pass: id::RenderPassEncoderId,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth_min: f32,
        depth_max: f32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.set_viewport(x, y, w, h, depth_min, depth_max)
    }

    pub fn render_pass_set_scissor_rect(
        &self,
        pass: id::RenderPassEncoderId,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.set_scissor_rect(x, y, w, h)
    }

    pub fn render_pass_set_immediates(
        &self,
        pass: id::RenderPassEncoderId,
        offset: u32,
        data: &[u8],
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.set_immediates(offset, data)
    }

    pub fn render_pass_draw(
        &self,
        pass: id::RenderPassEncoderId,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.draw(vertex_count, instance_count, first_vertex, first_instance)
    }

    pub fn render_pass_draw_indexed(
        &self,
        pass: id::RenderPassEncoderId,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    pub fn render_pass_draw_indirect(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            buffers,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.draw_indirect(buffers.get(buffer_id), offset)
    }

    pub fn render_pass_draw_indexed_indirect(
        &self,
        pass: id::RenderPassEncoderId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            buffers,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.draw_indexed_indirect(buffers.get(buffer_id), offset)
    }

    pub fn render_pass_push_debug_group(
        &self,
        pass: id::RenderPassEncoderId,
        label: &str,
        color: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.push_debug_group(label, color)
    }

    pub fn render_pass_pop_debug_group(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.pop_debug_group()
    }

    pub fn render_pass_insert_debug_marker(
        &self,
        pass: id::RenderPassEncoderId,
        label: &str,
        color: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.insert_debug_marker(label, color)
    }

    pub fn render_pass_write_timestamp(
        &self,
        pass: id::RenderPassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            query_sets,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.write_timestamp(query_sets.get(query_set_id), query_index)
    }

    pub fn render_pass_begin_occlusion_query(
        &self,
        pass: id::RenderPassEncoderId,
        query_index: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.begin_occlusion_query(query_index)
    }

    pub fn render_pass_end_occlusion_query(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.end_occlusion_query()
    }

    pub fn render_pass_begin_pipeline_statistics_query(
        &self,
        pass: id::RenderPassEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            query_sets,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        pass.begin_pipeline_statistics_query(query_sets.get(query_set_id), query_index)
    }

    pub fn render_pass_end_pipeline_statistics_query(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.end_pipeline_statistics_query()
    }

    pub fn render_pass_execute_bundles(
        &self,
        pass: id::RenderPassEncoderId,
        render_bundle_ids: &[id::RenderBundleId],
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_passes,
            render_bundles,
            ..
        } = &mut *hub;
        let pass = render_passes.get_mut(pass);
        let render_bundles = render_bundle_ids
            .iter()
            .map(|&id| render_bundles.get(id))
            .collect::<Vec<_>>();

        pass.execute_bundles(&render_bundles)
    }
}
