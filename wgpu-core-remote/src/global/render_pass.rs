use alloc::borrow::Cow;
use wgpu_core_remote_types::encoders::{
    BindingCommand, DebugCommand, RenderCommand, RenderPassDescriptor, RenderPassEncoderCommand,
};

use wgpu_core::command::{
    PassTimestampWrites, RenderPass, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    ResolvedRenderPassDescriptor,
};
use wgpu_core_remote_types::ffi::FfiOption;
use wgt::{BufferAddress, BufferSize, Color, DynamicOffset, IndexFormat};

use crate::global::Global;
use crate::hub::Hub;
use crate::id;

impl Global {
    /// Creates a render pass.
    ///
    /// If creation fails, an invalid pass is returned. Attempting to record
    /// commands into an invalid pass is permitted, but a validation error will
    /// ultimately be generated when the parent encoder is finished, and it is
    /// not possible to run any commands from the invalid pass.
    ///
    /// If successful, puts the encoder into the `Locked` state.
    pub fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &RenderPassDescriptor<'a>,
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
                            depth_slice: at.depth_slice.to_std(),
                            resolve_target: at
                                .resolve_target
                                .as_ref()
                                .map(|rt| texture_views.get(*rt)),
                            load_op: at.load_op.to_wgt(),
                            store_op: at.store_op,
                        })
                    })
                    .collect(),
            ),
            depth_stencil_attachment: desc.depth_stencil_attachment.as_ref().map(|at| {
                RenderPassDepthStencilAttachment {
                    view: texture_views.get(at.view),
                    depth: wgpu_core::command::PassChannel {
                        load_op: at
                            .depth
                            .load_op
                            .to_std()
                            .map(|x| x.map_clear_value(FfiOption::to_std).to_wgt()),
                        store_op: at.depth.store_op.to_std(),
                        read_only: at.depth.read_only,
                    },
                    stencil: wgpu_core::command::PassChannel {
                        load_op: at
                            .stencil
                            .load_op
                            .to_std()
                            .map(|x| x.map_clear_value(FfiOption::to_std).to_wgt()),
                        store_op: at.stencil.store_op.to_std(),
                        read_only: at.stencil.read_only,
                    },
                }
            }),
            timestamp_writes: desc
                .timestamp_writes
                .as_ref()
                .map(|tw| PassTimestampWrites {
                    query_set: query_sets.get(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index.to_std(),
                    end_of_pass_write_index: tw.end_of_pass_write_index.to_std(),
                }),
            occlusion_query_set: desc
                .occlusion_query_set
                .as_ref()
                .map(|query_set| query_sets.get(*query_set)),
            multiview_mask: None,
        };

        let render_pass = cmd_enc.begin_render_pass(desc);

        render_passes.assign(id_in, render_pass);
    }

    pub fn render_pass_end(&self, pass: id::RenderPassEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let pass = hub.render_passes.get_mut(pass);
        pass.end()
    }

    pub fn render_pass_remove(&self, pass: id::RenderPassEncoderId) -> RenderPass {
        let mut hub = self.hub.borrow_mut();
        hub.render_passes.remove(pass)
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

impl Global {
    pub fn handle_render_pass_command(
        &self,
        pass_id: id::RenderPassEncoderId,
        command: RenderPassEncoderCommand,
    ) {
        match command {
            RenderPassEncoderCommand::SetViewport {
                x,
                y,
                width,
                height,
                min_depth,
                max_depth,
            } => self.render_pass_set_viewport(pass_id, x, y, width, height, min_depth, max_depth),
            RenderPassEncoderCommand::SetScissorRect {
                x,
                y,
                width,
                height,
            } => self.render_pass_set_scissor_rect(pass_id, x, y, width, height),
            RenderPassEncoderCommand::SetBlendConstant(color) => {
                self.render_pass_set_blend_constant(pass_id, color)
            }
            RenderPassEncoderCommand::SetStencilReference(value) => {
                self.render_pass_set_stencil_reference(pass_id, value)
            }
            RenderPassEncoderCommand::BeginOcclusionQuery(query_index) => {
                self.render_pass_begin_occlusion_query(pass_id, query_index)
            }
            RenderPassEncoderCommand::EndOcclusionQuery => {
                self.render_pass_end_occlusion_query(pass_id)
            }
            RenderPassEncoderCommand::ExecuteBundles(ids) => {
                self.render_pass_execute_bundles(pass_id, &ids)
            }
            RenderPassEncoderCommand::BindingCommand(binding_command) => match binding_command {
                BindingCommand::SetBindGroup {
                    index,
                    bind_group,
                    dynamic_offsets,
                } => self.render_pass_set_bind_group(pass_id, index, bind_group, &dynamic_offsets),
                BindingCommand::SetImmediates { range_offset, data } => {
                    self.render_pass_set_immediates(pass_id, range_offset, &data)
                }
            },
            RenderPassEncoderCommand::RenderCommand(render_command) => match render_command {
                RenderCommand::SetPipeline(pipeline_id) => {
                    self.render_pass_set_pipeline(pass_id, pipeline_id)
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    index_format,
                    offset,
                    size,
                } => self.render_pass_set_index_buffer(
                    pass_id,
                    buffer,
                    index_format,
                    offset,
                    size.and_then(core::num::NonZeroU64::new), // pass size directly once https://github.com/gfx-rs/wgpu/issues/3170 is resolved
                ),
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer,
                    offset,
                    size,
                } => self.render_pass_set_vertex_buffer(
                    pass_id,
                    slot,
                    buffer,
                    offset,
                    size.and_then(core::num::NonZeroU64::new), // pass size directly once https://github.com/gfx-rs/wgpu/issues/3170 is resolved
                ),
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => self.render_pass_draw(
                    pass_id,
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                ),
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => self.render_pass_draw_indexed(
                    pass_id,
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                ),
                RenderCommand::DrawIndirect {
                    indirect_buffer,
                    indirect_offset,
                } => self.render_pass_draw_indirect(pass_id, indirect_buffer, indirect_offset),
                RenderCommand::DrawIndexedIndirect {
                    indirect_buffer,
                    indirect_offset,
                } => self.render_pass_draw_indexed_indirect(
                    pass_id,
                    indirect_buffer,
                    indirect_offset,
                ),
            },
            RenderPassEncoderCommand::DebugCommand(debug_command) => match debug_command {
                DebugCommand::PushDebugGroup(label) => {
                    self.render_pass_push_debug_group(pass_id, &label, 0)
                }
                DebugCommand::PopDebugGroup => self.render_pass_pop_debug_group(pass_id),
                DebugCommand::InsertDebugMarker(label) => {
                    self.render_pass_insert_debug_marker(pass_id, &label, 0)
                }
            },
            RenderPassEncoderCommand::End => self.render_pass_end(pass_id),
        }
    }
}
