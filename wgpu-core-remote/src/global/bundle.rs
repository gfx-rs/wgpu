use wgpu_core_remote_types::encoders::{
    BindingCommand, DebugCommand, RenderBundleEncoderCommand, RenderCommand,
};

use crate::hub::Hub;
use crate::id;

impl crate::global::Global {
    pub fn render_bundle_encoder_set_bind_group(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[wgt::DynamicOffset],
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            bind_groups,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.set_bind_group(index, bind_group_id.map(|id| bind_groups.get(id)), offsets)
    }

    pub fn render_bundle_encoder_set_pipeline(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        pipeline_id: id::RenderPipelineId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            render_pipelines,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.set_pipeline(render_pipelines.get(pipeline_id))
    }

    pub fn render_bundle_encoder_set_vertex_buffer(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            buffers,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.set_vertex_buffer(slot, buffer_id.map(|id| buffers.get(id)), offset, size)
    }

    pub fn render_bundle_encoder_set_index_buffer(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            buffers,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.set_index_buffer(buffers.get(buffer), index_format, offset, size)
    }

    pub fn render_bundle_encoder_set_immediates(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        offset: u32,
        data: &[u8],
    ) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.set_immediates(offset, data)
    }

    pub fn render_bundle_encoder_draw(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.draw(vertex_count, instance_count, first_vertex, first_instance)
    }

    pub fn render_bundle_encoder_draw_indexed(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    pub fn render_bundle_encoder_draw_indirect(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            buffers,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.draw_indirect(buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            render_bundle_encoders,
            buffers,
            ..
        } = &mut *hub;
        let bundle_encoder = render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.draw_indexed_indirect(buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_push_debug_group(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        label: &str,
    ) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.push_debug_group(label)
    }

    pub fn render_bundle_encoder_pop_debug_group(&self, bundle_encoder: id::RenderBundleEncoderId) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.pop_debug_group()
    }

    pub fn render_bundle_encoder_insert_debug_marker(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        label: &str,
    ) {
        let mut hub = self.hub.borrow_mut();
        let bundle_encoder = hub.render_bundle_encoders.get_mut(bundle_encoder);

        bundle_encoder.insert_debug_marker(label)
    }
}

impl crate::global::Global {
    pub fn handle_render_bundle_encoder_command<'a>(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        command: RenderBundleEncoderCommand<'a>,
    ) {
        match command {
            RenderBundleEncoderCommand::BindingCommand(binding_command) => match binding_command {
                BindingCommand::SetBindGroup {
                    index,
                    bind_group,
                    dynamic_offsets,
                } => self.render_bundle_encoder_set_bind_group(
                    bundle_encoder,
                    index,
                    bind_group,
                    &dynamic_offsets,
                ),
                BindingCommand::SetImmediates { range_offset, data } => {
                    self.render_bundle_encoder_set_immediates(bundle_encoder, range_offset, &data)
                }
            },
            RenderBundleEncoderCommand::RenderCommand(render_command) => match render_command {
                RenderCommand::SetPipeline(id) => {
                    self.render_bundle_encoder_set_pipeline(bundle_encoder, id)
                }
                RenderCommand::SetIndexBuffer {
                    buffer,
                    index_format,
                    offset,
                    size,
                } => self.render_bundle_encoder_set_index_buffer(
                    bundle_encoder,
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
                } => self.render_bundle_encoder_set_vertex_buffer(
                    bundle_encoder,
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
                } => self.render_bundle_encoder_draw(
                    bundle_encoder,
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
                } => self.render_bundle_encoder_draw_indexed(
                    bundle_encoder,
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                ),
                RenderCommand::DrawIndirect {
                    indirect_buffer,
                    indirect_offset,
                } => self.render_bundle_encoder_draw_indirect(
                    bundle_encoder,
                    indirect_buffer,
                    indirect_offset,
                ),
                RenderCommand::DrawIndexedIndirect {
                    indirect_buffer,
                    indirect_offset,
                } => self.render_bundle_encoder_draw_indexed_indirect(
                    bundle_encoder,
                    indirect_buffer,
                    indirect_offset,
                ),
            },
            RenderBundleEncoderCommand::DebugCommand(debug_command) => match debug_command {
                DebugCommand::PushDebugGroup(label) => {
                    self.render_bundle_encoder_push_debug_group(bundle_encoder, &label)
                }
                DebugCommand::PopDebugGroup => {
                    self.render_bundle_encoder_pop_debug_group(bundle_encoder)
                }
                DebugCommand::InsertDebugMarker(label) => {
                    self.render_bundle_encoder_insert_debug_marker(bundle_encoder, &label)
                }
            },
            RenderBundleEncoderCommand::Finish {
                desc,
                render_bundle_id,
            } => self.render_bundle_encoder_finish(bundle_encoder, desc, render_bundle_id),
        }
    }
}
