use crate::id;
use wgpu_core::command::{PassStateError, RenderBundleEncoder};

impl crate::global::Global {
    pub fn render_bundle_encoder_set_bind_group(
        &self,
        bundle: &mut RenderBundleEncoder,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), PassStateError> {
        bundle.set_bind_group(
            index,
            bind_group_id.map(|id| self.hub.bind_groups.get(id)),
            offsets,
        )
    }

    pub fn render_bundle_encoder_set_bind_group_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        index: u32,
        bind_group_id: Option<id::BindGroupId>,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.set_bind_group(
            index,
            bind_group_id.map(|id| self.hub.bind_groups.get(id)),
            offsets,
        )
    }

    pub fn render_bundle_encoder_set_pipeline(
        &self,
        bundle: &mut RenderBundleEncoder,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), PassStateError> {
        bundle.set_pipeline(self.hub.render_pipelines.get(pipeline_id))
    }

    pub fn render_bundle_encoder_set_pipeline_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        pipeline_id: id::RenderPipelineId,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.set_pipeline(self.hub.render_pipelines.get(pipeline_id))
    }

    pub fn render_bundle_encoder_set_vertex_buffer(
        &self,
        bundle: &mut RenderBundleEncoder,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), PassStateError> {
        bundle.set_vertex_buffer(
            slot,
            buffer_id.map(|id| self.hub.buffers.get(id)),
            offset,
            size,
        )
    }

    pub fn render_bundle_encoder_set_vertex_buffer_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        slot: u32,
        buffer_id: Option<id::BufferId>,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.set_vertex_buffer(
            slot,
            buffer_id.map(|id| self.hub.buffers.get(id)),
            offset,
            size,
        )
    }

    pub fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder: &mut RenderBundleEncoder,
        buffer: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), PassStateError> {
        encoder.set_index_buffer(self.hub.buffers.get(buffer), index_format, offset, size)
    }

    pub fn render_bundle_encoder_set_index_buffer_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.set_index_buffer(self.hub.buffers.get(buffer), index_format, offset, size)
    }

    pub fn render_bundle_encoder_set_immediates(
        &self,
        pass: &mut RenderBundleEncoder,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        pass.set_immediates(offset, data)
    }

    pub fn render_bundle_encoder_set_immediates_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        offset: u32,
        data: &[u8],
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.set_immediates(offset, data)
    }

    pub fn render_bundle_encoder_draw(
        &self,
        bundle: &mut RenderBundleEncoder,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        bundle.draw(vertex_count, instance_count, first_vertex, first_instance)
    }

    pub fn render_bundle_encoder_draw_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.draw(vertex_count, instance_count, first_vertex, first_instance)
    }

    pub fn render_bundle_encoder_draw_indexed(
        &self,
        bundle: &mut RenderBundleEncoder,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        bundle.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        )
    }

    pub fn render_bundle_encoder_draw_indexed_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

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
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), PassStateError> {
        bundle.draw_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_draw_indirect_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.draw_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), PassStateError> {
        bundle.draw_indexed_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_draw_indexed_indirect_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        buffer_id: id::BufferId,
        offset: wgt::BufferAddress,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.draw_indexed_indirect(self.hub.buffers.get(buffer_id), offset)
    }

    pub fn render_bundle_encoder_push_debug_group(
        &self,
        bundle: &mut RenderBundleEncoder,
        label: &str,
    ) -> Result<(), PassStateError> {
        bundle.push_debug_group(label)
    }

    pub fn render_bundle_encoder_push_debug_group_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        label: &str,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.push_debug_group(label)
    }

    pub fn render_bundle_encoder_pop_debug_group(
        &self,
        bundle: &mut RenderBundleEncoder,
    ) -> Result<(), PassStateError> {
        bundle.pop_debug_group()
    }

    pub fn render_bundle_encoder_pop_debug_group_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.pop_debug_group()
    }

    pub fn render_bundle_encoder_insert_debug_marker(
        &self,
        bundle: &mut RenderBundleEncoder,
        label: &str,
    ) -> Result<(), PassStateError> {
        bundle.insert_debug_marker(label)
    }

    pub fn render_bundle_encoder_insert_debug_marker_with_id(
        &self,
        bundle_encoder: id::RenderBundleEncoderId,
        label: &str,
    ) -> Result<(), PassStateError> {
        let bundle_encoder = self.hub.render_bundle_encoders.get(bundle_encoder);

        let mut bundle_encoder = bundle_encoder
            .try_lock()
            .expect("RenderBundleEncoders should not be accessed concurrently");

        bundle_encoder.insert_debug_marker(label)
    }
}
