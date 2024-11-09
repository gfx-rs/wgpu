use std::ops::Range;

use wgt::{BufferAddress, DynamicOffset, IndexFormat};

use crate::{BindGroup, Buffer, BufferSlice, RenderBundleEncoder, RenderPass, RenderPipeline};

/// Methods shared by [`RenderPass`] and [`RenderBundleEncoder`].
pub trait RenderEncoder<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&'a BindGroup>,
        offsets: &[DynamicOffset],
    );

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline);

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderEncoder::draw_indexed) on this [`RenderEncoder`] will
    /// use `buffer` as the source index buffer.
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat);

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderEncoder`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [VertexState::buffers](crate::VertexState::buffers).
    ///
    /// [`draw`]: RenderEncoder::draw
    /// [`draw_indexed`]: RenderEncoder::draw_indexed
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>);

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderEncoder::set_vertex_buffer`].
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderEncoder::set_vertex_buffer`].
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress);

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    );

    /// [`wgt::Features::PUSH_CONSTANTS`] must be enabled on the device in order to call this function.
    ///
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`wgt::PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must be aligned to the 4s, so we take an array of u32.
    /// For example, with an offset of 4 and an array of `[u32; 3]`, that will write to the range
    /// of 4..16.
    ///
    /// For each byte in the range of push constant data written, the union of the stages of all push constant
    /// ranges that covers that byte must be exactly `stages`. There's no good way of explaining this simply,
    /// so here are some examples:
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..4 Vertex
    /// - 4..8 Fragment
    /// ```
    ///
    /// You would need to upload this in two set_push_constants calls. First for the `Vertex` range, second for the `Fragment` range.
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..8  Vertex
    /// - 4..12 Fragment
    /// ```
    ///
    /// You would need to upload this in three set_push_constants calls. First for the `Vertex` only range 0..4, second
    /// for the `Vertex | Fragment` range 4..8, third for the `Fragment` range 8..12.
    fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]);
}

impl<'a> RenderEncoder<'a> for RenderPass<'a> {
    #[inline(always)]
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&'a BindGroup>,
        offsets: &[DynamicOffset],
    ) {
        Self::set_bind_group(self, index, bind_group, offsets);
    }

    #[inline(always)]
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        Self::set_pipeline(self, pipeline);
    }

    #[inline(always)]
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        Self::set_index_buffer(self, buffer_slice, index_format);
    }

    #[inline(always)]
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        Self::set_vertex_buffer(self, slot, buffer_slice);
    }

    #[inline(always)]
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        Self::draw(self, vertices, instances);
    }

    #[inline(always)]
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        Self::draw_indexed(self, indices, base_vertex, instances);
    }

    #[inline(always)]
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        Self::draw_indirect(self, indirect_buffer, indirect_offset);
    }

    #[inline(always)]
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        Self::draw_indexed_indirect(self, indirect_buffer, indirect_offset);
    }

    #[inline(always)]
    fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
        Self::set_push_constants(self, stages, offset, data);
    }
}

impl<'a> RenderEncoder<'a> for RenderBundleEncoder<'a> {
    #[inline(always)]
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&'a BindGroup>,
        offsets: &[DynamicOffset],
    ) {
        Self::set_bind_group(self, index, bind_group, offsets);
    }

    #[inline(always)]
    fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        Self::set_pipeline(self, pipeline);
    }

    #[inline(always)]
    fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        Self::set_index_buffer(self, buffer_slice, index_format);
    }

    #[inline(always)]
    fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        Self::set_vertex_buffer(self, slot, buffer_slice);
    }

    #[inline(always)]
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        Self::draw(self, vertices, instances);
    }

    #[inline(always)]
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        Self::draw_indexed(self, indices, base_vertex, instances);
    }

    #[inline(always)]
    fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        Self::draw_indirect(self, indirect_buffer, indirect_offset);
    }

    #[inline(always)]
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        Self::draw_indexed_indirect(self, indirect_buffer, indirect_offset);
    }

    #[inline(always)]
    fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
        Self::set_push_constants(self, stages, offset, data);
    }
}
