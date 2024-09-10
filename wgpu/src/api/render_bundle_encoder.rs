use std::{marker::PhantomData, num::NonZeroU32, ops::Range, sync::Arc};

use crate::context::DynContext;
use crate::*;

/// Encodes a series of GPU operations into a reusable "render bundle".
///
/// It only supports a handful of render commands, but it makes them reusable.
/// It can be created with [`Device::create_render_bundle_encoder`].
/// It can be executed onto a [`CommandEncoder`] using [`RenderPass::execute_bundles`].
///
/// Executing a [`RenderBundle`] is often more efficient than issuing the underlying commands
/// manually.
///
/// Corresponds to [WebGPU `GPURenderBundleEncoder`](
/// https://gpuweb.github.io/gpuweb/#gpurenderbundleencoder).
#[derive(Debug)]
pub struct RenderBundleEncoder<'a> {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
    pub(crate) parent: &'a Device,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    pub(crate) _p: PhantomData<*const u8>,
}
static_assertions::assert_not_impl_any!(RenderBundleEncoder<'_>: Send, Sync);

/// Describes a [`RenderBundleEncoder`].
///
/// For use with [`Device::create_render_bundle_encoder`].
///
/// Corresponds to [WebGPU `GPURenderBundleEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundleencoderdescriptor).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RenderBundleEncoderDescriptor<'a> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The formats of the color attachments that this render bundle is capable to rendering to. This
    /// must match the formats of the color attachments in the render pass this render bundle is executed in.
    pub color_formats: &'a [Option<TextureFormat>],
    /// Information about the depth attachment that this render bundle is capable to rendering to. This
    /// must match the format of the depth attachments in the render pass this render bundle is executed in.
    pub depth_stencil: Option<RenderBundleDepthStencil>,
    /// Sample count this render bundle is capable of rendering to. This must match the pipelines and
    /// the render passes it is used in.
    pub sample_count: u32,
    /// If this render bundle will rendering to multiple array layers in the attachments at the same time.
    pub multiview: Option<NonZeroU32>,
}
static_assertions::assert_impl_all!(RenderBundleEncoderDescriptor<'_>: Send, Sync);

impl<'a> RenderBundleEncoder<'a> {
    /// Finishes recording and returns a [`RenderBundle`] that can be executed in other render passes.
    pub fn finish(self, desc: &RenderBundleDescriptor<'_>) -> RenderBundle {
        let data = DynContext::render_bundle_encoder_finish(&*self.context, self.data, desc);
        RenderBundle {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&'a BindGroup>,
        offsets: &[DynamicOffset],
    ) {
        let bg = bind_group.map(|x| x.data.as_ref());
        DynContext::render_bundle_encoder_set_bind_group(
            &*self.parent.context,
            self.data.as_mut(),
            index,
            bg,
            offsets,
        )
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        DynContext::render_bundle_encoder_set_pipeline(
            &*self.parent.context,
            self.data.as_mut(),
            pipeline.data.as_ref(),
        )
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderBundleEncoder::draw_indexed) on this [`RenderBundleEncoder`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        DynContext::render_bundle_encoder_set_index_buffer(
            &*self.parent.context,
            self.data.as_mut(),
            buffer_slice.buffer.data.as_ref(),
            index_format,
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderBundleEncoder`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [`VertexState::buffers`].
    ///
    /// [`draw`]: RenderBundleEncoder::draw
    /// [`draw_indexed`]: RenderBundleEncoder::draw_indexed
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        DynContext::render_bundle_encoder_set_vertex_buffer(
            &*self.parent.context,
            self.data.as_mut(),
            slot,
            buffer_slice.buffer.data.as_ref(),
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    /// Does not use an Index Buffer. If you need this see [`RenderBundleEncoder::draw_indexed`]
    ///
    /// Panics if vertices Range is outside of the range of the vertices range of any set vertex buffer.
    ///
    /// vertices: The range of vertices to draw.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for vertex_id in vertex_range {
    ///         let vertex = vertex[vertex_id];
    ///         vertex_shader(vertex, vertex_id, instance_id);
    ///     }
    /// }
    /// ```
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        DynContext::render_bundle_encoder_draw(
            &*self.parent.context,
            self.data.as_mut(),
            vertices,
            instances,
        )
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffer(s).
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`].
    /// The active vertex buffer(s) can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// Panics if indices Range is outside of the range of the indices range of any set index buffer.
    ///
    /// indices: The range of indices to draw.
    /// base_vertex: value added to each index value before indexing into the vertex buffers.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for index_index in index_range {
    ///         let vertex_id = index_buffer[index_index];
    ///         let adjusted_vertex_id = vertex_id + base_vertex;
    ///         let vertex = vertex[adjusted_vertex_id];
    ///         vertex_shader(vertex, adjusted_vertex_id, instance_id);
    ///     }
    /// }
    /// ```
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        DynContext::render_bundle_encoder_draw_indexed(
            &*self.parent.context,
            self.data.as_mut(),
            indices,
            base_vertex,
            instances,
        );
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        DynContext::render_bundle_encoder_draw_indirect(
            &*self.parent.context,
            self.data.as_mut(),
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::render_bundle_encoder_draw_indexed_indirect(
            &*self.parent.context,
            self.data.as_mut(),
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> RenderBundleEncoder<'a> {
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must have an alignment of 4.
    /// For example, with an offset of 4 and an array of `[u8; 8]`, that will write to the range
    /// of 4..12.
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
    pub fn set_push_constants(&mut self, stages: ShaderStages, offset: u32, data: &[u8]) {
        DynContext::render_bundle_encoder_set_push_constants(
            &*self.parent.context,
            self.data.as_mut(),
            stages,
            offset,
            data,
        );
    }
}
