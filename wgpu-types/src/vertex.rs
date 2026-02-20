//! Types for defining vertex attributes and their buffers.

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: crate::BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: wst::VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [wst::VertexAttribute],
}
