//! Types for defining vertex attributes and their buffers.

use nt::VertexFormat;
#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

use crate::{link_to_wgpu_docs, link_to_wgpu_item};

#[cfg(doc)]
use crate::Features;

/// Whether a vertex buffer is indexed by vertex or by instance.
///
/// Consider a call to [`RenderPass::draw`] like this:
///
/// ```ignore
/// render_pass.draw(vertices, instances)
/// ```
///
/// where `vertices` is a `Range<u32>` of vertex indices, and
/// `instances` is a `Range<u32>` of instance indices.
///
/// For this call, `wgpu` invokes the vertex shader entry point once
/// for every possible `(v, i)` pair, where `v` is drawn from
/// `vertices` and `i` is drawn from `instances`. These invocations
/// may happen in any order, and will usually run in parallel.
///
/// Each vertex buffer has a step mode, established by the
/// [`step_mode`] field of its [`VertexBufferLayout`], given when the
/// pipeline was created. Buffers whose step mode is [`Vertex`] use
/// `v` as the index into their contents, whereas buffers whose step
/// mode is [`Instance`] use `i`. The indicated buffer element then
/// contributes zero or more attribute values for the `(v, i)` vertex
/// shader invocation to use, based on the [`VertexBufferLayout`]'s
/// [`attributes`] list.
///
/// You can visualize the results from all these vertex shader
/// invocations as a matrix with a row for each `i` from `instances`,
/// and with a column for each `v` from `vertices`. In one sense, `v`
/// and `i` are symmetrical: both are used to index vertex buffers and
/// provide attribute values.  But the key difference between `v` and
/// `i` is that line and triangle primitives are built from the values
/// of each row, along which `i` is constant and `v` varies, not the
/// columns.
///
/// An indexed draw call works similarly:
///
/// ```ignore
/// render_pass.draw_indexed(indices, base_vertex, instances)
/// ```
///
/// The only difference is that `v` values are drawn from the contents
/// of the index buffer&mdash;specifically, the subrange of the index
/// buffer given by `indices`&mdash;instead of simply being sequential
/// integers, as they are in a `draw` call.
///
/// A non-instanced call, where `instances` is `0..1`, is simply a
/// matrix with only one row.
///
/// Corresponds to [WebGPU `GPUVertexStepMode`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpuvertexstepmode).
///
#[doc = link_to_wgpu_docs!(["`RenderPass::draw`"]: "struct.RenderPass.html#method.draw")]
#[doc = link_to_wgpu_item!(struct VertexBufferLayout)]
#[doc = link_to_wgpu_docs!(["`step_mode`"]: "struct.VertexBufferLayout.html#structfield.step_mode")]
#[doc = link_to_wgpu_docs!(["`attributes`"]: "struct.VertexBufferLayout.html#structfield.attributes")]
/// [`Vertex`]: VertexStepMode::Vertex
/// [`Instance`]: VertexStepMode::Instance
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum VertexStepMode {
    /// Vertex data is advanced every vertex.
    #[default]
    Vertex = 0,
    /// Vertex data is advanced every instance.
    Instance = 1,
}

/// Vertex inputs (attributes) to shaders.
///
/// These are used to specify the individual attributes within a [`VertexBufferLayout`].
/// See its documentation for an example.
///
/// The [`vertex_attr_array!`] macro can help create these with appropriate offsets.
///
/// Corresponds to [WebGPU `GPUVertexAttribute`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexattribute).
///
#[doc = link_to_wgpu_docs!(["`vertex_attr_array!`"]: "macro.vertex_attr_array.html")]
#[doc = link_to_wgpu_item!(struct VertexBufferLayout)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct VertexAttribute {
    /// Format of the input
    pub format: VertexFormat,
    /// Byte offset of the start of the input
    pub offset: crate::BufferAddress,
    /// Location for this input. Must match the location in the shader.
    pub shader_location: crate::ShaderLocation,
}
