use crate::{
    binding_model::{LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    resource::{
        DestroyedResourceError, MissingBufferUsageError, MissingTextureUsageError,
        ResourceErrorIdent,
    },
    track::ResourceUsageCompatibilityError,
};
use wgt::VertexStepMode;

use thiserror::Error;

use super::bind::BinderError;

/// Error validating a draw call.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DrawError {
    #[error("Blend constant needs to be set")]
    MissingBlendConstant,
    #[error("Render pipeline must be set")]
    MissingPipeline,
    #[error("Currently set {pipeline} requires vertex buffer {index} to be set")]
    MissingVertexBuffer {
        pipeline: ResourceErrorIdent,
        index: u32,
    },
    #[error("Index buffer must be set")]
    MissingIndexBuffer,
    #[error(transparent)]
    IncompatibleBindGroup(#[from] Box<BinderError>),
    #[error("Vertex {last_vertex} extends beyond limit {vertex_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Vertex` step-rate vertex buffer?")]
    VertexBeyondLimit {
        last_vertex: u64,
        vertex_limit: u64,
        slot: u32,
    },
    #[error("{step_mode:?} buffer out of bounds at slot {slot}. Offset {offset} beyond limit {limit}. Did you bind the correct `Vertex` step-rate vertex buffer?")]
    VertexOutOfBounds {
        step_mode: VertexStepMode,
        offset: u64,
        limit: u64,
        slot: u32,
    },
    #[error("Instance {last_instance} extends beyond limit {instance_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Instance` step-rate vertex buffer?")]
    InstanceBeyondLimit {
        last_instance: u64,
        instance_limit: u64,
        slot: u32,
    },
    #[error("Index {last_index} extends beyond limit {index_limit}. Did you bind the correct index buffer?")]
    IndexBeyondLimit { last_index: u64, index_limit: u64 },
    #[error(
        "Index buffer format {buffer_format:?} doesn't match {pipeline}'s index format {pipeline_format:?}"
    )]
    UnmatchedIndexFormats {
        pipeline: ResourceErrorIdent,
        pipeline_format: wgt::IndexFormat,
        buffer_format: wgt::IndexFormat,
    },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
}

/// Error encountered when encoding a render command.
/// This is the shared error set between render bundles and passes.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RenderCommandError {
    #[error("Bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u32, max: u32 },
    #[error("Vertex buffer index {index} is greater than the device's requested `max_vertex_buffers` limit {max}")]
    VertexBufferIndexOutOfRange { index: u32, max: u32 },
    #[error("Render pipeline targets are incompatible with render pass")]
    IncompatiblePipelineTargets(#[from] crate::device::RenderPassCompatibilityError),
    #[error("{0} writes to depth, while the pass has read-only depth access")]
    IncompatibleDepthAccess(ResourceErrorIdent),
    #[error("{0} writes to stencil, while the pass has read-only stencil access")]
    IncompatibleStencilAccess(ResourceErrorIdent),
    #[error(transparent)]
    ResourceUsageCompatibility(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error(transparent)]
    PushConstants(#[from] PushConstantUploadError),
    #[error("Viewport has invalid rect {0:?}; origin and/or size is less than or equal to 0, and/or is not contained in the render target {1:?}")]
    InvalidViewportRect(Rect<f32>, wgt::Extent3d),
    #[error("Viewport minDepth {0} and/or maxDepth {1} are not in [0, 1]")]
    InvalidViewportDepth(f32, f32),
    #[error("Scissor {0:?} is not contained in the render target {1:?}")]
    InvalidScissorRect(Rect<u32>, wgt::Extent3d),
    #[error("Support for {0} is not implemented yet")]
    Unimplemented(&'static str),
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}
