use alloc::boxed::Box;

use thiserror::Error;

use wgt::error::{ErrorType, WebGpuError};

use super::bind::BinderError;
use crate::command::pass;
use crate::{
    binding_model::{BindingError, LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    resource::{
        DestroyedResourceError, MissingBufferUsageError, MissingTextureUsageError,
        ResourceErrorIdent,
    },
    track::ResourceUsageCompatibilityError,
};

/// Error validating a draw call.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DrawError {
    #[error("Blend constant needs to be set")]
    MissingBlendConstant,
    #[error("Render pipeline must be set")]
    MissingPipeline(#[from] pass::MissingPipeline),
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

impl WebGpuError for DrawError {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

/// Error encountered when encoding a render command.
/// This is the shared error set between render bundles and passes.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RenderCommandError {
    #[error(transparent)]
    BindGroupIndexOutOfRange(#[from] pass::BindGroupIndexOutOfRange),
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
    #[error(transparent)]
    BindingError(#[from] BindingError),
    #[error("Viewport size {{ w: {w}, h: {h} }} greater than device's requested `max_texture_dimension_2d` limit {max}, or less than zero")]
    InvalidViewportRectSize { w: f32, h: f32, max: u32 },
    #[error("Viewport has invalid rect {rect:?} for device's requested `max_texture_dimension_2d` limit; Origin less than -2 * `max_texture_dimension_2d` ({min}), or rect extends past 2 * `max_texture_dimension_2d` - 1 ({max})")]
    InvalidViewportRectPosition { rect: Rect<f32>, min: f32, max: f32 },
    #[error("Viewport minDepth {0} and/or maxDepth {1} are not in [0, 1]")]
    InvalidViewportDepth(f32, f32),
    #[error("Scissor {0:?} is not contained in the render target {1:?}")]
    InvalidScissorRect(Rect<u32>, wgt::Extent3d),
    #[error("Support for {0} is not implemented yet")]
    Unimplemented(&'static str),
}

impl WebGpuError for RenderCommandError {
    fn webgpu_error_type(&self) -> ErrorType {
        let e: &dyn WebGpuError = match self {
            Self::IncompatiblePipelineTargets(e) => e,
            Self::ResourceUsageCompatibility(e) => e,
            Self::DestroyedResource(e) => e,
            Self::MissingBufferUsage(e) => e,
            Self::MissingTextureUsage(e) => e,
            Self::PushConstants(e) => e,
            Self::BindingError(e) => e,

            Self::BindGroupIndexOutOfRange { .. }
            | Self::VertexBufferIndexOutOfRange { .. }
            | Self::IncompatibleDepthAccess(..)
            | Self::IncompatibleStencilAccess(..)
            | Self::InvalidViewportRectSize { .. }
            | Self::InvalidViewportRectPosition { .. }
            | Self::InvalidViewportDepth(..)
            | Self::InvalidScissorRect(..)
            | Self::Unimplemented(..) => return ErrorType::Validation,
        };
        e.webgpu_error_type()
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}
