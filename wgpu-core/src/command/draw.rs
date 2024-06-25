use crate::{
    binding_model::{LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    error::ErrorFormatter,
    id,
    resource::{DestroyedResourceError, MissingBufferUsageError, MissingTextureUsageError},
    track::ResourceUsageCompatibilityError,
};
use wgt::VertexStepMode;

use thiserror::Error;

/// Error validating a draw call.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum DrawError {
    #[error("Blend constant needs to be set")]
    MissingBlendConstant,
    #[error("Render pipeline must be set")]
    MissingPipeline,
    #[error("Vertex buffer {index} must be set")]
    MissingVertexBuffer { index: u32 },
    #[error("Index buffer must be set")]
    MissingIndexBuffer,
    #[error("Incompatible bind group at index {index} in the current render pipeline")]
    IncompatibleBindGroup { index: u32, diff: Vec<String> },
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
        "Pipeline index format ({pipeline:?}) and buffer index format ({buffer:?}) do not match"
    )]
    UnmatchedIndexFormats {
        pipeline: wgt::IndexFormat,
        buffer: wgt::IndexFormat,
    },
    #[error(transparent)]
    BindingSizeTooSmall(#[from] LateMinBufferBindingSizeMismatch),
}

/// Error encountered when encoding a render command.
/// This is the shared error set between render bundles and passes.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RenderCommandError {
    #[error("BufferId {0:?} is invalid")]
    InvalidBufferId(id::BufferId),
    #[error("BindGroupId {0:?} is invalid")]
    InvalidBindGroupId(id::BindGroupId),
    #[error("Render bundle {0:?} is invalid")]
    InvalidRenderBundle(id::RenderBundleId),
    #[error("Bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u32, max: u32 },
    #[error("Vertex buffer index {index} is greater than the device's requested `max_vertex_buffers` limit {max}")]
    VertexBufferIndexOutOfRange { index: u32, max: u32 },
    #[error("Dynamic buffer offset {0} does not respect device's requested `{1}` limit {2}")]
    UnalignedBufferOffset(u64, &'static str, u32),
    #[error("Number of buffer offsets ({actual}) does not match the number of dynamic bindings ({expected})")]
    InvalidDynamicOffsetCount { actual: usize, expected: usize },
    #[error("Render pipeline {0:?} is invalid")]
    InvalidPipeline(id::RenderPipelineId),
    #[error("QuerySet {0:?} is invalid")]
    InvalidQuerySet(id::QuerySetId),
    #[error("Render pipeline targets are incompatible with render pass")]
    IncompatiblePipelineTargets(#[from] crate::device::RenderPassCompatibilityError),
    #[error("Pipeline writes to depth/stencil, while the pass has read-only depth/stencil")]
    IncompatiblePipelineRods,
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
impl crate::error::PrettyError for RenderCommandError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::InvalidBindGroupId(id) => {
                fmt.bind_group_label(&id);
            }
            Self::InvalidPipeline(id) => {
                fmt.render_pipeline_label(&id);
            }
            _ => {}
        };
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
