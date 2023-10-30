/*! Draw structures - shared between render passes and bundles.
!*/

use crate::{
    binding_model::{LateMinBufferBindingSizeMismatch, PushConstantUploadError},
    error::ErrorFormatter,
    id,
    track::UsageConflict,
    validation::{MissingBufferUsageError, MissingTextureUsageError},
};
use wgt::{BufferAddress, BufferSize, Color};

use std::num::NonZeroU32;
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
    #[error("The pipeline layout, associated with the current render pipeline, contains a bind group layout at index {index} which is incompatible with the bind group layout associated with the bind group at {index}")]
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
    #[error("Vertex {last_vertex} extends beyond limit {vertex_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Vertex` step-rate vertex buffer?")]
    VertexBeyondLimit {
        last_vertex: u32,
        vertex_limit: u32,
        slot: u32,
    },
    #[error("Instance {last_instance} extends beyond limit {instance_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Instance` step-rate vertex buffer?")]
    InstanceBeyondLimit {
        last_instance: u32,
        instance_limit: u32,
        slot: u32,
    },
    #[error("Index {last_index} extends beyond limit {index_limit}. Did you bind the correct index buffer?")]
    IndexBeyondLimit { last_index: u32, index_limit: u32 },
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
    #[error("Bind group {0:?} is invalid")]
    InvalidBindGroup(id::BindGroupId),
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
    UsageConflict(#[from] UsageConflict),
    #[error("Buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
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
            Self::InvalidBindGroup(id) => {
                fmt.bind_group_label(&id);
            }
            Self::InvalidPipeline(id) => {
                fmt.render_pipeline_label(&id);
            }
            Self::UsageConflict(UsageConflict::TextureInvalid { id }) => {
                fmt.texture_label(&id);
            }
            Self::UsageConflict(UsageConflict::BufferInvalid { id })
            | Self::DestroyedBuffer(id) => {
                fmt.buffer_label(&id);
            }
            _ => {}
        };
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "trace"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "replay"),
    derive(serde::Deserialize)
)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "trace"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "replay"),
    derive(serde::Deserialize)
)]
pub enum RenderCommand {
    SetBindGroup {
        index: u32,
        num_dynamic_offsets: u8,
        bind_group_id: id::BindGroupId,
    },
    SetPipeline(id::RenderPipelineId),
    SetIndexBuffer {
        buffer_id: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    },
    SetBlendConstant(Color),
    SetStencilReference(u32),
    SetViewport {
        rect: Rect<f32>,
        //TODO: use half-float to reduce the size?
        depth_min: f32,
        depth_max: f32,
    },
    SetScissor(Rect<u32>),

    /// Set a range of push constants to values stored in [`BasePass::push_constant_data`].
    ///
    /// See [`wgpu::RenderPass::set_push_constants`] for a detailed explanation
    /// of the restrictions these commands must satisfy.
    SetPushConstant {
        /// Which stages we are setting push constant values for.
        stages: wgt::ShaderStages,

        /// The byte offset within the push constant storage to write to.  This
        /// must be a multiple of four.
        offset: u32,

        /// The number of bytes to write. This must be a multiple of four.
        size_bytes: u32,

        /// Index in [`BasePass::push_constant_data`] of the start of the data
        /// to be written.
        ///
        /// Note: this is not a byte offset like `offset`. Rather, it is the
        /// index of the first `u32` element in `push_constant_data` to read.
        ///
        /// `None` means zeros should be written to the destination range, and
        /// there is no corresponding data in `push_constant_data`. This is used
        /// by render bundles, which explicitly clear out any state that
        /// post-bundle code might see.
        values_offset: Option<u32>,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    MultiDrawIndirect {
        buffer_id: id::BufferId,
        offset: BufferAddress,
        /// Count of `None` represents a non-multi call.
        count: Option<NonZeroU32>,
        indexed: bool,
    },
    MultiDrawIndirectCount {
        buffer_id: id::BufferId,
        offset: BufferAddress,
        count_buffer_id: id::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
        indexed: bool,
    },
    PushDebugGroup {
        color: u32,
        len: usize,
    },
    PopDebugGroup,
    InsertDebugMarker {
        color: u32,
        len: usize,
    },
    WriteTimestamp {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    BeginOcclusionQuery {
        query_index: u32,
    },
    EndOcclusionQuery,
    BeginPipelineStatisticsQuery {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(id::RenderBundleId),
}
