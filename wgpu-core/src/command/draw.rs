/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! Draw structures - shared between render passes and bundles.
!*/

use crate::{
    binding_model::PushConstantUploadError,
    id,
    resource::BufferUse,
    track::UseExtendError,
    validation::{MissingBufferUsageError, MissingTextureUsageError},
};
use wgt::{BufferAddress, BufferSize, Color};

use std::num::NonZeroU32;
use thiserror::Error;

pub type BufferError = UseExtendError<BufferUse>;

/// Error validating a draw call.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DrawError {
    #[error("blend constant needs to be set")]
    MissingBlendConstant,
    #[error("render pipeline must be set")]
    MissingPipeline,
    #[error("vertex buffer {index} must be set")]
    MissingVertexBuffer { index: u32 },
    #[error("index buffer must be set")]
    MissingIndexBuffer,
    #[error("current render pipeline has a layout which is incompatible with a currently set bind group, first differing at entry index {index}")]
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
    #[error("vertex {last_vertex} extends beyond limit {vertex_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Vertex` step-rate vertex buffer?")]
    VertexBeyondLimit {
        last_vertex: u32,
        vertex_limit: u32,
        slot: u32,
    },
    #[error("instance {last_instance} extends beyond limit {instance_limit} imposed by the buffer in slot {slot}. Did you bind the correct `Instance` step-rate vertex buffer?")]
    InstanceBeyondLimit {
        last_instance: u32,
        instance_limit: u32,
        slot: u32,
    },
    #[error("index {last_index} extends beyond limit {index_limit}. Did you bind the correct index buffer?")]
    IndexBeyondLimit { last_index: u32, index_limit: u32 },
    #[error(
        "pipeline index format ({pipeline:?}) and buffer index format ({buffer:?}) do not match"
    )]
    UnmatchedIndexFormats {
        pipeline: wgt::IndexFormat,
        buffer: wgt::IndexFormat,
    },
}

/// Error encountered when encoding a render command.
/// This is the shared error set between render bundles and passes.
#[derive(Clone, Debug, Error)]
pub enum RenderCommandError {
    #[error("bind group {0:?} is invalid")]
    InvalidBindGroup(id::BindGroupId),
    #[error("render bundle {0:?} is invalid")]
    InvalidRenderBundle(id::RenderBundleId),
    #[error("bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u8, max: u32 },
    #[error("dynamic buffer offset {0} does not respect `BIND_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(u64),
    #[error("number of buffer offsets ({actual}) does not match the number of dynamic bindings ({expected})")]
    InvalidDynamicOffsetCount { actual: usize, expected: usize },
    #[error("render pipeline {0:?} is invalid")]
    InvalidPipeline(id::RenderPipelineId),
    #[error("QuerySet {0:?} is invalid")]
    InvalidQuerySet(id::QuerySetId),
    #[error("Render pipeline is incompatible with render pass")]
    IncompatiblePipeline(#[from] crate::device::RenderPassCompatibilityError),
    #[error("pipeline is not compatible with the depth-stencil read-only render pass")]
    IncompatibleReadOnlyDepthStencil,
    #[error("buffer {0:?} is in error {1:?}")]
    Buffer(id::BufferId, BufferError),
    #[error("buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error(transparent)]
    PushConstants(#[from] PushConstantUploadError),
    #[error("Invalid Viewport parameters")]
    InvalidViewport,
    #[error("Invalid ScissorRect parameters")]
    InvalidScissorRect,
    #[error("Support for {0} is not implemented yet")]
    Unimplemented(&'static str),
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
        index: u8,
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
    SetPushConstant {
        stages: wgt::ShaderStage,
        offset: u32,
        size_bytes: u32,
        /// None means there is no data and the data should be an array of zeros.
        ///
        /// Facilitates clears in renderbundles which explicitly do their clears.
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
    BeginPipelineStatisticsQuery {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(id::RenderBundleId),
}
