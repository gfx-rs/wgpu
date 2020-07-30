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
pub type BufferError = UseExtendError<BufferUse>;

use thiserror::Error;

/// Error validating a draw call.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum DrawError {
    #[error("blend color needs to be set")]
    MissingBlendColor,
    #[error("stencil reference needs to be set")]
    MissingStencilReference,
    #[error("render pipeline must be set")]
    MissingPipeline,
    #[error("current render pipeline has a layout which is incompatible with a currently set bind group, first differing at entry index {index}")]
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
    #[error("vertex {last_vertex} extends beyond limit {vertex_limit}")]
    VertexBeyondLimit { last_vertex: u32, vertex_limit: u32 },
    #[error("instance {last_instance} extends beyond limit {instance_limit}")]
    InstanceBeyondLimit {
        last_instance: u32,
        instance_limit: u32,
    },
    #[error("index {last_index} extends beyond limit {index_limit}")]
    IndexBeyondLimit { last_index: u32, index_limit: u32 },
}

/// Error encountered when encoding a render command.
/// This is the shared error set between render bundles and passes.
#[derive(Clone, Debug, Error)]
pub enum RenderCommandError {
    #[error("bind group {0:?} is invalid")]
    InvalidBindGroup(id::BindGroupId),
    #[error("bind group index {index} is greater than the device's requested `max_bind_group` limit {max}")]
    BindGroupIndexOutOfRange { index: u8, max: u32 },
    #[error("dynamic buffer offset {0} does not respect `BIND_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(u64),
    #[error("number of buffer offsets ({actual}) does not match the number of dynamic bindings ({expected})")]
    InvalidDynamicOffsetCount { actual: usize, expected: usize },
    #[error("render pipeline {0:?} is invalid")]
    InvalidPipeline(id::RenderPipelineId),
    #[error("render pipeline output formats and sample counts do not match render pass attachment formats")]
    IncompatiblePipeline,
    #[error("pipeline is not compatible with the depth-stencil read-only render pass")]
    IncompatibleReadOnlyDepthStencil,
    #[error("buffer {0:?} is in error {1:?}")]
    Buffer(id::BufferId, BufferError),
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error(transparent)]
    PushConstants(#[from] PushConstantUploadError),
}
