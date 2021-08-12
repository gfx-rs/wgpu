/*! Draw structures - shared between render passes and bundles.
!*/

use crate::{
    binding_model::PushConstantUploadError,
    error::ErrorFormatter,
    id::{self, AllResources, Hkt},
    track::UseExtendError,
    validation::{MissingBufferUsageError, MissingTextureUsageError},
};
use wgt::{BufferAddress, BufferSize, Color};

use std::num::NonZeroU32;
use thiserror::Error;

pub type BufferError = UseExtendError<hal::BufferUses>;

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
    #[error("Render pipeline targets are incompatible with render pass")]
    IncompatiblePipelineTargets(#[from] crate::device::RenderPassCompatibilityError),
    #[error("pipeline writes to depth/stencil, while the pass has read-only depth/stencil")]
    IncompatiblePipelineRods,
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
impl crate::error::PrettyError for RenderCommandError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match self {
            Self::InvalidBindGroup(id) => {
                fmt.bind_group_label(id);
            }
            Self::InvalidPipeline(id) => {
                fmt.render_pipeline_label(id);
            }
            Self::Buffer(id, ..) | Self::DestroyedBuffer(id) => {
                fmt.buffer_label(id);
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
#[derive(/* Clone, Copy, */Debug)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "trace"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "replay"),
    derive(serde::Deserialize)
)]
pub enum RenderCommand<A: hal::Api, F: AllResources<A>> {
    SetBindGroup {
        index: u8,
        num_dynamic_offsets: u8,
        #[cfg_attr(any(feature = "serial-pass", feature = "trace"),
          serde(bound(serialize = "<F as Hkt<crate::binding_model::BindGroup<A>>>::Output: serde::Serialize")))]
        #[cfg_attr(any(feature = "serial-pass", feature = "replay"),
          serde(bound(deserialize = "<F as Hkt<crate::binding_model::BindGroup<A>>>::Output: serde::Deserialize<'de>")))]
        bind_group_id: <F as Hkt<crate::binding_model::BindGroup<A>>>::Output,
    },
    SetPipeline(
        #[cfg_attr(any(feature = "serial-pass", feature = "trace"),
          serde(bound(serialize = "<F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output: serde::Serialize")))]
        #[cfg_attr(any(feature = "serial-pass", feature = "replay"),
          serde(bound(deserialize = "<F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output: serde::Deserialize<'de>")))]
        <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output,
    ),
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
        stages: wgt::ShaderStages,
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
        #[cfg_attr(any(feature = "serial-pass", feature = "trace"),
          serde(bound(serialize = "<F as Hkt<crate::resource::QuerySet<A>>>::Output: serde::Serialize")))]
        #[cfg_attr(any(feature = "serial-pass", feature = "replay"),
          serde(bound(deserialize = "<F as Hkt<crate::resource::QuerySet<A>>>::Output: serde::Deserialize<'de>")))]
        query_set_id: <F as Hkt<crate::resource::QuerySet<A>>>::Output,
        query_index: u32,
    },
    BeginPipelineStatisticsQuery {
        #[cfg_attr(any(feature = "serial-pass", feature = "trace"),
          serde(bound(serialize = "<F as Hkt<crate::resource::QuerySet<A>>>::Output: serde::Serialize")))]
        #[cfg_attr(any(feature = "serial-pass", feature = "replay"),
          serde(bound(deserialize = "<F as Hkt<crate::resource::QuerySet<A>>>::Output: serde::Deserialize<'de>")))]
        query_set_id: <F as Hkt<crate::resource::QuerySet<A>>>::Output,
        query_index: u32,
    },
    EndPipelineStatisticsQuery,
    ExecuteBundle(
        #[cfg_attr(any(feature = "serial-pass", feature = "trace"),
          serde(bound(serialize = "<F as Hkt<crate::command::RenderBundle<A>>>::Output: serde::Serialize")))]
        #[cfg_attr(any(feature = "serial-pass", feature = "replay"),
          serde(bound(deserialize = "<F as Hkt<crate::command::RenderBundle<A>>>::Output: serde::Deserialize<'de>")))]
        <F as Hkt<crate::command::RenderBundle<A>>>::Output,
    ),
}

impl<A: hal::Api, F: AllResources<A>> RenderCommand<A, F> {
    #[inline]
    #[cfg(feature = "replay")]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<A, &'b F>) -> Result<(), E>,
    ) -> Result<(), E>
        where id::Cached<A, &'b F>: 'b,
    {
        use self::RenderCommand::*;
        use id::Cached;

        match self {
            SetBindGroup { bind_group_id, .. } => f(Cached::BindGroup(bind_group_id)),
            SetPipeline(pipeline_id) => f(Cached::RenderPipeline(pipeline_id)),
            SetIndexBuffer { buffer_id: _, .. } =>
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(buffer_id)),
                Ok(()),
            SetVertexBuffer { buffer_id: _, .. } =>
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(buffer_id)),
                Ok(()),
            MultiDrawIndirect { buffer_id: _, .. } =>
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(buffer_id)),
                Ok(()),
            MultiDrawIndirectCount { buffer_id: _, count_buffer_id: _, .. } => {
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(buffer_id))?;
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(count_buffer_id)),
                Ok(())
            }
            WriteTimestamp { query_set_id, .. } => f(Cached::QuerySet(query_set_id)),
            BeginPipelineStatisticsQuery { query_set_id, .. } => f(Cached::QuerySet(query_set_id)),
            ExecuteBundle(bundle_id) => f(Cached::RenderBundle(bundle_id)),
            SetBlendConstant(..) | SetStencilReference(..)
            | SetViewport { .. } | SetScissor(..)
            | SetPushConstant { .. }
            | Draw { .. } | DrawIndexed { .. }
            | PushDebugGroup { .. } | PopDebugGroup | InsertDebugMarker { .. }
            | EndPipelineStatisticsQuery => Ok(()),
        }
    }
}


impl<A: hal::Api, F: AllResources<A>> Clone for RenderCommand<A, F>
    where
        <F as Hkt<crate::binding_model::BindGroup<A>>>::Output: Copy,
        <F as Hkt<crate::resource::QuerySet<A>>>::Output: Copy,
        <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output: Copy,
        <F as Hkt<crate::command::RenderBundle<A>>>::Output: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: hal::Api, F: AllResources<A>> Copy for RenderCommand<A, F>
    where
        <F as Hkt<crate::binding_model::BindGroup<A>>>::Output: Copy,
        <F as Hkt<crate::resource::QuerySet<A>>>::Output: Copy,
        <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output: Copy,
        <F as Hkt<crate::command::RenderBundle<A>>>::Output: Copy,
{}

impl<A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    super::FromCommand<RenderCommand<A, F>> for RenderCommand<B, G>
    where
        <F as Hkt<crate::binding_model::BindGroup<A>>>::Output:
            Into<<G as Hkt<crate::binding_model::BindGroup<B>>>::Output>,
        <F as Hkt<crate::resource::QuerySet<A>>>::Output:
            Into<<G as Hkt<crate::resource::QuerySet<B>>>::Output>,
        <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output:
            Into<<G as Hkt<crate::pipeline::RenderPipeline<B>>>::Output>,
        <F as Hkt<crate::command::RenderBundle<A>>>::Output:
            Into<<G as Hkt<crate::command::RenderBundle<B>>>::Output>,
{
    fn from(command: RenderCommand<A, F>) -> Self {
        use self::RenderCommand::*;

        match command {
            SetBindGroup { index, num_dynamic_offsets, bind_group_id } =>
                SetBindGroup { index, num_dynamic_offsets, bind_group_id: bind_group_id.into() },
            SetPipeline(pipeline_id) => SetPipeline(pipeline_id.into()),
            SetIndexBuffer { buffer_id, index_format, offset, size} =>
                SetIndexBuffer { buffer_id, index_format, offset, size},
            SetVertexBuffer { slot, buffer_id, offset, size } =>
                SetVertexBuffer { slot, buffer_id, offset, size },
            SetBlendConstant(color) => SetBlendConstant(color),
            SetStencilReference(value) => SetStencilReference(value),
            SetViewport { rect, depth_min, depth_max } =>
                SetViewport { rect, depth_min, depth_max },
            SetScissor(rect) => SetScissor(rect),
            SetPushConstant { stages, offset, size_bytes, values_offset } =>
                SetPushConstant { stages, offset, size_bytes, values_offset },
            Draw { vertex_count, instance_count, first_vertex, first_instance } =>
                Draw { vertex_count, instance_count, first_vertex, first_instance },
            DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance } =>
                DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance },
            MultiDrawIndirect { buffer_id, offset, count, indexed } =>
                MultiDrawIndirect { buffer_id, offset, count, indexed },
            MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed } =>
                MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed },
            PushDebugGroup { color, len } => PushDebugGroup { color, len },
            PopDebugGroup => PopDebugGroup,
            InsertDebugMarker { color, len } => InsertDebugMarker { color, len },
            WriteTimestamp { query_set_id, query_index } =>
                WriteTimestamp { query_set_id: query_set_id.into(), query_index },
            BeginPipelineStatisticsQuery { query_set_id, query_index } =>
                BeginPipelineStatisticsQuery { query_set_id: query_set_id.into(), query_index },
            EndPipelineStatisticsQuery => EndPipelineStatisticsQuery,
            ExecuteBundle(bundle_id) => ExecuteBundle(bundle_id.into()),
        }
    }
}

impl<'a, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>>
    super::FromCommand<&'a RenderCommand<A, F>> for RenderCommand<B, G>
    where
        &'a <F as Hkt<crate::binding_model::BindGroup<A>>>::Output:
            Into<<G as Hkt<crate::binding_model::BindGroup<B>>>::Output>,
        &'a <F as Hkt<crate::resource::QuerySet<A>>>::Output:
            Into<<G as Hkt<crate::resource::QuerySet<B>>>::Output>,
        &'a <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output:
            Into<<G as Hkt<crate::pipeline::RenderPipeline<B>>>::Output>,
        &'a <F as Hkt<crate::command::RenderBundle<A>>>::Output:
            Into<<G as Hkt<crate::command::RenderBundle<B>>>::Output>,
{
    fn from(command: &'a RenderCommand<A, F>) -> Self {
        use self::RenderCommand::*;

        match *command {
            SetBindGroup { index, num_dynamic_offsets, ref bind_group_id } =>
                SetBindGroup { index, num_dynamic_offsets, bind_group_id: bind_group_id.into() },
            SetPipeline(ref pipeline_id) => SetPipeline(pipeline_id.into()),
            SetIndexBuffer { buffer_id, index_format, offset, size} =>
                SetIndexBuffer { buffer_id, index_format, offset, size},
            SetVertexBuffer { slot, buffer_id, offset, size } =>
                SetVertexBuffer { slot, buffer_id, offset, size },
            SetBlendConstant(color) => SetBlendConstant(color),
            SetStencilReference(value) => SetStencilReference(value),
            SetViewport { rect, depth_min, depth_max } =>
                SetViewport { rect, depth_min, depth_max },
            SetScissor(rect) => SetScissor(rect),
            SetPushConstant { stages, offset, size_bytes, values_offset } =>
                SetPushConstant { stages, offset, size_bytes, values_offset },
            Draw { vertex_count, instance_count, first_vertex, first_instance } =>
                Draw { vertex_count, instance_count, first_vertex, first_instance },
            DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance } =>
                DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance },
            MultiDrawIndirect { buffer_id, offset, count, indexed } =>
                MultiDrawIndirect { buffer_id, offset, count, indexed },
            MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed } =>
                MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed },
            PushDebugGroup { color, len } => PushDebugGroup { color, len },
            PopDebugGroup => PopDebugGroup,
            InsertDebugMarker { color, len } => InsertDebugMarker { color, len },
            WriteTimestamp { ref query_set_id, query_index } =>
                WriteTimestamp { query_set_id: query_set_id.into(), query_index },
            BeginPipelineStatisticsQuery { ref query_set_id, query_index } =>
                BeginPipelineStatisticsQuery { query_set_id: query_set_id.into(), query_index },
            EndPipelineStatisticsQuery => EndPipelineStatisticsQuery,
            ExecuteBundle(ref bundle_id) => ExecuteBundle(bundle_id.into()),
        }
    }
}

#[cfg(feature = "replay")]
impl<'a, A: hal::Api, B: hal::Api, F: AllResources<A>, G: AllResources<B>, E>
    core::convert::TryFrom<(&'a id::IdCache2, RenderCommand<A, F>)> for RenderCommand<B, G>
    where
        (&'a id::IdCache2, <F as Hkt<crate::binding_model::BindGroup<A>>>::Output):
            core::convert::TryInto<<G as Hkt<crate::binding_model::BindGroup<B>>>::Output, Error=E>,
        (&'a id::IdCache2, <F as Hkt<crate::resource::QuerySet<A>>>::Output):
            core::convert::TryInto<<G as Hkt<crate::resource::QuerySet<B>>>::Output, Error=E>,
        (&'a id::IdCache2, <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output):
            core::convert::TryInto<<G as Hkt<crate::pipeline::RenderPipeline<B>>>::Output, Error=E>,
        (&'a id::IdCache2, <F as Hkt<crate::command::RenderBundle<A>>>::Output):
            core::convert::TryInto<<G as Hkt<crate::command::RenderBundle<B>>>::Output, Error=E>,
{
    type Error = /*<(&'a id::IdCache2,  <F as Hkt<crate::binding_model::BindGroup<A>>>::Output)
                  as core::convert::TryInto<<G as Hkt<crate::binding_model::BindGroup<B>>>::Output>>::Error*/E;

    fn try_from((cache, command): (&'a id::IdCache2, RenderCommand<A, F>)) -> Result<Self, Self::Error> {
        use core::convert::TryInto;
        use self::RenderCommand::*;

        Ok(match command {
            SetBindGroup { index, num_dynamic_offsets, bind_group_id } =>
                SetBindGroup { index, num_dynamic_offsets, bind_group_id: (cache, bind_group_id).try_into()? },
            SetPipeline(pipeline_id) => SetPipeline((cache, pipeline_id).try_into()?),
            SetIndexBuffer { buffer_id, index_format, offset, size} =>
                SetIndexBuffer { buffer_id, index_format, offset, size},
            SetVertexBuffer { slot, buffer_id, offset, size } =>
                SetVertexBuffer { slot, buffer_id, offset, size },
            SetBlendConstant(color) => SetBlendConstant(color),
            SetStencilReference(value) => SetStencilReference(value),
            SetViewport { rect, depth_min, depth_max } =>
                SetViewport { rect, depth_min, depth_max },
            SetScissor(rect) => SetScissor(rect),
            SetPushConstant { stages, offset, size_bytes, values_offset } =>
                SetPushConstant { stages, offset, size_bytes, values_offset },
            Draw { vertex_count, instance_count, first_vertex, first_instance } =>
                Draw { vertex_count, instance_count, first_vertex, first_instance },
            DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance } =>
                DrawIndexed { index_count, instance_count, first_index, base_vertex, first_instance },
            MultiDrawIndirect { buffer_id, offset, count, indexed } =>
                MultiDrawIndirect { buffer_id, offset, count, indexed },
            MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed } =>
                MultiDrawIndirectCount { buffer_id, offset, count_buffer_id, count_buffer_offset, max_count, indexed },
            PushDebugGroup { color, len } => PushDebugGroup { color, len },
            PopDebugGroup => PopDebugGroup,
            InsertDebugMarker { color, len } => InsertDebugMarker { color, len },
            WriteTimestamp { query_set_id, query_index } =>
                WriteTimestamp { query_set_id: (cache, query_set_id).try_into()?, query_index },
            BeginPipelineStatisticsQuery { query_set_id, query_index } =>
                BeginPipelineStatisticsQuery { query_set_id: (cache, query_set_id).try_into()?, query_index },
            EndPipelineStatisticsQuery => EndPipelineStatisticsQuery,
            ExecuteBundle(bundle_id) => ExecuteBundle((cache, bundle_id).try_into()?),
        })
    }
}
