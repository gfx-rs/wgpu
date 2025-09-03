use core::convert::Infallible;

use alloc::{string::String, sync::Arc, vec::Vec};

use crate::{
    id,
    resource::{Buffer, QuerySet, Texture},
};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Command {
    CopyBufferToBuffer {
        src: id::BufferId,
        src_offset: wgt::BufferAddress,
        dst: id::BufferId,
        dst_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    CopyBufferToTexture {
        src: wgt::TexelCopyBufferInfo<id::BufferId>,
        dst: wgt::TexelCopyTextureInfo<id::TextureId>,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: wgt::TexelCopyTextureInfo<id::TextureId>,
        dst: wgt::TexelCopyBufferInfo<id::BufferId>,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: wgt::TexelCopyTextureInfo<id::TextureId>,
        dst: wgt::TexelCopyTextureInfo<id::TextureId>,
        size: wgt::Extent3d,
    },
    ClearBuffer {
        dst: id::BufferId,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    ClearTexture {
        dst: id::TextureId,
        subresource_range: wgt::ImageSubresourceRange,
    },
    WriteTimestamp {
        query_set_id: id::QuerySetId,
        query_index: u32,
    },
    ResolveQuerySet {
        query_set_id: id::QuerySetId,
        start_query: u32,
        query_count: u32,
        destination: id::BufferId,
        destination_offset: wgt::BufferAddress,
    },
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
    RunComputePass {
        base: crate::command::BasePass<crate::command::ComputeCommand, Infallible>,
        timestamp_writes: Option<crate::command::PassTimestampWrites>,
    },
    RunRenderPass {
        base: crate::command::BasePass<crate::command::RenderCommand, Infallible>,
        target_colors: Vec<Option<crate::command::RenderPassColorAttachment>>,
        target_depth_stencil: Option<crate::command::RenderPassDepthStencilAttachment>,
        timestamp_writes: Option<crate::command::PassTimestampWrites>,
        occlusion_query_set_id: Option<id::QuerySetId>,
    },
    BuildAccelerationStructures {
        blas: Vec<crate::ray_tracing::TraceBlasBuildEntry>,
        tlas: Vec<crate::ray_tracing::TraceTlasPackage>,
    },
}

#[derive(Clone, Debug)]
pub enum ArcCommand {
    CopyBufferToBuffer {
        src: Arc<Buffer>,
        src_offset: wgt::BufferAddress,
        dst: Arc<Buffer>,
        dst_offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    CopyBufferToTexture {
        src: wgt::TexelCopyBufferInfo<Arc<Buffer>>,
        dst: wgt::TexelCopyTextureInfo<Arc<Texture>>,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: wgt::TexelCopyTextureInfo<Arc<Texture>>,
        dst: wgt::TexelCopyBufferInfo<Arc<Buffer>>,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: wgt::TexelCopyTextureInfo<Arc<Texture>>,
        dst: wgt::TexelCopyTextureInfo<Arc<Texture>>,
        size: wgt::Extent3d,
    },
    ClearBuffer {
        dst: Arc<Buffer>,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    },
    ClearTexture {
        dst: Arc<Texture>,
        subresource_range: wgt::ImageSubresourceRange,
    },
    WriteTimestamp {
        query_set: Arc<QuerySet>,
        query_index: u32,
    },
    ResolveQuerySet {
        query_set: Arc<QuerySet>,
        start_query: u32,
        query_count: u32,
        destination: Arc<Buffer>,
        destination_offset: wgt::BufferAddress,
    },
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
    RunComputePass {
        pass: super::BasePass<super::ArcComputeCommand, Infallible>,
        timestamp_writes: Option<super::ArcPassTimestampWrites>,
    },
    RunRenderPass {
        pass: super::BasePass<super::ArcRenderCommand, Infallible>,
        color_attachments: super::ArcRenderPassColorAttachmentArray,
        depth_stencil_attachment: Option<super::ArcRenderPassDepthStencilAttachment>,
        timestamp_writes: Option<super::ArcPassTimestampWrites>,
        occlusion_query_set: Option<Arc<QuerySet>>,
    },
    BuildAccelerationStructures {
        blas: Vec<crate::ray_tracing::ArcBlasBuildEntry>,
        tlas: Vec<crate::ray_tracing::ArcTlasPackage>,
    },
    TransitionResources {
        buffer_transitions: Vec<wgt::BufferTransition<Arc<Buffer>>>,
        texture_transitions: Vec<wgt::TextureTransition<Arc<Texture>>>,
    },
}
