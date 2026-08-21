use alloc::borrow::Cow;

use crate::{id, Label};

pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
pub type CommandBufferDescriptor<'a> = wgt::CommandBufferDescriptor<Label<'a>>;

/// Describes a color attachment to a render pass.
///
/// Corresponds to [`GPURenderColorAttachment`](https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpasscolorattachment)
#[repr(C)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RenderPassColorAttachment {
    /// The view to use as an attachment.
    pub view: id::TextureViewId,
    /// The depth slice index of a 3D view. It must not be provided if the view is not 3D.
    pub depth_slice: Option<u32>,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<id::TextureViewId>,
    /// Operation to perform to the output attachment at the start of a
    /// renderpass.
    ///
    /// This must be clear if it is the first renderpass rendering to a swap
    /// chain image.
    pub load_op: wgt::LoadOp<wgt::Color>,
    /// Operation to perform to the output attachment at the end of a renderpass.
    pub store_op: wgt::StoreOp,
}

/// Describes an individual channel within a render pass, such as color, depth, or stencil.
///
/// A channel must either be read-only, or it must specify both load and store
/// operations.
#[repr(C)]
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PassChannel<V> {
    /// Operation to perform to the output attachment at the start of a
    /// renderpass.
    ///
    /// This must be clear if it is the first renderpass rendering to a swap
    /// chain image.
    pub load_op: Option<wgt::LoadOp<V>>,
    /// Operation to perform to the output attachment at the end of a renderpass.
    pub store_op: Option<wgt::StoreOp>,
    /// If true, the relevant channel is not changed by a renderpass, and the
    /// corresponding attachment can be used inside the pass by other read-only
    /// usages.
    pub read_only: bool,
}

/// Describes a depth/stencil attachment to a render pass.
///
/// Corresponds to [`GPURenderDepthStencilAttachment`](https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdepthstencilattachment)
#[repr(C)]
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RenderPassDepthStencilAttachment {
    /// The view to use as an attachment.
    pub view: id::TextureViewId,
    /// What operations will be performed on the depth part of the attachment.
    pub depth: PassChannel<Option<f32>>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil: PassChannel<Option<u32>>,
}

/// Describes the writing of timestamp values in a render or compute pass.
///
/// Corresponds to [`GPURenderPassTimestampWrites`](https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpasstimestampwrites)
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PassTimestampWrites {
    /// The query set to write the timestamps to.
    pub query_set: id::QuerySetId,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}

/// Describes the attachments of a render pass.
///
/// Corresponds to [`GPURenderPassDescriptor`](https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdescriptor)
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RenderPassDescriptor<'a> {
    pub label: Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments: Cow<'a, [Option<RenderPassColorAttachment>]>,
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    /// Defines where the occlusion query results will be stored for this pass.
    pub occlusion_query_set: Option<id::QuerySetId>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PassTimestampWrites>,
}

/// Corresponds to [`GPUComputePassDescriptor`](https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepassdescriptor)
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ComputePassDescriptor<'a> {
    pub label: Label<'a>,
    /// Defines where and when timestamp values will be written for this pass.
    pub timestamp_writes: Option<PassTimestampWrites>,
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPUCommandEncoder`](https://www.w3.org/TR/webgpu/#gpucommandencoder).
pub enum CommandEncoderCommand<'a> {
    BeginRenderPass {
        desc: RenderPassDescriptor<'a>,
        render_pass_encoder_id: id::RenderPassEncoderId,
    },
    BeginComputePass {
        desc: ComputePassDescriptor<'a>, // optional, defaults to {}
        compute_pass_encoder_id: id::ComputePassEncoderId,
    },
    CopyBufferToBuffer {
        source: id::BufferId,
        source_offset: u64,
        destination: id::BufferId,
        destination_offset: u64,
        size: Option<u64>,
    },
    CopyBufferToTexture {
        source: wgt::TexelCopyBufferInfo<id::BufferId>,
        destination: wgt::TexelCopyTextureInfo<id::TextureId>,
        copy_size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        source: wgt::TexelCopyTextureInfo<id::TextureId>,
        destination: wgt::TexelCopyBufferInfo<id::BufferId>,
        copy_size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        source: wgt::TexelCopyTextureInfo<id::TextureId>,
        destination: wgt::TexelCopyTextureInfo<id::TextureId>,
        copy_size: wgt::Extent3d,
    },
    ClearBuffer {
        buffer: id::BufferId,
        offset: u64, // optional, defaults to 0
        size: Option<u64>,
    },
    ResolveQuerySet {
        query_set: id::QuerySetId,
        first_query: u32,
        query_count: u32,
        destination: id::BufferId,
        destination_offset: u64,
    },
    DebugCommand(DebugCommand),
    Finish {
        desc: CommandBufferDescriptor<'a>,
        command_buffer_id: id::CommandBufferId,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPURenderPassEncoder`](https://www.w3.org/TR/webgpu/#gpurenderpassencoder).
pub enum RenderPassEncoderCommand {
    SetViewport {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    },
    SetScissorRect {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    SetBlendConstant(wgt::Color),
    SetStencilReference(u32),
    BeginOcclusionQuery(u32),
    EndOcclusionQuery,
    ExecuteBundles(Vec<id::RenderBundleId>),
    BindingCommand(BindingCommand),
    RenderCommand(RenderCommand),
    DebugCommand(DebugCommand),
    End,
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPURenderBundleEncoder`](https://www.w3.org/TR/webgpu/#gpurenderbundleencoder).
pub enum RenderBundleEncoderCommand<'a> {
    BindingCommand(BindingCommand),
    RenderCommand(RenderCommand),
    DebugCommand(DebugCommand),
    Finish {
        desc: RenderBundleDescriptor<'a>, // optional, defaults to {}
        render_bundle_id: id::RenderBundleId,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPUComputePassEncoder`](https://www.w3.org/TR/webgpu/#gpucomputepassencoder).
pub enum ComputePassEncoderCommand {
    BindingCommand(BindingCommand),
    SetPipeline(id::ComputePipelineId),
    DispatchWorkgroups {
        workgroup_count_x: u32,
        workgroup_count_y: u32, // optional, defaults to 1
        workgroup_count_z: u32, // optional, defaults to 1
    },
    DispatchWorkgroupsIndirect {
        indirect_buffer: id::BufferId,
        indirect_offset: u64,
    },
    DebugCommand(DebugCommand),
    End,
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPUDebugCommandsMixin`](https://www.w3.org/TR/webgpu/#gpudebugcommandsmixin).
pub enum DebugCommand {
    PushDebugGroup(String),
    PopDebugGroup,
    InsertDebugMarker(String),
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPUBindingCommandsMixin`](https://www.w3.org/TR/webgpu/#gpubindingcommandsmixin).
pub enum BindingCommand {
    SetBindGroup {
        index: u32,
        bind_group: Option<id::BindGroupId>,
        dynamic_offsets: Vec<u32>, // optional, defaults to []
    },
    SetImmediates {
        range_offset: u32,
        data: Vec<u8>,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
/// Corresponds to [`GPURenderCommandsMixin`](https://www.w3.org/TR/webgpu/#gpurendercommandsmixin).
pub enum RenderCommand {
    SetPipeline(id::RenderPipelineId),
    SetIndexBuffer {
        buffer: id::BufferId,
        index_format: wgt::IndexFormat,
        offset: u64, // optional, defaults to 0
        size: Option<u64>,
    },
    SetVertexBuffer {
        slot: u32,
        buffer: Option<id::BufferId>,
        offset: u64, // optional, defaults to 0
        size: Option<u64>,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32, // optional, defaults to 1
        first_vertex: u32,   // optional, defaults to 0
        first_instance: u32, // optional, defaults to 0
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32, // optional, defaults to 1
        first_index: u32,    // optional, defaults to 0
        base_vertex: i32,    // optional, defaults to 0
        first_instance: u32, // optional, defaults to 0
    },
    DrawIndirect {
        indirect_buffer: id::BufferId,
        indirect_offset: u64,
    },
    DrawIndexedIndirect {
        indirect_buffer: id::BufferId,
        indirect_offset: u64,
    },
}
