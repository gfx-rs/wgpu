use crate::id;

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
