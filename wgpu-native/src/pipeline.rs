use hal;
use resource;

use {
    BlendStateId, ByteArray, DepthStencilStateId, PipelineLayoutId,
    ShaderModuleId,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    SrcAlpha = 4,
    OneMinusSrcAlpha = 5,
    DstColor = 6,
    OneMinusDstColor = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    SrcAlphaSaturated = 10,
    BlendColor = 11,
    OneMinusBlendColor = 12,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BlendOperation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

bitflags! {
    #[repr(transparent)]
    pub struct ColorWriteFlags: u32 {
        const RED = 1;
        const GREEN = 2;
        const BLUE = 4;
        const ALPHA = 8;
        const COLOR = 7;
        const ALL = 15;
    }
}

#[repr(C)]
pub struct BlendDescriptor {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl BlendDescriptor {
    pub const REPLACE: Self = BlendDescriptor {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };
}

#[repr(C)]
pub struct BlendStateDescriptor {
    pub blend_enabled: bool,
    pub alpha: BlendDescriptor,
    pub color: BlendDescriptor,
    pub write_mask: ColorWriteFlags,
}

impl BlendStateDescriptor {
    pub const REPLACE: Self = BlendStateDescriptor {
        blend_enabled: false,
        alpha: BlendDescriptor::REPLACE,
        color: BlendDescriptor::REPLACE,
        write_mask: ColorWriteFlags::ALL,
    };
}

pub(crate) struct BlendState {
    pub raw: hal::pso::ColorBlendDesc,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum StencilOperation {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    Invert = 3,
    IncrementClamp = 4,
    DecrementClamp = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
}

#[repr(C)]
pub struct StencilStateFaceDescriptor {
    pub compare: resource::CompareFunction,
    pub stencil_fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl StencilStateFaceDescriptor {
    pub const IGNORE: Self = StencilStateFaceDescriptor {
        compare: resource::CompareFunction::Always,
        stencil_fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

#[repr(C)]
pub struct DepthStencilStateDescriptor {
    pub depth_write_enabled: bool,
    pub depth_compare: resource::CompareFunction,
    pub front: StencilStateFaceDescriptor,
    pub back: StencilStateFaceDescriptor,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
}

impl DepthStencilStateDescriptor {
    pub const IGNORE: Self = DepthStencilStateDescriptor {
        depth_write_enabled: false,
        depth_compare: resource::CompareFunction::Always,
        front: StencilStateFaceDescriptor::IGNORE,
        back: StencilStateFaceDescriptor::IGNORE,
        stencil_read_mask: 0xFF,
        stencil_write_mask: 0xFF,
    };
}

pub(crate) struct DepthStencilState {
    pub raw: hal::pso::DepthStencilDesc,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum IndexFormat {
    Uint16 = 0,
    Uint32 = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum VertexFormat {
    FloatR32G32B32A32 = 0,
    FloatR32G32B32 = 1,
    FloatR32G32 = 2,
    FloatR32 = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum InputStepMode {
    Vertex = 0,
    Instance = 1,
}

#[repr(C)]
pub struct VertexAttributeDescriptor {
    pub shader_location: u32,
    pub input_slot: u32,
    pub offset: u32,
    pub format: VertexFormat,
}

#[repr(C)]
pub struct VertexInputDescriptor {
    pub input_slot: u32,
    pub stride: u32,
    pub step_mode: InputStepMode,
}

#[repr(C)]
pub struct InputStateDescriptor<'a> {
    pub index_format: IndexFormat,
    pub attributes: &'a [VertexAttributeDescriptor],
    pub inputs: &'a [VertexInputDescriptor],
}

pub struct InputState {
    // TODO
}

#[repr(C)]
pub struct ShaderModuleDescriptor {
    pub code: ByteArray,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum ShaderStage {
    Vertex = 0,
    Fragment = 1,
    Compute = 2,
}

#[repr(C)]
pub struct PipelineStageDescriptor {
    pub module: ShaderModuleId,
    pub stage: ShaderStage,
    pub entry_point: *const ::std::os::raw::c_char,
}

#[repr(C)]
pub struct ComputePipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub stages: *const PipelineStageDescriptor,
}

pub(crate) struct ComputePipeline<B: hal::Backend> {
    pub raw: B::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}

#[repr(C)]
pub struct Attachment {
    pub format: resource::TextureFormat,
    pub samples: u32,
}

#[repr(C)]
pub struct AttachmentsState {
    pub color_attachments: *const Attachment,
    pub color_attachments_length: usize,
    pub depth_stencil_attachment: *const Attachment,
}

#[repr(C)]
pub struct RenderPipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub stages: *const PipelineStageDescriptor,
    pub stages_length: usize,
    pub primitive_topology: PrimitiveTopology,
    pub attachments_state: AttachmentsState,
    pub blend_states: *const BlendStateId,
    pub blend_states_length: usize,
    pub depth_stencil_state: DepthStencilStateId,
}

pub(crate) struct RenderPipeline<B: hal::Backend> {
    pub raw: B::GraphicsPipeline,
}
