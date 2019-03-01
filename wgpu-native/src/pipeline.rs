use crate::resource;
use crate::{
    ByteArray, PipelineLayoutId, ShaderModuleId,
};

use bitflags::bitflags;

pub type ShaderAttributeIndex = u32;

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
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug)]
pub struct ColorStateDescriptor {
    pub format: resource::TextureFormat,
    pub alpha: BlendDescriptor,
    pub color: BlendDescriptor,
    pub write_mask: ColorWriteFlags,
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
#[derive(Clone, Debug, PartialEq)]
pub struct StencilStateFaceDescriptor {
    pub compare: resource::CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl StencilStateFaceDescriptor {
    pub const IGNORE: Self = StencilStateFaceDescriptor {
        compare: resource::CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct DepthStencilStateDescriptor {
    pub format: resource::TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: resource::CompareFunction,
    pub stencil_front: StencilStateFaceDescriptor,
    pub stencil_back: StencilStateFaceDescriptor,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
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
    Uchar = 0,
    Uchar2 = 1,
    Uchar3 = 2,
    Uchar4 = 3,
    Char = 4,
    Char2 = 5,
    Char3 = 6,
    Char4 = 7,
    UcharNorm = 8,
    Uchar2Norm = 9,
    Uchar3Norm = 10,
    Uchar4Norm = 11,
    Uchar4NormBgra = 12,
    CharNorm = 13,
    Char2Norm = 14,
    Char3Norm = 15,
    Char4Norm = 16,
    Ushort = 17,
    Ushort2 = 18,
    Ushort3 = 19,
    Ushort4 = 20,
    Short  = 21,
    Short2 = 22,
    Short3 = 23,
    Short4 = 24,
    UshortNorm = 25,
    Ushort2Norm = 26,
    Ushort3Norm = 27,
    Ushort4Norm = 28,
    ShortNorm = 29,
    Short2Norm = 30,
    Short3Norm = 31,
    Short4Norm = 32,
    Half = 33,
    Half2 = 34,
    Half3 = 35,
    Half4 = 36,
    Float = 37,
    Float2 = 38,
    Float3 = 39,
    Float4 = 40,
    Uint = 41,
    Uint2 = 42,
    Uint3 = 43,
    Uint4 = 44,
    Int = 45,
    Int2 = 46,
    Int3 = 47,
    Int4 = 48,
  }

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum InputStepMode {
    Vertex = 0,
    Instance = 1,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct VertexAttributeDescriptor {
    pub offset: u32,
    pub format: VertexFormat,
    pub attribute_index: ShaderAttributeIndex,
}

#[repr(C)]
pub struct VertexBufferDescriptor {
    pub stride: u32,
    pub step_mode: InputStepMode,
    pub attributes: *const VertexAttributeDescriptor,
    pub attributes_count: usize,
}

#[repr(C)]
pub struct VertexBufferStateDescriptor {
    pub index_format: IndexFormat,
    pub vertex_buffers: *const VertexBufferDescriptor,
    pub vertex_buffers_count: usize,
}

#[repr(C)]
pub struct ShaderModuleDescriptor {
    pub code: ByteArray,
}

#[repr(C)]
pub struct PipelineStageDescriptor {
    pub module: ShaderModuleId,
    pub entry_point: *const ::std::os::raw::c_char,
}

#[repr(C)]
pub struct ComputePipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub compute_stage: PipelineStageDescriptor,
}

pub struct ComputePipeline<B: hal::Backend> {
    pub(crate) raw: B::ComputePipeline,
    pub(crate) layout_id: PipelineLayoutId,
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
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum FrontFace {
    Ccw = 0,
    Cw = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum CullMode {
    None = 0,
    Front = 1,
    Back = 2,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct RasterizationStateDescriptor {
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

#[repr(C)]
pub struct RenderPipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub vertex_stage: PipelineStageDescriptor,
    pub fragment_stage: PipelineStageDescriptor,
    pub primitive_topology: PrimitiveTopology,
    pub rasterization_state: RasterizationStateDescriptor,
    pub color_states: *const ColorStateDescriptor,
    pub color_states_length: usize,
    pub depth_stencil_state: *const DepthStencilStateDescriptor,
    pub vertex_buffer_state: VertexBufferStateDescriptor,
    pub sample_count: u32,
}

pub struct RenderPipeline<B: hal::Backend> {
    pub(crate) raw: B::GraphicsPipeline,
    pub(crate) layout_id: PipelineLayoutId,
}
