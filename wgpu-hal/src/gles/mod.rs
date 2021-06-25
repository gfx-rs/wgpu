#[cfg(not(target_arch = "wasm32"))]
mod egl;

mod adapter;
mod command;
mod conv;
mod device;
mod queue;

#[cfg(not(target_arch = "wasm32"))]
use self::egl::{Instance, Surface};

use glow::HasContext;

use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct Api;

//Note: we can support more samplers if not every one of them is used at a time,
// but it probably doesn't worth it.
const MAX_TEXTURE_SLOTS: usize = 16;
const MAX_SAMPLERS: usize = 16;
const MAX_VERTEX_ATTRIBUTES: usize = 16;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = CommandEncoder;
    type CommandBuffer = CommandBuffer;

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = Texture;
    type TextureView = TextureView;
    type Sampler = Sampler;
    type QuerySet = QuerySet;
    type Fence = Fence;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = ShaderModule;
    type RenderPipeline = RenderPipeline;
    type ComputePipeline = ComputePipeline;
}

bitflags::bitflags! {
    /// Flags that affect internal code paths but do not
    /// change the exposed feature set.
    struct PrivateCapability: u32 {
        /// Support explicit layouts in shader.
        const EXPLICIT_LAYOUTS_IN_SHADER = 0x0001;
        /// Support memory barriers.
        const MEMORY_BARRIERS = 0x0002;
    }
}

type BindTarget = u32;

#[derive(Debug, Clone, Copy)]
enum VertexAttribKind {
    Float, // glVertexAttribPointer
    Integer, // glVertexAttribIPointer
           //Double,  // glVertexAttribLPointer
}

impl Default for VertexAttribKind {
    fn default() -> Self {
        Self::Float
    }
}

#[derive(Clone, Debug)]
struct TextureFormatDesc {
    internal: u32,
    external: u32,
    data_type: u32,
}

struct AdapterShared {
    context: glow::Context,
    private_caps: PrivateCapability,
    shading_language_version: naga::back::glsl::Version,
}

pub struct Adapter {
    shared: Arc<AdapterShared>,
}

pub struct Device {
    shared: Arc<AdapterShared>,
    main_vao: glow::VertexArray,
}

pub struct Queue {
    shared: Arc<AdapterShared>,
    draw_fbo: glow::Framebuffer,
    copy_fbo: glow::Framebuffer,
    temp_query_results: Vec<u64>,
}

#[derive(Debug)]
pub struct Buffer {
    raw: glow::Buffer,
    target: BindTarget,
    size: wgt::BufferAddress,
    map_flags: u32,
}

#[derive(Clone, Debug)]
enum TextureInner {
    Renderbuffer {
        raw: glow::Renderbuffer,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
    },
}

impl TextureInner {
    fn as_native(&self) -> (glow::Texture, BindTarget) {
        match *self {
            Self::Renderbuffer { raw, .. } => panic!("Unexpected renderbuffer {}", raw),
            Self::Texture { raw, target } => (raw, target),
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    inner: TextureInner,
    mip_level_count: u32,
    array_layer_count: u32,
    format: wgt::TextureFormat,
    format_desc: TextureFormatDesc,
}

#[derive(Clone, Debug)]
pub struct TextureView {
    inner: TextureInner,
    sample_type: wgt::TextureSampleType,
    aspects: crate::FormatAspect,
    mip_levels: Range<u32>,
    array_layers: Range<u32>,
}

#[derive(Debug)]
pub struct Sampler {
    raw: glow::Sampler,
}

pub struct BindGroupLayout {
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
}

struct BindGroupLayoutInfo {
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
    /// Mapping of resources, indexed by `binding`, into the whole layout space.
    /// For texture resources, the value is the texture slot index.
    /// For sampler resources, the value is the index of the sampler in the whole layout.
    /// For buffers, the value is the uniform or storage slot index.
    /// For unused bindings, the value is `!0`
    binding_to_slot: Box<[u8]>,
}

pub struct PipelineLayout {
    group_infos: Box<[BindGroupLayoutInfo]>,
}

impl PipelineLayout {
    fn get_slot(&self, br: &naga::ResourceBinding) -> u8 {
        let group_info = &self.group_infos[br.group as usize];
        group_info.binding_to_slot[br.binding as usize]
    }
}

#[derive(Debug)]
enum BindingRegister {
    UniformBuffers,
    StorageBuffers,
    Textures,
    Images,
}

#[derive(Debug)]
enum RawBinding {
    Buffer {
        raw: glow::Buffer,
        offset: i32,
        size: i32,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
        //TODO: mip levels, array layers
    },
    Image(ImageBinding),
    Sampler(glow::Sampler),
}

#[derive(Debug)]
pub struct BindGroup {
    contents: Box<[RawBinding]>,
}

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
}

#[derive(Clone, Debug, Default)]
struct VertexFormatDesc {
    element_count: i32,
    element_format: u32,
    attrib_kind: VertexAttribKind,
}

#[derive(Clone, Debug, Default)]
struct AttributeDesc {
    location: u32,
    offset: u32,
    buffer_index: u32,
    format_desc: VertexFormatDesc,
}

#[derive(Clone, Debug, Default)]
struct BufferBinding {
    raw: glow::Buffer,
    offset: wgt::BufferAddress,
}

#[derive(Clone, Debug)]
struct ImageBinding {
    raw: glow::Texture,
    mip_level: u32,
    array_layer: Option<u32>,
    access: u32,
    format: u32,
}

#[derive(Clone, Debug, Default)]
struct VertexBufferDesc {
    step: wgt::InputStepMode,
    stride: u32,
}

#[derive(Clone)]
struct UniformDesc {
    location: glow::UniformLocation,
    offset: u32,
    utype: u32,
}

/// For each texture in the pipeline layout, store the index of the only
/// sampler (in this layout) that the texture is used with.
type SamplerBindMap = [Option<u8>; MAX_TEXTURE_SLOTS];

struct PipelineInner {
    program: glow::Program,
    sampler_map: SamplerBindMap,
    uniforms: Box<[UniformDesc]>,
}

#[derive(Clone, Debug)]
struct DepthState {
    function: u32,
    mask: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct BlendComponent {
    src: u32,
    dst: u32,
    equation: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct BlendDesc {
    alpha: BlendComponent,
    color: BlendComponent,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ColorTargetDesc {
    mask: wgt::ColorWrite,
    blend: Option<BlendDesc>,
}

pub struct RenderPipeline {
    inner: PipelineInner,
    primitive: wgt::PrimitiveState,
    vertex_buffers: Box<[VertexBufferDesc]>,
    vertex_attributes: Box<[AttributeDesc]>,
    color_targets: Box<[ColorTargetDesc]>,
    depth: Option<DepthState>,
    depth_bias: wgt::DepthBiasState,
    stencil: Option<StencilState>,
}

pub struct ComputePipeline {
    inner: PipelineInner,
}

#[derive(Debug)]
pub struct QuerySet {
    queries: Box<[glow::Query]>,
    target: BindTarget,
}

#[derive(Debug)]
pub struct Fence {
    last_completed: crate::FenceValue,
    pending: Vec<(crate::FenceValue, glow::Fence)>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    fn get_latest(&self, gl: &glow::Context) -> crate::FenceValue {
        let mut max_value = self.last_completed;
        for &(value, sync) in self.pending.iter() {
            let status = unsafe { gl.get_sync_status(sync) };
            if status == glow::SIGNALED {
                max_value = value;
            }
        }
        max_value
    }

    fn maintain(&mut self, gl: &glow::Context) {
        let latest = self.get_latest(gl);
        for &(value, sync) in self.pending.iter() {
            if value <= latest {
                unsafe {
                    gl.delete_sync(sync);
                }
            }
        }
        self.pending.retain(|&(value, _)| value > latest);
        self.last_completed = latest;
    }
}

#[derive(Debug)]
struct TextureCopyInfo {
    external_format: u32,
    data_type: u32,
    texel_size: u8,
}

#[derive(Clone, Debug, PartialEq)]
struct StencilOps {
    pass: u32,
    fail: u32,
    depth_fail: u32,
}

impl Default for StencilOps {
    fn default() -> Self {
        Self {
            pass: glow::KEEP,
            fail: glow::KEEP,
            depth_fail: glow::KEEP,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StencilSide {
    function: u32,
    mask_read: u32,
    mask_write: u32,
    reference: u32,
    ops: StencilOps,
}

impl Default for StencilSide {
    fn default() -> Self {
        Self {
            function: glow::ALWAYS,
            mask_read: 0xFF,
            mask_write: 0xFF,
            reference: 0,
            ops: StencilOps::default(),
        }
    }
}

#[derive(Clone, Default)]
struct StencilState {
    front: StencilSide,
    back: StencilSide,
}

#[derive(Debug)]
enum Command {
    Draw {
        topology: u32,
        start_vertex: u32,
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        topology: u32,
        index_type: u32,
        index_count: u32,
        index_offset: wgt::BufferAddress,
        base_vertex: i32,
        instance_count: u32,
    },
    DrawIndirect {
        topology: u32,
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
    },
    DrawIndexedIndirect {
        topology: u32,
        index_type: u32,
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
    },
    Dispatch([u32; 3]),
    DispatchIndirect {
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
    },
    FillBuffer {
        dst: glow::Buffer,
        range: crate::MemoryRange,
        value: u8,
    },
    CopyBufferToBuffer {
        src: glow::Buffer,
        src_target: BindTarget,
        dst: glow::Buffer,
        dst_target: BindTarget,
        copy: crate::BufferCopy,
    },
    CopyTextureToTexture {
        src: glow::Texture,
        src_target: BindTarget,
        dst: glow::Texture,
        dst_target: BindTarget,
        copy: crate::TextureCopy,
    },
    CopyBufferToTexture {
        src: glow::Buffer,
        src_target: BindTarget,
        dst: glow::Texture,
        dst_target: BindTarget,
        dst_info: TextureCopyInfo,
        copy: crate::BufferTextureCopy,
    },
    CopyTextureToBuffer {
        src: glow::Texture,
        src_target: BindTarget,
        src_info: TextureCopyInfo,
        dst: glow::Buffer,
        dst_target: BindTarget,
        copy: crate::BufferTextureCopy,
    },
    SetIndexBuffer(glow::Buffer),
    BeginQuery(glow::Query, BindTarget),
    EndQuery(BindTarget),
    CopyQueryResults {
        query_range: Range<u32>,
        dst: glow::Buffer,
        dst_target: BindTarget,
        dst_offset: wgt::BufferAddress,
    },
    ResetFramebuffer,
    SetFramebufferAttachment {
        attachment: u32,
        view: TextureView,
    },
    SetDrawColorBuffers(u8),
    ClearColorF(u32, [f32; 4]),
    ClearColorU(u32, [u32; 4]),
    ClearColorI(u32, [i32; 4]),
    ClearDepth(f32),
    ClearStencil(u32),
    BufferBarrier(glow::Buffer, crate::BufferUse),
    TextureBarrier(crate::TextureUse),
    SetViewport {
        rect: crate::Rect<i32>,
        depth: Range<f32>,
    },
    SetScissor(crate::Rect<i32>),
    SetStencilFunc {
        face: u32,
        function: u32,
        reference: u32,
        read_mask: u32,
    },
    SetStencilOps {
        face: u32,
        write_mask: u32,
        ops: StencilOps,
    },
    SetDepth(DepthState),
    SetDepthBias(wgt::DepthBiasState),
    ConfigureDepthStencil(crate::FormatAspect),
    SetVertexAttribute {
        buffer: BufferBinding,
        buffer_desc: VertexBufferDesc,
        attribute_desc: AttributeDesc,
    },
    SetProgram(glow::Program),
    SetBlendConstant([f32; 4]),
    SetColorTarget {
        draw_buffer_index: Option<u32>,
        desc: ColorTargetDesc,
    },
    BindBuffer {
        target: BindTarget,
        slot: u32,
        buffer: glow::Buffer,
        offset: i32,
        size: i32,
    },
    BindSampler(u32, glow::Sampler),
    BindTexture {
        slot: u32,
        texture: glow::Texture,
        target: BindTarget,
    },
    BindImage {
        slot: u32,
        binding: ImageBinding,
    },
    InsertDebugMarker(Range<u32>),
    PushDebugGroup(Range<u32>),
    PopDebugGroup,
}

#[derive(Default)]
pub struct CommandBuffer {
    label: Option<String>,
    commands: Vec<Command>,
    data_bytes: Vec<u8>,
    data_words: Vec<u32>,
}

//TODO: we would have something like `Arc<typed_arena::Arena>`
// here and in the command buffers. So that everything grows
// inside the encoder and stays there until `reset_all`.

pub struct CommandEncoder {
    cmd_buffer: CommandBuffer,
    state: command::State,
    private_caps: PrivateCapability,
}
