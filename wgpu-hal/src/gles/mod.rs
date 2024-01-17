/*!
# OpenGL ES3 API (aka GLES3).

Designed to work on Linux and Android, with context provided by EGL.

## Texture views

GLES3 doesn't really have separate texture view objects. We have to remember the
original texture and the sub-range into it. Problem is, however, that there is
no way to expose a subset of array layers or mip levels of a sampled texture.

## Binding model

Binding model is very different from WebGPU, especially with regards to samplers.
GLES3 has sampler objects, but they aren't separately bindable to the shaders.
Each sampled texture is exposed to the shader as a combined texture-sampler binding.

When building the pipeline layout, we linearize binding entries based on the groups
(uniform/storage buffers, uniform/storage textures), and record the mapping into
`BindGroupLayoutInfo`.
When a pipeline gets created, and we track all the texture-sampler associations
from the static use in the shader.
We only support at most one sampler used with each texture so far. The linear index
of this sampler is stored per texture slot in `SamplerBindMap` array.

The texture-sampler pairs get potentially invalidated in 2 places:
  - when a new pipeline is set, we update the linear indices of associated samplers
  - when a new bind group is set, we update both the textures and the samplers

We expect that the changes to sampler states between any 2 pipelines of the same layout
will be minimal, if any.

## Vertex data

Generally, vertex buffers are marked as dirty and lazily bound on draw.

GLES3 doesn't support `first_instance` semantics. However, it's easy to support,
since we are forced to do late binding anyway. We just adjust the offsets
into the vertex data.

### Old path

In GLES-3.0 and WebGL2, vertex buffer layout is provided
together with the actual buffer binding.
We invalidate the attributes on the vertex buffer change, and re-bind them.

### New path

In GLES-3.1 and higher, the vertex buffer layout can be declared separately
from the vertex data itself. This mostly matches WebGPU, however there is a catch:
`stride` needs to be specified with the data, not as a part of the layout.

To address this, we invalidate the vertex buffers based on:
  - whether or not `first_instance` is used
  - stride has changed

## Handling of `base_vertex`, `first_instance`, and `first_vertex`

Between indirect, the lack of `first_instance` semantics, and the availability of `gl_BaseInstance`
in shaders, getting buffers and builtins to work correctly is a bit tricky.

We never emulate `base_vertex` and gl_VertexID behaves as `@builtin(vertex_index)` does, so we
never need to do anything about that.

We always advertise support for `VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW`.

### GL 4.2+ with ARB_shader_draw_parameters

- `@builtin(instance_index)` translates to `gl_InstanceID + gl_BaseInstance`
- We bind instance buffers without any offset emulation.
- We advertise support for the `INDIRECT_FIRST_INSTANCE` feature.

While we can theoretically have a card with 4.2+ support but without ARB_shader_draw_parameters,
we don't bother with that combination.

### GLES & GL 4.1

- `@builtin(instance_index)` translates to `gl_InstanceID + naga_vs_first_instance`
- We bind instance buffers with offset emulation.
- We _do not_ advertise support for `INDIRECT_FIRST_INSTANCE` and cpu-side pretend the `first_instance` is 0 on indirect calls.

*/

///cbindgen:ignore
#[cfg(not(any(windows, webgl)))]
mod egl;
#[cfg(Emscripten)]
mod emscripten;
#[cfg(webgl)]
mod web;
#[cfg(windows)]
mod wgl;

mod adapter;
mod command;
mod conv;
mod device;
mod queue;

use crate::{CopyExtent, TextureDescriptor};

#[cfg(not(any(windows, webgl)))]
pub use self::egl::{AdapterContext, AdapterContextLock};
#[cfg(not(any(windows, webgl)))]
use self::egl::{Instance, Surface};

#[cfg(webgl)]
pub use self::web::AdapterContext;
#[cfg(webgl)]
use self::web::{Instance, Surface};

#[cfg(windows)]
use self::wgl::AdapterContext;
#[cfg(windows)]
use self::wgl::{Instance, Surface};

use arrayvec::ArrayVec;

use glow::HasContext;

use naga::FastHashMap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU8};
use std::{fmt, ops::Range, sync::Arc};

#[derive(Clone, Debug)]
pub struct Api;

//Note: we can support more samplers if not every one of them is used at a time,
// but it probably doesn't worth it.
const MAX_TEXTURE_SLOTS: usize = 16;
const MAX_SAMPLERS: usize = 16;
const MAX_VERTEX_ATTRIBUTES: usize = 16;
const ZERO_BUFFER_SIZE: usize = 256 << 10;
const MAX_PUSH_CONSTANTS: usize = 64;
// We have to account for each push constant may need to be set for every shader.
const MAX_PUSH_CONSTANT_COMMANDS: usize = MAX_PUSH_CONSTANTS * crate::MAX_CONCURRENT_SHADER_STAGES;

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
    type AccelerationStructure = ();

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
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct PrivateCapabilities: u32 {
        /// Indicates support for `glBufferStorage` allocation.
        const BUFFER_ALLOCATION = 1 << 0;
        /// Support explicit layouts in shader.
        const SHADER_BINDING_LAYOUT = 1 << 1;
        /// Support extended shadow sampling instructions.
        const SHADER_TEXTURE_SHADOW_LOD = 1 << 2;
        /// Support memory barriers.
        const MEMORY_BARRIERS = 1 << 3;
        /// Vertex buffer layouts separate from the data.
        const VERTEX_BUFFER_LAYOUT = 1 << 4;
        /// Indicates that buffers used as `GL_ELEMENT_ARRAY_BUFFER` may be created / initialized / used
        /// as other targets, if not present they must not be mixed with other targets.
        const INDEX_BUFFER_ROLE_CHANGE = 1 << 5;
        /// Supports `glGetBufferSubData`
        const GET_BUFFER_SUB_DATA = 1 << 7;
        /// Supports `f16` color buffers
        const COLOR_BUFFER_HALF_FLOAT = 1 << 8;
        /// Supports `f11/f10` and `f32` color buffers
        const COLOR_BUFFER_FLOAT = 1 << 9;
        /// Supports query buffer objects.
        const QUERY_BUFFERS = 1 << 11;
        /// Supports 64 bit queries via `glGetQueryObjectui64v`
        const QUERY_64BIT = 1 << 12;
        /// Supports `glTexStorage2D`, etc.
        const TEXTURE_STORAGE = 1 << 13;
        /// Supports `push_debug_group`, `pop_debug_group` and `debug_message_insert`.
        const DEBUG_FNS = 1 << 14;
        /// Supports framebuffer invalidation.
        const INVALIDATE_FRAMEBUFFER = 1 << 15;
        /// Indicates support for `glDrawElementsInstancedBaseVertexBaseInstance` and `ARB_shader_draw_parameters`
        ///
        /// When this is true, instance offset emulation via vertex buffer rebinding and a shader uniform will be disabled.
        const FULLY_FEATURED_INSTANCING = 1 << 16;
    }
}

bitflags::bitflags! {
    /// Flags that indicate necessary workarounds for specific devices or driver bugs
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Workarounds: u32 {
        // Needs workaround for Intel Mesa bug:
        // https://gitlab.freedesktop.org/mesa/mesa/-/issues/2565.
        //
        // This comment
        // (https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/4972/diffs?diff_id=75888#22f5d1004713c9bbf857988c7efb81631ab88f99_323_327)
        // seems to indicate all skylake models are effected.
        const MESA_I915_SRGB_SHADER_CLEAR = 1 << 0;
        /// Buffer map must emulated becuase it is not supported natively
        const EMULATE_BUFFER_MAP = 1 << 1;
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
pub struct TextureFormatDesc {
    pub internal: u32,
    pub external: u32,
    pub data_type: u32,
}

struct AdapterShared {
    context: AdapterContext,
    private_caps: PrivateCapabilities,
    features: wgt::Features,
    workarounds: Workarounds,
    shading_language_version: naga::back::glsl::Version,
    next_shader_id: AtomicU32,
    program_cache: Mutex<ProgramCache>,
    es: bool,
}

pub struct Adapter {
    shared: Arc<AdapterShared>,
}

pub struct Device {
    shared: Arc<AdapterShared>,
    main_vao: glow::VertexArray,
    #[cfg(all(native, feature = "renderdoc"))]
    render_doc: crate::auxil::renderdoc::RenderDoc,
}

pub struct Queue {
    shared: Arc<AdapterShared>,
    features: wgt::Features,
    draw_fbo: glow::Framebuffer,
    copy_fbo: glow::Framebuffer,
    /// Shader program used to clear the screen for [`Workarounds::MESA_I915_SRGB_SHADER_CLEAR`]
    /// devices.
    shader_clear_program: glow::Program,
    /// The uniform location of the color uniform in the shader clear program
    shader_clear_program_color_uniform_location: glow::UniformLocation,
    /// Keep a reasonably large buffer filled with zeroes, so that we can implement `ClearBuffer` of
    /// zeroes by copying from it.
    zero_buffer: glow::Buffer,
    temp_query_results: Mutex<Vec<u64>>,
    draw_buffer_count: AtomicU8,
    current_index_buffer: Mutex<Option<glow::Buffer>>,
}

#[derive(Clone, Debug)]
pub struct Buffer {
    raw: Option<glow::Buffer>,
    target: BindTarget,
    size: wgt::BufferAddress,
    map_flags: u32,
    data: Option<Arc<std::sync::Mutex<Vec<u8>>>>,
}

#[cfg(send_sync)]
unsafe impl Sync for Buffer {}
#[cfg(send_sync)]
unsafe impl Send for Buffer {}

#[derive(Clone, Debug)]
pub enum TextureInner {
    Renderbuffer {
        raw: glow::Renderbuffer,
    },
    DefaultRenderbuffer,
    Texture {
        raw: glow::Texture,
        target: BindTarget,
    },
    #[cfg(webgl)]
    ExternalFramebuffer {
        inner: web_sys::WebGlFramebuffer,
    },
}

#[cfg(send_sync)]
unsafe impl Sync for TextureInner {}
#[cfg(send_sync)]
unsafe impl Send for TextureInner {}

impl TextureInner {
    fn as_native(&self) -> (glow::Texture, BindTarget) {
        match *self {
            Self::Renderbuffer { .. } | Self::DefaultRenderbuffer => {
                panic!("Unexpected renderbuffer");
            }
            Self::Texture { raw, target } => (raw, target),
            #[cfg(webgl)]
            Self::ExternalFramebuffer { .. } => panic!("Unexpected external framebuffer"),
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    pub inner: TextureInner,
    pub drop_guard: Option<crate::DropGuard>,
    pub mip_level_count: u32,
    pub array_layer_count: u32,
    pub format: wgt::TextureFormat,
    #[allow(unused)]
    pub format_desc: TextureFormatDesc,
    pub copy_size: CopyExtent,
}

impl Texture {
    pub fn default_framebuffer(format: wgt::TextureFormat) -> Self {
        Self {
            inner: TextureInner::DefaultRenderbuffer,
            drop_guard: None,
            mip_level_count: 1,
            array_layer_count: 1,
            format,
            format_desc: TextureFormatDesc {
                internal: 0,
                external: 0,
                data_type: 0,
            },
            copy_size: CopyExtent {
                width: 0,
                height: 0,
                depth: 0,
            },
        }
    }

    /// Returns the `target`, whether the image is 3d and whether the image is a cubemap.
    fn get_info_from_desc(desc: &TextureDescriptor) -> u32 {
        match desc.dimension {
            wgt::TextureDimension::D1 => glow::TEXTURE_2D,
            wgt::TextureDimension::D2 => {
                // HACK: detect a cube map; forces cube compatible textures to be cube textures
                match (desc.is_cube_compatible(), desc.size.depth_or_array_layers) {
                    (false, 1) => glow::TEXTURE_2D,
                    (false, _) => glow::TEXTURE_2D_ARRAY,
                    (true, 6) => glow::TEXTURE_CUBE_MAP,
                    (true, _) => glow::TEXTURE_CUBE_MAP_ARRAY,
                }
            }
            wgt::TextureDimension::D3 => glow::TEXTURE_3D,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextureView {
    inner: TextureInner,
    aspects: crate::FormatAspects,
    mip_levels: Range<u32>,
    array_layers: Range<u32>,
    format: wgt::TextureFormat,
}

#[derive(Debug)]
pub struct Sampler {
    raw: glow::Sampler,
}

#[derive(Debug)]
pub struct BindGroupLayout {
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
}

#[derive(Debug)]
struct BindGroupLayoutInfo {
    entries: Arc<[wgt::BindGroupLayoutEntry]>,
    /// Mapping of resources, indexed by `binding`, into the whole layout space.
    /// For texture resources, the value is the texture slot index.
    /// For sampler resources, the value is the index of the sampler in the whole layout.
    /// For buffers, the value is the uniform or storage slot index.
    /// For unused bindings, the value is `!0`
    binding_to_slot: Box<[u8]>,
}

#[derive(Debug)]
pub struct PipelineLayout {
    group_infos: Box<[BindGroupLayoutInfo]>,
    naga_options: naga::back::glsl::Options,
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
        aspects: crate::FormatAspects,
        mip_levels: Range<u32>,
        //TODO: array layers
    },
    Image(ImageBinding),
    Sampler(glow::Sampler),
}

#[derive(Debug)]
pub struct BindGroup {
    contents: Box<[RawBinding]>,
}

type ShaderId = u32;

#[derive(Debug)]
pub struct ShaderModule {
    naga: crate::NagaShader,
    label: Option<String>,
    id: ShaderId,
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Default, PartialEq)]
struct VertexBufferDesc {
    step: wgt::VertexStepMode,
    stride: u32,
}

#[derive(Clone, Debug)]
struct PushConstantDesc {
    location: glow::UniformLocation,
    ty: naga::TypeInner,
    offset: u32,
    size_bytes: u32,
}

#[cfg(send_sync)]
unsafe impl Sync for PushConstantDesc {}
#[cfg(send_sync)]
unsafe impl Send for PushConstantDesc {}

/// For each texture in the pipeline layout, store the index of the only
/// sampler (in this layout) that the texture is used with.
type SamplerBindMap = [Option<u8>; MAX_TEXTURE_SLOTS];

#[derive(Debug)]
struct PipelineInner {
    program: glow::Program,
    sampler_map: SamplerBindMap,
    first_instance_location: Option<glow::UniformLocation>,
    push_constant_descs: ArrayVec<PushConstantDesc, MAX_PUSH_CONSTANT_COMMANDS>,
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
    mask: wgt::ColorWrites,
    blend: Option<BlendDesc>,
}

#[derive(PartialEq, Eq, Hash)]
struct ProgramStage {
    naga_stage: naga::ShaderStage,
    shader_id: ShaderId,
    entry_point: String,
}

#[derive(PartialEq, Eq, Hash)]
struct ProgramCacheKey {
    stages: ArrayVec<ProgramStage, 3>,
    group_to_binding_to_slot: Box<[Box<[u8]>]>,
}

type ProgramCache = FastHashMap<ProgramCacheKey, Result<Arc<PipelineInner>, crate::PipelineError>>;

#[derive(Debug)]
pub struct RenderPipeline {
    inner: Arc<PipelineInner>,
    primitive: wgt::PrimitiveState,
    vertex_buffers: Box<[VertexBufferDesc]>,
    vertex_attributes: Box<[AttributeDesc]>,
    color_targets: Box<[ColorTargetDesc]>,
    depth: Option<DepthState>,
    depth_bias: wgt::DepthBiasState,
    stencil: Option<StencilState>,
    alpha_to_coverage_enabled: bool,
}

#[cfg(send_sync)]
unsafe impl Sync for RenderPipeline {}
#[cfg(send_sync)]
unsafe impl Send for RenderPipeline {}

#[derive(Debug)]
pub struct ComputePipeline {
    inner: Arc<PipelineInner>,
}

#[cfg(send_sync)]
unsafe impl Sync for ComputePipeline {}
#[cfg(send_sync)]
unsafe impl Send for ComputePipeline {}

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

#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl Send for Fence {}
#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
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

#[derive(Debug, Clone, Default)]
struct StencilState {
    front: StencilSide,
    back: StencilSide,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct PrimitiveState {
    front_face: u32,
    cull_face: u32,
    unclipped_depth: bool,
    polygon_mode: u32,
}

type InvalidatedAttachments = ArrayVec<u32, { crate::MAX_COLOR_ATTACHMENTS + 2 }>;

#[derive(Debug)]
enum Command {
    Draw {
        topology: u32,
        first_vertex: u32,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
        first_instance_location: Option<glow::UniformLocation>,
    },
    DrawIndexed {
        topology: u32,
        index_type: u32,
        index_count: u32,
        index_offset: wgt::BufferAddress,
        base_vertex: i32,
        first_instance: u32,
        instance_count: u32,
        first_instance_location: Option<glow::UniformLocation>,
    },
    DrawIndirect {
        topology: u32,
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
        first_instance_location: Option<glow::UniformLocation>,
    },
    DrawIndexedIndirect {
        topology: u32,
        index_type: u32,
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
        first_instance_location: Option<glow::UniformLocation>,
    },
    Dispatch([u32; 3]),
    DispatchIndirect {
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
    },
    ClearBuffer {
        dst: Buffer,
        dst_target: BindTarget,
        range: crate::MemoryRange,
    },
    CopyBufferToBuffer {
        src: Buffer,
        src_target: BindTarget,
        dst: Buffer,
        dst_target: BindTarget,
        copy: crate::BufferCopy,
    },
    #[cfg(webgl)]
    CopyExternalImageToTexture {
        src: wgt::ImageCopyExternalImage,
        dst: glow::Texture,
        dst_target: BindTarget,
        dst_format: wgt::TextureFormat,
        dst_premultiplication: bool,
        copy: crate::TextureCopy,
    },
    CopyTextureToTexture {
        src: glow::Texture,
        src_target: BindTarget,
        dst: glow::Texture,
        dst_target: BindTarget,
        copy: crate::TextureCopy,
    },
    CopyBufferToTexture {
        src: Buffer,
        #[allow(unused)]
        src_target: BindTarget,
        dst: glow::Texture,
        dst_target: BindTarget,
        dst_format: wgt::TextureFormat,
        copy: crate::BufferTextureCopy,
    },
    CopyTextureToBuffer {
        src: glow::Texture,
        src_target: BindTarget,
        src_format: wgt::TextureFormat,
        dst: Buffer,
        #[allow(unused)]
        dst_target: BindTarget,
        copy: crate::BufferTextureCopy,
    },
    SetIndexBuffer(glow::Buffer),
    BeginQuery(glow::Query, BindTarget),
    EndQuery(BindTarget),
    TimestampQuery(glow::Query),
    CopyQueryResults {
        query_range: Range<u32>,
        dst: Buffer,
        dst_target: BindTarget,
        dst_offset: wgt::BufferAddress,
    },
    ResetFramebuffer {
        is_default: bool,
    },
    BindAttachment {
        attachment: u32,
        view: TextureView,
    },
    ResolveAttachment {
        attachment: u32,
        dst: TextureView,
        size: wgt::Extent3d,
    },
    InvalidateAttachments(InvalidatedAttachments),
    SetDrawColorBuffers(u8),
    ClearColorF {
        draw_buffer: u32,
        color: [f32; 4],
        is_srgb: bool,
    },
    ClearColorU(u32, [u32; 4]),
    ClearColorI(u32, [i32; 4]),
    ClearDepth(f32),
    ClearStencil(u32),
    // Clearing both the depth and stencil buffer individually appears to
    // result in the stencil buffer failing to clear, atleast in WebGL.
    // It is also more efficient to emit a single command instead of two for
    // this.
    ClearDepthAndStencil(f32, u32),
    BufferBarrier(glow::Buffer, crate::BufferUses),
    TextureBarrier(crate::TextureUses),
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
    ConfigureDepthStencil(crate::FormatAspects),
    SetAlphaToCoverage(bool),
    SetVertexAttribute {
        buffer: Option<glow::Buffer>,
        buffer_desc: VertexBufferDesc,
        attribute_desc: AttributeDesc,
    },
    UnsetVertexAttribute(u32),
    SetVertexBuffer {
        index: u32,
        buffer: BufferBinding,
        buffer_desc: VertexBufferDesc,
    },
    SetProgram(glow::Program),
    SetPrimitive(PrimitiveState),
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
    BindSampler(u32, Option<glow::Sampler>),
    BindTexture {
        slot: u32,
        texture: glow::Texture,
        target: BindTarget,
        aspects: crate::FormatAspects,
        mip_levels: Range<u32>,
    },
    BindImage {
        slot: u32,
        binding: ImageBinding,
    },
    InsertDebugMarker(Range<u32>),
    PushDebugGroup(Range<u32>),
    PopDebugGroup,
    SetPushConstants {
        uniform: PushConstantDesc,
        /// Offset from the start of the `data_bytes`
        offset: u32,
    },
}

#[derive(Default)]
pub struct CommandBuffer {
    label: Option<String>,
    commands: Vec<Command>,
    data_bytes: Vec<u8>,
    queries: Vec<glow::Query>,
}

impl fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = f.debug_struct("CommandBuffer");
        if let Some(ref label) = self.label {
            builder.field("label", label);
        }
        builder.finish()
    }
}

#[cfg(send_sync)]
unsafe impl Sync for CommandBuffer {}
#[cfg(send_sync)]
unsafe impl Send for CommandBuffer {}

//TODO: we would have something like `Arc<typed_arena::Arena>`
// here and in the command buffers. So that everything grows
// inside the encoder and stays there until `reset_all`.

pub struct CommandEncoder {
    cmd_buffer: CommandBuffer,
    state: command::State,
    private_caps: PrivateCapabilities,
}

impl fmt::Debug for CommandEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandEncoder")
            .field("cmd_buffer", &self.cmd_buffer)
            .finish()
    }
}

#[cfg(send_sync)]
unsafe impl Sync for CommandEncoder {}
#[cfg(send_sync)]
unsafe impl Send for CommandEncoder {}

#[cfg(not(webgl))]
fn gl_debug_message_callback(source: u32, gltype: u32, id: u32, severity: u32, message: &str) {
    let source_str = match source {
        glow::DEBUG_SOURCE_API => "API",
        glow::DEBUG_SOURCE_WINDOW_SYSTEM => "Window System",
        glow::DEBUG_SOURCE_SHADER_COMPILER => "ShaderCompiler",
        glow::DEBUG_SOURCE_THIRD_PARTY => "Third Party",
        glow::DEBUG_SOURCE_APPLICATION => "Application",
        glow::DEBUG_SOURCE_OTHER => "Other",
        _ => unreachable!(),
    };

    let log_severity = match severity {
        glow::DEBUG_SEVERITY_HIGH => log::Level::Error,
        glow::DEBUG_SEVERITY_MEDIUM => log::Level::Warn,
        glow::DEBUG_SEVERITY_LOW => log::Level::Info,
        glow::DEBUG_SEVERITY_NOTIFICATION => log::Level::Trace,
        _ => unreachable!(),
    };

    let type_str = match gltype {
        glow::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "Deprecated Behavior",
        glow::DEBUG_TYPE_ERROR => "Error",
        glow::DEBUG_TYPE_MARKER => "Marker",
        glow::DEBUG_TYPE_OTHER => "Other",
        glow::DEBUG_TYPE_PERFORMANCE => "Performance",
        glow::DEBUG_TYPE_POP_GROUP => "Pop Group",
        glow::DEBUG_TYPE_PORTABILITY => "Portability",
        glow::DEBUG_TYPE_PUSH_GROUP => "Push Group",
        glow::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "Undefined Behavior",
        _ => unreachable!(),
    };

    let _ = std::panic::catch_unwind(|| {
        log::log!(
            log_severity,
            "GLES: [{}/{}] ID {} : {}",
            source_str,
            type_str,
            id,
            message
        );
    });

    if cfg!(debug_assertions) && log_severity == log::Level::Error {
        // Set canary and continue
        crate::VALIDATION_CANARY.add(message.to_string());
    }
}
