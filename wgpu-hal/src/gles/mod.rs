#![allow(unused_variables)]

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
#[derive(Debug)]
pub struct Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

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
    type QuerySet = Resource;
    type Fence = Resource;

    type BindGroupLayout = BindGroupLayout;
    type BindGroup = BindGroup;
    type PipelineLayout = PipelineLayout;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

bitflags::bitflags! {
    /// Flags that affect internal code paths but do not
    /// change the exposed feature set.
    struct PrivateCapability: u32 {
        /// Support explicit layouts in shader.
        const EXPLICIT_LAYOUTS_IN_SHADER = 0x00002000;
    }
}

type BindTarget = u32;
type TextureFormat = u32;

trait Sampled {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32);
    // see https://github.com/grovesNL/glow/issues/170
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]);
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32);

    unsafe fn configure_sampling(&self, gl: &glow::Context, desc: &crate::SamplerDescriptor) {
        let (min, mag) =
            conv::map_filter_modes(desc.min_filter, desc.mag_filter, desc.mipmap_filter);

        self.set_param_int(gl, glow::TEXTURE_MIN_FILTER, min as i32);
        self.set_param_int(gl, glow::TEXTURE_MAG_FILTER, mag as i32);

        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_S,
            conv::map_address_mode(desc.address_modes[0]) as i32,
        );
        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_T,
            conv::map_address_mode(desc.address_modes[1]) as i32,
        );
        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_R,
            conv::map_address_mode(desc.address_modes[2]) as i32,
        );

        if let Some(border_color) = desc.border_color {
            let mut border = match border_color {
                wgt::SamplerBorderColor::TransparentBlack => [0.0; 4],
                wgt::SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                wgt::SamplerBorderColor::OpaqueWhite => [1.0; 4],
            };
            self.set_param_float_vec(gl, glow::TEXTURE_BORDER_COLOR, &mut border);
        }

        if let Some(ref range) = desc.lod_clamp {
            self.set_param_float(gl, glow::TEXTURE_MIN_LOD, range.start);
            self.set_param_float(gl, glow::TEXTURE_MAX_LOD, range.end);
        }

        //TODO: `desc.anisotropy_clamp` depends on the downlevel flag
        // self.set_param_float(glow::TEXTURE_MAX_ANISOTROPY, aniso as f32);

        //set_param_float(glow::TEXTURE_LOD_BIAS, info.lod_bias.0);

        if let Some(compare) = desc.compare {
            self.set_param_int(
                gl,
                glow::TEXTURE_COMPARE_MODE,
                glow::COMPARE_REF_TO_TEXTURE as i32,
            );
            self.set_param_int(
                gl,
                glow::TEXTURE_COMPARE_FUNC,
                conv::map_compare_func(compare) as i32,
            );
        }
    }
}

struct SamplerBinding(glow::Sampler);
impl Sampled for SamplerBinding {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32) {
        gl.sampler_parameter_f32(self.0, key, value);
    }
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]) {
        gl.sampler_parameter_f32_slice(self.0, key, values);
    }
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32) {
        gl.sampler_parameter_i32(self.0, key, value);
    }
}

struct SampledTextureBinding(BindTarget);
impl Sampled for SampledTextureBinding {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32) {
        gl.tex_parameter_f32(self.0, key, value);
    }
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]) {
        gl.tex_parameter_f32_slice(self.0, key, values);
    }
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32) {
        gl.tex_parameter_i32(self.0, key, value);
    }
}

#[derive(Debug, Clone, Copy)]
enum VertexAttribKind {
    Float,   // glVertexAttribPointer
    Integer, // glVertexAttribIPointer
    Double,  // glVertexAttribLPointer
}

#[derive(Debug)]
struct FormatDescription {
    tex_internal: u32,
    tex_external: u32,
    data_type: u32,
    num_components: u8,
    va_kind: VertexAttribKind,
}

struct AdapterShared {
    context: glow::Context,
    private_caps: PrivateCapability,
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
    features: wgt::Features,
    copy_fbo: glow::Framebuffer,
}

#[derive(Debug)]
pub struct Buffer {
    raw: glow::Buffer,
    target: BindTarget,
    size: wgt::BufferAddress,
    map_flags: u32,
}

#[derive(Debug)]
enum TextureInner {
    Renderbuffer {
        raw: glow::Renderbuffer,
        aspects: crate::FormatAspect,
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
    format_desc: FormatDescription,
    format_info: wgt::TextureFormatInfo,
}

#[derive(Debug)]
pub enum TextureView {
    Renderbuffer {
        raw: glow::Renderbuffer,
        aspects: crate::FormatAspect,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
        range: wgt::ImageSubresourceRange,
    },
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

#[derive(Debug)]
enum BindingRegister {
    Textures,
    UniformBuffers,
    StorageBuffers,
}

#[derive(Debug)]
enum RawBinding {
    Buffer {
        register: BindingRegister,
        raw: glow::Buffer,
        offset: i32,
        size: i32,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
        range: wgt::ImageSubresourceRange,
    },
    Sampler(glow::Sampler),
}

#[derive(Debug)]
pub struct BindGroup {
    layout_entries: Arc<[wgt::BindGroupLayoutEntry]>,
    contents: Box<[RawBinding]>,
}

#[derive(Debug)]
struct TextureCopyInfo {
    external_format: u32,
    data_type: u32,
    texel_size: u8,
}

#[derive(Debug)]
enum Command {
    Draw {
        primitive: u32,
        start_vertex: u32,
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        primitive: u32,
        index_type: u32,
        index_count: u32,
        index_offset: wgt::BufferAddress,
        base_vertex: i32,
        instance_count: u32,
    },
    DrawIndirect {
        primitive: u32,
        indirect_buf: glow::Buffer,
        indirect_offset: wgt::BufferAddress,
    },
    DrawIndexedIndirect {
        primitive: u32,
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
    InsertDebugMarker(Range<u32>),
    PushDebugGroup(Range<u32>),
    PopDebugGroup,
}

#[derive(Default)]
pub struct CommandBuffer {
    label: Option<String>,
    commands: Vec<Command>,
    data: Vec<u8>,
}

#[derive(Default)]
struct CommandState {
    primitive: u32,
    index_format: wgt::IndexFormat,
    index_offset: wgt::BufferAddress,
}

//TODO: we would have something like `Arc<typed_arena::Arena>`
// here and in the command buffers. So that everything grows
// inside the encoder and stays there until `reset_all`.

pub struct CommandEncoder {
    cmd_buffer: CommandBuffer,
    state: CommandState,
}
