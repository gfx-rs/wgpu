/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! This library describes the internal unsafe graphics abstraction API.
 *  It follows WebGPU for the most part, re-using wgpu-types,
 *  with the following deviations:
 *  - Fully unsafe: zero overhead, zero validation.
 *  - Compile-time backend selection via traits.
 *  - Objects are passed by references and returned by value. No IDs.
 *  - Mapping is persistent, with explicit synchronization.
 *  - Resource transitions are explicit.
 *  - All layouts are explicit.
 */

#![allow(
    // We use loops for getting early-out of scope without closures.
    clippy::never_loop,
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

pub mod empty;

use std::{borrow::Cow, fmt, num::NonZeroU8, ops::Range, ptr::NonNull};

use bitflags::bitflags;
use smallvec::SmallVec;

pub const MAX_ANISOTROPY: u8 = 16;
pub const MAX_BIND_GROUPS: usize = 8;

pub type Label<'a> = Option<&'a str>;
pub type MemoryRange = Range<wgt::BufferAddress>;
pub type MipLevel = u8;
pub type ArrayLayer = u16;

#[derive(Debug)]
pub enum Error {
    OutOfMemory,
    DeviceLost,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::DeviceLost => write!(f, "device is lost"),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Debug)]
pub enum ShaderError {
    Compilation(String),
    Device(Error),
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Compilation(ref message) => write!(f, "compilation failed: {}", message),
            Self::Device(_) => Ok(()),
        }
    }
}
impl std::error::Error for ShaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Self::Compilation(..) => None,
            Self::Device(ref parent) => Some(parent),
        }
    }
}

#[derive(Debug)]
pub enum PipelineError {
    Linkage(wgt::ShaderStage, String),
    Device(Error),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Linkage(stage, ref message) => {
                write!(f, "linkage failed for stage {:?}: {}", stage, message)
            }
            Self::Device(_) => Ok(()),
        }
    }
}
impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            Self::Linkage(..) => None,
            Self::Device(ref parent) => Some(parent),
        }
    }
}

pub trait Api: Clone + Sized {
    type Instance: Instance<Self>;
    type Surface: Surface<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;
    type Queue: Queue<Self>;

    type CommandBuffer: CommandBuffer<Self>;
    type RenderPass: RenderPass<Self>;
    type ComputePass: ComputePass<Self>;

    type Buffer: fmt::Debug + Send + Sync;
    type QuerySet: fmt::Debug + Send + Sync;
    type Texture: fmt::Debug + Send + Sync;
    type SwapChainTexture: fmt::Debug + Send + Sync;
    type TextureView: fmt::Debug + Send + Sync;
    type Sampler: fmt::Debug + Send + Sync;

    type BindGroupLayout;
    type BindGroup: fmt::Debug + Send + Sync;
    type PipelineLayout;
    type ShaderModule: fmt::Debug + Send + Sync;
    type RenderPipeline;
    type ComputePipeline;
}

pub trait Instance<A: Api> {
    unsafe fn enumerate_adapters(&self) -> Vec<ExposedAdapter<A>>;
}

pub trait Surface<A: Api> {}

pub trait Adapter<A: Api> {
    unsafe fn open(&self, features: wgt::Features) -> Result<OpenDevice<A>, Error>;
    unsafe fn close(&self, device: A::Device);

    /// Return the set of supported capabilities for a texture format.
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapability;
    /// Returns the list of surface formats supported for presentation, if any.
    unsafe fn surface_formats(&self, surface: &A::Surface) -> Vec<wgt::TextureFormat>;
}

pub trait Device<A: Api> {
    unsafe fn create_buffer(&self, desc: &wgt::BufferDescriptor<Label>)
        -> Result<A::Buffer, Error>;
    unsafe fn destroy_buffer(&self, buffer: A::Buffer);
    unsafe fn map_buffer(
        &self,
        buffer: &A::Buffer,
        range: MemoryRange,
    ) -> Result<NonNull<u8>, Error>;
    unsafe fn unmap_buffer(&self, buffer: &A::Buffer);
    unsafe fn flush_mapped_ranges<I: Iterator<Item = MemoryRange>>(
        &self,
        buffer: &A::Buffer,
        ranges: I,
    );
    unsafe fn invalidate_mapped_ranges<I: Iterator<Item = MemoryRange>>(
        &self,
        buffer: &A::Buffer,
        ranges: I,
    );

    unsafe fn create_texture(
        &self,
        desc: &wgt::TextureDescriptor<Label>,
    ) -> Result<A::Texture, Error>;
    unsafe fn destroy_texture(&self, texture: A::Texture);
    unsafe fn create_texture_view(
        &self,
        texture: &A::Texture,
        desc: &TextureViewDescriptor<Label>,
    ) -> Result<A::TextureView, Error>;
    unsafe fn destroy_texture_view(&self, view: A::TextureView);
    unsafe fn create_sampler(&self, desc: &SamplerDescriptor) -> Result<A::Sampler, Error>;
    unsafe fn destroy_sampler(&self, sampler: A::Sampler);

    unsafe fn create_command_buffer(&self) -> Result<A::CommandBuffer, Error>;
    unsafe fn destroy_command_buffer(&self, cmd_buf: A::CommandBuffer);

    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<A::BindGroupLayout, Error>;
    unsafe fn destroy_bind_group_layout(&self, bg_layout: A::BindGroupLayout);
    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor<A>,
    ) -> Result<A::PipelineLayout, Error>;
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: A::PipelineLayout);
    unsafe fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<A>,
    ) -> Result<A::BindGroup, Error>;
    unsafe fn destroy_bind_group(&self, group: A::BindGroup);

    unsafe fn create_shader_module(
        &self,
        desc: &ShaderModuleDescriptor,
        shader: NagaShader,
    ) -> Result<A::ShaderModule, (ShaderError, NagaShader)>;
    unsafe fn destroy_shader_module(&self, module: A::ShaderModule);
    unsafe fn create_render_pipeline(
        &self,
        desc: &RenderPipelineDescriptor<A>,
    ) -> Result<A::RenderPipeline, PipelineError>;
    unsafe fn destroy_render_pipeline(&self, pipeline: A::RenderPipeline);
    unsafe fn create_compute_pipeline(
        &self,
        desc: &ComputePipelineDescriptor<A>,
    ) -> Result<A::ComputePipeline, PipelineError>;
    unsafe fn destroy_compute_pipeline(&self, pipeline: A::ComputePipeline);
}

pub trait Queue<A: Api> {
    unsafe fn submit<I: Iterator<Item = A::CommandBuffer>>(&mut self, command_buffers: I);
}

pub trait CommandBuffer<A: Api> {
    unsafe fn begin(&mut self);
    unsafe fn end(&mut self);

    unsafe fn begin_render_pass(&mut self) -> A::RenderPass;
    unsafe fn end_render_pass(&mut self, pass: A::RenderPass);
    unsafe fn begin_compute_pass(&mut self) -> A::ComputePass;
    unsafe fn end_compute_pass(&mut self, pass: A::ComputePass);
}

pub trait RenderPass<A: Api> {}
pub trait ComputePass<A: Api> {}

bitflags!(
    /// Texture format capability flags.
    pub struct TextureFormatCapability: u32 {
        /// Format can be sampled.
        const SAMPLED = 0x1;
        /// Format can be sampled with a linear sampler.
        const SAMPLED_LINEAR = 0x2;
        /// Format can be sampled with a min/max reduction sampler.
        const SAMPLED_MINMAX = 0x4;

        /// Format can be used as storage with exclusive read & write access.
        const STORAGE = 0x10;
        /// Format can be used as storage with simultaneous read/write access.
        const STORAGE_READ_WRITE = 0x20;
        /// Format can be used as storage with atomics.
        const STORAGE_ATOMIC = 0x40;

        /// Format can be used as color and input attachment.
        const COLOR_ATTACHMENT = 0x100;
        /// Format can be used as color (with blending) and input attachment.
        const COLOR_ATTACHMENT_BLEND = 0x200;
        /// Format can be used as depth-stencil and input attachment.
        const DEPTH_STENCIL_ATTACHMENT = 0x400;

        /// Format can be copied from.
        const COPY_SRC = 0x1000;
        /// Format can be copied to.
        const COPY_DST = 0x2000;
    }
);

bitflags!(
    /// Texture format capability flags.
    pub struct FormatAspect: u8 {
        const COLOR = 1;
        const DEPTH = 2;
        const STENCIL = 4;
    }
);

impl From<wgt::TextureFormat> for FormatAspect {
    fn from(format: wgt::TextureFormat) -> Self {
        match format {
            wgt::TextureFormat::Depth32Float | wgt::TextureFormat::Depth24Plus => Self::DEPTH,
            wgt::TextureFormat::Depth24PlusStencil8 => Self::DEPTH | Self::STENCIL,
            _ => Self::COLOR,
        }
    }
}

bitflags::bitflags! {
    /// Similar to `wgt::BufferUsage` but for internal use.
    pub struct BufferUse: u32 {
        const MAP_READ = 1;
        const MAP_WRITE = 2;
        const COPY_SRC = 4;
        const COPY_DST = 8;
        const INDEX = 16;
        const VERTEX = 32;
        const UNIFORM = 64;
        const STORAGE_LOAD = 128;
        const STORAGE_STORE = 256;
        const INDIRECT = 512;
        /// The combination of all read-only usages.
        const READ_ALL = Self::MAP_READ.bits | Self::COPY_SRC.bits |
            Self::INDEX.bits | Self::VERTEX.bits | Self::UNIFORM.bits |
            Self::STORAGE_LOAD.bits | Self::INDIRECT.bits;
        /// The combination of all write-only and read-write usages.
        const WRITE_ALL = Self::MAP_WRITE.bits | Self::COPY_DST.bits | Self::STORAGE_STORE.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is not ordered, then even if it doesn't change between draw calls, there
        /// still need to be pipeline barriers inserted for synchronization.
        const ORDERED = Self::READ_ALL.bits | Self::MAP_WRITE.bits | Self::COPY_DST.bits;
    }
}

bitflags::bitflags! {
    /// Similar to `wgt::TextureUsage` but for internal use.
    pub struct TextureUse: u32 {
        const COPY_SRC = 1;
        const COPY_DST = 2;
        const SAMPLED = 4;
        const COLOR_TARGET = 8;
        const DEPTH_STENCIL_READ = 16;
        const DEPTH_STENCIL_WRITE = 32;
        const STORAGE_LOAD = 64;
        const STORAGE_STORE = 128;
        /// The combination of all read-only usages.
        const READ_ALL = Self::COPY_SRC.bits | Self::SAMPLED.bits | Self::DEPTH_STENCIL_READ.bits | Self::STORAGE_LOAD.bits;
        /// The combination of all write-only and read-write usages.
        const WRITE_ALL = Self::COPY_DST.bits | Self::COLOR_TARGET.bits | Self::DEPTH_STENCIL_WRITE.bits | Self::STORAGE_STORE.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is not ordered, then even if it doesn't change between draw calls, there
        /// still need to be pipeline barriers inserted for synchronization.
        const ORDERED = Self::READ_ALL.bits | Self::COPY_DST.bits | Self::COLOR_TARGET.bits | Self::DEPTH_STENCIL_WRITE.bits;
        const UNINITIALIZED = 0xFFFF;
    }
}

#[derive(Debug)]
pub struct Alignments {
    /// The alignment of the start of the buffer used as a GPU copy source.
    pub buffer_copy_offset: wgt::BufferSize,
    /// The alignment of the row pitch of the texture data stored in a buffer that is
    /// used in a GPU copy operation.
    pub buffer_copy_pitch: wgt::BufferSize,
    pub storage_buffer_offset: wgt::BufferSize,
    pub uniform_buffer_offset: wgt::BufferSize,
}

#[derive(Debug)]
pub struct Capabilities {
    pub limits: wgt::Limits,
    pub alignments: Alignments,
    pub downlevel: wgt::DownlevelCapabilities,
}

#[derive(Debug)]
pub struct ExposedAdapter<A: Api> {
    pub adapter: A::Adapter,
    pub info: wgt::AdapterInfo,
    pub features: wgt::Features,
    pub capabilities: Capabilities,
}

#[derive(Debug)]
pub struct OpenDevice<A: Api> {
    pub device: A::Device,
    pub queue: A::Queue,
}

#[derive(Clone, Debug)]
pub struct TextureViewDescriptor<L> {
    pub label: L,
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub range: wgt::ImageSubresourceRange,
}

impl<L> TextureViewDescriptor<L> {
    ///
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> TextureViewDescriptor<K> {
        TextureViewDescriptor {
            label: fun(&self.label),
            format: self.format,
            dimension: self.dimension,
            range: self.range.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SamplerDescriptor<'a> {
    pub label: Label<'a>,
    pub address_modes: [wgt::AddressMode; 3],
    pub mag_filter: wgt::FilterMode,
    pub min_filter: wgt::FilterMode,
    pub mipmap_filter: wgt::FilterMode,
    pub lod_clamp: Option<Range<f32>>,
    pub compare: Option<wgt::CompareFunction>,
    pub anisotropy_clamp: Option<NonZeroU8>,
    pub border_color: Option<wgt::SamplerBorderColor>,
}

#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    pub label: Label<'a>,
    pub entries: Cow<'a, [wgt::BindGroupLayoutEntry]>,
}

#[derive(Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub bind_group_layouts: Cow<'a, [&'a A::BindGroupLayout]>,
    pub push_constant_ranges: Cow<'a, [wgt::PushConstantRange]>,
}

#[derive(Debug)]
pub struct BufferBinding<'a, A: Api> {
    pub buffer: &'a A::Buffer,
    pub offset: wgt::BufferAddress,
    pub size: Option<wgt::BufferSize>,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for BufferBinding<'_, A> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
        }
    }
}

#[derive(Debug)]
pub enum BindingResource<'a, A: Api> {
    Buffers(SmallVec<[BufferBinding<'a, A>; 1]>),
    Sampler(&'a A::Sampler),
    TextureViews(SmallVec<[&'a A::TextureView; 1]>, TextureUse),
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for BindingResource<'_, A> {
    fn clone(&self) -> Self {
        match *self {
            Self::Buffers(ref slice) => Self::Buffers(slice.clone()),
            Self::Sampler(sampler) => Self::Sampler(sampler),
            Self::TextureViews(ref slice, usage) => Self::TextureViews(slice.clone(), usage),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindGroupEntry<'a, A: Api> {
    pub binding: u32,
    pub resource: BindingResource<'a, A>,
}

#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub layout: &'a A::BindGroupLayout,
    pub entries: Cow<'a, [BindGroupEntry<'a, A>]>,
}

/// Naga shader module.
pub struct NagaShader {
    /// Shader module IR.
    pub module: naga::Module,
    /// Analysis information of the module.
    pub info: naga::valid::ModuleInfo,
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
}

/// Describes a programmable pipeline stage.
#[derive(Debug)]
pub struct ProgrammableStage<'a, A: Api> {
    /// The compiled shader module for this stage.
    pub module: &'a A::ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: Cow<'a, str>,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for ProgrammableStage<'_, A> {
    fn clone(&self) -> Self {
        Self {
            module: self.module,
            entry_point: self.entry_point.clone(),
        }
    }
}

/// Describes a compute pipeline.
#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: &'a A::PipelineLayout,
    /// The compiled compute stage and its entry point.
    pub stage: ProgrammableStage<'a, A>,
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: wgt::BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: wgt::InputStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: Cow<'a, [wgt::VertexAttribute]>,
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: &'a A::PipelineLayout,
    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers: Cow<'a, [VertexBufferLayout<'a>]>,
    /// The vertex stage for this pipeline.
    pub vertex_stage: ProgrammableStage<'a, A>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: wgt::MultisampleState,
    /// The fragment stage for this pipeline.
    pub fragment_stage: ProgrammableStage<'a, A>,
    /// The effect of draw calls on the color aspect of the output target.
    pub color_targets: Cow<'a, [wgt::ColorTargetState]>,
}

#[test]
fn test_default_limits() {
    let limits = wgt::Limits::default();
    assert!(limits.max_bind_groups <= MAX_BIND_GROUPS as u32);
}
