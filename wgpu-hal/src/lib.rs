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
 *  - All layouts are explicit. Binding model has compatibility.
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

use std::{
    borrow::{Borrow, Cow},
    fmt,
    num::NonZeroU8,
    ops::{Range, RangeInclusive},
    ptr::NonNull,
};

use bitflags::bitflags;
use smallvec::SmallVec;
use thiserror::Error;

pub const MAX_ANISOTROPY: u8 = 16;
pub const MAX_BIND_GROUPS: usize = 8;

pub type Label<'a> = Option<&'a str>;
pub type MemoryRange = Range<wgt::BufferAddress>;
pub type MipLevel = u8;
pub type ArrayLayer = u16;

#[derive(Clone, Debug, PartialEq, Error)]
pub enum DeviceError {
    #[error("out of memory")]
    OutOfMemory,
    #[error("device is lost")]
    Lost,
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum ShaderError {
    #[error("compilation failed: {0:?}")]
    Compilation(String),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum PipelineError {
    #[error("linkage failed for stage {0:?}: {1}")]
    Linkage(wgt::ShaderStage, String),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum SurfaceError {
    #[error("surface is lost")]
    Lost,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("other reason: {0}")]
    Other(&'static str),
}

/// Marker value returned if the presentation configuration no longer matches
/// the surface properties exactly, but can still be used to present
/// to the surface successfully.
#[derive(Debug)]
pub struct Suboptimal;

pub trait Api: Clone + Sized {
    type Instance: Instance<Self>;
    type Surface: Surface<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;
    type Queue: Queue<Self>;

    type CommandBuffer: CommandBuffer<Self>;
    type RenderPass: RenderPass<Self>;
    type ComputePass: ComputePass<Self>;

    type Buffer: fmt::Debug + Send + Sync + 'static;
    type QuerySet: fmt::Debug + Send + Sync;
    type Texture: fmt::Debug + Send + Sync + 'static;
    type SurfaceTexture: fmt::Debug + Send + Sync + Borrow<Self::Texture>;
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

pub trait Surface<A: Api> {
    unsafe fn configure(
        &mut self,
        device: &A::Device,
        config: &SurfaceConfiguration,
    ) -> Result<(), SurfaceError>;

    unsafe fn unconfigure(&mut self, device: &A::Device);

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<(A::SurfaceTexture, Option<Suboptimal>), SurfaceError>;
}

pub trait Adapter<A: Api> {
    unsafe fn open(&self, features: wgt::Features) -> Result<OpenDevice<A>, DeviceError>;
    unsafe fn close(&self, device: A::Device);

    /// Return the set of supported capabilities for a texture format.
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapability;

    /// Returns the capabilities of working with a specified surface.
    ///
    /// `None` means presentation is not supported for it.
    unsafe fn surface_capabilities(&self, surface: &A::Surface) -> Option<SurfaceCapabilities>;
}

pub trait Device<A: Api> {
    /// Creates a new buffer.
    ///
    /// The initial usage is `BufferUse::empty()`.
    unsafe fn create_buffer(&self, desc: &BufferDescriptor) -> Result<A::Buffer, DeviceError>;
    unsafe fn destroy_buffer(&self, buffer: A::Buffer);
    unsafe fn map_buffer(
        &self,
        buffer: &A::Buffer,
        range: MemoryRange,
    ) -> Result<NonNull<u8>, DeviceError>;
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

    /// Creates a new texture.
    ///
    /// The initial usage for all subresources is `TextureUse::UNINITIALIZED`.
    unsafe fn create_texture(&self, desc: &TextureDescriptor) -> Result<A::Texture, DeviceError>;
    unsafe fn destroy_texture(&self, texture: A::Texture);
    unsafe fn create_texture_view(
        &self,
        texture: &A::Texture,
        desc: &TextureViewDescriptor,
    ) -> Result<A::TextureView, DeviceError>;
    unsafe fn destroy_texture_view(&self, view: A::TextureView);
    unsafe fn create_sampler(&self, desc: &SamplerDescriptor) -> Result<A::Sampler, DeviceError>;
    unsafe fn destroy_sampler(&self, sampler: A::Sampler);

    //TODO: consider making DX12-style command pools
    unsafe fn create_command_buffer(
        &self,
        desc: &CommandBufferDescriptor,
    ) -> Result<A::CommandBuffer, DeviceError>;
    unsafe fn destroy_command_buffer(&self, cmd_buf: A::CommandBuffer);

    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<A::BindGroupLayout, DeviceError>;
    unsafe fn destroy_bind_group_layout(&self, bg_layout: A::BindGroupLayout);
    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor<A>,
    ) -> Result<A::PipelineLayout, DeviceError>;
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: A::PipelineLayout);
    unsafe fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<A>,
    ) -> Result<A::BindGroup, DeviceError>;
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

pub trait SwapChain<A: Api> {}

pub trait CommandBuffer<A: Api> {
    unsafe fn begin(&mut self);
    unsafe fn end(&mut self);

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = BufferBarrier<'a, A>>;

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = TextureBarrier<'a, A>>;

    unsafe fn fill_buffer(&mut self, buffer: &A::Buffer, range: MemoryRange, value: u8);

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &A::Buffer, dst: &A::Buffer, regions: T)
    where
        T: Iterator<Item = BufferCopy>;

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &A::Texture,
        src_usage: TextureUse,
        dst: &A::Texture,
        regions: T,
    ) where
        T: Iterator<Item = TextureCopy>;

    /// Note: `dst` current usage has to be `TextureUse::COPY_DST`.
    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &A::Buffer, dst: &A::Texture, regions: T)
    where
        T: Iterator<Item = BufferTextureCopy>;

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &A::Texture,
        src_usage: TextureUse,
        dst: &A::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = BufferTextureCopy>;

    unsafe fn begin_render_pass(&mut self) -> A::RenderPass;
    unsafe fn end_render_pass(&mut self, pass: A::RenderPass);
    unsafe fn begin_compute_pass(&mut self) -> A::ComputePass;
    unsafe fn end_compute_pass(&mut self, pass: A::ComputePass);
}

pub trait RenderPass<A: Api> {
    unsafe fn set_pipeline(&mut self, pipeline: &A::RenderPipeline);

    /// Sets the bind group at `index` to `group`, assuming the layout
    /// of all the preceeding groups to be taken from `layout`.
    unsafe fn set_bind_group(
        &mut self,
        layout: &A::PipelineLayout,
        index: u32,
        group: &A::BindGroup,
    );

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, A>,
        format: wgt::IndexFormat,
    );
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: BufferBinding<'a, A>);
    unsafe fn set_viewport(&mut self, rect: &Rect, depth_range: Range<f32>);
    unsafe fn set_scissor_rect(&mut self, rect: &Rect);
    unsafe fn set_stencil_reference(&mut self, value: u32);
    unsafe fn set_blend_constants(&mut self, color: wgt::Color);

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    );
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    );
    unsafe fn draw_indirect(
        &mut self,
        buffer: &A::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    );
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &A::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    );
}

pub trait ComputePass<A: Api> {
    unsafe fn set_pipeline(&mut self, pipeline: &A::ComputePipeline);

    /// Sets the bind group at `index` to `group`, assuming the layout
    /// of all the preceeding groups to be taken from `layout`.
    unsafe fn set_bind_group(
        &mut self,
        layout: &A::PipelineLayout,
        index: u32,
        group: &A::BindGroup,
    );

    unsafe fn dispatch(&mut self, count: [u32; 3]);
    unsafe fn dispatch_indirect(&mut self, buffer: &A::Buffer, offset: wgt::BufferAddress);
}

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

impl From<wgt::TextureAspect> for FormatAspect {
    fn from(aspect: wgt::TextureAspect) -> Self {
        match aspect {
            wgt::TextureAspect::All => Self::all(),
            wgt::TextureAspect::DepthOnly => Self::DEPTH,
            wgt::TextureAspect::StencilOnly => Self::STENCIL,
        }
    }
}

impl From<wgt::TextureFormat> for FormatAspect {
    fn from(format: wgt::TextureFormat) -> Self {
        match format {
            wgt::TextureFormat::Depth32Float | wgt::TextureFormat::Depth24Plus => Self::DEPTH,
            wgt::TextureFormat::Depth24PlusStencil8 => Self::DEPTH | Self::STENCIL,
            _ => Self::COLOR,
        }
    }
}

bitflags!(
    pub struct MemoryFlag: u32 {
        const TRANSIENT = 1;
    }
);

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

/// Describes information about what a `Surface`'s presentation capabilities are.
/// Fetch this with [Adapter::surface_capabilities].
#[derive(Debug, Clone)]
pub struct SurfaceCapabilities {
    /// List of supported texture formats.
    ///
    /// Must be at least one.
    pub texture_formats: Vec<wgt::TextureFormat>,

    /// Range for the swap chain sizes.
    ///
    /// - `swap_chain_sizes.start` must be at least 1.
    /// - `swap_chain_sizes.end` must be larger or equal to `swap_chain_sizes.start`.
    pub swap_chain_sizes: RangeInclusive<u32>,

    /// Current extent of the surface, if known.
    pub current_extent: Option<wgt::Extent3d>,

    /// Range of supported extents.
    ///
    /// `current_extent` must be inside this range.
    pub extents: RangeInclusive<wgt::Extent3d>,

    /// Supported texture usage flags.
    ///
    /// Must have at least `TextureUse::COLOR_TARGET`
    pub texture_uses: TextureUse,

    /// List of supported V-sync modes.
    ///
    /// Must be at least one.
    pub vsync_modes: Vec<VsyncMode>,

    /// List of supported alpha composition modes.
    ///
    /// Must be at least one.
    pub composite_alpha_modes: Vec<CompositeAlphaMode>,
}

#[derive(Debug)]
pub struct OpenDevice<A: Api> {
    pub device: A::Device,
    pub queue: A::Queue,
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor<'a> {
    pub label: Label<'a>,
    pub size: wgt::BufferAddress,
    pub usage: BufferUse,
    pub memory_flags: MemoryFlag,
}

#[derive(Clone, Debug)]
pub struct TextureDescriptor<'a> {
    pub label: Label<'a>,
    pub size: wgt::Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: wgt::TextureDimension,
    pub format: wgt::TextureFormat,
    pub usage: TextureUse,
    pub memory_flags: MemoryFlag,
}

#[derive(Clone, Debug)]
pub struct TextureViewDescriptor<'a> {
    pub label: Label<'a>,
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub range: wgt::ImageSubresourceRange,
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

#[derive(Clone, Debug)]
pub struct CommandBufferDescriptor<'a> {
    pub label: Label<'a>,
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

/// Specifies the mode regulating how a surface presents frames.
#[derive(Debug, Clone)]
pub enum VsyncMode {
    /// Don't ever wait for v-sync.
    Immediate,
    /// Wait for v-sync, overwrite the last rendered frame.
    Mailbox,
    /// Present frames in the same order they are rendered.
    Fifo,
    /// Don't wait for the next v-sync if we just missed it.
    Relaxed,
}

/// Specifies how the alpha channel of the textures should be handled during (martin mouv i step)
/// compositing.
#[derive(Debug, Clone)]
pub enum CompositeAlphaMode {
    /// The alpha channel, if it exists, of the textures is ignored in the
    /// compositing process. Instead, the textures is treated as if it has a
    /// constant alpha of 1.0.
    Opaque,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are not
    /// expected to already be multiplied by the alpha channel by the
    /// application; instead, the compositor will multiply the non-alpha
    /// channels of the texture by the alpha channel during compositing.
    Alpha,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are
    /// expected to already be multiplied by the alpha channel by the
    /// application.
    PremultipliedAlpha,
}

#[derive(Debug, Clone)]
pub struct SurfaceConfiguration {
    /// Number of textures in the swap chain. Must be in
    /// `SurfaceCapabilities::swap_chain_size` range.
    pub swap_chain_size: u32,
    /// Vertical synchronization mode.
    pub vsync_mode: VsyncMode,
    /// Alpha composition mode.
    pub composite_alpha_mode: CompositeAlphaMode,
    /// Format of the surface textures.
    pub format: wgt::TextureFormat,
    /// Requested texture extent. Must be in
    /// `SurfaceCapabilities::extents` range.
    pub extent: wgt::Extent3d,
    /// Allowed usage of surface textures,
    pub usage: TextureUse,
}

#[derive(Debug, Clone)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone)]
pub struct BufferBarrier<'a, A: Api> {
    pub buffer: &'a A::Buffer,
    pub usage: Range<BufferUse>,
}

#[derive(Debug, Clone)]
pub struct TextureBarrier<'a, A: Api> {
    pub texture: &'a A::Texture,
    pub subresource: wgt::ImageSubresourceRange,
    pub usage: Range<TextureUse>,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferCopy {
    pub src_offset: wgt::BufferAddress,
    pub dst_offset: wgt::BufferAddress,
    pub size: wgt::BufferSize,
}

#[derive(Clone, Debug)]
pub struct TextureCopy {
    pub src_subresource: wgt::ImageSubresourceRange,
    pub src_origin: wgt::Origin3d,
    pub dst_subresource: wgt::ImageSubresourceRange,
    pub dst_origin: wgt::Origin3d,
    pub size: wgt::Extent3d,
}

#[derive(Clone, Debug)]
pub struct BufferTextureCopy {
    pub buffer_layout: wgt::ImageDataLayout,
    pub texture_mip_level: u32,
    pub texture_origin: wgt::Origin3d,
    pub size: wgt::Extent3d,
}

#[test]
fn test_default_limits() {
    let limits = wgt::Limits::default();
    assert!(limits.max_bind_groups <= MAX_BIND_GROUPS as u32);
}
