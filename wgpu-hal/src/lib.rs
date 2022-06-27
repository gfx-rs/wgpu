/*! This library describes the internal unsafe graphics abstraction API.
 *  It follows WebGPU for the most part, re-using wgpu-types,
 *  with the following deviations:
 *  - Fully unsafe: zero overhead, zero validation.
 *  - Compile-time backend selection via traits.
 *  - Objects are passed by references and returned by value. No IDs.
 *  - Mapping is persistent, with explicit synchronization.
 *  - Resource transitions are explicit.
 *  - All layouts are explicit. Binding model has compatibility.
 *
 *  General design direction is to follow the majority by the following weights:
 *  - wgpu-core: 1.5
 *  - primary backends (Vulkan/Metal/DX12): 1.0 each
 *  - secondary backends (DX11/GLES): 0.5 each
 */

#![allow(
    // for `if_then_panic` until it reaches stable
    unknown_lints,
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
    // Matches are good and extendable, no need to make an exception here.
    clippy::single_match,
    // Push commands are more regular than macros.
    clippy::vec_init_then_push,
    // "if panic" is a good uniform construct.
    clippy::if_then_panic,
    // We unsafe impl `Send` for a reason.
    clippy::non_send_fields_in_send_ty,
    // TODO!
    clippy::missing_safety_doc,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

#[cfg(all(feature = "metal", not(any(target_os = "macos", target_os = "ios"))))]
compile_error!("Metal API enabled on non-Apple OS. If your project is not using resolver=\"2\" in Cargo.toml, it should.");
#[cfg(all(feature = "dx12", not(windows)))]
compile_error!("DX12 API enabled on non-Windows OS. If your project is not using resolver=\"2\" in Cargo.toml, it should.");

#[cfg(all(feature = "dx11", windows))]
mod dx11;
#[cfg(all(feature = "dx12", windows))]
mod dx12;
mod empty;
#[cfg(all(feature = "gles"))]
mod gles;
#[cfg(all(feature = "metal"))]
mod metal;
#[cfg(feature = "vulkan")]
mod vulkan;

pub mod auxil;
pub mod api {
    #[cfg(feature = "dx11")]
    pub use super::dx11::Api as Dx11;
    #[cfg(feature = "dx12")]
    pub use super::dx12::Api as Dx12;
    pub use super::empty::Api as Empty;
    #[cfg(feature = "gles")]
    pub use super::gles::Api as Gles;
    #[cfg(feature = "metal")]
    pub use super::metal::Api as Metal;
    #[cfg(feature = "vulkan")]
    pub use super::vulkan::Api as Vulkan;
}

#[cfg(feature = "vulkan")]
pub use vulkan::UpdateAfterBindTypes;

use std::{
    borrow::Borrow,
    fmt,
    num::{NonZeroU32, NonZeroU8},
    ops::{Range, RangeInclusive},
    ptr::NonNull,
    sync::atomic::AtomicBool,
};

use bitflags::bitflags;
use thiserror::Error;

pub const MAX_ANISOTROPY: u8 = 16;
pub const MAX_BIND_GROUPS: usize = 8;
pub const MAX_VERTEX_BUFFERS: usize = 16;
pub const MAX_COLOR_ATTACHMENTS: usize = 8;
pub const MAX_MIP_LEVELS: u32 = 16;
/// Size of a single occlusion/timestamp query, when copied into a buffer, in bytes.
pub const QUERY_SIZE: wgt::BufferAddress = 8;

pub type Label<'a> = Option<&'a str>;
pub type MemoryRange = Range<wgt::BufferAddress>;
pub type FenceValue = u64;

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
    Linkage(wgt::ShaderStages, String),
    #[error("entry point for stage {0:?} is invalid")]
    EntryPoint(naga::ShaderStage),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

#[derive(Clone, Debug, PartialEq, Error)]
pub enum SurfaceError {
    #[error("surface is lost")]
    Lost,
    #[error("surface is outdated, needs to be re-created")]
    Outdated,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("other reason: {0}")]
    Other(&'static str),
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("Not supported")]
pub struct InstanceError;

pub trait Api: Clone + Sized {
    type Instance: Instance<Self>;
    type Surface: Surface<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;

    type Queue: Queue<Self>;
    type CommandEncoder: CommandEncoder<Self>;
    type CommandBuffer: Send + Sync;

    type Buffer: fmt::Debug + Send + Sync + 'static;
    type Texture: fmt::Debug + Send + Sync + 'static;
    type SurfaceTexture: fmt::Debug + Send + Sync + Borrow<Self::Texture>;
    type TextureView: fmt::Debug + Send + Sync;
    type Sampler: fmt::Debug + Send + Sync;
    type QuerySet: fmt::Debug + Send + Sync;
    type Fence: fmt::Debug + Send + Sync;

    type BindGroupLayout: Send + Sync;
    type BindGroup: fmt::Debug + Send + Sync;
    type PipelineLayout: Send + Sync;
    type ShaderModule: fmt::Debug + Send + Sync;
    type RenderPipeline: Send + Sync;
    type ComputePipeline: Send + Sync;
}

pub trait Instance<A: Api>: Sized + Send + Sync {
    unsafe fn init(desc: &InstanceDescriptor) -> Result<Self, InstanceError>;
    unsafe fn create_surface(
        &self,
        rwh: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<A::Surface, InstanceError>;
    unsafe fn destroy_surface(&self, surface: A::Surface);
    unsafe fn enumerate_adapters(&self) -> Vec<ExposedAdapter<A>>;
}

pub trait Surface<A: Api>: Send + Sync {
    unsafe fn configure(
        &mut self,
        device: &A::Device,
        config: &SurfaceConfiguration,
    ) -> Result<(), SurfaceError>;

    unsafe fn unconfigure(&mut self, device: &A::Device);

    /// Returns the next texture to be presented by the swapchain for drawing
    ///
    /// A `timeout` of `None` means to wait indefinitely, with no timeout.
    ///
    /// # Portability
    ///
    /// Some backends can't support a timeout when acquiring a texture and
    /// the timeout will be ignored.
    ///
    /// Returns `None` on timing out.
    unsafe fn acquire_texture(
        &mut self,
        timeout: Option<std::time::Duration>,
    ) -> Result<Option<AcquiredSurfaceTexture<A>>, SurfaceError>;
    unsafe fn discard_texture(&mut self, texture: A::SurfaceTexture);
}

pub trait Adapter<A: Api>: Send + Sync {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
    ) -> Result<OpenDevice<A>, DeviceError>;

    /// Return the set of supported capabilities for a texture format.
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapabilities;

    /// Returns the capabilities of working with a specified surface.
    ///
    /// `None` means presentation is not supported for it.
    unsafe fn surface_capabilities(&self, surface: &A::Surface) -> Option<SurfaceCapabilities>;
}

pub trait Device<A: Api>: Send + Sync {
    /// Exit connection to this logical device.
    unsafe fn exit(self, queue: A::Queue);
    /// Creates a new buffer.
    ///
    /// The initial usage is `BufferUses::empty()`.
    unsafe fn create_buffer(&self, desc: &BufferDescriptor) -> Result<A::Buffer, DeviceError>;
    unsafe fn destroy_buffer(&self, buffer: A::Buffer);
    //TODO: clarify if zero-sized mapping is allowed
    unsafe fn map_buffer(
        &self,
        buffer: &A::Buffer,
        range: MemoryRange,
    ) -> Result<BufferMapping, DeviceError>;
    unsafe fn unmap_buffer(&self, buffer: &A::Buffer) -> Result<(), DeviceError>;
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &A::Buffer, ranges: I)
    where
        I: Iterator<Item = MemoryRange>;
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &A::Buffer, ranges: I)
    where
        I: Iterator<Item = MemoryRange>;

    /// Creates a new texture.
    ///
    /// The initial usage for all subresources is `TextureUses::UNINITIALIZED`.
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

    unsafe fn create_command_encoder(
        &self,
        desc: &CommandEncoderDescriptor<A>,
    ) -> Result<A::CommandEncoder, DeviceError>;
    unsafe fn destroy_command_encoder(&self, pool: A::CommandEncoder);

    /// Creates a bind group layout.
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
        shader: ShaderInput,
    ) -> Result<A::ShaderModule, ShaderError>;
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

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<Label>,
    ) -> Result<A::QuerySet, DeviceError>;
    unsafe fn destroy_query_set(&self, set: A::QuerySet);
    unsafe fn create_fence(&self) -> Result<A::Fence, DeviceError>;
    unsafe fn destroy_fence(&self, fence: A::Fence);
    unsafe fn get_fence_value(&self, fence: &A::Fence) -> Result<FenceValue, DeviceError>;
    /// Calling wait with a lower value than the current fence value will immediately return.
    unsafe fn wait(
        &self,
        fence: &A::Fence,
        value: FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, DeviceError>;

    unsafe fn start_capture(&self) -> bool;
    unsafe fn stop_capture(&self);
}

pub trait Queue<A: Api>: Send + Sync {
    /// Submits the command buffers for execution on GPU.
    ///
    /// Valid usage:
    /// - all of the command buffers were created from command pools
    ///   that are associated with this queue.
    /// - all of the command buffers had `CommadBuffer::finish()` called.
    unsafe fn submit(
        &mut self,
        command_buffers: &[&A::CommandBuffer],
        signal_fence: Option<(&mut A::Fence, FenceValue)>,
    ) -> Result<(), DeviceError>;
    unsafe fn present(
        &mut self,
        surface: &mut A::Surface,
        texture: A::SurfaceTexture,
    ) -> Result<(), SurfaceError>;
    unsafe fn get_timestamp_period(&self) -> f32;
}

/// Encoder for commands in command buffers.
/// Serves as a parent for all the encoded command buffers.
/// Works in bursts of action: one or more command buffers are recorded,
/// then submitted to a queue, and then it needs to be `reset_all()`.
pub trait CommandEncoder<A: Api>: Send + Sync {
    /// Begin encoding a new command buffer.
    unsafe fn begin_encoding(&mut self, label: Label) -> Result<(), DeviceError>;
    /// Discard currently recorded list, if any.
    unsafe fn discard_encoding(&mut self);
    unsafe fn end_encoding(&mut self) -> Result<A::CommandBuffer, DeviceError>;
    /// Reclaims all resources that are allocated for this encoder.
    /// Must get all of the produced command buffers back,
    /// and they must not be used by GPU at this moment.
    unsafe fn reset_all<I>(&mut self, command_buffers: I)
    where
        I: Iterator<Item = A::CommandBuffer>;

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = BufferBarrier<'a, A>>;

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = TextureBarrier<'a, A>>;

    // copy operations

    unsafe fn clear_buffer(&mut self, buffer: &A::Buffer, range: MemoryRange);

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &A::Buffer, dst: &A::Buffer, regions: T)
    where
        T: Iterator<Item = BufferCopy>;

    /// Copy from one texture to another.
    /// Works with a single array layer.
    /// Note: `dst` current usage has to be `TextureUses::COPY_DST`.
    /// Note: the copy extent is in physical size (rounded to the block size)
    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &A::Texture,
        src_usage: TextureUses,
        dst: &A::Texture,
        regions: T,
    ) where
        T: Iterator<Item = TextureCopy>;

    /// Copy from buffer to texture.
    /// Works with a single array layer.
    /// Note: `dst` current usage has to be `TextureUses::COPY_DST`.
    /// Note: the copy extent is in physical size (rounded to the block size)
    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &A::Buffer, dst: &A::Texture, regions: T)
    where
        T: Iterator<Item = BufferTextureCopy>;

    /// Copy from texture to buffer.
    /// Works with a single array layer.
    /// Note: the copy extent is in physical size (rounded to the block size)
    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &A::Texture,
        src_usage: TextureUses,
        dst: &A::Buffer,
        regions: T,
    ) where
        T: Iterator<Item = BufferTextureCopy>;

    // pass common

    /// Sets the bind group at `index` to `group`, assuming the layout
    /// of all the preceeding groups to be taken from `layout`.
    unsafe fn set_bind_group(
        &mut self,
        layout: &A::PipelineLayout,
        index: u32,
        group: &A::BindGroup,
        dynamic_offsets: &[wgt::DynamicOffset],
    );

    unsafe fn set_push_constants(
        &mut self,
        layout: &A::PipelineLayout,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u32],
    );

    unsafe fn insert_debug_marker(&mut self, label: &str);
    unsafe fn begin_debug_marker(&mut self, group_label: &str);
    unsafe fn end_debug_marker(&mut self);

    // queries

    unsafe fn begin_query(&mut self, set: &A::QuerySet, index: u32);
    unsafe fn end_query(&mut self, set: &A::QuerySet, index: u32);
    unsafe fn write_timestamp(&mut self, set: &A::QuerySet, index: u32);
    unsafe fn reset_queries(&mut self, set: &A::QuerySet, range: Range<u32>);
    unsafe fn copy_query_results(
        &mut self,
        set: &A::QuerySet,
        range: Range<u32>,
        buffer: &A::Buffer,
        offset: wgt::BufferAddress,
        stride: wgt::BufferSize,
    );

    // render passes

    // Begins a render pass, clears all active bindings.
    unsafe fn begin_render_pass(&mut self, desc: &RenderPassDescriptor<A>);
    unsafe fn end_render_pass(&mut self);

    unsafe fn set_render_pipeline(&mut self, pipeline: &A::RenderPipeline);

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, A>,
        format: wgt::IndexFormat,
    );
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: BufferBinding<'a, A>);
    unsafe fn set_viewport(&mut self, rect: &Rect<f32>, depth_range: Range<f32>);
    unsafe fn set_scissor_rect(&mut self, rect: &Rect<u32>);
    unsafe fn set_stencil_reference(&mut self, value: u32);
    unsafe fn set_blend_constants(&mut self, color: &[f32; 4]);

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
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &A::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &A::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    );
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &A::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &A::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    );

    // compute passes

    // Begins a compute pass, clears all active bindings.
    unsafe fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor);
    unsafe fn end_compute_pass(&mut self);

    unsafe fn set_compute_pipeline(&mut self, pipeline: &A::ComputePipeline);

    unsafe fn dispatch(&mut self, count: [u32; 3]);
    unsafe fn dispatch_indirect(&mut self, buffer: &A::Buffer, offset: wgt::BufferAddress);
}

bitflags!(
    /// Instance initialization flags.
    pub struct InstanceFlags: u32 {
        /// Generate debug information in shaders and objects.
        const DEBUG = 1 << 0;
        /// Enable validation, if possible.
        const VALIDATION = 1 << 1;
    }
);

bitflags!(
    /// Pipeline layout creation flags.
    pub struct PipelineLayoutFlags: u32 {
        /// Include support for base vertex/instance drawing.
        const BASE_VERTEX_INSTANCE = 1 << 0;
        /// Include support for num work groups builtin.
        const NUM_WORK_GROUPS = 1 << 1;
    }
);

bitflags!(
    /// Pipeline layout creation flags.
    pub struct BindGroupLayoutFlags: u32 {
        /// Allows for bind group binding arrays to be shorter than the array in the BGL.
        const PARTIALLY_BOUND = 1 << 0;
    }
);

bitflags!(
    /// Texture format capability flags.
    pub struct TextureFormatCapabilities: u32 {
        /// Format can be sampled.
        const SAMPLED = 1 << 0;
        /// Format can be sampled with a linear sampler.
        const SAMPLED_LINEAR = 1 << 1;
        /// Format can be sampled with a min/max reduction sampler.
        const SAMPLED_MINMAX = 1 << 2;

        /// Format can be used as storage with write-only access.
        const STORAGE = 1 << 3;
        /// Format can be used as storage with read and read/write access.
        const STORAGE_READ_WRITE = 1 << 4;
        /// Format can be used as storage with atomics.
        const STORAGE_ATOMIC = 1 << 5;

        /// Format can be used as color and input attachment.
        const COLOR_ATTACHMENT = 1 << 6;
        /// Format can be used as color (with blending) and input attachment.
        const COLOR_ATTACHMENT_BLEND = 1 << 7;
        /// Format can be used as depth-stencil and input attachment.
        const DEPTH_STENCIL_ATTACHMENT = 1 << 8;

        /// Format can be multisampled.
        const MULTISAMPLE = 1 << 9;
        /// Format can be used for render pass resolve targets.
        const MULTISAMPLE_RESOLVE = 1 << 10;

        /// Format can be copied from.
        const COPY_SRC = 1 << 11;
        /// Format can be copied to.
        const COPY_DST = 1 << 12;
    }
);

bitflags!(
    /// Texture format capability flags.
    pub struct FormatAspects: u8 {
        const COLOR = 1 << 0;
        const DEPTH = 1 << 1;
        const STENCIL = 1 << 2;
    }
);

impl From<wgt::TextureAspect> for FormatAspects {
    fn from(aspect: wgt::TextureAspect) -> Self {
        match aspect {
            wgt::TextureAspect::All => Self::all(),
            wgt::TextureAspect::DepthOnly => Self::DEPTH,
            wgt::TextureAspect::StencilOnly => Self::STENCIL,
        }
    }
}

impl From<wgt::TextureFormat> for FormatAspects {
    fn from(format: wgt::TextureFormat) -> Self {
        match format {
            wgt::TextureFormat::Depth32Float | wgt::TextureFormat::Depth24Plus => Self::DEPTH,
            wgt::TextureFormat::Depth32FloatStencil8
            | wgt::TextureFormat::Depth24PlusStencil8
            | wgt::TextureFormat::Depth24UnormStencil8 => Self::DEPTH | Self::STENCIL,
            _ => Self::COLOR,
        }
    }
}

bitflags!(
    pub struct MemoryFlags: u32 {
        const TRANSIENT = 1 << 0;
        const PREFER_COHERENT = 1 << 1;
    }
);

//TODO: it's not intuitive for the backends to consider `LOAD` being optional.

bitflags!(
    pub struct AttachmentOps: u8 {
        const LOAD = 1 << 0;
        const STORE = 1 << 1;
    }
);

bitflags::bitflags! {
    /// Similar to `wgt::BufferUsages` but for internal use.
    pub struct BufferUses: u16 {
        /// The argument to a read-only mapping.
        const MAP_READ = 1 << 0;
        /// The argument to a write-only mapping.
        const MAP_WRITE = 1 << 1;
        /// The source of a hardware copy.
        const COPY_SRC = 1 << 2;
        /// The destination of a hardware copy.
        const COPY_DST = 1 << 3;
        /// The index buffer used for drawing.
        const INDEX = 1 << 4;
        /// A vertex buffer used for drawing.
        const VERTEX = 1 << 5;
        /// A uniform buffer bound in a bind group.
        const UNIFORM = 1 << 6;
        /// A read-only storage buffer used in a bind group.
        const STORAGE_READ = 1 << 7;
        /// A read-write or write-only buffer used in a bind group.
        const STORAGE_READ_WRITE = 1 << 8;
        /// The indirect or count buffer in a indirect draw or dispatch.
        const INDIRECT = 1 << 9;
        /// The combination of states that a buffer may be in _at the same time_.
        const INCLUSIVE = Self::MAP_READ.bits | Self::COPY_SRC.bits |
            Self::INDEX.bits | Self::VERTEX.bits | Self::UNIFORM.bits |
            Self::STORAGE_READ.bits | Self::INDIRECT.bits;
        /// The combination of states that a buffer must exclusively be in.
        const EXCLUSIVE = Self::MAP_WRITE.bits | Self::COPY_DST.bits | Self::STORAGE_READ_WRITE.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is ordered, then if the buffer state doesn't change between draw calls, there
        /// are no barriers needed for synchronization.
        const ORDERED = Self::INCLUSIVE.bits | Self::MAP_WRITE.bits;
    }
}

bitflags::bitflags! {
    /// Similar to `wgt::TextureUsages` but for internal use.
    pub struct TextureUses: u16 {
        /// The texture is in unknown state.
        const UNINITIALIZED = 1 << 0;
        /// Ready to present image to the surface.
        const PRESENT = 1 << 1;
        /// The source of a hardware copy.
        const COPY_SRC = 1 << 2;
        /// The destination of a hardware copy.
        const COPY_DST = 1 << 3;
        /// Read-only sampled or fetched resource.
        const RESOURCE = 1 << 4;
        /// The color target of a renderpass.
        const COLOR_TARGET = 1 << 5;
        /// Read-only depth stencil usage.
        const DEPTH_STENCIL_READ = 1 << 6;
        /// Read-write depth stencil usage
        const DEPTH_STENCIL_WRITE = 1 << 7;
        /// Read-only storage buffer usage. Corresponds to a UAV in d3d, so is exclusive, despite being read only.
        const STORAGE_READ = 1 << 8;
        /// Read-write or write-only storage buffer usage.
        const STORAGE_READ_WRITE = 1 << 9;
        /// The combination of states that a texture may be in _at the same time_.
        const INCLUSIVE = Self::COPY_SRC.bits | Self::RESOURCE.bits | Self::DEPTH_STENCIL_READ.bits;
        /// The combination of states that a texture must exclusively be in.
        const EXCLUSIVE = Self::COPY_DST.bits | Self::COLOR_TARGET.bits | Self::DEPTH_STENCIL_WRITE.bits | Self::STORAGE_READ.bits | Self::STORAGE_READ_WRITE.bits | Self::PRESENT.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is ordered, then if the texture state doesn't change between draw calls, there
        /// are no barriers needed for synchronization.
        const ORDERED = Self::INCLUSIVE.bits | Self::COLOR_TARGET.bits | Self::DEPTH_STENCIL_WRITE.bits | Self::STORAGE_READ.bits;

        /// Flag used by the wgpu-core texture tracker to say a texture is in different states for every sub-resource
        const COMPLEX = 1 << 10;
        /// Flag used by the wgpu-core texture tracker to say that the tracker does not know the state of the sub-resource.
        /// This is different from UNINITIALIZED as that says the tracker does know, but the texture has not been initialized.
        const UNKNOWN = 1 << 11;
    }
}

#[derive(Clone, Debug)]
pub struct InstanceDescriptor<'a> {
    pub name: &'a str,
    pub flags: InstanceFlags,
}

#[derive(Clone, Debug)]
pub struct Alignments {
    /// The alignment of the start of the buffer used as a GPU copy source.
    pub buffer_copy_offset: wgt::BufferSize,
    /// The alignment of the row pitch of the texture data stored in a buffer that is
    /// used in a GPU copy operation.
    pub buffer_copy_pitch: wgt::BufferSize,
}

#[derive(Clone, Debug)]
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
    pub formats: Vec<wgt::TextureFormat>,

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
    /// Must have at least `TextureUses::COLOR_TARGET`
    pub usage: TextureUses,

    /// List of supported V-sync modes.
    ///
    /// Must be at least one.
    pub present_modes: Vec<wgt::PresentMode>,

    /// List of supported alpha composition modes.
    ///
    /// Must be at least one.
    pub composite_alpha_modes: Vec<CompositeAlphaMode>,
}

#[derive(Debug)]
pub struct AcquiredSurfaceTexture<A: Api> {
    pub texture: A::SurfaceTexture,
    /// The presentation configuration no longer matches
    /// the surface properties exactly, but can still be used to present
    /// to the surface successfully.
    pub suboptimal: bool,
}

#[derive(Debug)]
pub struct OpenDevice<A: Api> {
    pub device: A::Device,
    pub queue: A::Queue,
}

#[derive(Clone, Debug)]
pub struct BufferMapping {
    pub ptr: NonNull<u8>,
    pub is_coherent: bool,
}

#[derive(Clone, Debug)]
pub struct BufferDescriptor<'a> {
    pub label: Label<'a>,
    pub size: wgt::BufferAddress,
    pub usage: BufferUses,
    pub memory_flags: MemoryFlags,
}

#[derive(Clone, Debug)]
pub struct TextureDescriptor<'a> {
    pub label: Label<'a>,
    pub size: wgt::Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: wgt::TextureDimension,
    pub format: wgt::TextureFormat,
    pub usage: TextureUses,
    pub memory_flags: MemoryFlags,
}

/// TextureView descriptor.
///
/// Valid usage:
///. - `format` has to be the same as `TextureDescriptor::format`
///. - `dimension` has to be compatible with `TextureDescriptor::dimension`
///. - `usage` has to be a subset of `TextureDescriptor::usage`
///. - `range` has to be a subset of parent texture
#[derive(Clone, Debug)]
pub struct TextureViewDescriptor<'a> {
    pub label: Label<'a>,
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub usage: TextureUses,
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

/// BindGroupLayout descriptor.
///
/// Valid usage:
/// - `entries` are sorted by ascending `wgt::BindGroupLayoutEntry::binding`
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    pub label: Label<'a>,
    pub flags: BindGroupLayoutFlags,
    pub entries: &'a [wgt::BindGroupLayoutEntry],
}

#[derive(Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub flags: PipelineLayoutFlags,
    pub bind_group_layouts: &'a [&'a A::BindGroupLayout],
    pub push_constant_ranges: &'a [wgt::PushConstantRange],
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
pub struct TextureBinding<'a, A: Api> {
    pub view: &'a A::TextureView,
    pub usage: TextureUses,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for TextureBinding<'_, A> {
    fn clone(&self) -> Self {
        Self {
            view: self.view,
            usage: self.usage,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource_index: u32,
    pub count: u32,
}

/// BindGroup descriptor.
///
/// Valid usage:
///. - `entries` has to be sorted by ascending `BindGroupEntry::binding`
///. - `entries` has to have the same set of `BindGroupEntry::binding` as `layout`
///. - each entry has to be compatible with the `layout`
///. - each entry's `BindGroupEntry::resource_index` is within range
///    of the corresponding resource array, selected by the relevant
///    `BindGroupLayoutEntry`.
#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub layout: &'a A::BindGroupLayout,
    pub buffers: &'a [BufferBinding<'a, A>],
    pub samplers: &'a [&'a A::Sampler],
    pub textures: &'a [TextureBinding<'a, A>],
    pub entries: &'a [BindGroupEntry],
}

#[derive(Clone, Debug)]
pub struct CommandEncoderDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub queue: &'a A::Queue,
}

/// Naga shader module.
pub struct NagaShader {
    /// Shader module IR.
    pub module: naga::Module,
    /// Analysis information of the module.
    pub info: naga::valid::ModuleInfo,
}

// Custom implementation avoids the need to generate Debug impl code
// for the whole Naga module and info.
impl fmt::Debug for NagaShader {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Naga shader")
    }
}

/// Shader input.
#[allow(clippy::large_enum_variant)]
pub enum ShaderInput<'a> {
    Naga(NagaShader),
    SpirV(&'a [u32]),
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Label<'a>,
    pub runtime_checks: bool,
}

/// Describes a programmable pipeline stage.
#[derive(Debug)]
pub struct ProgrammableStage<'a, A: Api> {
    /// The compiled shader module for this stage.
    pub module: &'a A::ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for ProgrammableStage<'_, A> {
    fn clone(&self) -> Self {
        Self {
            module: self.module,
            entry_point: self.entry_point,
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
    pub step_mode: wgt::VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [wgt::VertexAttribute],
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: &'a A::PipelineLayout,
    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers: &'a [VertexBufferLayout<'a>],
    /// The vertex stage for this pipeline.
    pub vertex_stage: ProgrammableStage<'a, A>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: wgt::PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<wgt::DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: wgt::MultisampleState,
    /// The fragment stage for this pipeline.
    pub fragment_stage: Option<ProgrammableStage<'a, A>>,
    /// The effect of draw calls on the color aspect of the output target.
    pub color_targets: &'a [Option<wgt::ColorTargetState>],
    /// If the pipeline will be used with a multiview render pass, this indicates how many array
    /// layers the attachments will have.
    pub multiview: Option<NonZeroU32>,
}

/// Specifies how the alpha channel of the textures should be handled during (martin mouv i step)
/// compositing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositeAlphaMode {
    /// The alpha channel, if it exists, of the textures is ignored in the
    /// compositing process. Instead, the textures is treated as if it has a
    /// constant alpha of 1.0.
    Opaque,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are
    /// expected to already be multiplied by the alpha channel by the
    /// application.
    PreMultiplied,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are not
    /// expected to already be multiplied by the alpha channel by the
    /// application; instead, the compositor will multiply the non-alpha
    /// channels of the texture by the alpha channel during compositing.
    PostMultiplied,
}

#[derive(Debug, Clone)]
pub struct SurfaceConfiguration {
    /// Number of textures in the swap chain. Must be in
    /// `SurfaceCapabilities::swap_chain_size` range.
    pub swap_chain_size: u32,
    /// Vertical synchronization mode.
    pub present_mode: wgt::PresentMode,
    /// Alpha composition mode.
    pub composite_alpha_mode: CompositeAlphaMode,
    /// Format of the surface textures.
    pub format: wgt::TextureFormat,
    /// Requested texture extent. Must be in
    /// `SurfaceCapabilities::extents` range.
    pub extent: wgt::Extent3d,
    /// Allowed usage of surface textures,
    pub usage: TextureUses,
}

#[derive(Debug, Clone)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub w: T,
    pub h: T,
}

#[derive(Debug, Clone)]
pub struct BufferBarrier<'a, A: Api> {
    pub buffer: &'a A::Buffer,
    pub usage: Range<BufferUses>,
}

#[derive(Debug, Clone)]
pub struct TextureBarrier<'a, A: Api> {
    pub texture: &'a A::Texture,
    pub range: wgt::ImageSubresourceRange,
    pub usage: Range<TextureUses>,
}

#[derive(Clone, Copy, Debug)]
pub struct BufferCopy {
    pub src_offset: wgt::BufferAddress,
    pub dst_offset: wgt::BufferAddress,
    pub size: wgt::BufferSize,
}

#[derive(Clone, Debug)]
pub struct TextureCopyBase {
    pub mip_level: u32,
    pub array_layer: u32,
    /// Origin within a texture.
    /// Note: for 1D and 2D textures, Z must be 0.
    pub origin: wgt::Origin3d,
    pub aspect: FormatAspects,
}

#[derive(Clone, Copy, Debug)]
pub struct CopyExtent {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Clone, Debug)]
pub struct TextureCopy {
    pub src_base: TextureCopyBase,
    pub dst_base: TextureCopyBase,
    pub size: CopyExtent,
}

#[derive(Clone, Debug)]
pub struct BufferTextureCopy {
    pub buffer_layout: wgt::ImageDataLayout,
    pub texture_base: TextureCopyBase,
    pub size: CopyExtent,
}

#[derive(Debug)]
pub struct Attachment<'a, A: Api> {
    pub view: &'a A::TextureView,
    /// Contains either a single mutating usage as a target,
    /// or a valid combination of read-only usages.
    pub usage: TextureUses,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for Attachment<'_, A> {
    fn clone(&self) -> Self {
        Self {
            view: self.view,
            usage: self.usage,
        }
    }
}

#[derive(Debug)]
pub struct ColorAttachment<'a, A: Api> {
    pub target: Attachment<'a, A>,
    pub resolve_target: Option<Attachment<'a, A>>,
    pub ops: AttachmentOps,
    pub clear_value: wgt::Color,
}

// Rust gets confused about the impl requirements for `A`
impl<A: Api> Clone for ColorAttachment<'_, A> {
    fn clone(&self) -> Self {
        Self {
            target: self.target.clone(),
            resolve_target: self.resolve_target.clone(),
            ops: self.ops,
            clear_value: self.clear_value,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DepthStencilAttachment<'a, A: Api> {
    pub target: Attachment<'a, A>,
    pub depth_ops: AttachmentOps,
    pub stencil_ops: AttachmentOps,
    pub clear_value: (f32, u32),
}

#[derive(Clone, Debug)]
pub struct RenderPassDescriptor<'a, A: Api> {
    pub label: Label<'a>,
    pub extent: wgt::Extent3d,
    pub sample_count: u32,
    pub color_attachments: &'a [Option<ColorAttachment<'a, A>>],
    pub depth_stencil_attachment: Option<DepthStencilAttachment<'a, A>>,
    pub multiview: Option<NonZeroU32>,
}

#[derive(Clone, Debug)]
pub struct ComputePassDescriptor<'a> {
    pub label: Label<'a>,
}

/// Stores if any API validation error has occurred in this process
/// since it was last reset.
///
/// This is used for internal wgpu testing only and _must not_ be used
/// as a way to check for errors.
///
/// This works as a static because `cargo nextest` runs all of our
/// tests in separate processes, so each test gets its own canary.
///
/// This prevents the issue of one validation error terminating the
/// entire process.
pub static VALIDATION_CANARY: ValidationCanary = ValidationCanary {
    inner: AtomicBool::new(false),
};

/// Flag for internal testing.
pub struct ValidationCanary {
    inner: AtomicBool,
}

impl ValidationCanary {
    #[allow(dead_code)] // in some configurations this function is dead
    fn set(&self) {
        self.inner.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns true if any API validation error has occurred in this process
    /// since the last call to this function.
    pub fn get_and_reset(&self) -> bool {
        self.inner.swap(false, std::sync::atomic::Ordering::SeqCst)
    }
}

#[test]
fn test_default_limits() {
    let limits = wgt::Limits::default();
    assert!(limits.max_bind_groups <= MAX_BIND_GROUPS as u32);
}
