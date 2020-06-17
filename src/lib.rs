//! A cross-platform graphics and compute library based on WebGPU.

#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu-rs/master/logo.png")]

mod backend;
pub mod util;
#[macro_use]
mod macros;

use std::{
    future::Future,
    marker::PhantomData,
    ops::{Bound, Range, RangeBounds},
    sync::Arc,
    thread,
};

use futures::FutureExt as _;
use parking_lot::Mutex;

#[cfg(not(target_arch = "wasm32"))]
pub use wgc::instance::{AdapterInfo, DeviceType};
pub use wgt::{
    AddressMode, Backend, BackendBit, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BlendDescriptor, BlendFactor, BlendOperation, BufferAddress, BufferSize, BufferUsage,
    Capabilities, Color, ColorStateDescriptor, ColorWrite, CommandBufferDescriptor,
    CompareFunction, CullMode, DepthStencilStateDescriptor, DeviceDescriptor, DynamicOffset,
    Extensions, Extent3d, FilterMode, FrontFace, IndexFormat, InputStepMode, Limits, LoadOp,
    NonZeroBufferAddress, Origin3d, PowerPreference, PresentMode, PrimitiveTopology,
    RasterizationStateDescriptor, RenderBundleEncoderDescriptor, ShaderLocation, ShaderStage,
    StencilOperation, StencilStateFaceDescriptor, StoreOp, SwapChainDescriptor, SwapChainStatus,
    TextureAspect, TextureComponentType, TextureDataLayout, TextureDimension, TextureFormat,
    TextureUsage, TextureViewDimension, UnsafeExtensions, VertexAttributeDescriptor, VertexFormat,
    BIND_BUFFER_ALIGNMENT, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
};

use backend::Context as C;

trait ComputePassInner<Ctx: Context> {
    fn set_pipeline(&mut self, pipeline: &Ctx::ComputePipelineId);
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &Ctx::BindGroupId,
        offsets: &[DynamicOffset],
    );
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn dispatch_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
    );
}

trait RenderInner<Ctx: Context> {
    fn set_pipeline(&mut self, pipeline: &Ctx::RenderPipelineId);
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &Ctx::BindGroupId,
        offsets: &[DynamicOffset],
    );
    fn set_index_buffer(&mut self, buffer: &Ctx::BufferId, offset: BufferAddress, size: BufferSize);
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Ctx::BufferId,
        offset: BufferAddress,
        size: BufferSize,
    );
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(&mut self, indirect_buffer: &Ctx::BufferId, indirect_offset: BufferAddress);
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
    );
}

trait RenderPassInner<Ctx: Context>: RenderInner<Ctx> {
    fn set_blend_color(&mut self, color: Color);
    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32);
    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    );
    fn set_stencil_reference(&mut self, reference: u32);
    fn insert_debug_marker(&mut self, label: &str);
    fn push_debug_group(&mut self, group_label: &str);
    fn pop_debug_group(&mut self);
    fn execute_bundles<'a, I: Iterator<Item = &'a Ctx::RenderBundleId>>(
        &mut self,
        render_bundles: I,
    );
}

trait Context: Sized {
    type AdapterId: Send + Sync + 'static;
    type DeviceId: Send + Sync + 'static;
    type QueueId: Send + Sync + 'static;
    type ShaderModuleId: Send + Sync + 'static;
    type BindGroupLayoutId: Send + Sync + 'static;
    type BindGroupId: Send + Sync + 'static;
    type TextureViewId: Send + Sync + 'static;
    type SamplerId: Send + Sync + 'static;
    type BufferId: Send + Sync + 'static;
    type TextureId: Send + Sync + 'static;
    type PipelineLayoutId: Send + Sync + 'static;
    type RenderPipelineId: Send + Sync + 'static;
    type ComputePipelineId: Send + Sync + 'static;
    type CommandEncoderId;
    type ComputePassId: ComputePassInner<Self>;
    type RenderPassId: RenderPassInner<Self>;
    type CommandBufferId: Send + Sync;
    type RenderBundleEncoderId: RenderInner<Self>;
    type RenderBundleId: Send + Sync + 'static;
    type SurfaceId: Send + Sync + 'static;
    type SwapChainId: Send + Sync + 'static;

    type SwapChainOutputDetail: Send;

    type RequestAdapterFuture: Future<Output = Option<Self::AdapterId>> + Send;
    type RequestDeviceFuture: Future<Output = Result<(Self::DeviceId, Self::QueueId), RequestDeviceError>>
        + Send;
    type MapAsyncFuture: Future<Output = Result<(), BufferAsyncError>> + Send;

    fn init(backends: BackendBit) -> Self;
    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId;
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
        unsafe_extensions: UnsafeExtensions,
    ) -> Self::RequestAdapterFuture;
    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture;
    fn adapter_extensions(&self, adapter: &Self::AdapterId) -> Extensions;
    fn adapter_limits(&self, adapter: &Self::AdapterId) -> Limits;
    fn adapter_capabilities(&self, adapter: &Self::AdapterId) -> Capabilities;

    fn device_extensions(&self, device: &Self::DeviceId) -> Extensions;
    fn device_limits(&self, device: &Self::DeviceId) -> Limits;
    fn device_capabilities(&self, device: &Self::DeviceId) -> Capabilities;
    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &SwapChainDescriptor,
    ) -> Self::SwapChainId;
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        source: ShaderModuleSource,
    ) -> Self::ShaderModuleId;
    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId;
    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId;
    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId;
    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId;
    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId;
    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> Self::BufferId;
    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        desc: &TextureDescriptor,
    ) -> Self::TextureId;
    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        desc: &SamplerDescriptor,
    ) -> Self::SamplerId;
    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &CommandEncoderDescriptor,
    ) -> Self::CommandEncoderId;
    fn device_create_render_bundle_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &RenderBundleEncoderDescriptor,
    ) -> Self::RenderBundleEncoderId;
    fn device_drop(&self, device: &Self::DeviceId);
    fn device_poll(&self, device: &Self::DeviceId, maintain: Maintain);

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        mode: MapMode,
        range: Range<BufferAddress>,
    ) -> Self::MapAsyncFuture;
    //TODO: we might be able to merge these, depending on how Web backend
    // turns out to be implemented.
    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<BufferAddress>,
    ) -> &[u8];
    fn buffer_get_mapped_range_mut(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<BufferAddress>,
    ) -> &mut [u8];
    fn buffer_unmap(&self, buffer: &Self::BufferId);
    fn swap_chain_get_next_texture(
        &self,
        swap_chain: &Self::SwapChainId,
    ) -> (
        Option<Self::TextureViewId>,
        SwapChainStatus,
        Self::SwapChainOutputDetail,
    );
    fn swap_chain_present(&self, view: &Self::TextureViewId, detail: &Self::SwapChainOutputDetail);
    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        desc: Option<&TextureViewDescriptor>,
    ) -> Self::TextureViewId;
    fn texture_drop(&self, texture: &Self::TextureId);
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId);
    fn sampler_drop(&self, sampler: &Self::SamplerId);
    fn buffer_drop(&self, buffer: &Self::BufferId);
    fn bind_group_drop(&self, bind_group: &Self::BindGroupId);
    fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId);
    fn pipeline_layout_drop(&self, pipeline_layout: &Self::PipelineLayoutId);
    fn shader_module_drop(&self, shader_module: &Self::ShaderModuleId);
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId);
    fn render_bundle_drop(&self, render_bundle: &Self::RenderBundleId);
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId);
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId);

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: BufferAddress,
        destination: &Self::BufferId,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: BufferCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: TextureCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
    ) -> Self::ComputePassId;
    fn command_encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    );
    fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId;
    fn command_encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    );
    fn command_encoder_finish(&self, encoder: &Self::CommandEncoderId) -> Self::CommandBufferId;
    fn render_bundle_encoder_finish(
        &self,
        encoder: Self::RenderBundleEncoderId,
        desc: &RenderBundleDescriptor,
    ) -> Self::RenderBundleId;
    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        buffer: &Self::BufferId,
        offset: BufferAddress,
        data: &[u8],
    );
    fn queue_write_texture(
        &self,
        queue: &Self::QueueId,
        texture: TextureCopyView,
        data: &[u8],
        data_layout: TextureDataLayout,
        size: Extent3d,
    );
    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    );
}

/// Instance of wgpu. First thing you create when using wgpu.
///
/// An instance sets up the context for all other wgpu objects.
///
/// Instances and adapters do not have to be kept alive, only devices.
pub struct Instance {
    context: Arc<C>,
}

/// Handle to a physical graphics and/or compute device.
///
/// An `Adapter` can be used to open a connection to the corresponding device on the host system,
/// yielding a [`Device`] object.
///
/// Instances and adapters do not have to be kept alive, only devices.
pub struct Adapter {
    context: Arc<C>,
    id: <C as Context>::AdapterId,
}

/// Options for requesting adapter.
#[derive(Clone)]
pub struct RequestAdapterOptions<'a> {
    /// Power preference for the adapter.
    pub power_preference: PowerPreference,
    /// Surface that is required to be presentable with the requested adapter. This does not
    /// create the surface, only guarantees that the adapter can present to said surface.
    pub compatible_surface: Option<&'a Surface>,
}

/// Open connection to a graphics and/or compute device.
///
/// The `Device` is the responsible for the creation of most rendering and compute resources, as
/// well as exposing [`Queue`] objects.
///
/// A device may be requested from an adapter with [`Adapter::request_device`].
pub struct Device {
    context: Arc<C>,
    id: <C as Context>::DeviceId,
}

/// Passed to [`Device::poll`] to control if it should block or not. This has no effect on
/// the web.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Maintain {
    /// Block
    Wait,
    /// Don't block
    Poll,
}

/// The main purpose of this struct is to resolve mapped ranges
/// (convert sizes to end points), and to ensure that the sub-ranges
/// don't intersect.
#[derive(Debug)]
struct MapContext {
    total_size: BufferAddress,
    initial_range: Range<BufferAddress>,
    sub_ranges: Vec<Range<BufferAddress>>,
}

impl MapContext {
    fn new(total_size: BufferAddress) -> Self {
        MapContext {
            total_size,
            initial_range: 0..0,
            sub_ranges: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.initial_range = 0..0;

        assert!(
            self.sub_ranges.is_empty(),
            "You cannot unmap a buffer that still has accessible mapped views"
        );
    }

    fn add(&mut self, offset: BufferAddress, size: BufferSize) -> BufferAddress {
        let end = if size == BufferSize::WHOLE {
            self.initial_range.end
        } else {
            offset + size.0
        };
        assert!(self.initial_range.start <= offset && end <= self.initial_range.end);
        for sub in self.sub_ranges.iter() {
            assert!(
                end <= sub.start || offset >= sub.end,
                "Intersecting map range with {:?}",
                sub
            );
        }
        self.sub_ranges.push(offset..end);
        end
    }

    fn remove(&mut self, offset: BufferAddress, size: BufferSize) {
        let end = if size == BufferSize::WHOLE {
            self.initial_range.end
        } else {
            offset + size.0
        };

        // Switch this out with `Vec::remove_item` once that stabilizes.
        let index = self
            .sub_ranges
            .iter()
            .position(|r| *r == (offset..end))
            .expect("unable to remove range from map context");
        self.sub_ranges.swap_remove(index);
    }
}

/// Handle to a GPU-accessible buffer.
///
/// Created with [`Device::create_buffer`] or [`Device::create_buffer_with_data`]
pub struct Buffer {
    context: Arc<C>,
    id: <C as Context>::BufferId,
    map_context: Mutex<MapContext>,
    usage: BufferUsage,
}

/// Slice into a [`Buffer`].
///
/// Created by calling [`Buffer::slice`]. To use the whole buffer, call with unbounded slice:
///
/// `buffer.slice(..)`
#[derive(Copy, Clone)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: BufferSize,
}

/// Handle to a texture on the GPU.
///
/// Created by calling [`Device::create_texture`]
pub struct Texture {
    context: Arc<C>,
    id: <C as Context>::TextureId,
    owned: bool,
}

/// Handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
pub struct TextureView {
    context: Arc<C>,
    id: <C as Context>::TextureViewId,
    owned: bool,
}

/// Handle to a sampler.
///
/// A `Sampler` object defines how a pipeline will sample from a [`TextureView`]. Samplers define
/// image filters (including anisotropy) and address (wrapping) modes, among other things. See
/// the documentation for [`SamplerDescriptor`] for more information.
pub struct Sampler {
    context: Arc<C>,
    id: <C as Context>::SamplerId,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.sampler_drop(&self.id);
        }
    }
}

/// Handle to a presentable surface.
///
/// A `Surface` represents a platform-specific surface (e.g. a window) onto which rendered images may
/// be presented. A `Surface` may be created with the unsafe function [`Instance::create_surface`].
pub struct Surface {
    id: <C as Context>::SurfaceId,
}

/// Handle to a swap chain.
///
/// A `SwapChain` represents the image or series of images that will be presented to a [`Surface`].
/// A `SwapChain` may be created with [`Device::create_swap_chain`].
pub struct SwapChain {
    context: Arc<C>,
    id: <C as Context>::SwapChainId,
}

/// Handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
pub struct BindGroupLayout {
    context: Arc<C>,
    id: <C as Context>::BindGroupLayoutId,
}

impl Drop for BindGroupLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.bind_group_layout_drop(&self.id);
        }
    }
}

/// Handle to a binding group.
///
/// A `BindGroup` represents the set of resources bound to the bindings described by a
/// [`BindGroupLayout`]. It can be created with [`Device::create_bind_group`]. A `BindGroup` can
/// be bound to a particular [`RenderPass`] with [`RenderPass::set_bind_group`], or to a
/// [`ComputePass`] with [`ComputePass::set_bind_group`].
pub struct BindGroup {
    context: Arc<C>,
    id: <C as Context>::BindGroupId,
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.bind_group_drop(&self.id);
        }
    }
}

/// Handle to a compiled shader module.
///
/// A `ShaderModule` represents a compiled shader module on the GPU. It can be created by passing
/// valid SPIR-V source code to [`Device::create_shader_module`]. Shader modules are used to define
/// programmable stages of a pipeline.
pub struct ShaderModule {
    context: Arc<C>,
    id: <C as Context>::ShaderModuleId,
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.shader_module_drop(&self.id);
        }
    }
}

/// Source of a shader module.
pub enum ShaderModuleSource<'a> {
    /// SPIR-V module represented as a slice of words.
    /// wgpu-rs will try to reflect it and use for validation, but the
    /// original data is passed to gfx-rs and spirv_cross for translation.
    SpirV(&'a [u32]),
    /// WGSL module as a string slice.
    /// wgpu-rs will parse it and use for validation. It will attempt
    /// to build a SPIR-V module internally and panic otherwise.
    ///
    /// Note: WGSL is not yet supported on the Web.
    Wgsl(&'a str),
}

/// Handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
pub struct PipelineLayout {
    context: Arc<C>,
    id: <C as Context>::PipelineLayoutId,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.pipeline_layout_drop(&self.id);
        }
    }
}

/// Handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. A `RenderPipeline` may be created with [`Device::create_render_pipeline`].
pub struct RenderPipeline {
    context: Arc<C>,
    id: <C as Context>::RenderPipelineId,
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.render_pipeline_drop(&self.id);
        }
    }
}

/// Handle to a compute pipeline.
///
/// A `ComputePipeline` object represents a compute pipeline and its single shader stage.
/// A `ComputePipeline` may be created with [`Device::create_compute_pipeline`].
pub struct ComputePipeline {
    context: Arc<C>,
    id: <C as Context>::ComputePipelineId,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.compute_pipeline_drop(&self.id);
        }
    }
}

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
pub struct CommandBuffer {
    context: Arc<C>,
    id: Option<<C as Context>::CommandBufferId>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            if let Some(ref id) = self.id {
                self.context.command_buffer_drop(id);
            }
        }
    }
}

/// Encodes a series of GPU operations.
///
/// A `CommandEncoder` can record [`RenderPass`]es, [`ComputePass`]es, and transfer operations
/// between driver-managed resources like [`Buffer`]s and [`Texture`]s.
///
/// When finished recording, call [`CommandEncoder::finish`] to obtain a [`CommandBuffer`] which may
/// be submitted for execution.
pub struct CommandEncoder {
    context: Arc<C>,
    id: <C as Context>::CommandEncoderId,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: PhantomData<*const u8>,
}

/// In-progress recording of a render pass.
pub struct RenderPass<'a> {
    id: <C as Context>::RenderPassId,
    parent: &'a mut CommandEncoder,
}

/// In-progress recording of a compute pass.
pub struct ComputePass<'a> {
    id: <C as Context>::ComputePassId,
    parent: &'a mut CommandEncoder,
}

/// Encodes a series of GPU operations into a reusable "render bundle".
///
/// It only supports a handful of render commands, but it makes them reusable. [`RenderBundle`]s
/// can be executed onto a [`CommandEncoder`] using [`RenderPass::execute_bundles`].
///
/// Executing a [`RenderBundle`] is often more efficient then issuing the underlying commands manually.
pub struct RenderBundleEncoder<'a> {
    context: Arc<C>,
    id: <C as Context>::RenderBundleEncoderId,
    _parent: &'a Device,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: PhantomData<*const u8>,
}

/// Pre-prepared reusable bundle of GPU operations.
///
/// It only supports a handful of render commands, but it makes them reusable. [`RenderBundle`]s
/// can be executed onto a [`CommandEncoder`] using [`RenderPass::execute_bundles`].
///
/// Executing a [`RenderBundle`] is often more efficient then issuing the underlying commands manually.
pub struct RenderBundle {
    context: Arc<C>,
    id: <C as Context>::RenderBundleId,
}

impl Drop for RenderBundle {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.render_bundle_drop(&self.id);
        }
    }
}

/// Handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects and provides convenience methods
/// for writing to [buffers](Queue::write_buffer) and [textures](Queue::write_texture).
pub struct Queue {
    context: Arc<C>,
    id: <C as Context>::QueueId,
}

/// Resource that can be bound to a pipeline.
#[non_exhaustive]
pub enum BindingResource<'a> {
    /// Binding is backed by a buffer.
    ///
    /// Corresponds to [`BindingType::UniformBuffer`] and [`BindingType::StorageBuffer`]
    /// with [`BindGroupLayoutEntry::count`] set to None.
    Buffer(BufferSlice<'a>),
    /// Binding is a sampler.
    ///
    /// Corresponds to [`BindingType::Sampler`] with [`BindGroupLayoutEntry::count`] set to None.
    Sampler(&'a Sampler),
    /// Binding is backed by a texture.
    ///
    /// Corresponds to [`BindingType::SampledTexture`] and [`BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to None.
    TextureView(&'a TextureView),
    /// Binding is backed by an array of textures.
    ///
    /// [`Capabilities::SAMPLED_TEXTURE_BINDING_ARRAY`] must be supported to use this feature.
    ///
    /// Corresponds to [`BindingType::SampledTexture`] and [`BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to Some.
    TextureViewArray(&'a [TextureView]),
}

/// Bindable resource and the slot to bind it to.
pub struct Binding<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource<'a>,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone)]
pub struct BindGroupDescriptor<'a> {
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: &'a BindGroupLayout,

    /// The resources to bind to this bind group.
    pub bindings: &'a [Binding<'a>],

    /// Debug label of the bind group. This will show up in graphics debuggers for easy identification.
    pub label: Option<&'a str>,
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be passed to [`Device::create_pipeline_layout`] to obtain a
/// [`PipelineLayout`].
#[derive(Clone)]
pub struct PipelineLayoutDescriptor<'a> {
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

/// Describes a programmable pipeline stage.
#[derive(Clone)]
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,

    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
}

/// Describes vertex input state for a render pipeline.
#[derive(Clone, Debug)]
pub struct VertexStateDescriptor<'a> {
    /// The format of any index buffers used with this pipeline.
    pub index_format: IndexFormat,

    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers: &'a [VertexBufferDescriptor<'a>],
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug)]
pub struct VertexBufferDescriptor<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub stride: BufferAddress,

    /// How often this vertex buffer is "stepped" forward. Can be per-vertex
    /// or per-instance.
    pub step_mode: InputStepMode,

    /// The list of attributes which comprise a single vertex, this can be made with [`vertex_attr_array`].
    pub attributes: &'a [VertexAttributeDescriptor],
}

/// Describes a render (graphics) pipeline.
#[derive(Clone)]
pub struct RenderPipelineDescriptor<'a> {
    /// The layout of bind groups for this pipeline.
    pub layout: &'a PipelineLayout,

    /// The compiled vertex stage and its entry point.
    pub vertex_stage: ProgrammableStageDescriptor<'a>,

    /// The compiled fragment stage and its entry point, if any.
    pub fragment_stage: Option<ProgrammableStageDescriptor<'a>>,

    /// The rasterization process for this pipeline.
    pub rasterization_state: Option<RasterizationStateDescriptor>,

    /// The primitive topology used to interpret vertices.
    pub primitive_topology: PrimitiveTopology,

    /// The effect of draw calls on the color aspect of the output target.
    pub color_states: &'a [ColorStateDescriptor],

    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil_state: Option<DepthStencilStateDescriptor>,

    /// The vertex input state for this pipeline.
    pub vertex_state: VertexStateDescriptor<'a>,

    /// The number of samples calculated per pixel (for MSAA). For non-multisampled textures,
    /// this should be `1`
    pub sample_count: u32,

    /// Bitmask that restricts the samples of a pixel modified by this pipeline. All samples
    /// can be enabled using the value `!0`
    pub sample_mask: u32,

    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    ///
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

/// Describes a compute pipeline.
#[derive(Clone)]
pub struct ComputePipelineDescriptor<'a> {
    /// The layout of bind groups for this pipeline.
    pub layout: &'a PipelineLayout,

    /// The compiled compute stage and its entry point.
    pub compute_stage: ProgrammableStageDescriptor<'a>,
}

// The underlying types are also exported so that documentation shows up for them

pub use wgt::RenderPassColorAttachmentDescriptorBase;
/// Describes a color attachment to a [`RenderPass`].
pub type RenderPassColorAttachmentDescriptor<'a> =
    wgt::RenderPassColorAttachmentDescriptorBase<&'a TextureView>;
pub use wgt::RenderPassDepthStencilAttachmentDescriptorBase;
/// Describes a depth/stencil attachment to a [`RenderPass`].
pub type RenderPassDepthStencilAttachmentDescriptor<'a> =
    wgt::RenderPassDepthStencilAttachmentDescriptorBase<&'a TextureView>;

/// Describes the attachments of a [`RenderPass`].
pub struct RenderPassDescriptor<'a, 'b> {
    /// The color attachments of the render pass.
    pub color_attachments: &'b [RenderPassColorAttachmentDescriptor<'a>],

    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachmentDescriptor<'a>>,
}

// The underlying types are also exported so that documentation shows up for them

pub use wgt::BufferDescriptor as BufferDescriptorBase;
/// Describes a [`Buffer`].
pub type BufferDescriptor<'a> = BufferDescriptorBase<Option<&'a str>>;

pub use wgt::CommandEncoderDescriptor as CommandEncoderDescriptorBase;
/// Describes a [`CommandEncoder`].
pub type CommandEncoderDescriptor<'a> = CommandEncoderDescriptorBase<Option<&'a str>>;

pub use wgt::RenderBundleDescriptor as RenderBundleDescriptorBase;
/// Describes a [`RenderBundle`].
pub type RenderBundleDescriptor<'a> = RenderBundleDescriptorBase<Option<&'a str>>;

pub use wgt::TextureDescriptor as TextureDescriptorBase;
/// Describes a [`Texture`].
pub type TextureDescriptor<'a> = TextureDescriptorBase<Option<&'a str>>;

pub use wgt::TextureViewDescriptor as TextureViewDescriptorBase;
/// Describes a [`TextureView`].
pub type TextureViewDescriptor<'a> = TextureViewDescriptorBase<Option<&'a str>>;

pub use wgt::SamplerDescriptor as SamplerDescriptorBase;
/// Describes a [`Sampler`].
pub type SamplerDescriptor<'a> = SamplerDescriptorBase<Option<&'a str>>;

/// Swap chain image that can be rendered to.
pub struct SwapChainTexture {
    pub view: TextureView,
    detail: <C as Context>::SwapChainOutputDetail,
}

/// Result of a successful call to [`SwapChain::get_next_frame`].
pub struct SwapChainFrame {
    pub output: SwapChainTexture,
    pub suboptimal: bool,
}

/// Result of an unsuccessful call to [`SwapChain::get_next_frame`].
#[derive(Debug)]
pub enum SwapChainError {
    Timeout,
    Outdated,
    Lost,
    OutOfMemory,
}

/// View of a buffer which can be used to copy to/from a texture.
#[derive(Clone)]
pub struct BufferCopyView<'a> {
    /// The buffer to be copied to/from.
    pub buffer: &'a Buffer,

    /// The layout of the texture data in this buffer.
    pub layout: TextureDataLayout,
}

/// View of a texture which can be used to copy to/from a buffer/texture.
#[derive(Clone)]
pub struct TextureCopyView<'a> {
    /// The texture to be copied to/from.
    pub texture: &'a Texture,

    /// The target mip level of the texture.
    pub mip_level: u32,

    /// The base texel of the texture in the selected `mip_level`.
    pub origin: Origin3d,
}

impl Instance {
    /// Create an new instance of wgpu. The `backends` parameter controls
    /// among which backends wgpu will decide during instantiation.
    pub fn new(backends: BackendBit) -> Self {
        Instance {
            context: Arc::new(C::init(backends)),
        }
    }

    /// Retrieves all available [`Adapter`]s that match the given backends.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enumerate_adapters(
        &self,
        unsafe_extensions: UnsafeExtensions,
        backends: BackendBit,
    ) -> impl Iterator<Item = Adapter> {
        let context = Arc::clone(&self.context);
        self.context
            .enumerate_adapters(
                unsafe_extensions,
                wgc::instance::AdapterInputs::Mask(backends, |_| PhantomData),
            )
            .into_iter()
            .map(move |id| crate::Adapter {
                id,
                context: Arc::clone(&context),
            })
    }

    /// Creates a surface from a raw window handle.
    ///
    /// # Safety
    ///
    /// - Raw Window Handle must be a valid object to create a surface upon.
    pub unsafe fn create_surface<W: raw_window_handle::HasRawWindowHandle>(
        &self,
        window: &W,
    ) -> Surface {
        Surface {
            id: Context::instance_create_surface(&*self.context, window),
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> Surface {
        let surface = wgc::instance::Surface {
            #[cfg(feature = "vulkan-portability")]
            vulkan: self.context.instance.vulkan.as_ref().map(|inst| {
                inst.create_surface_from_layer(layer as *mut _, cfg!(debug_assertions))
            }),
            metal: self.context.instance.metal.as_ref().map(|inst| {
                inst.create_surface_from_layer(layer as *mut _, cfg!(debug_assertions))
            }),
        };

        crate::Surface {
            id: self.context.surfaces.register_identity(
                PhantomData,
                surface,
                &mut wgc::hub::Token::root(),
            ),
        }
    }

    /// Retrieves an [`Adapter`] which matches the given options.
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that suffice all the "hard" options, `None` is returned.
    pub fn request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
        unsafe_extensions: UnsafeExtensions,
    ) -> impl Future<Output = Option<Adapter>> + Send {
        let context = Arc::clone(&self.context);
        self.context
            .instance_request_adapter(options, unsafe_extensions)
            .map(|option| option.map(|id| Adapter { context, id }))
    }
}

impl Adapter {
    /// Requests a connection to a physical device, creating a logical device.
    /// Returns the device together with a queue that executes command buffers.
    ///
    /// # Panics
    ///
    /// - Extensions specified by `desc` are not supported by this adapter.
    /// - Unsafe extensions were requested but enabled when requesting the adapter.
    /// - Limits requested exceed the values provided by the adapter.
    /// - Adapter does not support all features wgpu requires to safely operate.
    pub fn request_device(
        &self,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> impl Future<Output = Result<(Device, Queue), RequestDeviceError>> + Send {
        let context = Arc::clone(&self.context);
        Context::adapter_request_device(&*self.context, &self.id, desc, trace_path).map(|result| {
            result.map(|(device_id, queue_id)| {
                (
                    Device {
                        context: Arc::clone(&context),
                        id: device_id,
                    },
                    Queue {
                        context,
                        id: queue_id,
                    },
                )
            })
        })
    }

    /// List all extensions that are supported with this adapter.
    ///
    /// Extensions must be explicitly requested in [`Adapter::request_device`] in order
    /// to use them.
    pub fn extensions(&self) -> Extensions {
        Context::adapter_extensions(&*self.context, &self.id)
    }

    /// List the "best" limits that are supported by this adapter.
    ///
    /// Limits must be explicitly requested in [`Adapter::request_device`] in to set
    /// the values that you are allowed to use.
    pub fn limits(&self) -> Limits {
        Context::adapter_limits(&*self.context, &self.id)
    }

    /// List all capabilities that may be used wth this adapter.
    pub fn capabilities(&self) -> Capabilities {
        Context::adapter_capabilities(&*self.context, &self.id)
    }

    /// Get info about the adapter itself.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_info(&self) -> AdapterInfo {
        let context = &self.context;
        wgc::gfx_select!(self.id => context.adapter_get_info(self.id))
    }
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    ///
    /// no-op on the web, device is automatically polled.
    pub fn poll(&self, maintain: Maintain) {
        Context::device_poll(&*self.context, &self.id, maintain);
    }

    /// List all extensions that may be used with this device.
    ///
    /// Functions may panic if you use unsupported extensions.
    pub fn extensions(&self) -> Extensions {
        Context::device_extensions(&*self.context, &self.id)
    }

    /// List all limits that were requested of this device.
    ///
    /// If any of these limits are exceeded, functions may panic.
    pub fn limits(&self) -> Limits {
        Context::device_limits(&*self.context, &self.id)
    }

    /// List all capabilities that may be used wth this device.
    ///
    /// Functions may panic if you use unsupported capabilities.
    pub fn capabilities(&self) -> Capabilities {
        Context::device_capabilities(&*self.context, &self.id)
    }

    /// Creates a shader module from either SPIR-V or WGSL source code.
    pub fn create_shader_module(&self, source: ShaderModuleSource) -> ShaderModule {
        ShaderModule {
            context: Arc::clone(&self.context),
            id: Context::device_create_shader_module(&*self.context, &self.id, source),
        }
    }

    /// Creates an empty [`CommandEncoder`].
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        CommandEncoder {
            context: Arc::clone(&self.context),
            id: Context::device_create_command_encoder(&*self.context, &self.id, desc),
            _p: Default::default(),
        }
    }

    /// Creates an empty [`RenderBundleEncoder`].
    pub fn create_render_bundle_encoder(
        &self,
        desc: &RenderBundleEncoderDescriptor,
    ) -> RenderBundleEncoder {
        RenderBundleEncoder {
            context: Arc::clone(&self.context),
            id: Context::device_create_render_bundle_encoder(&*self.context, &self.id, desc),
            _parent: self,
            _p: Default::default(),
        }
    }

    /// Creates a new [`BindGroup`].
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        BindGroup {
            context: Arc::clone(&self.context),
            id: Context::device_create_bind_group(&*self.context, &self.id, desc),
        }
    }

    /// Creates a [`BindGroupLayout`].
    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        BindGroupLayout {
            context: Arc::clone(&self.context),
            id: Context::device_create_bind_group_layout(&*self.context, &self.id, desc),
        }
    }

    /// Creates a [`PipelineLayout`].
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        PipelineLayout {
            context: Arc::clone(&self.context),
            id: Context::device_create_pipeline_layout(&*self.context, &self.id, desc),
        }
    }

    /// Creates a [`RenderPipeline`].
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        RenderPipeline {
            context: Arc::clone(&self.context),
            id: Context::device_create_render_pipeline(&*self.context, &self.id, desc),
        }
    }

    /// Creates a [`ComputePipeline`].
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        ComputePipeline {
            context: Arc::clone(&self.context),
            id: Context::device_create_compute_pipeline(&*self.context, &self.id, desc),
        }
    }

    /// Creates a [`Buffer`].
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        let mut map_context = MapContext::new(desc.size);
        if desc.mapped_at_creation {
            map_context.initial_range = 0..desc.size;
        }
        Buffer {
            context: Arc::clone(&self.context),
            id: Context::device_create_buffer(&*self.context, &self.id, desc),
            map_context: Mutex::new(map_context),
            usage: desc.usage,
        }
    }

    /// Creates a new buffer, maps it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    pub fn create_buffer_with_data(&self, data: &[u8], usage: BufferUsage) -> Buffer {
        let size = data.len() as u64;
        let buffer = self.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: true,
        });
        Context::buffer_get_mapped_range_mut(&*self.context, &buffer.id, 0..size)
            .copy_from_slice(data);
        buffer.unmap();
        buffer
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture {
        Texture {
            context: Arc::clone(&self.context),
            id: Context::device_create_texture(&*self.context, &self.id, desc),
            owned: true,
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler {
        Sampler {
            context: Arc::clone(&self.context),
            id: Context::device_create_sampler(&*self.context, &self.id, desc),
        }
    }

    /// Create a new [`SwapChain`] which targets `surface`.
    ///
    /// # Panics
    ///
    /// - A old [`SwapChainFrame`] is still alive referencing an old swapchain.
    /// - Texture format requested is unsupported on the swap chain.
    pub fn create_swap_chain(&self, surface: &Surface, desc: &SwapChainDescriptor) -> SwapChain {
        SwapChain {
            context: Arc::clone(&self.context),
            id: Context::device_create_swap_chain(&*self.context, &self.id, &surface.id, desc),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.device_drop(&self.id);
        }
    }
}

/// Requesting a device failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RequestDeviceError;

/// Error occurred when trying to async map a number.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;

/// Type of buffer mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapMode {
    /// Map only for reading
    Read,
    /// Map only for writing
    Write,
}

fn range_to_offset_size<S: RangeBounds<BufferAddress>>(bounds: S) -> (BufferAddress, BufferSize) {
    let offset = match bounds.start_bound() {
        Bound::Included(&bound) => bound,
        Bound::Excluded(&bound) => bound + 1,
        Bound::Unbounded => 0,
    };
    let size = match bounds.end_bound() {
        Bound::Included(&bound) => BufferSize(bound + 1 - offset),
        Bound::Excluded(&bound) => BufferSize(bound - offset),
        Bound::Unbounded => BufferSize::WHOLE,
    };

    (offset, size)
}

/// Read only view into a mapped buffer.
pub struct BufferView<'a> {
    slice: BufferSlice<'a>,
    data: &'a [u8],
}

/// Write only view into mapped buffer.
pub struct BufferViewMut<'a> {
    slice: BufferSlice<'a>,
    data: &'a mut [u8],
    readable: bool,
}

impl std::ops::Deref for BufferView<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.data
    }
}

impl std::ops::Deref for BufferViewMut<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        assert!(
            self.readable,
            "Attempting to read a write-only mapping for buffer {:?}",
            self.slice.buffer.id
        );
        self.data
    }
}

impl std::ops::DerefMut for BufferViewMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl Drop for BufferView<'_> {
    fn drop(&mut self) {
        self.slice
            .buffer
            .map_context
            .lock()
            .remove(self.slice.offset, self.slice.size);
    }
}

impl Drop for BufferViewMut<'_> {
    fn drop(&mut self) {
        self.slice
            .buffer
            .map_context
            .lock()
            .remove(self.slice.offset, self.slice.size);
    }
}

impl Buffer {
    /// Use only a portion of this Buffer for a given operation. Choosing a range with no end
    /// will use the rest of the buffer. Using a totally unbounded range will use the entire buffer.
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice {
        let (offset, size) = range_to_offset_size(bounds);
        BufferSlice {
            buffer: self,
            offset,
            size,
        }
    }

    /// Flushes any pending write operations and unmaps the buffer from host memory.
    pub fn unmap(&self) {
        self.map_context.lock().reset();
        Context::buffer_unmap(&*self.context, &self.id);
    }
}

impl BufferSlice<'_> {
    /// Use only a portion of this BufferSlice for a given operation. Choosing a range with no end
    /// will use the rest of the buffer. Using a totally unbounded range will use the entire BufferSlice.
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Self {
        let (sub_offset, sub_size) = range_to_offset_size(bounds);
        let new_offset = self.offset + sub_offset;
        let new_size = if sub_size == BufferSize::WHOLE {
            BufferSize(
                self.size
                    .0
                    .checked_sub(sub_offset)
                    .expect("underflow when slicing `BufferSlice`"),
            )
        } else {
            assert!(
                new_offset + sub_size.0 <= self.offset + self.size.0,
                "offset and size must stay within the bounds of the parent slice"
            );
            sub_size
        };
        Self {
            buffer: self.buffer,
            offset: new_offset,
            size: new_size,
        }
    }

    /// Map the buffer. Buffer is ready to map once the future is resolved.
    ///
    /// For the future to complete, `device.poll(...)` must be called elsewhere in the runtime, possibly integrated
    /// into an event loop, run on a separate thread, or continually polled in the same task runtime that this
    /// future will be run on.
    ///
    /// It's expected that wgpu will eventually supply its own event loop infrastructure that will be easy to integrate
    /// into other event loops, like winit's.
    pub fn map_async(
        &self,
        mode: MapMode,
    ) -> impl Future<Output = Result<(), BufferAsyncError>> + Send {
        let end = {
            let mut mc = self.buffer.map_context.lock();
            assert_eq!(
                mc.initial_range,
                0..0,
                "Buffer {:?} is already mapped",
                self.buffer.id
            );
            let end = if self.size == BufferSize::WHOLE {
                mc.total_size
            } else {
                self.offset + self.size.0
            };
            mc.initial_range = self.offset..end;
            end
        };
        Context::buffer_map_async(
            &*self.buffer.context,
            &self.buffer.id,
            mode,
            self.offset..end,
        )
    }

    /// Synchronously and immediately map a buffer for reading. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will panic.
    pub fn get_mapped_range(&self) -> BufferView {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = Context::buffer_get_mapped_range(
            &*self.buffer.context,
            &self.buffer.id,
            self.offset..end,
        );
        BufferView { slice: *self, data }
    }

    /// Synchronously and immediately map a buffer for writing. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will panic.
    pub fn get_mapped_range_mut(&self) -> BufferViewMut {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = Context::buffer_get_mapped_range_mut(
            &*self.buffer.context,
            &self.buffer.id,
            self.offset..end,
        );
        BufferViewMut {
            slice: *self,
            data,
            readable: self.buffer.usage.contains(BufferUsage::MAP_READ),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.buffer_drop(&self.id);
        }
    }
}

impl Texture {
    /// Creates a view of this texture.
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        TextureView {
            context: Arc::clone(&self.context),
            id: Context::texture_create_view(&*self.context, &self.id, Some(desc)),
            owned: true,
        }
    }

    /// Creates the default view of this whole texture. This is likely what you want.
    pub fn create_default_view(&self) -> TextureView {
        TextureView {
            context: Arc::clone(&self.context),
            id: Context::texture_create_view(&*self.context, &self.id, None),
            owned: true,
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if self.owned && !thread::panicking() {
            self.context.texture_drop(&self.id);
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if self.owned && !thread::panicking() {
            self.context.texture_view_drop(&self.id);
        }
    }
}

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            context: Arc::clone(&self.context),
            id: Some(Context::command_encoder_finish(&*self.context, &self.id)),
        }
    }

    /// Begins recording of a render pass.
    ///
    /// This function returns a [`RenderPass`] object which records a single render pass.
    pub fn begin_render_pass<'a>(
        &'a mut self,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> RenderPass<'a> {
        RenderPass {
            id: Context::command_encoder_begin_render_pass(&*self.context, &self.id, desc),
            parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: Context::command_encoder_begin_compute_pass(&*self.context, &self.id),
            parent: self,
        }
    }

    /// Copy data from one buffer to another.
    ///
    /// # Panics
    ///
    /// - Buffer offsets or copy size not a multiple of [`COPY_BUFFER_ALIGNMENT`].
    /// - Copy would overrun buffer.
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        Context::command_encoder_copy_buffer_to_buffer(
            &*self.context,
            &self.id,
            &source.id,
            source_offset,
            &destination.id,
            destination_offset,
            copy_size,
        );
    }

    /// Copy data from a buffer to a texture.
    ///
    /// # Panics
    ///
    /// - Copy would overrun buffer.
    /// - Copy would overrun texture.
    /// - `source.layout.bytes_per_row` isn't divisible by [`COPY_BYTES_PER_ROW_ALIGNMENT`].
    pub fn copy_buffer_to_texture(
        &mut self,
        source: BufferCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_buffer_to_texture(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from a texture to a buffer.
    ///
    /// # Panics
    ///
    /// - Copy would overrun buffer.
    /// - Copy would overrun texture.
    /// - `source.layout.bytes_per_row` isn't divisible by [`COPY_BYTES_PER_ROW_ALIGNMENT`].
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_texture_to_buffer(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from one texture to another.
    ///
    /// # Panics
    ///
    /// - Textures are not the same type
    /// - If a depth texture, or a multisampled texture, the entire texture must be copied
    /// - Copy would overrun either texture
    pub fn copy_texture_to_texture(
        &mut self,
        source: TextureCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_texture_to_texture(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }
}

impl<'a> RenderPass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        RenderInner::set_bind_group(&mut self.id, index, &bind_group.id, offsets)
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        RenderInner::set_pipeline(&mut self.id, &pipeline.id)
    }

    /// Sets the blend color as used by some of the blending modes.
    ///
    /// Subsequent blending tests will test against this value.
    pub fn set_blend_color(&mut self, color: Color) {
        self.id.set_blend_color(color)
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>) {
        RenderInner::set_index_buffer(
            &mut self.id,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderPass`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [`VertexStateDescriptor::vertex_buffers`].
    ///
    /// [`draw`]: RenderPass::draw
    /// [`draw_indexed`]: RenderPass::draw_indexed
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        RenderInner::set_vertex_buffer(
            &mut self.id,
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Sets the scissor region.
    ///
    /// Subsequent draw calls will discard any fragments that fall outside this region.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.id.set_scissor_rect(x, y, width, height);
    }

    /// Sets the viewport region.
    ///
    /// Subsequent draw calls will draw any fragments in this region.
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.id.set_viewport(x, y, w, h, min_depth, max_depth);
    }

    /// Sets the stencil reference.
    ///
    /// Subsequent stencil tests will test against this value.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        self.id.set_stencil_reference(reference);
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        RenderInner::draw(&mut self.id, vertices, instances)
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        self.id.insert_debug_marker(label);
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        self.id.push_debug_group(label);
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        self.id.pop_debug_group();
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        RenderInner::draw_indexed(&mut self.id, indices, base_vertex, instances);
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndirect {
    ///     vertex_count: u32, // The number of vertices to draw.
    ///     instance_count: u32, // The number of instances to draw.
    ///     base_vertex: u32, // The Index of the first vertex to draw.
    ///     base_instance: u32, // The instance ID of the first instance to draw.
    /// }
    /// ```
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        self.id.draw_indirect(&indirect_buffer.id, indirect_offset);
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndexedIndirect {
    ///     vertex_count: u32, // The number of vertices to draw.
    ///     instance_count: u32, // The number of instances to draw.
    ///     base_index: u32, // The base index within the index buffer.
    ///     vertex_offset: i32, // The value added to the vertex index before indexing into the vertex buffer.
    ///     base_instance: u32, // The instance ID of the first instance to draw.
    /// }
    /// ```
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        self.id
            .draw_indexed_indirect(&indirect_buffer.id, indirect_offset);
    }

    pub fn execute_bundles<I: Iterator<Item = &'a RenderBundle>>(&mut self, render_bundles: I) {
        self.id
            .execute_bundles(render_bundles.into_iter().map(|rb| &rb.id))
    }
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.parent
                .context
                .command_encoder_end_render_pass(&self.parent.id, &mut self.id);
        }
    }
}

impl<'a> ComputePass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `dispatch()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        ComputePassInner::set_bind_group(&mut self.id, index, &bind_group.id, offsets);
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &'a ComputePipeline) {
        ComputePassInner::set_pipeline(&mut self.id, &pipeline.id);
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        ComputePassInner::dispatch(&mut self.id, x, y, z);
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    pub fn dispatch_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        ComputePassInner::dispatch_indirect(&mut self.id, &indirect_buffer.id, indirect_offset);
    }
}

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.parent
                .context
                .command_encoder_end_compute_pass(&self.parent.id, &mut self.id);
        }
    }
}

impl<'a> RenderBundleEncoder<'a> {
    /// Finishes recording and returns a [`RenderBundle`] that can be executed in other render passes.
    pub fn finish(self, desc: &RenderBundleDescriptor) -> RenderBundle {
        RenderBundle {
            context: Arc::clone(&self.context),
            id: Context::render_bundle_encoder_finish(&*self.context, self.id, desc),
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        RenderInner::set_bind_group(&mut self.id, index, &bind_group.id, offsets)
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        RenderInner::set_pipeline(&mut self.id, &pipeline.id)
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderBundleEncoder::draw_indexed) on this [`RenderBundleEncoder`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>) {
        RenderInner::set_index_buffer(
            &mut self.id,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderBundleEncoder`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [`VertexStateDescriptor::vertex_buffers`].
    ///
    /// [`draw`]: RenderBundleEncoder::draw
    /// [`draw_indexed`]: RenderBundleEncoder::draw_indexed
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        RenderInner::set_vertex_buffer(
            &mut self.id,
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        RenderInner::draw(&mut self.id, vertices, instances)
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        RenderInner::draw_indexed(&mut self.id, indices, base_vertex, instances);
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndirect {
    ///     vertex_count: u32, // The number of vertices to draw.
    ///     instance_count: u32, // The number of instances to draw.
    ///     base_vertex: u32, // The Index of the first vertex to draw.
    ///     base_instance: u32, // The instance ID of the first instance to draw.
    /// }
    /// ```
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        self.id.draw_indirect(&indirect_buffer.id, indirect_offset);
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndexedIndirect {
    ///     vertex_count: u32, // The number of vertices to draw.
    ///     instance_count: u32, // The number of instances to draw.
    ///     base_index: u32, // The base index within the index buffer.
    ///     vertex_offset: i32, // The value added to the vertex index before indexing into the vertex buffer.
    ///     base_instance: u32, // The instance ID of the first instance to draw.
    /// }
    /// ```
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        self.id
            .draw_indexed_indirect(&indirect_buffer.id, indirect_offset);
    }
}

impl Queue {
    /// Schedule a data write into `buffer` starting at `offset`.
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        Context::queue_write_buffer(&*self.context, &self.id, &buffer.id, offset, data)
    }

    /// Schedule a data write into `texture`.
    pub fn write_texture(
        &self,
        texture: TextureCopyView,
        data: &[u8],
        data_layout: TextureDataLayout,
        size: Extent3d,
    ) {
        Context::queue_write_texture(&*self.context, &self.id, texture, data, data_layout, size)
    }

    /// Submits a series of finished command buffers for execution.
    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(&self, command_buffers: I) {
        Context::queue_submit(
            &*self.context,
            &self.id,
            command_buffers
                .into_iter()
                .map(|mut comb| comb.id.take().unwrap()),
        );
    }
}

impl Drop for SwapChainTexture {
    fn drop(&mut self) {
        if !thread::panicking() {
            Context::swap_chain_present(&*self.view.context, &self.view.id, &self.detail);
        }
    }
}

impl SwapChain {
    /// Returns the next texture to be presented by the swapchain for drawing.
    ///
    /// When the [`SwapChainFrame`] returned by this method is dropped, the swapchain will present
    /// the texture to the associated [`Surface`].
    ///
    /// If a SwapChainFrame referencing this surface is alive when the swapchain is recreated,
    /// recreating the swapchain will panic.
    pub fn get_next_frame(&mut self) -> Result<SwapChainFrame, SwapChainError> {
        let (view_id, status, detail) =
            Context::swap_chain_get_next_texture(&*self.context, &self.id);
        let output = view_id.map(|id| SwapChainTexture {
            view: TextureView {
                context: Arc::clone(&self.context),
                id: id,
                owned: false,
            },
            detail,
        });

        match status {
            SwapChainStatus::Good => Ok(SwapChainFrame {
                output: output.unwrap(),
                suboptimal: false,
            }),
            SwapChainStatus::Suboptimal => Ok(SwapChainFrame {
                output: output.unwrap(),
                suboptimal: true,
            }),
            SwapChainStatus::Timeout => Err(SwapChainError::Timeout),
            SwapChainStatus::Outdated => Err(SwapChainError::Outdated),
            SwapChainStatus::Lost => Err(SwapChainError::Lost),
            SwapChainStatus::OutOfMemory => Err(SwapChainError::OutOfMemory),
        }
    }
}
