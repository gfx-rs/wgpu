//! A cross-platform graphics and compute library based on [WebGPU](https://gpuweb.github.io/gpuweb/).
//!
//! To start using the API, create an [`Instance`].

#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu-rs/master/logo.png")]
#![warn(missing_docs)]

mod backend;
pub mod util;
#[macro_use]
mod macros;

use std::{
    borrow::Cow,
    error,
    fmt::{Debug, Display},
    future::Future,
    marker::PhantomData,
    num::{NonZeroU32, NonZeroU8},
    ops::{Bound, Range, RangeBounds},
    sync::Arc,
    thread,
};

use parking_lot::Mutex;

pub use wgt::{
    AdapterInfo, AddressMode, Backend, BackendBit, BindGroupLayoutEntry, BindingType,
    BlendComponent, BlendFactor, BlendOperation, BlendState, BufferAddress, BufferBindingType,
    BufferSize, BufferUsage, Color, ColorTargetState, ColorWrite, CommandBufferDescriptor,
    CompareFunction, DepthBiasState, DepthStencilState, DeviceType, DownlevelFlags,
    DownlevelProperties, DynamicOffset, Extent3d, Face, Features, FilterMode, FrontFace,
    ImageDataLayout, IndexFormat, InputStepMode, Limits, MultisampleState, Origin3d,
    PipelineStatisticsTypes, PolygonMode, PowerPreference, PresentMode, PrimitiveState,
    PrimitiveTopology, PushConstantRange, QuerySetDescriptor, QueryType, SamplerBorderColor,
    ShaderFlags, ShaderLocation, ShaderModel, ShaderStage, StencilFaceState, StencilOperation,
    StencilState, StorageTextureAccess, SwapChainDescriptor, SwapChainStatus, TextureAspect,
    TextureDimension, TextureFormat, TextureFormatFeatureFlags, TextureFormatFeatures,
    TextureSampleType, TextureUsage, TextureViewDimension, VertexAttribute, VertexFormat,
    BIND_BUFFER_ALIGNMENT, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT, MAP_ALIGNMENT,
    PUSH_CONSTANT_ALIGNMENT, QUERY_SET_MAX_QUERIES, QUERY_SIZE, VERTEX_STRIDE_ALIGNMENT,
};

use backend::{BufferMappedRange, Context as C};

trait ComputePassInner<Ctx: Context> {
    fn set_pipeline(&mut self, pipeline: &Ctx::ComputePipelineId);
    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &Ctx::BindGroupId,
        offsets: &[DynamicOffset],
    );
    fn set_push_constants(&mut self, offset: u32, data: &[u8]);
    fn insert_debug_marker(&mut self, label: &str);
    fn push_debug_group(&mut self, group_label: &str);
    fn pop_debug_group(&mut self);
    fn write_timestamp(&mut self, query_set: &Ctx::QuerySetId, query_index: u32);
    fn begin_pipeline_statistics_query(&mut self, query_set: &Ctx::QuerySetId, query_index: u32);
    fn end_pipeline_statistics_query(&mut self);
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
    fn set_index_buffer(
        &mut self,
        buffer: &Ctx::BufferId,
        index_format: IndexFormat,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Ctx::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    );
    fn set_push_constants(&mut self, stages: wgt::ShaderStage, offset: u32, data: &[u8]);
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(&mut self, indirect_buffer: &Ctx::BufferId, indirect_offset: BufferAddress);
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
    );
    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
        count: u32,
    );
    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
        count_buffer: &Ctx::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
        count_buffer: &Ctx::BufferId,
        count_buffer_offset: BufferAddress,
        max_count: u32,
    );
}

trait RenderPassInner<Ctx: Context>: RenderInner<Ctx> {
    fn set_blend_constant(&mut self, color: Color);
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
    fn write_timestamp(&mut self, query_set: &Ctx::QuerySetId, query_index: u32);
    fn begin_pipeline_statistics_query(&mut self, query_set: &Ctx::QuerySetId, query_index: u32);
    fn end_pipeline_statistics_query(&mut self);
    fn execute_bundles<'a, I: Iterator<Item = &'a Ctx::RenderBundleId>>(
        &mut self,
        render_bundles: I,
    );
}

trait Context: Debug + Send + Sized + Sync {
    type AdapterId: Debug + Send + Sync + 'static;
    type DeviceId: Debug + Send + Sync + 'static;
    type QueueId: Debug + Send + Sync + 'static;
    type ShaderModuleId: Debug + Send + Sync + 'static;
    type BindGroupLayoutId: Debug + Send + Sync + 'static;
    type BindGroupId: Debug + Send + Sync + 'static;
    type TextureViewId: Debug + Send + Sync + 'static;
    type SamplerId: Debug + Send + Sync + 'static;
    type BufferId: Debug + Send + Sync + 'static;
    type TextureId: Debug + Send + Sync + 'static;
    type QuerySetId: Debug + Send + Sync + 'static;
    type PipelineLayoutId: Debug + Send + Sync + 'static;
    type RenderPipelineId: Debug + Send + Sync + 'static;
    type ComputePipelineId: Debug + Send + Sync + 'static;
    type CommandEncoderId: Debug;
    type ComputePassId: Debug + ComputePassInner<Self>;
    type RenderPassId: Debug + RenderPassInner<Self>;
    type CommandBufferId: Debug + Send + Sync;
    type RenderBundleEncoderId: Debug + RenderInner<Self>;
    type RenderBundleId: Debug + Send + Sync + 'static;
    type SurfaceId: Debug + Send + Sync + 'static;
    type SwapChainId: Debug + Send + Sync + 'static;

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
    ) -> Self::RequestAdapterFuture;
    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture;
    fn instance_poll_all_devices(&self, force_wait: bool);
    fn adapter_get_swap_chain_preferred_format(
        &self,
        adapter: &Self::AdapterId,
        surface: &Self::SurfaceId,
    ) -> Option<TextureFormat>;
    fn adapter_features(&self, adapter: &Self::AdapterId) -> Features;
    fn adapter_limits(&self, adapter: &Self::AdapterId) -> Limits;
    fn adapter_downlevel_properties(&self, adapter: &Self::AdapterId) -> DownlevelProperties;
    fn adapter_get_info(&self, adapter: &Self::AdapterId) -> AdapterInfo;
    fn adapter_get_texture_format_features(
        &self,
        adapter: &Self::AdapterId,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures;

    fn device_features(&self, device: &Self::DeviceId) -> Features;
    fn device_limits(&self, device: &Self::DeviceId) -> Limits;
    fn device_downlevel_properties(&self, device: &Self::DeviceId) -> DownlevelProperties;
    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &SwapChainDescriptor,
    ) -> Self::SwapChainId;
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        desc: &ShaderModuleDescriptor,
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
    fn device_create_query_set(
        &self,
        device: &Self::DeviceId,
        desc: &QuerySetDescriptor,
    ) -> Self::QuerySetId;
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
    fn device_on_uncaptured_error(
        &self,
        device: &Self::DeviceId,
        handler: impl UncapturedErrorHandler,
    );

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        mode: MapMode,
        range: Range<BufferAddress>,
    ) -> Self::MapAsyncFuture;
    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<BufferAddress>,
    ) -> BufferMappedRange;
    fn buffer_unmap(&self, buffer: &Self::BufferId);
    fn swap_chain_get_current_texture_view(
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
        desc: &TextureViewDescriptor,
    ) -> Self::TextureViewId;

    fn surface_drop(&self, surface: &Self::SurfaceId);
    fn adapter_drop(&self, adapter: &Self::AdapterId);
    fn buffer_destroy(&self, buffer: &Self::BufferId);
    fn buffer_drop(&self, buffer: &Self::BufferId);
    fn texture_destroy(&self, buffer: &Self::TextureId);
    fn texture_drop(&self, texture: &Self::TextureId);
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId);
    fn sampler_drop(&self, sampler: &Self::SamplerId);
    fn query_set_drop(&self, query_set: &Self::QuerySetId);
    fn bind_group_drop(&self, bind_group: &Self::BindGroupId);
    fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId);
    fn pipeline_layout_drop(&self, pipeline_layout: &Self::PipelineLayoutId);
    fn shader_module_drop(&self, shader_module: &Self::ShaderModuleId);
    fn command_encoder_drop(&self, command_encoder: &Self::CommandEncoderId);
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId);
    fn render_bundle_drop(&self, render_bundle: &Self::RenderBundleId);
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId);
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId);

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::ComputePipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId;
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::RenderPipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId;

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
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    );
    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    );

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &ComputePassDescriptor,
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
    fn command_encoder_finish(&self, encoder: Self::CommandEncoderId) -> Self::CommandBufferId;

    fn command_encoder_insert_debug_marker(&self, encoder: &Self::CommandEncoderId, label: &str);
    fn command_encoder_push_debug_group(&self, encoder: &Self::CommandEncoderId, label: &str);
    fn command_encoder_pop_debug_group(&self, encoder: &Self::CommandEncoderId);

    fn command_encoder_write_timestamp(
        &self,
        encoder: &Self::CommandEncoderId,
        query_set: &Self::QuerySetId,
        query_index: u32,
    );
    fn command_encoder_resolve_query_set(
        &self,
        encoder: &Self::CommandEncoderId,
        query_set: &Self::QuerySetId,
        first_query: u32,
        query_count: u32,
        destination: &Self::BufferId,
        destination_offset: BufferAddress,
    );

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
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    );
    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    );
    fn queue_get_timestamp_period(&self, queue: &Self::QueueId) -> f32;

    fn device_start_capture(&self, device: &Self::DeviceId);
    fn device_stop_capture(&self, device: &Self::DeviceId);
}

/// Context for all other wgpu objects. Instance of wgpu.
///
/// This is the first thing you create when using wgpu.
/// Its primary use is to create [`Adapter`]s and [`Surface`]s.
///
/// Does not have to be kept alive.
#[derive(Debug)]
pub struct Instance {
    context: Arc<C>,
}

/// Handle to a physical graphics and/or compute device.
///
/// Adapters can be used to open a connection to the corresponding [`Device`]
/// on the host system by using [`Adapter::request_device`].
///
/// Does not have to be kept alive.
#[derive(Debug)]
pub struct Adapter {
    context: Arc<C>,
    id: <C as Context>::AdapterId,
}

impl Drop for Adapter {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.adapter_drop(&self.id)
        }
    }
}

/// Open connection to a graphics and/or compute device.
///
/// Responsible for the creation of most rendering and compute resources.
/// These are then used in commands, which are submitted to a [`Queue`].
///
/// A device may be requested from an adapter with [`Adapter::request_device`].
#[derive(Debug)]
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

/// The main purpose of this struct is to resolve mapped ranges (convert sizes
/// to end points), and to ensure that the sub-ranges don't intersect.
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

    fn add(&mut self, offset: BufferAddress, size: Option<BufferSize>) -> BufferAddress {
        let end = match size {
            Some(s) => offset + s.get(),
            None => self.initial_range.end,
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

    fn remove(&mut self, offset: BufferAddress, size: Option<BufferSize>) {
        let end = match size {
            Some(s) => offset + s.get(),
            None => self.initial_range.end,
        };

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
/// Created with [`Device::create_buffer`] or [DeviceExt::create_buffer_init](util::DeviceExt::create_buffer_init)
#[derive(Debug)]
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
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: Option<BufferSize>,
}

/// Handle to a texture on the GPU.
///
/// Created by calling [`Device::create_texture`]
#[derive(Debug)]
pub struct Texture {
    context: Arc<C>,
    id: <C as Context>::TextureId,
    owned: bool,
}

/// Handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
pub struct Surface {
    context: Arc<C>,
    id: <C as Context>::SurfaceId,
}

impl Drop for Surface {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.surface_drop(&self.id)
        }
    }
}

/// Handle to a swap chain.
///
/// A `SwapChain` represents the image or series of images that will be presented to a [`Surface`].
/// A `SwapChain` may be created with [`Device::create_swap_chain`].
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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
pub enum ShaderSource<'a> {
    /// SPIR-V module represented as a slice of words.
    ///
    /// wgpu will attempt to parse and validate it, but the original binary
    /// is passed to `gfx-rs` and `spirv_cross` for translation.
    SpirV(Cow<'a, [u32]>),
    /// WGSL module as a string slice.
    ///
    /// wgpu-rs will parse it and use for validation. It will attempt
    /// to build a SPIR-V module internally and panic otherwise.
    ///
    /// Note: WGSL is not yet supported on the Web.
    Wgsl(Cow<'a, str>),
}

/// Descriptor for a shader module.
pub struct ShaderModuleDescriptor<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Source code for the shader.
    pub source: ShaderSource<'a>,
    /// Shader handling flags.
    pub flags: ShaderFlags,
}

/// Handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
#[derive(Debug)]
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
#[derive(Debug)]
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

impl RenderPipeline {
    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        BindGroupLayout {
            context,
            id: self
                .context
                .render_pipeline_get_bind_group_layout(&self.id, index),
        }
    }
}

/// Handle to a compute pipeline.
///
/// A `ComputePipeline` object represents a compute pipeline and its single shader stage.
/// A `ComputePipeline` may be created with [`Device::create_compute_pipeline`].
#[derive(Debug)]
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

impl ComputePipeline {
    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        BindGroupLayout {
            context,
            id: self
                .context
                .compute_pipeline_get_bind_group_layout(&self.id, index),
        }
    }
}

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
#[derive(Debug)]
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
/// A command encoder can record [`RenderPass`]es, [`ComputePass`]es,
/// and transfer operations between driver-managed resources like [`Buffer`]s and [`Texture`]s.
///
/// When finished recording, call [`CommandEncoder::finish`] to obtain a [`CommandBuffer`] which may
/// be submitted for execution.
#[derive(Debug)]
pub struct CommandEncoder {
    context: Arc<C>,
    id: Option<<C as Context>::CommandEncoderId>,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: PhantomData<*const u8>,
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if !thread::panicking() {
            if let Some(id) = self.id.take() {
                self.context.command_encoder_drop(&id);
            }
        }
    }
}

/// In-progress recording of a render pass.
#[derive(Debug)]
pub struct RenderPass<'a> {
    id: <C as Context>::RenderPassId,
    parent: &'a mut CommandEncoder,
}

/// In-progress recording of a compute pass.
#[derive(Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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

/// Handle to a query set.
pub struct QuerySet {
    context: Arc<C>,
    id: <C as Context>::QuerySetId,
}

impl Drop for QuerySet {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.query_set_drop(&self.id);
        }
    }
}

/// Handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects and provides convenience methods
/// for writing to [buffers](Queue::write_buffer) and [textures](Queue::write_texture).
#[derive(Debug)]
pub struct Queue {
    context: Arc<C>,
    id: <C as Context>::QueueId,
}

/// Resource that can be bound to a pipeline.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum BindingResource<'a> {
    /// Binding is backed by a buffer.
    ///
    /// Corresponds to [`wgt::BufferBindingType::Uniform`] and [`wgt::BufferBindingType::Storage`]
    /// with [`BindGroupLayoutEntry::count`] set to None.
    Buffer(BufferBinding<'a>),
    /// Binding is backed by an array of buffers.
    ///
    /// [`Features::BUFFER_BINDING_ARRAY`] must be supported to use this feature.
    ///
    /// Corresponds to [`wgt::BufferBindingType::Uniform`] and [`wgt::BufferBindingType::Storage`]
    /// with [`BindGroupLayoutEntry::count`] set to Some.
    BufferArray(&'a [BufferBinding<'a>]),
    /// Binding is a sampler.
    ///
    /// Corresponds to [`wgt::BindingType::Sampler`] with [`BindGroupLayoutEntry::count`] set to None.
    Sampler(&'a Sampler),
    /// Binding is backed by a texture.
    ///
    /// Corresponds to [`wgt::BindingType::Texture`] and [`wgt::BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to None.
    TextureView(&'a TextureView),
    /// Binding is backed by an array of textures.
    ///
    /// [`Features::SAMPLED_TEXTURE_BINDING_ARRAY`] must be supported to use this feature.
    ///
    /// Corresponds to [`wgt::BindingType::Texture`] and [`wgt::BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to Some.
    TextureViewArray(&'a [&'a TextureView]),
}

/// Describes the segment of a buffer to bind.
#[derive(Clone, Debug)]
pub struct BufferBinding<'a> {
    /// The buffer to bind.
    pub buffer: &'a Buffer,
    /// Base offset of the buffer. For bindings with `dynamic == true`, this offset
    /// will be added to the dynamic offset provided in [`RenderPass::set_bind_group`].
    ///
    /// The offset has to be aligned to [`BIND_BUFFER_ALIGNMENT`].
    pub offset: BufferAddress,
    /// Size of the binding, or `None` for using the rest of the buffer.
    pub size: Option<BufferSize>,
}

/// Operation to perform to the output attachment at the start of a renderpass.
///
/// The render target must be cleared at least once before its content is loaded.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum LoadOp<V> {
    /// Clear with a specified value.
    Clear(V),
    /// Load from memory.
    Load,
}

impl<V: Default> Default for LoadOp<V> {
    fn default() -> Self {
        Self::Clear(Default::default())
    }
}

/// Pair of load and store operations for an attachment aspect.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct Operations<V> {
    /// How data should be read through this attachment.
    pub load: LoadOp<V>,
    /// Whether data will be written to through this attachment.
    pub store: bool,
}

impl<V: Default> Default for Operations<V> {
    fn default() -> Self {
        Self {
            load: Default::default(),
            store: true,
        }
    }
}

/// Describes a color attachment to a [`RenderPass`].
#[derive(Clone, Debug)]
pub struct RenderPassColorAttachment<'a> {
    /// The view to use as an attachment.
    pub view: &'a TextureView,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<&'a TextureView>,
    /// What operations will be performed on this color attachment.
    pub ops: Operations<Color>,
}

/// Describes a depth/stencil attachment to a [`RenderPass`].
#[derive(Clone, Debug)]
pub struct RenderPassDepthStencilAttachment<'a> {
    /// The view to use as an attachment.
    pub view: &'a TextureView,
    /// What operations will be performed on the depth part of the attachment.
    pub depth_ops: Option<Operations<f32>>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil_ops: Option<Operations<u32>>,
}

// The underlying types are also exported so that documentation shows up for them

/// Object label.
pub type Label<'a> = Option<&'a str>;
pub use wgt::RequestAdapterOptions as RequestAdapterOptionsBase;
/// Additional information required when requesting an adapter.
pub type RequestAdapterOptions<'a> = RequestAdapterOptionsBase<&'a Surface>;
/// Describes a [`Device`].
pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
/// Describes a [`Buffer`].
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
/// Describes a [`CommandEncoder`].
pub type CommandEncoderDescriptor<'a> = wgt::CommandEncoderDescriptor<Label<'a>>;
/// Describes a [`RenderBundle`].
pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
/// Describes a [`Texture`].
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>>;

/// Describes a [`TextureView`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view. At this time, it must be the same as the underlying format of the texture.
    pub format: Option<TextureFormat>,
    /// The dimension of the texture view. For 1D textures, this must be `1D`. For 2D textures it must be one of
    /// `D2`, `D2Array`, `Cube`, and `CubeArray`. For 3D textures it must be `3D`
    pub dimension: Option<TextureViewDimension>,
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`].
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<NonZeroU32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<NonZeroU32>,
}

/// Describes a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be used to create a pipeline layout.
#[derive(Clone, Debug, Default)]
pub struct PipelineLayoutDescriptor<'a> {
    /// Debug label of the pipeline layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Bind groups that this pipeline uses. The first entry will provide all the bindings for
    /// "set = 0", second entry will provide all the bindings for "set = 1" etc.
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
    /// Set of push constant ranges this pipeline uses. Each shader stage that uses push constants
    /// must define the range in push constant memory that corresponds to its single `layout(push_constant)`
    /// uniform block.
    ///
    /// If this array is non-empty, the [`Features::PUSH_CONSTANTS`] must be enabled.
    pub push_constant_ranges: &'a [PushConstantRange],
}

/// Describes a [`Sampler`]
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerDescriptor<'a> {
    /// Debug label of the sampler. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_mode_u: AddressMode,
    /// How to deal with out of bounds accesses in the v (i.e. y) direction
    pub address_mode_v: AddressMode,
    /// How to deal with out of bounds accesses in the w (i.e. z) direction
    pub address_mode_w: AddressMode,
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: FilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<CompareFunction>,
    /// Valid values: 1, 2, 4, 8, and 16.
    pub anisotropy_clamp: Option<NonZeroU8>,
    /// Border color to use when address_mode is [`AddressMode::ClampToBorder`]
    pub border_color: Option<SamplerBorderColor>,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: Default::default(),
            address_mode_v: Default::default(),
            address_mode_w: Default::default(),
            mag_filter: Default::default(),
            min_filter: Default::default(),
            mipmap_filter: Default::default(),
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        }
    }
}

/// Bindable resource and the slot to bind it to.
#[derive(Clone, Debug)]
pub struct BindGroupEntry<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource<'a>,
}

/// Describes a group of bindings and the resources to be bound.
#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: &'a BindGroupLayout,
    /// The resources to bind to this bind group.
    pub entries: &'a [BindGroupEntry<'a>],
}

/// Describes the attachments of a render pass.
///
/// Note: separate lifetimes are needed because the texture views
/// have to live as long as the pass is recorded, while everything else doesn't.
#[derive(Clone, Debug, Default)]
pub struct RenderPassDescriptor<'a, 'b> {
    /// Debug label of the render pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments: &'b [RenderPassColorAttachment<'a>],
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
}

/// Describes how the vertex buffer is interpreted.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: InputStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [VertexAttribute],
}

/// Describes the vertex process in a render pipeline.
#[derive(Clone, Debug)]
pub struct VertexState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}

/// Describes the fragment process in a render pipeline.
#[derive(Clone, Debug)]
pub struct FragmentState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The color state of the render targets.
    pub targets: &'a [ColorTargetState],
}

/// Describes a render (graphics) pipeline.
#[derive(Clone, Debug)]
pub struct RenderPipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// The compiled vertex stage, its entry point, and the input buffers layout.
    pub vertex: VertexState<'a>,
    /// The properties of the pipeline at the primitive assembly and rasterization level.
    pub primitive: PrimitiveState,
    /// The effect of draw calls on the depth and stencil aspects of the output target, if any.
    pub depth_stencil: Option<DepthStencilState>,
    /// The multi-sampling properties of the pipeline.
    pub multisample: MultisampleState,
    /// The compiled fragment stage, its entry point, and the color targets.
    pub fragment: Option<FragmentState<'a>>,
}

/// Describes the attachments of a compute pass.
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a> {
    /// Debug label of the compute pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
}

/// Describes a compute pipeline.
#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
}

pub use wgt::ImageCopyBuffer as ImageCopyBufferBase;
/// View of a buffer which can be used to copy to/from a texture.
pub type ImageCopyBuffer<'a> = ImageCopyBufferBase<&'a Buffer>;

pub use wgt::ImageCopyTexture as ImageCopyTextureBase;
/// View of a texture which can be used to copy to/from a buffer/texture.
pub type ImageCopyTexture<'a> = ImageCopyTextureBase<&'a Texture>;

/// Describes a [`BindGroupLayout`].
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,

    /// Array of entries in this BindGroupLayout
    pub entries: &'a [BindGroupLayoutEntry],
}

/// Describes a [`RenderBundleEncoder`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RenderBundleEncoderDescriptor<'a> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The formats of the color attachments that this render bundle is capable to rendering to. This
    /// must match the formats of the color attachments in the renderpass this render bundle is executed in.
    pub color_formats: &'a [TextureFormat],
    /// The formats of the depth attachment that this render bundle is capable to rendering to. This
    /// must match the formats of the depth attachments in the renderpass this render bundle is executed in.
    pub depth_stencil_format: Option<TextureFormat>,
    /// Sample count this render bundle is capable of rendering to. This must match the pipelines and
    /// the renderpasses it is used in.
    pub sample_count: u32,
}

/// Swap chain image that can be rendered to.
#[derive(Debug)]
pub struct SwapChainTexture {
    /// Accessible view of the frame.
    pub view: TextureView,
    detail: <C as Context>::SwapChainOutputDetail,
}

/// Result of a successful call to [`SwapChain::get_current_frame`].
#[derive(Debug)]
pub struct SwapChainFrame {
    /// The texture into which the next frame should be rendered.
    pub output: SwapChainTexture,
    /// `true` if the acquired buffer can still be used for rendering,
    /// but should be recreated for maximum performance.
    pub suboptimal: bool,
}

/// Result of an unsuccessful call to [`SwapChain::get_current_frame`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SwapChainError {
    /// A timeout was encountered while trying to acquire the next frame.
    Timeout,
    /// The underlying surface has changed, and therefore the swap chain must be updated.
    Outdated,
    /// The swap chain has been lost and needs to be recreated.
    Lost,
    /// There is no more memory left to allocate a new frame.
    OutOfMemory,
}

impl Display for SwapChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::Timeout => "A timeout was encountered while trying to acquire the next frame",
            Self::Outdated => "The underlying surface has changed, and therefore the swap chain must be updated",
            Self::Lost =>  "The swap chain has been lost and needs to be recreated",
            Self::OutOfMemory => "There is no more memory left to allocate a new frame",
        })
    }
}

impl error::Error for SwapChainError {}

impl Instance {
    /// Create an new instance of wgpu.
    ///
    /// # Arguments
    ///
    /// - `backends` - Controls from which [backends][BackendBit] wgpu will choose
    ///   during instantiation.
    pub fn new(backends: BackendBit) -> Self {
        Instance {
            context: Arc::new(C::init(backends)),
        }
    }

    /// Retrieves all available [`Adapter`]s that match the given [`BackendBit`].
    ///
    /// # Arguments
    ///
    /// - `backends` - Backends from which to enumerate adapters.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enumerate_adapters(&self, backends: BackendBit) -> impl Iterator<Item = Adapter> {
        let context = Arc::clone(&self.context);
        self.context
            .enumerate_adapters(backends)
            .into_iter()
            .map(move |id| crate::Adapter {
                id,
                context: Arc::clone(&context),
            })
    }

    /// Retrieves an [`Adapter`] which matches the given [`RequestAdapterOptions`].
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that suffice all the "hard" options, `None` is returned.
    pub fn request_adapter(
        &self,
        options: &RequestAdapterOptions,
    ) -> impl Future<Output = Option<Adapter>> + Send {
        let context = Arc::clone(&self.context);
        let adapter = self.context.instance_request_adapter(options);
        async move { adapter.await.map(|id| Adapter { context, id }) }
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
            context: Arc::clone(&self.context),
            id: Context::instance_create_surface(&*self.context, window),
        }
    }

    /// Creates a surface from `CoreAnimationLayer`.
    ///
    /// # Safety
    ///
    /// - layer must be a valid object to create a surface upon.
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> Surface {
        self.context.create_surface_from_core_animation_layer(layer)
    }

    /// Polls all devices.
    /// If `force_wait` is true and this is not running on the web,
    /// then this function will block until all in-flight buffers have been mapped.
    pub fn poll_all(&self, force_wait: bool) {
        self.context.instance_poll_all_devices(force_wait);
    }
}

impl Adapter {
    /// Requests a connection to a physical device, creating a logical device.
    ///
    /// Returns the [`Device`] together with a [`Queue`] that executes command buffers.
    ///
    /// # Arguments
    ///
    /// - `desc` - Description of the features and limits requested from the given device.
    /// - `trace_path` - Can be used for API call tracing, if that feature is
    ///   enabled in `wgpu-core`.
    ///
    /// # Panics
    ///
    /// - Features specified by `desc` are not supported by this adapter.
    /// - Unsafe features were requested but not enabled when requesting the adapter.
    /// - Limits requested exceed the values provided by the adapter.
    /// - Adapter does not support all features wgpu requires to safely operate.
    pub fn request_device(
        &self,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> impl Future<Output = Result<(Device, Queue), RequestDeviceError>> + Send {
        let context = Arc::clone(&self.context);
        let device = Context::adapter_request_device(&*self.context, &self.id, desc, trace_path);
        async move {
            device.await.map(|(device_id, queue_id)| {
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
        }
    }

    /// Returns an optimal texture format to use for the [`SwapChain`] with this adapter.
    ///
    /// Returns None if the surface is incompatible with the adapter.
    pub fn get_swap_chain_preferred_format(&self, surface: &Surface) -> Option<TextureFormat> {
        Context::adapter_get_swap_chain_preferred_format(&*self.context, &self.id, &surface.id)
    }

    /// List all features that are supported with this adapter.
    ///
    /// Features must be explicitly requested in [`Adapter::request_device`] in order
    /// to use them.
    pub fn features(&self) -> Features {
        Context::adapter_features(&*self.context, &self.id)
    }

    /// List the "best" limits that are supported by this adapter.
    ///
    /// Limits must be explicitly requested in [`Adapter::request_device`] to set
    /// the values that you are allowed to use.
    pub fn limits(&self) -> Limits {
        Context::adapter_limits(&*self.context, &self.id)
    }

    /// Get info about the adapter itself.
    pub fn get_info(&self) -> AdapterInfo {
        Context::adapter_get_info(&*self.context, &self.id)
    }

    /// Returns the features supported for a given texture format by this adapter.
    ///
    /// Note that the WebGPU spec further restricts the available usages/features.
    /// To disable these restrictions on a device, request the [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] feature.
    pub fn get_texture_format_features(
        &self,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        Context::adapter_get_texture_format_features(&*self.context, &self.id, format)
    }
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    ///
    /// no-op on the web, device is automatically polled.
    pub fn poll(&self, maintain: Maintain) {
        Context::device_poll(&*self.context, &self.id, maintain);
    }

    /// List all features that may be used with this device.
    ///
    /// Functions may panic if you use unsupported features.
    pub fn features(&self) -> Features {
        Context::device_features(&*self.context, &self.id)
    }

    /// List all limits that were requested of this device.
    ///
    /// If any of these limits are exceeded, functions may panic.
    pub fn limits(&self) -> Limits {
        Context::device_limits(&*self.context, &self.id)
    }

    /// Creates a shader module from either SPIR-V or WGSL source code.
    pub fn create_shader_module(&self, desc: &ShaderModuleDescriptor) -> ShaderModule {
        ShaderModule {
            context: Arc::clone(&self.context),
            id: Context::device_create_shader_module(&*self.context, &self.id, desc),
        }
    }

    /// Creates an empty [`CommandEncoder`].
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        CommandEncoder {
            context: Arc::clone(&self.context),
            id: Some(Context::device_create_command_encoder(
                &*self.context,
                &self.id,
                desc,
            )),
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

    /// Creates a new [`QuerySet`].
    pub fn create_query_set(&self, desc: &QuerySetDescriptor) -> QuerySet {
        QuerySet {
            context: Arc::clone(&self.context),
            id: Context::device_create_query_set(&*self.context, &self.id, desc),
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

    /// Set a callback for errors that are not handled in error scopes.
    pub fn on_uncaptured_error(&self, handler: impl UncapturedErrorHandler) {
        self.context.device_on_uncaptured_error(&self.id, handler);
    }

    /// Starts frame capture.
    pub fn start_capture(&self) {
        Context::device_start_capture(&*self.context, &self.id)
    }

    /// Stops frame capture.
    pub fn stop_capture(&self) {
        Context::device_stop_capture(&*self.context, &self.id)
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

impl Display for RequestDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Requesting a device failed")
    }
}

impl error::Error for RequestDeviceError {}

/// Error occurred when trying to async map a buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;

impl Display for BufferAsyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error occurred when trying to async map a buffer")
    }
}

impl error::Error for BufferAsyncError {}

/// Type of buffer mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapMode {
    /// Map only for reading
    Read,
    /// Map only for writing
    Write,
}

fn range_to_offset_size<S: RangeBounds<BufferAddress>>(
    bounds: S,
) -> (BufferAddress, Option<BufferSize>) {
    let offset = match bounds.start_bound() {
        Bound::Included(&bound) => bound,
        Bound::Excluded(&bound) => bound + 1,
        Bound::Unbounded => 0,
    };
    let size = match bounds.end_bound() {
        Bound::Included(&bound) => Some(bound + 1 - offset),
        Bound::Excluded(&bound) => Some(bound - offset),
        Bound::Unbounded => None,
    }
    .map(|size| BufferSize::new(size).expect("Buffer slices can not be empty"));

    (offset, size)
}

#[cfg(test)]
mod tests {
    use crate::BufferSize;

    #[test]
    fn range_to_offset_size_works() {
        assert_eq!(crate::range_to_offset_size(0..2), (0, BufferSize::new(2)));
        assert_eq!(crate::range_to_offset_size(2..5), (2, BufferSize::new(3)));
        assert_eq!(crate::range_to_offset_size(..), (0, None));
        assert_eq!(crate::range_to_offset_size(21..), (21, None));
        assert_eq!(crate::range_to_offset_size(0..), (0, None));
        assert_eq!(crate::range_to_offset_size(..21), (0, BufferSize::new(21)));
    }

    #[test]
    #[should_panic]
    fn range_to_offset_size_panics_for_empty_range() {
        crate::range_to_offset_size(123..123);
    }

    #[test]
    #[should_panic]
    fn range_to_offset_size_panics_for_unbounded_empty_range() {
        crate::range_to_offset_size(..0);
    }
}

trait BufferMappedRangeSlice {
    fn slice(&self) -> &[u8];
    fn slice_mut(&mut self) -> &mut [u8];
}

/// Read only view into a mapped buffer.
#[derive(Debug)]
pub struct BufferView<'a> {
    slice: BufferSlice<'a>,
    data: BufferMappedRange,
}

/// Write only view into mapped buffer.
#[derive(Debug)]
pub struct BufferViewMut<'a> {
    slice: BufferSlice<'a>,
    data: BufferMappedRange,
    readable: bool,
}

impl std::ops::Deref for BufferView<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.data.slice()
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
        self.data.slice()
    }
}

impl std::ops::DerefMut for BufferViewMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.slice_mut()
    }
}

impl AsRef<[u8]> for BufferView<'_> {
    fn as_ref(&self) -> &[u8] {
        self.data.slice()
    }
}

impl AsMut<[u8]> for BufferViewMut<'_> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.slice_mut()
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
    /// Return the binding view of the entire buffer.
    pub fn as_entire_binding(&self) -> BindingResource {
        BindingResource::Buffer(self.as_entire_buffer_binding())
    }

    /// Return the binding view of the entire buffer.
    pub fn as_entire_buffer_binding(&self) -> BufferBinding {
        BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        }
    }

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

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        Context::buffer_destroy(&*self.context, &self.id);
    }
}

impl<'a> BufferSlice<'a> {
    //TODO: fn slice(&self) -> Self

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
        let mut mc = self.buffer.map_context.lock();
        assert_eq!(
            mc.initial_range,
            0..0,
            "Buffer {:?} is already mapped",
            self.buffer.id
        );
        let end = match self.size {
            Some(s) => self.offset + s.get(),
            None => mc.total_size,
        };
        mc.initial_range = self.offset..end;

        Context::buffer_map_async(
            &*self.buffer.context,
            &self.buffer.id,
            mode,
            self.offset..end,
        )
    }

    /// Synchronously and immediately map a buffer for reading. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will panic.
    pub fn get_mapped_range(&self) -> BufferView<'a> {
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
    pub fn get_mapped_range_mut(&self) -> BufferViewMut<'a> {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = Context::buffer_get_mapped_range(
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
            id: Context::texture_create_view(&*self.context, &self.id, desc),
            owned: true,
        }
    }

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        Context::texture_destroy(&*self.context, &self.id);
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
    pub fn finish(mut self) -> CommandBuffer {
        CommandBuffer {
            context: Arc::clone(&self.context),
            id: Some(Context::command_encoder_finish(
                &*self.context,
                self.id.take().unwrap(),
            )),
        }
    }

    /// Begins recording of a render pass.
    ///
    /// This function returns a [`RenderPass`] object which records a single render pass.
    pub fn begin_render_pass<'a>(
        &'a mut self,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> RenderPass<'a> {
        let id = self.id.as_ref().unwrap();
        RenderPass {
            id: Context::command_encoder_begin_render_pass(&*self.context, id, desc),
            parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor) -> ComputePass {
        let id = self.id.as_ref().unwrap();
        ComputePass {
            id: Context::command_encoder_begin_compute_pass(&*self.context, id, desc),
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
            self.id.as_ref().unwrap(),
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
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_buffer_to_texture(
            &*self.context,
            self.id.as_ref().unwrap(),
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
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_texture_to_buffer(
            &*self.context,
            self.id.as_ref().unwrap(),
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
        source: ImageCopyTexture,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        Context::command_encoder_copy_texture_to_texture(
            &*self.context,
            self.id.as_ref().unwrap(),
            source,
            destination,
            copy_size,
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        let id = self.id.as_ref().unwrap();
        Context::command_encoder_insert_debug_marker(&*self.context, id, label);
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        let id = self.id.as_ref().unwrap();
        Context::command_encoder_push_debug_group(&*self.context, id, label);
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        let id = self.id.as_ref().unwrap();
        Context::command_encoder_pop_debug_group(&*self.context, id);
    }
}

/// [`Features::TIMESTAMP_QUERY`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Issue a timestamp command at this point in the queue.
    /// The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Device::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        Context::command_encoder_write_timestamp(
            &*self.context,
            self.id.as_ref().unwrap(),
            &query_set.id,
            query_index,
        )
    }
}

/// [`Features::TIMESTAMP_QUERY`] or [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Resolve a query set, writing the results into the supplied destination buffer.
    ///
    /// Queries may be between 8 and 40 bytes each. See [`PipelineStatisticsType`] for more information.
    pub fn resolve_query_set(
        &mut self,
        query_set: &QuerySet,
        query_range: Range<u32>,
        destination: &Buffer,
        destination_offset: BufferAddress,
    ) {
        Context::command_encoder_resolve_query_set(
            &*self.context,
            self.id.as_ref().unwrap(),
            &query_set.id,
            query_range.start,
            query_range.end - query_range.start,
            &destination.id,
            destination_offset,
        )
    }
}

impl<'a> RenderPass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    /// These offsets have to be aligned to [`BIND_BUFFER_ALIGNMENT`].
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
    pub fn set_blend_constant(&mut self, color: Color) {
        self.id.set_blend_constant(color)
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        RenderInner::set_index_buffer(
            &mut self.id,
            &buffer_slice.buffer.id,
            index_format,
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

    /// Execute a [render bundle][RenderBundle], which is a set of pre-recorded commands
    /// that can be run together.
    pub fn execute_bundles<I: Iterator<Item = &'a RenderBundle>>(&mut self, render_bundles: I) {
        self.id
            .execute_bundles(render_bundles.into_iter().map(|rb| &rb.id))
    }
}

/// [`Features::MULTI_DRAW_INDIRECT`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Dispatches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// `count` draw calls are issued.
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
    ///
    /// These draw structures are expected to be tightly packed.
    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        self.id
            .multi_draw_indirect(&indirect_buffer.id, indirect_offset, count);
    }

    /// Dispatches multiple draw calls from the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`. `count` draw calls are issued.
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
    ///
    /// These draw structures are expected to be tightly packed.
    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        self.id
            .multi_draw_indexed_indirect(&indirect_buffer.id, indirect_offset, count);
    }
}

/// [`Features::MULTI_DRAW_INDIRECT_COUNT`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Disptaches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// The count buffer is read to determine how many draws to issue.
    ///
    /// The indirect buffer must be long enough to account for `max_count` draws, however only `count` will
    /// draws will be read. If `count` is greater than `max_count`, `max_count` will be used.
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
    ///
    /// These draw structures are expected to be tightly packed.
    ///
    /// The structure expected in `count_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndirectCount {
    ///     count: u32, // Number of draw calls to issue.
    /// }
    /// ```
    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &'a Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    ) {
        self.id.multi_draw_indirect_count(
            &indirect_buffer.id,
            indirect_offset,
            &count_buffer.id,
            count_offset,
            max_count,
        );
    }

    /// Dispatches multiple draw calls from the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`. The count buffer is read to determine how many draws to issue.
    ///
    /// The indirect buffer must be long enough to account for `max_count` draws, however only `count` will
    /// draws will be read. If `count` is greater than `max_count`, `max_count` will be used.
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
    ///
    /// These draw structures are expected to be tightly packed.
    ///
    /// The structure expected in `count_buffer` is the following:
    ///
    /// ```rust
    /// #[repr(C)]
    /// struct DrawIndexedIndirectCount {
    ///     count: u32, // Number of draw calls to issue.
    /// }
    /// ```
    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &'a Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    ) {
        self.id.multi_draw_indexed_indirect_count(
            &indirect_buffer.id,
            indirect_offset,
            &count_buffer.id,
            count_offset,
            max_count,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must be aligned to the 4s, so we take an array of u32.
    /// For example, with an offset of 4 and an array of `[u32; 3]`, that will write to the range
    /// of 4..16.
    ///
    /// For each byte in the range of push constant data written, the union of the stages of all push constant
    /// ranges that covers that byte must be exactly `stages`. There's no good way of explaining this simply,
    /// so here are some examples:
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..4 Vertex
    /// - 4..8 Fragment
    /// ```
    ///
    /// You would need to upload this in two set_push_constants calls. First for the `Vertex` range, second for the `Fragment` range.
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..8  Vertex
    /// - 4..12 Fragment
    /// ```
    ///
    /// You would need to upload this in three set_push_constants calls. First for the `Vertex` only range 0..4, second
    /// for the `Vertex | Fragment` range 4..8, third for the `Fragment` range 8..12.
    pub fn set_push_constants(&mut self, stages: wgt::ShaderStage, offset: u32, data: &[u8]) {
        self.id.set_push_constants(stages, offset, data);
    }
}

/// [`Features::TIMESTAMP_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Issue a timestamp command at this point in the queue. The
    /// timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Device::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        self.id.write_timestamp(&query_set.id, query_index)
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Start a pipeline statistics query on this render pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        self.id
            .begin_pipeline_statistics_query(&query_set.id, query_index);
    }

    /// End the pipeline statistics query on this render pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        self.id.end_pipeline_statistics_query();
    }
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            let parent_id = self.parent.id.as_ref().unwrap();
            self.parent
                .context
                .command_encoder_end_render_pass(parent_id, &mut self.id);
        }
    }
}

impl<'a> ComputePass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `dispatch()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in order of their declaration.
    /// These offsets have to be aligned to [`BIND_BUFFER_ALIGNMENT`].
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

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must be aligned to the 4s, so we take an array of u32.
    /// For example, with an offset of 4 and an array of `[u32; 3]`, that will write to the range
    /// of 4..16.
    pub fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
        self.id.set_push_constants(offset, data);
    }
}

/// [`Features::TIMESTAMP_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Issue a timestamp command at this point in the queue. The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Device::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        self.id.write_timestamp(&query_set.id, query_index)
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Start a pipeline statistics query on this render pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        self.id
            .begin_pipeline_statistics_query(&query_set.id, query_index);
    }

    /// End the pipeline statistics query on this render pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        self.id.end_pipeline_statistics_query();
    }
}

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            let parent_id = self.parent.id.as_ref().unwrap();
            self.parent
                .context
                .command_encoder_end_compute_pass(parent_id, &mut self.id);
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
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        RenderInner::set_index_buffer(
            &mut self.id,
            &buffer_slice.buffer.id,
            index_format,
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

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> RenderBundleEncoder<'a> {
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must be aligned to the 4s, so we take an array of u32.
    /// For example, with an offset of 4 and an array of `[u32; 3]`, that will write to the range
    /// of 4..16.
    ///
    /// For each byte in the range of push constant data written, the union of the stages of all push constant
    /// ranges that covers that byte must be exactly `stages`. There's no good way of explaining this simply,
    /// so here are some examples:
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..4 Vertex
    /// - 4..8 Fragment
    /// ```
    ///
    /// You would need to upload this in two set_push_constants calls. First for the `Vertex` range, second for the `Fragment` range.
    ///
    /// ```text
    /// For the given ranges:
    /// - 0..8  Vertex
    /// - 4..12 Fragment
    /// ```
    ///
    /// You would need to upload this in three set_push_constants calls. First for the `Vertex` only range 0..4, second
    /// for the `Vertex | Fragment` range 4..8, third for the `Fragment` range 8..12.
    pub fn set_push_constants(&mut self, stages: wgt::ShaderStage, offset: u32, data: &[u8]) {
        self.id.set_push_constants(stages, offset, data);
    }
}

impl Queue {
    /// Schedule a data write into `buffer` starting at `offset`.
    ///
    /// This method is intended to have low performance costs.
    /// As such, the write is not immediately submitted, and instead enqueued
    /// internally to happen at the start of the next `submit()` call.
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        Context::queue_write_buffer(&*self.context, &self.id, &buffer.id, offset, data)
    }

    /// Schedule a data write into `texture`.
    ///
    /// This method is intended to have low performance costs.
    /// As such, the write is not immediately submitted, and instead enqueued
    /// internally to happen at the start of the next `submit()` call.
    pub fn write_texture(
        &self,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
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

    /// Gets the amount of nanoseconds each tick of a timestamp query represents.
    ///
    /// Returns zero if timestamp queries are unsupported.
    pub fn get_timestamp_period(&self) -> f32 {
        Context::queue_get_timestamp_period(&*self.context, &self.id)
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
    pub fn get_current_frame(&self) -> Result<SwapChainFrame, SwapChainError> {
        let (view_id, status, detail) =
            Context::swap_chain_get_current_texture_view(&*self.context, &self.id);
        let output = view_id.map(|id| SwapChainTexture {
            view: TextureView {
                context: Arc::clone(&self.context),
                id,
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
        }
    }
}

/// Type for the callback of uncaptured error handler
pub trait UncapturedErrorHandler: Fn(Error) + Send + 'static {}
impl<T> UncapturedErrorHandler for T where T: Fn(Error) + Send + 'static {}

/// Error type
#[derive(Debug)]
pub enum Error {
    /// Out of memory error
    OutOfMemoryError {
        ///
        source: Box<dyn error::Error + Send + 'static>,
    },
    /// Validation error, signifying a bug in code or data
    ValidationError {
        ///
        source: Box<dyn error::Error + Send + 'static>,
        ///
        description: String,
    },
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::OutOfMemoryError { source } => Some(source.as_ref()),
            Error::ValidationError { source, .. } => Some(source.as_ref()),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OutOfMemoryError { .. } => f.write_str("Out of Memory"),
            Error::ValidationError { description, .. } => f.write_str(description),
        }
    }
}
