//! A cross-platform graphics and compute library based on WebGPU.

mod backend;

#[macro_use]
mod macros;

use futures::FutureExt as _;
use std::{
    future::Future,
    marker::PhantomData,
    ops::{Bound, Range, RangeBounds},
    sync::Arc,
    thread,
};

#[cfg(not(target_arch = "wasm32"))]
pub use wgc::instance::{AdapterInfo, DeviceType};
pub use wgt::{
    read_spirv, AddressMode, Backend, BackendBit, BlendDescriptor, BlendFactor, BlendOperation,
    BufferAddress, BufferUsage, Color, ColorStateDescriptor, ColorWrite, CommandBufferDescriptor,
    CompareFunction, CullMode, DepthStencilStateDescriptor, DeviceDescriptor, DynamicOffset,
    Extensions, Extent3d, FilterMode, FrontFace, IndexFormat, InputStepMode, Limits, LoadOp,
    Origin3d, PowerPreference, PresentMode, PrimitiveTopology, RasterizationStateDescriptor,
    ShaderLocation, ShaderStage, StencilOperation, StencilStateFaceDescriptor, StoreOp,
    SwapChainDescriptor, TextureAspect, TextureComponentType, TextureDimension, TextureFormat,
    TextureUsage, TextureViewDimension, VertexAttributeDescriptor, VertexFormat,
    BIND_BUFFER_ALIGNMENT, MAX_BIND_GROUPS,
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

trait RenderPassInner<Ctx: Context> {
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
        offset: BufferAddress,
        size: BufferAddress,
    );
    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &Ctx::BufferId,
        offset: BufferAddress,
        size: BufferAddress,
    );
    fn set_blend_color(&mut self, color: wgt::Color);
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
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indirect(&mut self, indirect_buffer: &Ctx::BufferId, indirect_offset: BufferAddress);
    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Ctx::BufferId,
        indirect_offset: BufferAddress,
    );
}

trait Context: Sized {
    type AdapterId: Send + Sync;
    type DeviceId: Send + Sync;
    type QueueId: Send + Sync;
    type ShaderModuleId: Send + Sync;
    type BindGroupLayoutId: Send + Sync;
    type BindGroupId: Send + Sync;
    type TextureViewId: Send + Sync;
    type SamplerId: Send + Sync;
    type BufferId: Send + Sync;
    type TextureId: Send + Sync;
    type PipelineLayoutId: Send + Sync;
    type RenderPipelineId: Send + Sync;
    type ComputePipelineId: Send + Sync;
    type CommandEncoderId;
    type ComputePassId: ComputePassInner<Self>;
    type CommandBufferId: Send + Sync;
    type SurfaceId: Send + Sync;
    type SwapChainId: Send + Sync;
    type RenderPassId: RenderPassInner<Self>;

    type CreateBufferMappedDetail: Send;
    type BufferReadMappingDetail: Send;
    type BufferWriteMappingDetail: Send;
    type SwapChainOutputDetail: Send;

    type RequestAdapterFuture: Future<Output = Option<Self::AdapterId>> + Send;
    type RequestDeviceFuture: Future<Output = Result<(Self::DeviceId, Self::QueueId), RequestDeviceError>>
        + Send;
    type MapReadFuture: Future<Output = Result<Self::BufferReadMappingDetail, BufferAsyncError>>
        + Send;
    type MapWriteFuture: Future<Output = Result<Self::BufferWriteMappingDetail, BufferAsyncError>>
        + Send;

    fn init() -> Self;
    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId;
    fn instance_request_adapter(
        &self,
        options: &RequestAdapterOptions<'_>,
        backends: wgt::BackendBit,
    ) -> Self::RequestAdapterFuture;
    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture;

    fn device_create_swap_chain(
        &self,
        device: &Self::DeviceId,
        surface: &Self::SurfaceId,
        desc: &SwapChainDescriptor,
    ) -> Self::SwapChainId;
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        spv: &[u32],
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
    fn device_create_buffer_mapped<'a>(
        &self,
        device: &Self::DeviceId,
        desc: &BufferDescriptor,
    ) -> (Self::BufferId, &'a mut [u8], Self::CreateBufferMappedDetail);
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
    fn device_drop(&self, device: &Self::DeviceId);
    fn device_poll(&self, device: &Self::DeviceId, maintain: Maintain);

    fn buffer_map_read(
        &self,
        buffer: &Self::BufferId,
        start: BufferAddress,
        size: BufferAddress,
    ) -> Self::MapReadFuture;
    fn buffer_map_write(
        &self,
        buffer: &Self::BufferId,
        start: BufferAddress,
        size: BufferAddress,
    ) -> Self::MapWriteFuture;
    fn buffer_unmap(&self, buffer: &Self::BufferId);
    fn swap_chain_get_next_texture(
        &self,
        swap_chain: &Self::SwapChainId,
    ) -> Result<(Self::TextureViewId, Self::SwapChainOutputDetail), TimeOut>;
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
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId);
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId);

    fn encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: BufferAddress,
        destination: &Self::BufferId,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
    fn encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: BufferCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    );
    fn encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    );
    fn encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: TextureCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    );

    fn flush_mapped_data(data: &mut [u8], detail: Self::CreateBufferMappedDetail);
    fn encoder_begin_compute_pass(&self, encoder: &Self::CommandEncoderId) -> Self::ComputePassId;
    fn encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    );
    fn encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId;
    fn encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    );
    fn encoder_finish(&self, encoder: &Self::CommandEncoderId) -> Self::CommandBufferId;
    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        data: &[u8],
        buffer: &Self::BufferId,
        offset: BufferAddress,
    );
    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    );
}

/// An instance sets up the context for all other wgpu objects.
///
/// An `Adapter` can be used to open a connection to the corresponding device on the host system,
/// yielding a [`Device`] object.
pub struct Instance {
    context: Arc<C>,
}

/// A handle to a physical graphics and/or compute device.
///
/// An `Adapter` can be used to open a connection to the corresponding device on the host system,
/// yielding a [`Device`] object.
pub struct Adapter {
    context: Arc<C>,
    id: <C as Context>::AdapterId,
}

/// Options for requesting adapter.
#[derive(Clone)]
pub struct RequestAdapterOptions<'a> {
    /// Power preference for the adapter.
    pub power_preference: PowerPreference,
    /// Surface that is required to be presentable with the requested adapter.
    pub compatible_surface: Option<&'a Surface>,
}

/// An open connection to a graphics and/or compute device.
///
/// The `Device` is the responsible for the creation of most rendering and compute resources, as
/// well as exposing [`Queue`] objects.
pub struct Device {
    context: Arc<C>,
    id: <C as Context>::DeviceId,
}

/// This is passed to `Device::poll` to control whether
/// it should block or not.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Maintain {
    Wait,
    Poll,
}

/// A handle to a GPU-accessible buffer.
pub struct Buffer {
    context: Arc<C>,
    id: <C as Context>::BufferId,
    //detail: <C as Context>::BufferDetail,
}

/// A description of what portion of a buffer to use
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: Option<BufferAddress>,
}

impl<'a> BufferSlice<'a> {
    /// This fn can be used for calling lower-level APIs where `0` denotes that the slice should
    /// extend to the end of the buffer.
    fn size_or_0(&self) -> BufferAddress {
        self.size.unwrap_or(0)
    }
}

/// A handle to a texture on the GPU.
pub struct Texture {
    context: Arc<C>,
    id: <C as Context>::TextureId,
    owned: bool,
}

/// A handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
pub struct TextureView {
    context: Arc<C>,
    id: <C as Context>::TextureViewId,
    owned: bool,
}

/// A handle to a sampler.
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
        self.context.sampler_drop(&self.id);
    }
}

/// A handle to a presentable surface.
///
/// A `Surface` represents a platform-specific surface (e.g. a window) to which rendered images may
/// be presented. A `Surface` may be created with [`Surface::create`].
pub struct Surface {
    id: <C as Context>::SurfaceId,
}

/// A handle to a swap chain.
///
/// A `SwapChain` represents the image or series of images that will be presented to a [`Surface`].
/// A `SwapChain` may be created with [`Device::create_swap_chain`].
pub struct SwapChain {
    context: Arc<C>,
    id: <C as Context>::SwapChainId,
}

/// An opaque handle to a binding group layout.
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
        self.context.bind_group_layout_drop(&self.id);
    }
}

/// An opaque handle to a binding group.
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
        self.context.bind_group_drop(&self.id);
    }
}

/// A handle to a compiled shader module.
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
        self.context.shader_module_drop(&self.id);
    }
}

/// An opaque handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
pub struct PipelineLayout {
    context: Arc<C>,
    id: <C as Context>::PipelineLayoutId,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        self.context.pipeline_layout_drop(&self.id);
    }
}

/// A handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. A `RenderPipeline` may be created with [`Device::create_render_pipeline`].
pub struct RenderPipeline {
    context: Arc<C>,
    id: <C as Context>::RenderPipelineId,
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        self.context.render_pipeline_drop(&self.id);
    }
}

/// A handle to a compute pipeline.
pub struct ComputePipeline {
    context: Arc<C>,
    id: <C as Context>::ComputePipelineId,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.context.compute_pipeline_drop(&self.id);
    }
}

/// An opaque handle to a command buffer on the GPU.
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
        if let Some(ref id) = self.id {
            self.context.command_buffer_drop(id);
        }
    }
}

/// An object that encodes GPU operations.
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

/// An in-progress recording of a render pass.
pub struct RenderPass<'a> {
    id: <C as Context>::RenderPassId,
    parent: &'a mut CommandEncoder,
}

/// An in-progress recording of a compute pass.
pub struct ComputePass<'a> {
    id: <C as Context>::ComputePassId,
    parent: &'a mut CommandEncoder,
}

/// A handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects.
pub struct Queue {
    context: Arc<C>,
    id: <C as Context>::QueueId,
}

/// A resource that can be bound to a pipeline.
pub enum BindingResource<'a> {
    Buffer(BufferSlice<'a>),
    Sampler(&'a Sampler),
    TextureView(&'a TextureView),
}

/// A bindable resource and the slot to bind it to.
pub struct Binding<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

/// Specific type of a binding.
/// WebGPU spec: https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutentry
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BindingType {
    /// A buffer for uniform values.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(std140, binding = 0)
    /// uniform Globals {
    ///     vec2 aUniform;
    ///     vec2 anotherUniform;
    /// };
    /// ```
    UniformBuffer {
        /// Indicates that the binding has a dynamic offset.
        /// One offset must be passed to [RenderPass::set_bind_group] for each dynamic binding in increasing order of binding number.
        dynamic: bool,
    },
    /// A storage buffer.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout (set=0, binding=0) buffer myStorageBuffer {
    ///     vec4 myElement[];
    /// };
    /// ```
    StorageBuffer {
        /// Indicates that the binding has a dynamic offset.
        /// One offset must be passed to [RenderPass::set_bind_group] for each dynamic binding in increasing order of binding number.
        dynamic: bool,
        /// The buffer can only be read in the shader and it must be annotated with `readonly`.
        ///
        /// Example GLSL syntax:
        /// ```cpp,ignore
        /// layout (set=0, binding=0) readonly buffer myStorageBuffer {
        ///     vec4 myElement[];
        /// };
        /// ```
        readonly: bool,
    },
    /// A sampler that can be used to sample a texture.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform sampler s;
    /// ```
    Sampler {
        /// Use as a comparison sampler instead of a normal sampler.
        /// For more info take a look at the analogous functionality in OpenGL: https://www.khronos.org/opengl/wiki/Sampler_Object#Comparison_mode.
        comparison: bool,
    },
    /// A texture.
    ///
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(binding = 0)
    /// uniform texture2D t;
    /// ```
    SampledTexture {
        /// Dimension of the texture view that is going to be sampled.
        dimension: TextureViewDimension,
        /// Component type of the texture.
        /// This must be compatible with the format of the texture.
        component_type: TextureComponentType,
        /// True if the texture has a sample count greater than 1.
        multisampled: bool,
    },
    /// A storage texture.
    /// Example GLSL syntax:
    /// ```cpp,ignore
    /// layout(set=0, binding=0, r32f) uniform image2D myStorageImage;
    /// ```
    /// Note that the texture format must be specified in the shader as well.
    /// A list of valid formats can be found in the specification here: https://www.khronos.org/registry/OpenGL/specs/gl/GLSLangSpec.4.60.html#layout-qualifiers
    StorageTexture {
        /// Dimension of the texture view that is going to be sampled.
        dimension: TextureViewDimension,
        /// Component type of the texture.
        /// This must be compatible with the format of the texture.
        component_type: TextureComponentType,
        /// Format of the texture.
        format: TextureFormat,
        /// The texture can only be read in the shader and it must be annotated with `readonly`.
        ///
        /// Example GLSL syntax:
        /// ```cpp,ignore
        /// layout(set=0, binding=0, r32f) readonly uniform image2D myStorageImage;
        /// ```
        readonly: bool,
    },
}

/// A description of a single binding inside a bind group.
#[derive(Clone, Debug, Hash)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStage,
    pub ty: BindingType,
}

/// A description of a bind group layout.
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    pub bindings: &'a [BindGroupLayoutEntry],

    /// An optional label to apply to the bind group layout.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,
}

/// A description of a group of bindings and the resources to be bound.
#[derive(Clone)]
pub struct BindGroupDescriptor<'a> {
    /// The layout for this bind group.
    pub layout: &'a BindGroupLayout,

    /// The resources to bind to this bind group.
    pub bindings: &'a [Binding<'a>],

    /// An optional label to apply to the bind group.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,
}

/// A description of a pipeline layout.
///
/// A `PipelineLayoutDescriptor` can be passed to [`Device::create_pipeline_layout`] to obtain a
/// [`PipelineLayout`].
#[derive(Clone)]
pub struct PipelineLayoutDescriptor<'a> {
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

/// A description of a programmable pipeline stage.
#[derive(Clone)]
pub struct ProgrammableStageDescriptor<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,

    /// The name of the entry point in the compiled shader.
    pub entry_point: &'a str,
}

/// The vertex input state for a render pipeline.
#[derive(Clone, Debug)]
pub struct VertexStateDescriptor<'a> {
    /// The format of any index buffers used with this pipeline.
    pub index_format: IndexFormat,

    /// The format of any vertex buffers used with this pipeline.
    pub vertex_buffers: &'a [VertexBufferDescriptor<'a>],
}

/// A description of a vertex buffer.
#[derive(Clone, Debug)]
pub struct VertexBufferDescriptor<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub stride: BufferAddress,

    pub step_mode: InputStepMode,

    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [VertexAttributeDescriptor],
}

/// A complete description of a render (graphics) pipeline.
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

    /// The number of samples calculated per pixel (for MSAA).
    pub sample_count: u32,

    /// Bitmask that restricts the samples of a pixel modified by this pipeline.
    pub sample_mask: u32,

    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

/// A complete description of a compute pipeline.
#[derive(Clone)]
pub struct ComputePipelineDescriptor<'a> {
    /// The layout of bind groups for this pipeline.
    pub layout: &'a PipelineLayout,

    /// The compiled compute stage and its entry point.
    pub compute_stage: ProgrammableStageDescriptor<'a>,
}

pub type RenderPassColorAttachmentDescriptor<'a> =
    wgt::RenderPassColorAttachmentDescriptorBase<&'a TextureView>;
pub type RenderPassDepthStencilAttachmentDescriptor<'a> =
    wgt::RenderPassDepthStencilAttachmentDescriptorBase<&'a TextureView>;

/// A description of all the attachments of a render pass.
pub struct RenderPassDescriptor<'a, 'b> {
    /// The color attachments of the render pass.
    pub color_attachments: &'b [RenderPassColorAttachmentDescriptor<'a>],

    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachmentDescriptor<'a>>,
}

/// A description of a buffer.
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Option<&'a str>>;

/// A description of a command encoder.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CommandEncoderDescriptor<'a> {
    /// An optional label to apply to the command encoder.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,
}

/// A description of a texture.
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Option<&'a str>>;

/// A description of a texture view.
pub type TextureViewDescriptor<'a> = wgt::TextureViewDescriptor<Option<&'a str>>;

/// A description of a sampler.
pub type SamplerDescriptor<'a> = wgt::SamplerDescriptor<Option<&'a str>>;

/// A swap chain image that can be rendered to.
pub struct SwapChainOutput {
    pub view: TextureView,
    detail: <C as Context>::SwapChainOutputDetail,
}

/// A view of a buffer which can be used to copy to or from a texture.
#[derive(Clone)]
pub struct BufferCopyView<'a> {
    /// The buffer to be copied to or from.
    pub buffer: &'a Buffer,

    /// The offset in bytes from the start of the buffer.
    /// In the future this must be aligned to 512 bytes, however the requirement is currently unimplemented.
    pub offset: BufferAddress,

    /// The size in bytes of a single row of the texture.
    /// In the future this must be a multiple of 256 bytes, however the requirement is currently unimplemented.
    pub bytes_per_row: u32,

    /// The height in texels of the imaginary texture view overlaid on the buffer.
    /// Must be zero for copies where `copy_size.depth == 1`.
    pub rows_per_image: u32,
}

/// A view of a texture which can be used to copy to or from a buffer or another texture.
#[derive(Clone)]
pub struct TextureCopyView<'a> {
    /// The texture to be copied to or from.
    pub texture: &'a Texture,

    /// The target mip level of the texture.
    pub mip_level: u32,

    /// The target layer of the texture.
    pub array_layer: u32,

    /// The base texel of the texture in the selected `mip_level`.
    pub origin: Origin3d,
}

/// A buffer being created, mapped in host memory.
pub struct CreateBufferMapped<'a> {
    context: Arc<C>,
    id: <C as Context>::BufferId,
    /// The backing field for `data()`. This isn't `pub` because users shouldn't
    /// be able to replace it to point somewhere else. We rely on it pointing to
    /// to the correct memory later during `unmap()`.
    mapped_data: &'a mut [u8],
    detail: <C as Context>::CreateBufferMappedDetail,
}

impl CreateBufferMapped<'_> {
    /// The mapped data.
    pub fn data(&mut self) -> &mut [u8] {
        self.mapped_data
    }

    /// Unmaps the buffer from host memory and returns a [`Buffer`].
    pub fn finish(self) -> Buffer {
        <C as Context>::flush_mapped_data(self.mapped_data, self.detail);
        Context::buffer_unmap(&*self.context, &self.id);
        Buffer {
            context: self.context,
            id: self.id,
            //detail: self.detail,
        }
    }
}

impl Instance {
    /// Create an new instance.
    pub fn new() -> Self {
        Instance {
            context: Arc::new(C::init()),
        }
    }

    /// Retrieves all available [`Adapter`]s that match the given backends.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn enumerate_adapters(&self, backends: wgt::BackendBit) -> impl Iterator<Item = Adapter> {
        let context = Arc::clone(&self.context);
        self.context
            .enumerate_adapters(wgc::instance::AdapterInputs::Mask(backends, |_| {
                PhantomData
            }))
            .into_iter()
            .map(move |id| crate::Adapter {
                id,
                context: Arc::clone(&context),
            })
    }

    /// Creates a surface from a raw window handle.
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
            vulkan: self
                .context
                .instance
                .vulkan
                .create_surface_from_layer(layer as *mut _, cfg!(debug_assertions)),
            metal: self
                .context
                .instance
                .metal
                .create_surface_from_layer(layer as *mut _, cfg!(debug_assertions)),
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
        backends: BackendBit,
    ) -> impl Future<Output = Option<Adapter>> + Send {
        let context = Arc::clone(&self.context);
        self.context
            .instance_request_adapter(options, backends)
            .map(|option| option.map(|id| Adapter { context, id }))
    }
}

impl Adapter {
    /// Requests a connection to a physical device, creating a logical device.
    /// Returns the device together with a queue that executes command buffers.
    ///
    /// # Panics
    ///
    /// Panics if the extensions specified by `desc` are not supported by this adapter.
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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_info(&self) -> AdapterInfo {
        //wgn::adapter_get_info(self.id)
        unimplemented!()
    }
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    pub fn poll(&self, maintain: Maintain) {
        Context::device_poll(&*self.context, &self.id, maintain);
    }

    /// Creates a shader module from SPIR-V source code.
    pub fn create_shader_module(&self, spv: &[u32]) -> ShaderModule {
        ShaderModule {
            context: Arc::clone(&self.context),
            id: Context::device_create_shader_module(&*self.context, &self.id, spv),
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

    /// Creates a new bind group.
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        BindGroup {
            context: Arc::clone(&self.context),
            id: Context::device_create_bind_group(&*self.context, &self.id, desc),
        }
    }

    /// Creates a bind group layout.
    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        BindGroupLayout {
            context: Arc::clone(&self.context),
            id: Context::device_create_bind_group_layout(&*self.context, &self.id, desc),
        }
    }

    /// Creates a pipeline layout.
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        PipelineLayout {
            context: Arc::clone(&self.context),
            id: Context::device_create_pipeline_layout(&*self.context, &self.id, desc),
        }
    }

    /// Creates a render pipeline.
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        RenderPipeline {
            context: Arc::clone(&self.context),
            id: Context::device_create_render_pipeline(&*self.context, &self.id, desc),
        }
    }

    /// Creates a compute pipeline.
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        ComputePipeline {
            context: Arc::clone(&self.context),
            id: Context::device_create_compute_pipeline(&*self.context, &self.id, desc),
        }
    }

    /// Creates a new buffer.
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        Buffer {
            context: Arc::clone(&self.context),
            id: Context::device_create_buffer(&*self.context, &self.id, desc),
        }
    }

    /// Creates a new buffer and maps it into host-visible memory.
    ///
    /// This returns a [`CreateBufferMapped`], which exposes a `&mut [u8]`. The actual [`Buffer`]
    /// will not be created until calling [`CreateBufferMapped::finish`].
    pub fn create_buffer_mapped(&self, desc: &BufferDescriptor) -> CreateBufferMapped<'_> {
        assert_ne!(desc.size, 0);
        let (id, mapped_data, detail) =
            Context::device_create_buffer_mapped(&*self.context, &self.id, desc);
        CreateBufferMapped {
            context: Arc::clone(&self.context),
            id,
            mapped_data,
            detail,
        }
    }

    /// Creates a new buffer, maps it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    pub fn create_buffer_with_data(&self, data: &[u8], usage: BufferUsage) -> Buffer {
        let mut mapped = self.create_buffer_mapped(&BufferDescriptor {
            size: data.len() as u64,
            usage,
            label: None,
        });
        mapped.data().copy_from_slice(data);
        mapped.finish()
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
    pub fn create_swap_chain(&self, surface: &Surface, desc: &SwapChainDescriptor) -> SwapChain {
        SwapChain {
            context: Arc::clone(&self.context),
            id: Context::device_create_swap_chain(&*self.context, &self.id, &surface.id, desc),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.context.device_drop(&self.id);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RequestDeviceError;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;

pub struct BufferReadMapping {
    context: Arc<C>,
    detail: <C as Context>::BufferReadMappingDetail,
}

unsafe impl Send for BufferReadMapping {}
unsafe impl Sync for BufferReadMapping {}

impl BufferReadMapping {
    pub fn as_slice(&self) -> &[u8] {
        self.detail.as_slice()
    }
}

impl Drop for BufferReadMapping {
    fn drop(&mut self) {
        Context::buffer_unmap(&*self.context, &self.detail.buffer_id);
    }
}

pub struct BufferWriteMapping {
    context: Arc<C>,
    detail: <C as Context>::BufferWriteMappingDetail,
}

unsafe impl Send for BufferWriteMapping {}
unsafe impl Sync for BufferWriteMapping {}

impl BufferWriteMapping {
    pub fn as_slice(&mut self) -> &mut [u8] {
        self.detail.as_slice()
    }
}

impl Drop for BufferWriteMapping {
    fn drop(&mut self) {
        Context::buffer_unmap(&*self.context, &self.detail.buffer_id);
    }
}

impl Buffer {
    /// Use only a portion of this Buffer for a given operation. Choosing a range with 0 size will
    /// return a slice that extends to the end of the buffer.
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice {
        let offset = match bounds.start_bound() {
            Bound::Included(&bound) => bound,
            Bound::Excluded(&bound) => bound + 1,
            Bound::Unbounded => 0,
        };
        let size = match bounds.end_bound() {
            Bound::Included(&bound) => Some(bound + 1 - offset),
            Bound::Excluded(&bound) => Some(bound - offset),
            Bound::Unbounded => None,
        };
        BufferSlice {
            buffer: self,
            offset,
            size,
        }
    }

    /// Map the buffer for reading. The result is returned in a future.
    ///
    /// For the future to complete, `device.poll(...)` must be called elsewhere in the runtime, possibly integrated
    /// into an event loop, run on a separate thread, or continually polled in the same task runtime that this
    /// future will be run on.
    ///
    /// It's expected that wgpu will eventually supply its own event loop infrastructure that will be easy to integrate
    /// into other event loops, like winit's.
    pub fn map_read(
        &self,
        start: BufferAddress,
        size: BufferAddress,
    ) -> impl Future<Output = Result<BufferReadMapping, BufferAsyncError>> + Send {
        let context = Arc::clone(&self.context);
        self.context
            .buffer_map_read(&self.id, start, size)
            .map(|result| result.map(|detail| BufferReadMapping { context, detail }))
    }

    /// Map the buffer for writing. The result is returned in a future.
    ///
    /// See the documentation of (map_read)[#method.map_read] for more information about
    /// how to run this future.
    pub fn map_write(
        &self,
        start: BufferAddress,
        size: BufferAddress,
    ) -> impl Future<Output = Result<BufferWriteMapping, BufferAsyncError>> + Send {
        let context = Arc::clone(&self.context);
        self.context
            .buffer_map_write(&self.id, start, size)
            .map(|result| result.map(|detail| BufferWriteMapping { context, detail }))
    }

    /// Flushes any pending write operations and unmaps the buffer from host memory.
    pub fn unmap(&self) {
        Context::buffer_unmap(&*self.context, &self.id);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.context.buffer_drop(&self.id);
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

    /// Creates a default view of this whole texture.
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
        if self.owned {
            self.context.texture_drop(&self.id);
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if self.owned {
            self.context.texture_view_drop(&self.id);
        }
    }
}

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            context: Arc::clone(&self.context),
            id: Some(Context::encoder_finish(&*self.context, &self.id)),
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
            id: Context::encoder_begin_render_pass(&*self.context, &self.id, desc),
            parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: Context::encoder_begin_compute_pass(&*self.context, &self.id),
            parent: self,
        }
    }

    /// Copy data from one buffer to another.
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        Context::encoder_copy_buffer_to_buffer(
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
    pub fn copy_buffer_to_texture(
        &mut self,
        source: BufferCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        Context::encoder_copy_buffer_to_texture(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from a texture to a buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    ) {
        Context::encoder_copy_texture_to_buffer(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from one texture to another.
    pub fn copy_texture_to_texture(
        &mut self,
        source: TextureCopyView,
        destination: TextureCopyView,
        copy_size: Extent3d,
    ) {
        Context::encoder_copy_texture_to_texture(
            &*self.context,
            &self.id,
            source,
            destination,
            copy_size,
        );
    }
}

impl<'a> RenderPass<'a> {
    /// Sets the active bind group for a given bind group index.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        RenderPassInner::set_bind_group(&mut self.id, index, &bind_group.id, offsets)
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        RenderPassInner::set_pipeline(&mut self.id, &pipeline.id)
    }

    pub fn set_blend_color(&mut self, color: Color) {
        self.id.set_blend_color(color)
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>) {
        RenderPassInner::set_index_buffer(
            &mut self.id,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size_or_0(),
        )
    }

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderPass`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [`RenderPipelineDescriptor::vertex_buffers`].
    ///
    /// [`draw`]: #method.draw
    /// [`draw_indexed`]: #method.draw_indexed
    /// [`RenderPass`]: struct.RenderPass.html
    /// [`RenderPipelineDescriptor::vertex_buffers`]: struct.RenderPipelineDescriptor.html#structfield.vertex_buffers
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        RenderPassInner::set_vertex_buffer(
            &mut self.id,
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.offset,
            buffer_slice.size_or_0(),
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
        RenderPassInner::draw(&mut self.id, vertices, instances)
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        RenderPassInner::draw_indexed(&mut self.id, indices, base_vertex, instances);
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
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.parent
                .context
                .encoder_end_render_pass(&self.parent.id, &mut self.id);
        }
    }
}

impl<'a> ComputePass<'a> {
    /// Sets the active bind group for a given bind group index.
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
                .encoder_end_compute_pass(&self.parent.id, &mut self.id);
        }
    }
}

impl Queue {
    /// Schedule a data write into `buffer` starting at `offset`.
    pub fn write_buffer(&self, data: &[u8], buffer: &Buffer, offset: BufferAddress) {
        Context::queue_write_buffer(&*self.context, &self.id, data, &buffer.id, offset)
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

impl Drop for SwapChainOutput {
    fn drop(&mut self) {
        if !thread::panicking() {
            Context::swap_chain_present(&*self.view.context, &self.view.id, &self.detail);
        }
    }
}

/// The GPU timed out when attempting to acquire the next texture or if a
/// previous output is still alive.
#[derive(Clone, Debug)]
pub struct TimeOut;

impl SwapChain {
    /// Returns the next texture to be presented by the swapchain for drawing.
    ///
    /// When the [`SwapChainOutput`] returned by this method is dropped, the swapchain will present
    /// the texture to the associated [`Surface`].
    pub fn get_next_texture(&mut self) -> Result<SwapChainOutput, TimeOut> {
        Context::swap_chain_get_next_texture(&*self.context, &self.id).map(|(id, detail)| {
            SwapChainOutput {
                view: TextureView {
                    context: Arc::clone(&self.context),
                    id,
                    owned: false,
                },
                detail,
            }
        })
    }
}
