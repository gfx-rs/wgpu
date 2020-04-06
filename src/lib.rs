//! A cross-platform graphics and compute library based on WebGPU.

mod backend;

#[macro_use]
mod macros;

use std::{future::Future, ops::Range, thread};

pub use wgc::instance::{AdapterInfo, DeviceType};
pub use wgt::{
    read_spirv, AddressMode, Backend, BackendBit, BlendDescriptor, BlendFactor, BlendOperation,
    BufferAddress, BufferUsage, Color, ColorStateDescriptor, ColorWrite, CommandBufferDescriptor,
    CompareFunction, CullMode, DepthStencilStateDescriptor, DeviceDescriptor, DynamicOffset,
    Extensions, Extent3d, FilterMode, FrontFace, IndexFormat, InputStepMode, Limits, LoadOp,
    Origin3d, PowerPreference, PresentMode, PrimitiveTopology, RasterizationStateDescriptor,
    SamplerDescriptor, ShaderLocation, ShaderStage, StencilOperation, StencilStateFaceDescriptor,
    StoreOp, SwapChainDescriptor, TextureAspect, TextureComponentType, TextureDimension,
    TextureFormat, TextureUsage, TextureViewDescriptor, TextureViewDimension,
    VertexAttributeDescriptor, VertexFormat, BIND_BUFFER_ALIGNMENT, MAX_BIND_GROUPS,
};
/*
pub use wgc::instance::{
    AdapterInfo,
    DeviceType,
};
*/

//TODO: avoid heap allocating vectors during resource creation.
#[derive(Default, Debug)]
struct Temp {
    //bind_group_descriptors: Vec<wgn::BindGroupDescriptor>,
//vertex_buffers: Vec<wgn::VertexBufferDescriptor>,
}

/// A handle to a physical graphics and/or compute device.
///
/// An `Adapter` can be used to open a connection to the corresponding device on the host system,
/// yielding a [`Device`] object.
#[derive(Debug, PartialEq)]
pub struct Adapter {
    id: backend::AdapterId,
}

/// Options for requesting adapter.
#[derive(Clone, Debug)]
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
#[derive(Debug)]
pub struct Device {
    id: backend::DeviceId,
    temp: Temp,
}

/// This is passed to `Device::poll` to control whether
/// it should block or not.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Maintain {
    Wait,
    Poll,
}

/// A handle to a GPU-accessible buffer.
#[derive(Debug, PartialEq)]
pub struct Buffer {
    id: backend::BufferId,
    detail: backend::BufferDetail,
}

/// A handle to a texture on the GPU.
#[derive(Debug, PartialEq)]
pub struct Texture {
    id: backend::TextureId,
    owned: bool,
}

/// A handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
#[derive(Debug, PartialEq)]
pub struct TextureView {
    id: backend::TextureViewId,
    owned: bool,
}

/// A handle to a sampler.
///
/// A `Sampler` object defines how a pipeline will sample from a [`TextureView`]. Samplers define
/// image filters (including anisotropy) and address (wrapping) modes, among other things. See
/// the documentation for [`SamplerDescriptor`] for more information.
#[derive(Debug, PartialEq)]
pub struct Sampler {
    id: backend::SamplerId,
}

/// A handle to a presentable surface.
///
/// A `Surface` represents a platform-specific surface (e.g. a window) to which rendered images may
/// be presented. A `Surface` may be created with [`Surface::create`].
#[derive(Debug, PartialEq)]
pub struct Surface {
    id: backend::SurfaceId,
}

/// A handle to a swap chain.
///
/// A `SwapChain` represents the image or series of images that will be presented to a [`Surface`].
/// A `SwapChain` may be created with [`Device::create_swap_chain`].
#[derive(Debug, PartialEq)]
pub struct SwapChain {
    id: backend::SwapChainId,
}

/// An opaque handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
#[derive(Debug, PartialEq)]
pub struct BindGroupLayout {
    id: backend::BindGroupLayoutId,
}

/// An opaque handle to a binding group.
///
/// A `BindGroup` represents the set of resources bound to the bindings described by a
/// [`BindGroupLayout`]. It can be created with [`Device::create_bind_group`]. A `BindGroup` can
/// be bound to a particular [`RenderPass`] with [`RenderPass::set_bind_group`], or to a
/// [`ComputePass`] with [`ComputePass::set_bind_group`].
#[derive(Debug, PartialEq)]
pub struct BindGroup {
    id: backend::BindGroupId,
}

impl Drop for BindGroup {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        wgn::wgpu_bind_group_destroy(self.id);
    }
}

/// A handle to a compiled shader module.
///
/// A `ShaderModule` represents a compiled shader module on the GPU. It can be created by passing
/// valid SPIR-V source code to [`Device::create_shader_module`]. Shader modules are used to define
/// programmable stages of a pipeline.
#[derive(Debug, PartialEq)]
pub struct ShaderModule {
    id: backend::ShaderModuleId,
}

/// An opaque handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
#[derive(Debug, PartialEq)]
pub struct PipelineLayout {
    id: backend::PipelineLayoutId,
}

/// A handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. A `RenderPipeline` may be created with [`Device::create_render_pipeline`].
#[derive(Debug, PartialEq)]
pub struct RenderPipeline {
    id: backend::RenderPipelineId,
}

/// A handle to a compute pipeline.
#[derive(Debug, PartialEq)]
pub struct ComputePipeline {
    id: backend::ComputePipelineId,
}

/// An opaque handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
#[derive(Debug, PartialEq)]
pub struct CommandBuffer {
    id: backend::CommandBufferId,
}

/// An object that encodes GPU operations.
///
/// A `CommandEncoder` can record [`RenderPass`]es, [`ComputePass`]es, and transfer operations
/// between driver-managed resources like [`Buffer`]s and [`Texture`]s.
///
/// When finished recording, call [`CommandEncoder::finish`] to obtain a [`CommandBuffer`] which may
/// be submitted for execution.
#[derive(Debug)]
pub struct CommandEncoder {
    id: backend::CommandEncoderId,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: std::marker::PhantomData<*const u8>,
}

/// An in-progress recording of a render pass.
#[derive(Debug)]
pub struct RenderPass<'a> {
    id: backend::RenderPassEncoderId,
    _parent: &'a mut CommandEncoder,
}

/// An in-progress recording of a compute pass.
#[derive(Debug)]
pub struct ComputePass<'a> {
    id: backend::ComputePassId,
    _parent: &'a mut CommandEncoder,
}

/// A handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects.
#[derive(Debug, PartialEq)]
pub struct Queue {
    id: backend::QueueId,
}

/// A resource that can be bound to a pipeline.
#[derive(Clone, Debug)]
pub enum BindingResource<'a> {
    Buffer {
        buffer: &'a Buffer,
        range: Range<BufferAddress>,
    },
    Sampler(&'a Sampler),
    TextureView(&'a TextureView),
}

/// A bindable resource and the slot to bind it to.
#[derive(Clone, Debug)]
pub struct Binding<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

/// Specific type of a binding..
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BindingType {
    UniformBuffer {
        dynamic: bool,
    },
    StorageBuffer {
        dynamic: bool,
        readonly: bool,
    },
    Sampler {
        comparison: bool,
    },
    SampledTexture {
        dimension: TextureViewDimension,
        component_type: TextureComponentType,
        multisampled: bool,
    },
    StorageTexture {
        dimension: TextureViewDimension,
        component_type: TextureComponentType,
        format: TextureFormat,
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a> {
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

/// A description of a programmable pipeline stage.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Debug)]
pub struct RenderPassDescriptor<'a, 'b> {
    /// The color attachments of the render pass.
    pub color_attachments: &'b [RenderPassColorAttachmentDescriptor<'a>],

    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachmentDescriptor<'a>>,
}

/// A description of a buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferDescriptor<'a> {
    /// An optional label to apply to the buffer.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,

    /// The size of the buffer (in bytes).
    pub size: BufferAddress,

    /// All possible ways the buffer can be used.
    pub usage: BufferUsage,
}

/// A description of a command encoder.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CommandEncoderDescriptor<'a> {
    /// An optional label to apply to the command encoder.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,
}

/// A description of a texture.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureDescriptor<'a> {
    /// An optional label to apply to the texture.
    /// This can be useful for debugging and performance analysis.
    pub label: Option<&'a str>,

    /// The size of the texture.
    pub size: Extent3d,

    /// The mip level count.
    pub mip_level_count: u32,

    /// The sample count.
    pub sample_count: u32,

    /// The texture dimension.
    pub dimension: TextureDimension,

    /// The texture format.
    pub format: TextureFormat,

    /// All possible ways the texture can be used.
    pub usage: TextureUsage,
}

/// A swap chain image that can be rendered to.
#[derive(Debug)]
pub struct SwapChainOutput {
    pub view: TextureView,
    swap_chain_id: backend::SwapChainId,
}

/// A view of a buffer which can be used to copy to or from a texture.
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
    id: backend::BufferId,
    /// The backing field for `data()`. This isn't `pub` because users shouldn't
    /// be able to replace it to point somewhere else. We rely on it pointing to
    /// to the correct memory later during `unmap()`.
    mapped_data: &'a mut [u8],
    detail: backend::CreateBufferMappedDetail,
}

impl CreateBufferMapped<'_> {
    /// The mapped data.
    pub fn data(&mut self) -> &mut [u8] {
        self.mapped_data
    }

    /// Unmaps the buffer from host memory and returns a [`Buffer`].
    pub fn finish(self) -> Buffer {
        backend::device_create_buffer_mapped_finish(self)
    }
}

impl Surface {
    /// Creates a surface from a raw window handle.
    pub fn create<W: raw_window_handle::HasRawWindowHandle>(window: &W) -> Self {
        Surface {
            id: backend::device_create_surface(window),
        }
    }

    /*
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        pub fn create_surface_from_core_animation_layer(layer: *mut std::ffi::c_void) -> Self {
            Surface {
                id: wgn::wgpu_create_surface_from_metal_layer(layer),
            }
        }
    */
}

impl Adapter {
    /*
        /// Retrieves all available [`Adapter`]s that match the given backends.
        pub fn enumerate(backends: BackendBit) -> Vec<Self> {
            wgn::wgpu_enumerate_adapters(backends)
                .into_iter()
                .map(|id| Adapter { id })
                .collect()
        }
    */

    /// Retrieves an [`Adapter`] which matches the given options.
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that suffice all the "hard" options, `None` is returned.
    pub async fn request(
        options: &RequestAdapterOptions<'_>,
        backends: BackendBit,
    ) -> Option<Self> {
        backend::request_adapter(options, backends)
            .await
            .map(|id| Adapter { id })
    }

    /// Requests a connection to a physical device, creating a logical device.
    /// Returns the device together with a queue that executes command buffers.
    ///
    /// # Panics
    ///
    /// Panics if the extensions specified by `desc` are not supported by this adapter.
    pub async fn request_device(&self, desc: &DeviceDescriptor) -> (Device, Queue) {
        let (device_id, queue_id) = backend::request_device_and_queue(&self.id, Some(desc)).await;
        let device = Device {
            id: device_id,
            temp: Temp::default(),
        };
        let queue = Queue { id: queue_id };
        (device, queue)
    }

    /*
        pub fn get_info(&self) -> AdapterInfo {
            wgn::adapter_get_info(self.id)
        }
    */
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    pub fn poll(&self, maintain: Maintain) {
        backend::device_poll(&self.id, maintain);
    }

    /// Creates a shader module from SPIR-V source code.
    pub fn create_shader_module(&self, spv: &[u32]) -> ShaderModule {
        ShaderModule {
            id: backend::create_shader_module(&self.id, spv),
        }
    }

    /// Creates an empty [`CommandEncoder`].
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        CommandEncoder {
            id: backend::create_command_encoder(&self.id, desc),
            _p: Default::default(),
        }
    }

    /// Creates a new bind group.
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        let id = backend::create_bind_group(&self.id, desc);
        BindGroup { id }
    }

    /// Creates a bind group layout.
    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        let id = backend::create_bind_group_layout(&self.id, desc);
        BindGroupLayout { id }
    }

    /// Creates a pipeline layout.
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        let id = backend::create_pipeline_layout(&self.id, desc);
        PipelineLayout { id }
    }

    /// Creates a render pipeline.
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        let id = backend::create_render_pipeline(&self.id, desc);
        RenderPipeline { id }
    }

    /// Creates a compute pipeline.
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        let id = backend::create_compute_pipeline(&self.id, desc);
        ComputePipeline { id }
    }

    /// Creates a new buffer.
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        backend::device_create_buffer(&self.id, desc)
    }

    /// Creates a new buffer and maps it into host-visible memory.
    ///
    /// This returns a [`CreateBufferMapped`], which exposes a `&mut [u8]`. The actual [`Buffer`]
    /// will not be created until calling [`CreateBufferMapped::finish`].
    pub fn create_buffer_mapped(&self, desc: &BufferDescriptor) -> CreateBufferMapped<'_> {
        assert_ne!(desc.size, 0);
        backend::device_create_buffer_mapped(&self.id, desc)
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
            id: backend::device_create_texture(&self.id, desc),
            owned: true,
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler {
        Sampler {
            id: backend::device_create_sampler(&self.id, desc),
        }
    }

    /// Create a new [`SwapChain`] which targets `surface`.
    pub fn create_swap_chain(&self, surface: &Surface, desc: &SwapChainDescriptor) -> SwapChain {
        SwapChain {
            id: backend::device_create_swap_chain(&self.id, &surface.id, desc),
        }
    }
}

// TODO
impl Drop for Device {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        wgn::wgpu_device_poll(self.id, true);
        //TODO: make this work in general
        #[cfg(not(target_arch = "wasm32"))]
        #[cfg(feature = "metal-auto-capture")]
        wgn::wgpu_device_destroy(self.id);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncErr;

pub struct BufferReadMapping {
    detail: backend::BufferReadMappingDetail,
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
        backend::buffer_unmap(&self.detail.buffer_id);
    }
}

/*
pub struct BufferWriteMapping {
    data: *mut u8,
    size: usize,
    buffer_id: wgc::id::BufferId,
}

unsafe impl Send for BufferWriteMapping {}
unsafe impl Sync for BufferWriteMapping {}

impl BufferWriteMapping {
    pub fn as_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data as *mut u8, self.size) }
    }
}

impl Drop for BufferWriteMapping {
    fn drop(&mut self) {
        wgn::wgpu_buffer_unmap(self.buffer_id);
    }
}
*/

impl Buffer {
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
    ) -> impl Future<Output = Result<BufferReadMapping, BufferAsyncErr>> + '_ {
        backend::buffer_map_read(self, start, size)
    }

    /*
        /// Map the buffer for writing. The result is returned in a future.
        ///
        /// See the documentation of (map_read)[#method.map_read] for more information about
        /// how to run this future.
        pub fn map_write(
            &self,
            start: BufferAddress,
            size: BufferAddress,
        ) -> impl Future<Output = Result<BufferWriteMapping, BufferAsyncErr>> {
            let (future, completion) = native_gpu_future::new_gpu_future(self.id, size);

            extern "C" fn buffer_map_write_future_wrapper(
                status: wgc::resource::BufferMapAsyncStatus,
                data: *mut u8,
                user_data: *mut u8,
            ) {
                let completion =
                    unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
                let (buffer_id, size) = completion.get_buffer_info();

                if let wgc::resource::BufferMapAsyncStatus::Success = status {
                    completion.complete(Ok(BufferWriteMapping {
                        data,
                        size: size as usize,
                        buffer_id,
                    }));
                } else {
                    completion.complete(Err(BufferAsyncErr));
                }
            }

            wgn::wgpu_buffer_map_write_async(
                self.id,
                start,
                size,
                buffer_map_write_future_wrapper,
                completion.to_raw() as _,
            );

            future
        }

        /// Flushes any pending write operations and unmaps the buffer from host memory.
        pub fn unmap(&self) {
            wgn::wgpu_buffer_unmap(self.id);
        }
    */
}

/*
impl Drop for Buffer {
    fn drop(&mut self) {
        wgn::wgpu_buffer_destroy(self.id);
    }
}
*/

impl Texture {
    /// Creates a view of this texture.
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        TextureView {
            id: backend::texture_create_view(&self.id, Some(desc)),
            owned: true,
        }
    }

    /// Creates a default view of this whole texture.
    pub fn create_default_view(&self) -> TextureView {
        TextureView {
            id: backend::texture_create_view(&self.id, None),
            owned: true,
        }
    }
}

/*
impl Drop for Texture {
    fn drop(&mut self) {
        if self.owned {
            wgn::wgpu_texture_destroy(self.id);
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if self.owned {
            wgn::wgpu_texture_view_destroy(self.id);
        }
    }
}
*/

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            id: backend::command_encoder_finish(&self.id),
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
            id: backend::command_encoder_begin_render_pass(&self.id, desc),
            _parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: backend::begin_compute_pass(&self.id),
            _parent: self,
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
        backend::command_encoder_copy_buffer_to_buffer(
            &self.id,
            source,
            source_offset,
            destination,
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
        backend::command_encoder_copy_buffer_to_texture(&self.id, source, destination, copy_size);
    }

    /// Copy data from a texture to a buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: TextureCopyView,
        destination: BufferCopyView,
        copy_size: Extent3d,
    ) {
        backend::command_encoder_copy_texture_to_buffer(&self.id, source, destination, copy_size);
    }

    /*
        /// Copy data from one texture to another.
        pub fn copy_texture_to_texture(
            &mut self,
            source: TextureCopyView,
            destination: TextureCopyView,
            copy_size: Extent3d,
        ) {
            wgn::wgpu_command_encoder_copy_texture_to_texture(
                self.id,
                &source.into_native(),
                &destination.into_native(),
                copy_size,
            );
        }
    */
}

impl<'a> RenderPass<'a> {
    /// Sets the active bind group for a given bind group index.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        backend::render_pass_set_bind_group(&self.id, index, &bind_group.id, offsets)
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        backend::render_pass_set_pipeline(&self.id, &pipeline.id)
    }

    /*
            pub fn set_blend_color(&mut self, color: Color) {
                unsafe {
                    wgn::wgpu_render_pass_set_blend_color(self.id.as_mut().unwrap(), &color);
                }
            }
        }
    */

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    ///
    /// If `size == 0`, the remaining part of the buffer is considered.
    pub fn set_index_buffer(
        &mut self,
        buffer: &'a Buffer,
        offset: BufferAddress,
        size: BufferAddress,
    ) {
        backend::render_pass_set_index_buffer(&self.id, buffer, offset, size)
    }

    /// Assign a vertex buffer to a slot.
    ///
    /// Subsequent calls to [`draw`] and [`draw_indexed`] on this
    /// [`RenderPass`] will use `buffer` as one of the source vertex buffers.
    ///
    /// The `slot` refers to the index of the matching descriptor in
    /// [`RenderPipelineDescriptor::vertex_buffers`].
    ///
    /// If `size == 0`, the remaining part of the buffer is considered.
    ///
    /// [`draw`]: #method.draw
    /// [`draw_indexed`]: #method.draw_indexed
    /// [`RenderPass`]: struct.RenderPass.html
    /// [`RenderPipelineDescriptor::vertex_buffers`]: struct.RenderPipelineDescriptor.html#structfield.vertex_buffers
    pub fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &'a Buffer,
        offset: BufferAddress,
        size: BufferAddress,
    ) {
        backend::render_pass_set_vertex_buffer(&self.id, slot, buffer, offset, size)
    }

    /*
        /// Sets the scissor region.
        ///
        /// Subsequent draw calls will discard any fragments that fall outside this region.
        pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
            unsafe {
                wgn::wgpu_render_pass_set_scissor_rect(self.id.as_mut().unwrap(), x, y, w, h);
            }
        }

        /// Sets the viewport region.
        ///
        /// Subsequent draw calls will draw any fragments in this region.
        pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
            unsafe {
                wgn::wgpu_render_pass_set_viewport(
                    self.id.as_mut().unwrap(),
                    x,
                    y,
                    w,
                    h,
                    min_depth,
                    max_depth,
                );
            }
        }

        /// Sets the stencil reference.
        ///
        /// Subsequent stencil tests will test against this value.
        pub fn set_stencil_reference(&mut self, reference: u32) {
            unsafe {
                wgn::wgpu_render_pass_set_stencil_reference(self.id.as_mut().unwrap(), reference);
            }
        }
    */

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        backend::render_pass_draw(&self.id, vertices, instances)
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        backend::render_pass_draw_indexed(&self.id, indices, base_vertex, instances);
    }

    /*
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
        unsafe {
            wgn::wgpu_render_pass_draw_indirect(
                self.id.as_mut().unwrap(),
                indirect_buffer.id,
                indirect_offset,
            );
        }
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
        unsafe {
            wgn::wgpu_render_pass_draw_indexed_indirect(
                self.id.as_mut().unwrap(),
                indirect_buffer.id,
                indirect_offset,
            );
        }
    }
    */
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            backend::render_pass_end_pass(&self.id);
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
        backend::compute_pass_set_bind_group(&self.id, index, &bind_group.id, offsets);
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &'a ComputePipeline) {
        backend::compute_pass_set_pipeline(&self.id, &pipeline.id);
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        backend::compute_pass_dispatch(&self.id, x, y, z);
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    pub fn dispatch_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        backend::compute_pass_dispatch_indirect(&self.id, &indirect_buffer.id, indirect_offset);
    }
}

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            backend::compute_pass_end_pass(&self.id);
        }
    }
}

impl Queue {
    /// Submits a series of finished command buffers for execution.
    pub fn submit(&self, command_buffers: &[CommandBuffer]) {
        backend::queue_submit(&self.id, command_buffers);
    }
}

impl Drop for SwapChainOutput {
    fn drop(&mut self) {
        if !thread::panicking() {
            backend::swap_chain_present(&self.swap_chain_id);
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
        match backend::swap_chain_get_next_texture(&self.id) {
            Some(id) => Ok(SwapChainOutput {
                view: TextureView { id, owned: false },
                swap_chain_id: self.id,
            }),
            None => Err(TimeOut),
        }
    }
}
