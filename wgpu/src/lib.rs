//! A cross-platform graphics and compute library based on WebGPU.

mod backend;
use crate::backend::native_gpu_future;

#[macro_use]
mod macros;

use arrayvec::ArrayVec;
use smallvec::SmallVec;

use std::{
    ffi::CString,
    ops::Range,
    future::Future,
    ptr,
    slice,
    thread,
};

pub use wgt::{
    AddressMode,
    Backend,
    BackendBit,
    BlendDescriptor,
    BlendFactor,
    BlendOperation,
    BufferAddress,
    BufferUsage,
    Color,
    ColorStateDescriptor,
    ColorWrite,
    CommandBufferDescriptor,
    CompareFunction,
    CullMode,
    DepthStencilStateDescriptor,
    DeviceDescriptor,
    DynamicOffset,
    Extensions,
    Extent3d,
    FilterMode,
    FrontFace,
    IndexFormat,
    InputStepMode,
    Limits,
    LoadOp,
    Origin3d,
    PowerPreference,
    PresentMode,
    PrimitiveTopology,
    RasterizationStateDescriptor,
    RequestAdapterOptions,
    SamplerDescriptor,
    ShaderLocation,
    ShaderStage,
    StencilOperation,
    StencilStateFaceDescriptor,
    StoreOp,
    SwapChainDescriptor,
    TextureAspect,
    TextureComponentType,
    TextureDimension,
    TextureFormat,
    TextureUsage,
    TextureViewDescriptor,
    TextureViewDimension,
    VertexAttributeDescriptor,
    VertexFormat,
    BIND_BUFFER_ALIGNMENT,
    MAX_BIND_GROUPS,
    read_spirv,
};
pub use wgc::instance::{
    AdapterInfo,
    DeviceType,
};

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
    id: wgc::id::AdapterId,
}

/// An open connection to a graphics and/or compute device.
///
/// The `Device` is the responsible for the creation of most rendering and compute resources, as
/// well as exposing [`Queue`] objects.
#[derive(Debug)]
pub struct Device {
    id: wgc::id::DeviceId,
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
    id: wgc::id::BufferId,
    device_id: wgc::id::DeviceId,
}

/// A handle to a texture on the GPU.
#[derive(Debug, PartialEq)]
pub struct Texture {
    id: wgc::id::TextureId,
    owned: bool,
}

/// A handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
#[derive(Debug, PartialEq)]
pub struct TextureView {
    id: wgc::id::TextureViewId,
    owned: bool,
}

/// A handle to a sampler.
///
/// A `Sampler` object defines how a pipeline will sample from a [`TextureView`]. Samplers define
/// image filters (including anisotropy) and address (wrapping) modes, among other things. See
/// the documentation for [`SamplerDescriptor`] for more information.
#[derive(Debug, PartialEq)]
pub struct Sampler {
    id: wgc::id::SamplerId,
}

/// A handle to a presentable surface.
///
/// A `Surface` represents a platform-specific surface (e.g. a window) to which rendered images may
/// be presented. A `Surface` may be created with [`Surface::create`].
#[derive(Debug, PartialEq)]
pub struct Surface {
    id: wgc::id::SurfaceId,
}

/// A handle to a swap chain.
///
/// A `SwapChain` represents the image or series of images that will be presented to a [`Surface`].
/// A `SwapChain` may be created with [`Device::create_swap_chain`].
#[derive(Debug, PartialEq)]
pub struct SwapChain {
    id: wgc::id::SwapChainId,
}

/// An opaque handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
#[derive(Debug, PartialEq)]
pub struct BindGroupLayout {
    id: wgc::id::BindGroupLayoutId,
}

/// An opaque handle to a binding group.
///
/// A `BindGroup` represents the set of resources bound to the bindings described by a
/// [`BindGroupLayout`]. It can be created with [`Device::create_bind_group`]. A `BindGroup` can
/// be bound to a particular [`RenderPass`] with [`RenderPass::set_bind_group`], or to a
/// [`ComputePass`] with [`ComputePass::set_bind_group`].
#[derive(Debug, PartialEq)]
pub struct BindGroup {
    id: wgc::id::BindGroupId,
}

impl Drop for BindGroup {
    fn drop(&mut self) {
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
    id: wgc::id::ShaderModuleId,
}

/// An opaque handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
#[derive(Debug, PartialEq)]
pub struct PipelineLayout {
    id: wgc::id::PipelineLayoutId,
}

/// A handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. A `RenderPipeline` may be created with [`Device::create_render_pipeline`].
#[derive(Debug, PartialEq)]
pub struct RenderPipeline {
    id: wgc::id::RenderPipelineId,
}

/// A handle to a compute pipeline.
#[derive(Debug, PartialEq)]
pub struct ComputePipeline {
    id: wgc::id::ComputePipelineId,
}

/// An opaque handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
#[derive(Debug, PartialEq)]
pub struct CommandBuffer {
    id: wgc::id::CommandBufferId,
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
    id: wgc::id::CommandEncoderId,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: std::marker::PhantomData<*const u8>,
}

/// An in-progress recording of a render pass.
#[derive(Debug)]
pub struct RenderPass<'a> {
    id: wgc::id::RenderPassId,
    _parent: &'a mut CommandEncoder,
}

/// An in-progress recording of a compute pass.
#[derive(Debug)]
pub struct ComputePass<'a> {
    id: wgc::id::ComputePassId,
    _parent: &'a mut CommandEncoder,
}

/// A handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects.
#[derive(Debug, PartialEq)]
pub struct Queue {
    id: wgc::id::QueueId,
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
    wgt::RenderPassColorAttachmentDescriptorBase<&'a TextureView, Option<&'a TextureView>>;
pub type RenderPassDepthStencilAttachmentDescriptor<'a> =
    wgt::RenderPassDepthStencilAttachmentDescriptorBase<&'a TextureView>;

/// A description of all the attachments of a render pass.
#[derive(Debug)]
pub struct RenderPassDescriptor<'a, 'b> {
    /// The color attachments of the render pass.
    pub color_attachments: &'b [RenderPassColorAttachmentDescriptor<'a>],

    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment:
        Option<RenderPassDepthStencilAttachmentDescriptor<'a>>,
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// The array layer count.
    pub array_layer_count: u32,

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
    swap_chain_id: wgc::id::SwapChainId,
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

impl BufferCopyView<'_> {
    fn into_native(self) -> wgc::command::BufferCopyView {
        wgc::command::BufferCopyView {
            buffer: self.buffer.id,
            offset: self.offset,
            bytes_per_row: self.bytes_per_row,
            rows_per_image: self.rows_per_image,
        }
    }
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

impl<'a> TextureCopyView<'a> {
    fn into_native(self) -> wgc::command::TextureCopyView {
        wgc::command::TextureCopyView {
            texture: self.texture.id,
            mip_level: self.mip_level,
            array_layer: self.array_layer,
            origin: self.origin,
        }
    }
}

/// A buffer being created, mapped in host memory.
pub struct CreateBufferMapped<'a> {
    id: wgc::id::BufferId,
    pub data: &'a mut [u8],
    device_id: wgc::id::DeviceId,
}

impl CreateBufferMapped<'_> {
    /// Unmaps the buffer from host memory and returns a [`Buffer`].
    pub fn finish(self) -> Buffer {
        wgn::wgpu_buffer_unmap(self.id);
        Buffer { device_id: self.device_id, id: self.id }
    }
}

impl Surface {
    /// Creates a surface from a raw window handle.
    pub fn create<W: raw_window_handle::HasRawWindowHandle>(window: &W) -> Self {
        Surface {
            id: wgn::wgpu_create_surface(window.raw_window_handle()),
        }
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub fn create_surface_from_core_animation_layer(layer: *mut std::ffi::c_void) -> Self {
        Surface {
            id: wgn::wgpu_create_surface_from_metal_layer(layer),
        }
    }
}

impl Adapter {
    /// Retrieves all available [`Adapter`]s that match the given backends.
    pub fn enumerate(backends: BackendBit) -> Vec<Self> {
        wgn::wgpu_enumerate_adapters(backends)
            .into_iter()
            .map(|id| Adapter { id })
            .collect()
    }

    /// Retrieves an [`Adapter`] which matches the given options.
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that suffice all the "hard" options, `None` is returned.
    pub async fn request(options: &RequestAdapterOptions, backends: BackendBit) -> Option<Self> {
        unsafe extern "C" fn adapter_callback(
            id: wgc::id::AdapterId,
            user_data: *mut std::ffi::c_void,
        ) {
            *(user_data as *mut wgc::id::AdapterId) = id;
        }

        let mut id = wgc::id::AdapterId::ERROR;
        unsafe {
            wgn::wgpu_request_adapter_async(
                Some(options),
                backends,
                adapter_callback,
                &mut id as *mut _ as *mut std::ffi::c_void,
            )
        };
        Some(Adapter { id })
    }

    /// Requests a connection to a physical device, creating a logical device.
    /// Returns the device together with a queue that executes command buffers.
    ///
    /// # Panics
    ///
    /// Panics if the extensions specified by `desc` are not supported by this adapter.
    pub async fn request_device(&self, desc: &DeviceDescriptor) -> (Device, Queue) {
        let device = Device {
            id: wgn::wgpu_adapter_request_device(self.id, Some(desc)),
            temp: Temp::default(),
        };
        let queue = Queue {
            id: wgn::wgpu_device_get_default_queue(device.id),
        };
        (device, queue)
    }

    pub fn get_info(&self) -> AdapterInfo {
        wgn::adapter_get_info(self.id)
    }
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    pub fn poll(&self, maintain: Maintain) {
        wgn::wgpu_device_poll(self.id, match maintain {
            Maintain::Poll => false,
            Maintain::Wait => true,
        });
    }

    /// Creates a shader module from SPIR-V source code.
    pub fn create_shader_module(&self, spv: &[u32]) -> ShaderModule {
        let desc = wgc::pipeline::ShaderModuleDescriptor {
            code: wgc::U32Array {
                bytes: spv.as_ptr(),
                length: spv.len(),
            },
        };
        ShaderModule {
            id: wgn::wgpu_device_create_shader_module(self.id, &desc),
        }
    }

    /// Creates an empty [`CommandEncoder`].
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        CommandEncoder {
            id: wgn::wgpu_device_create_command_encoder(
                self.id,
                Some(&wgt::CommandEncoderDescriptor {
                    label: owned_label.as_ptr(),
                }),
            ),
            _p: Default::default(),
        }
    }

    /// Creates a new bind group.
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        use wgc::binding_model as bm;

        let bindings = desc
            .bindings
            .iter()
            .map(|binding| bm::BindGroupEntry {
                binding: binding.binding,
                resource: match binding.resource {
                    BindingResource::Buffer {
                        ref buffer,
                        ref range,
                    } => bm::BindingResource::Buffer(bm::BufferBinding {
                        buffer: buffer.id,
                        offset: range.start,
                        size: range.end - range.start,
                    }),
                    BindingResource::Sampler(ref sampler) => {
                        bm::BindingResource::Sampler(sampler.id)
                    }
                    BindingResource::TextureView(ref texture_view) => {
                        bm::BindingResource::TextureView(texture_view.id)
                    }
                },
            })
            .collect::<Vec<_>>();

        let owned_label = OwnedLabel::new(desc.label.as_deref());
        BindGroup {
            id: wgn::wgpu_device_create_bind_group(
                self.id,
                &bm::BindGroupDescriptor {
                    layout: desc.layout.id,
                    entries: bindings.as_ptr(),
                    entries_length: bindings.len(),
                    label: owned_label.as_ptr(),
                },
            ),
        }
    }

    /// Creates a bind group layout.
    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        use wgc::binding_model as bm;

        let temp_layouts = desc
            .bindings
            .iter()
            .map(|bind| bm::BindGroupLayoutEntry {
                binding: bind.binding,
                visibility: bind.visibility,
                ty: match bind.ty {
                    BindingType::UniformBuffer { .. } => bm::BindingType::UniformBuffer,
                    BindingType::StorageBuffer {
                        readonly: false, ..
                    } => bm::BindingType::StorageBuffer,
                    BindingType::StorageBuffer { readonly: true, .. } => {
                        bm::BindingType::ReadonlyStorageBuffer
                    }
                    BindingType::Sampler { comparison: false } => bm::BindingType::Sampler,
                    BindingType::Sampler { .. } => bm::BindingType::ComparisonSampler,
                    BindingType::SampledTexture { .. } => bm::BindingType::SampledTexture,
                    BindingType::StorageTexture { readonly: true, .. } => {
                        bm::BindingType::ReadonlyStorageTexture
                    }
                    BindingType::StorageTexture { .. } => {
                        bm::BindingType::WriteonlyStorageTexture
                    }
                },
                has_dynamic_offset: match bind.ty {
                    BindingType::UniformBuffer { dynamic } |
                    BindingType::StorageBuffer { dynamic, .. } => dynamic,
                    _ => false,
                },
                multisampled: match bind.ty {
                    BindingType::SampledTexture { multisampled, .. } => multisampled,
                    _ => false,
                },
                view_dimension: match bind.ty {
                    BindingType::SampledTexture { dimension, .. } |
                    BindingType::StorageTexture { dimension, .. } => dimension,
                    _ => TextureViewDimension::D2,
                },
                texture_component_type: match bind.ty {
                    BindingType::SampledTexture { component_type, .. } |
                    BindingType::StorageTexture { component_type, .. } => component_type,
                    _ => TextureComponentType::Float,
                },
                storage_texture_format: match bind.ty {
                    BindingType::StorageTexture { format, .. } => format,
                    _ => TextureFormat::Rgb10a2Unorm, // doesn't matter
                },
            })
            .collect::<Vec<_>>();
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        BindGroupLayout {
            id: wgn::wgpu_device_create_bind_group_layout(
                self.id,
                &bm::BindGroupLayoutDescriptor {
                    entries: temp_layouts.as_ptr(),
                    entries_length: temp_layouts.len(),
                    label: owned_label.as_ptr(),
                },
            ),
        }
    }

    /// Creates a pipeline layout.
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        //TODO: avoid allocation here
        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id)
            .collect::<Vec<_>>();
        PipelineLayout {
            id: wgn::wgpu_device_create_pipeline_layout(
                self.id,
                &wgc::binding_model::PipelineLayoutDescriptor {
                    bind_group_layouts: temp_layouts.as_ptr(),
                    bind_group_layouts_length: temp_layouts.len(),
                },
            ),
        }
    }

    /// Creates a render pipeline.
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        use wgc::pipeline as pipe;

        let vertex_entry_point = CString::new(desc.vertex_stage.entry_point).unwrap();
        let vertex_stage = pipe::ProgrammableStageDescriptor {
            module: desc.vertex_stage.module.id,
            entry_point: vertex_entry_point.as_ptr(),
        };
        let (_fragment_entry_point, fragment_stage) =
            if let Some(fragment_stage) = &desc.fragment_stage {
                let fragment_entry_point = CString::new(fragment_stage.entry_point).unwrap();
                let fragment_stage = pipe::ProgrammableStageDescriptor {
                    module: fragment_stage.module.id,
                    entry_point: fragment_entry_point.as_ptr(),
                };
                (fragment_entry_point, Some(fragment_stage))
            } else {
                (CString::default(), None)
            };

        let temp_color_states = desc.color_states.to_vec();
        let temp_vertex_buffers = desc
            .vertex_state.vertex_buffers
            .iter()
            .map(|vbuf| pipe::VertexBufferLayoutDescriptor {
                array_stride: vbuf.stride,
                step_mode: vbuf.step_mode,
                attributes: vbuf.attributes.as_ptr(),
                attributes_length: vbuf.attributes.len(),
            })
            .collect::<Vec<_>>();

        RenderPipeline {
            id: wgn::wgpu_device_create_render_pipeline(
                self.id,
                &pipe::RenderPipelineDescriptor {
                    layout: desc.layout.id,
                    vertex_stage,
                    fragment_stage: fragment_stage
                        .as_ref()
                        .map_or(ptr::null(), |fs| fs as *const _),
                    rasterization_state: desc
                        .rasterization_state
                        .as_ref()
                        .map_or(ptr::null(), |p| p as *const _),
                    primitive_topology: desc.primitive_topology,
                    color_states: temp_color_states.as_ptr(),
                    color_states_length: temp_color_states.len(),
                    depth_stencil_state: desc
                        .depth_stencil_state
                        .as_ref()
                        .map_or(ptr::null(), |p| p as *const _),
                    vertex_state: pipe::VertexStateDescriptor {
                        index_format: desc.vertex_state.index_format,
                        vertex_buffers: temp_vertex_buffers.as_ptr(),
                        vertex_buffers_length: temp_vertex_buffers.len(),
                    },
                    sample_count: desc.sample_count,
                    sample_mask: desc.sample_mask,
                    alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
                },
            ),
        }
    }

    /// Creates a compute pipeline.
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        use wgc::pipeline as pipe;

        let entry_point = CString::new(desc.compute_stage.entry_point).unwrap();

        ComputePipeline {
            id: wgn::wgpu_device_create_compute_pipeline(
                self.id,
                &pipe::ComputePipelineDescriptor {
                    layout: desc.layout.id,
                    compute_stage: pipe::ProgrammableStageDescriptor {
                        module: desc.compute_stage.module.id,
                        entry_point: entry_point.as_ptr(),
                    },
                },
            ),
        }
    }

    /// Creates a new buffer.
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        Buffer {
            device_id: self.id,
            id: wgn::wgpu_device_create_buffer(
                self.id,
                &wgt::BufferDescriptor {
                    label: owned_label.as_ptr(),
                    size: desc.size,
                    usage: desc.usage,
                }
            ),
        }
    }

    /// Creates a new buffer and maps it into host-visible memory.
    ///
    /// This returns a [`CreateBufferMapped`], which exposes a `&mut [u8]`. The actual [`Buffer`]
    /// will not be created until calling [`CreateBufferMapped::finish`].
    pub fn create_buffer_mapped(&self, desc: &BufferDescriptor) -> CreateBufferMapped<'_> {
        assert_ne!(desc.size, 0);

        let owned_label = OwnedLabel::new(desc.label.as_deref());
        let mut data_ptr: *mut u8 = std::ptr::null_mut();

        let (id, data) = unsafe {
            let id = wgn::wgpu_device_create_buffer_mapped(
                self.id,
                &wgt::BufferDescriptor {
                    label: owned_label.as_ptr(),
                    size: desc.size,
                    usage: desc.usage,
                },
                &mut data_ptr as *mut *mut u8);
            let data = std::slice::from_raw_parts_mut(data_ptr as *mut u8, desc.size as usize);
            (id, data)
        };

        CreateBufferMapped { device_id: self.id, id, data }
    }

    /// Creates a new buffer, maps it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    pub fn create_buffer_with_data(&self, data: &[u8], usage: BufferUsage) -> Buffer {
        let mapped = self.create_buffer_mapped(&BufferDescriptor {
            size: data.len() as u64,
            usage,
            label: None,
        });
        mapped.data.copy_from_slice(data);
        mapped.finish()
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture {
        let owned_label = OwnedLabel::new(desc.label.as_deref());
        Texture {
            id: wgn::wgpu_device_create_texture(self.id, &wgt::TextureDescriptor {
                label: owned_label.as_ptr(),
                size: desc.size,
                array_layer_count: desc.array_layer_count,
                mip_level_count: desc.mip_level_count,
                sample_count: desc.sample_count,
                dimension: desc.dimension,
                format: desc.format,
                usage: desc.usage,
            }),
            owned: true,
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler {
        Sampler {
            id: wgn::wgpu_device_create_sampler(self.id, desc),
        }
    }

    /// Create a new [`SwapChain`] which targets `surface`.
    pub fn create_swap_chain(&self, surface: &Surface, desc: &SwapChainDescriptor) -> SwapChain {
        SwapChain {
            id: wgn::wgpu_device_create_swap_chain(self.id, surface.id, desc),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        wgn::wgpu_device_poll(self.id, true);
        //TODO: make this work in general
        #[cfg(feature = "metal-auto-capture")]
        wgn::wgpu_device_destroy(self.id);
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncErr;

pub struct BufferReadMapping {
    data: *const u8,
    size: usize,
    buffer_id: wgc::id::BufferId,
}

impl BufferReadMapping
{
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.data as *const u8, self.size)
        }
    }
}

impl Drop for BufferReadMapping {
    fn drop(&mut self) {
        wgn::wgpu_buffer_unmap(self.buffer_id);
    }
}

pub struct BufferWriteMapping {
    data: *mut u8,
    size: usize,
    buffer_id: wgc::id::BufferId,
}

impl BufferWriteMapping
{
    pub fn as_slice(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(self.data as *mut u8, self.size)
        }
    }
}

impl Drop for BufferWriteMapping {
    fn drop(&mut self) {
        wgn::wgpu_buffer_unmap(self.buffer_id);
    }
}



impl Buffer {
    /// Map the buffer for reading. The result is returned in a future.
    /// 
    /// For the future to complete, `device.poll(...)` must be called elsewhere in the runtime, possibly integrated
    /// into an event loop, run on a separate thread, or continually polled in the same task runtime that this
    /// future will be run on.
    /// 
    /// It's expected that wgpu will eventually supply its own event loop infrastructure that will be easy to integrate
    /// into other event loops, like winit's.
    pub fn map_read(&self, start: BufferAddress, size: BufferAddress) -> impl Future<Output = Result<BufferReadMapping, BufferAsyncErr>>
    {
        let (future, completion) = native_gpu_future::new_gpu_future(
            self.id,
            size,
        );

        extern "C" fn buffer_map_read_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            data: *const u8,
            user_data: *mut u8,
        )
        {
            let completion = unsafe {
                native_gpu_future::GpuFutureCompletion::from_raw(user_data as _)
            };
            let (buffer_id, size) = completion.get_buffer_info();

            if let wgc::resource::BufferMapAsyncStatus::Success = status {
                completion.complete(Ok(BufferReadMapping {
                    data,
                    size: size as usize,
                    buffer_id,
                }));
            } else {
                completion.complete(Err(BufferAsyncErr));
            }
        }

        wgn::wgpu_buffer_map_read_async(
            self.id,
            start,
            size,
            buffer_map_read_future_wrapper,
            completion.to_raw() as _,
        );

        future
    }

    /// Map the buffer for writing. The result is returned in a future.
    /// 
    /// See the documentation of (map_read)[#method.map_read] for more information about
    /// how to run this future.
    pub fn map_write(&self, start: BufferAddress, size: BufferAddress) -> impl Future<Output = Result<BufferWriteMapping, BufferAsyncErr>>
    {
        let (future, completion) = native_gpu_future::new_gpu_future(
            self.id,
            size,
        );

        extern "C" fn buffer_map_write_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            data: *mut u8,
            user_data: *mut u8,
        )
        {
            let completion = unsafe {
                native_gpu_future::GpuFutureCompletion::from_raw(user_data as _)
            };
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
}

impl Drop for Buffer {
    fn drop(&mut self) {
        wgn::wgpu_buffer_destroy(self.id);
    }
}

impl Texture {
    /// Creates a view of this texture.
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_view(self.id, Some(desc)),
            owned: true,
        }
    }

    /// Creates a default view of this whole texture.
    pub fn create_default_view(&self) -> TextureView {
        TextureView {
            id: wgn::wgpu_texture_create_view(self.id, None),
            owned: true,
        }
    }
}

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

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(self) -> CommandBuffer {
        CommandBuffer {
            id: wgn::wgpu_command_encoder_finish(self.id, None),
        }
    }

    /// Begins recording of a render pass.
    ///
    /// This function returns a [`RenderPass`] object which records a single render pass.
    pub fn begin_render_pass<'a>(
        &'a mut self,
        desc: &RenderPassDescriptor<'a, '_>,
    ) -> RenderPass<'a> {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| wgc::command::RenderPassColorAttachmentDescriptor {
                attachment: ca.attachment.id,
                resolve_target: ca.resolve_target.map(|rt| &rt.id),
                load_op: ca.load_op,
                store_op: ca.store_op,
                clear_color: ca.clear_color,
            })
            .collect::<ArrayVec<[_; 4]>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachmentDescriptor {
                attachment: dsa.attachment.id,
                depth_load_op: dsa.depth_load_op,
                depth_store_op: dsa.depth_store_op,
                clear_depth: dsa.clear_depth,
                stencil_load_op: dsa.stencil_load_op,
                stencil_store_op: dsa.stencil_store_op,
                clear_stencil: dsa.clear_stencil,
            }
        });

        RenderPass {
            id: unsafe {
                wgn::wgpu_command_encoder_begin_render_pass(
                    self.id,
                    &wgc::command::RenderPassDescriptor {
                        color_attachments: colors.as_ptr(),
                        color_attachments_length: colors.len(),
                        depth_stencil_attachment: depth_stencil.as_ref(),
                    },
                )
            },
            _parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self) -> ComputePass {
        ComputePass {
            id: unsafe {
                wgn::wgpu_command_encoder_begin_compute_pass(self.id, None)
            },
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
        wgn::wgpu_command_encoder_copy_buffer_to_buffer(
            self.id,
            source.id,
            source_offset,
            destination.id,
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
        wgn::wgpu_command_encoder_copy_buffer_to_texture(
            self.id,
            &source.into_native(),
            &destination.into_native(),
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
        wgn::wgpu_command_encoder_copy_texture_to_buffer(
            self.id,
            &source.into_native(),
            &destination.into_native(),
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
        wgn::wgpu_command_encoder_copy_texture_to_texture(
            self.id,
            &source.into_native(),
            &destination.into_native(),
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
        unsafe {
            wgn::wgpu_render_pass_set_bind_group(
                self.id.as_mut().unwrap(),
                index,
                bind_group.id,
                offsets.as_ptr(),
                offsets.len(),
            );
        }
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        unsafe {
            wgn::wgpu_render_pass_set_pipeline(
                self.id.as_mut().unwrap(),
                pipeline.id,
            );
        }
    }

    pub fn set_blend_color(&mut self, color: Color) {
        unsafe {
            wgn::wgpu_render_pass_set_blend_color(
                self.id.as_mut().unwrap(),
                &color,
            );
        }
    }

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
        unsafe {
            wgn::wgpu_render_pass_set_index_buffer(
                self.id.as_mut().unwrap(),
                buffer.id,
                offset,
                size,
            );
        }
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
        unsafe {
            wgn::wgpu_render_pass_set_vertex_buffer(
                self.id.as_mut().unwrap(),
                slot,
                buffer.id,
                offset,
                size,
            )
        };
    }

    /// Sets the scissor region.
    ///
    /// Subsequent draw calls will discard any fragments that fall outside this region.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        unsafe {
            wgn::wgpu_render_pass_set_scissor_rect(
                self.id.as_mut().unwrap(),
                x, y, w, h,
            );
        }
    }

    /// Sets the viewport region.
    ///
    /// Subsequent draw calls will draw any fragments in this region.
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        unsafe {
            wgn::wgpu_render_pass_set_viewport(
                self.id.as_mut().unwrap(),
                x, y, w, h,
                min_depth, max_depth,
            );
        }
    }

    /// Sets the stencil reference.
    ///
    /// Subsequent stencil tests will test against this value.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        unsafe {
            wgn::wgpu_render_pass_set_stencil_reference(
                self.id.as_mut().unwrap(),
                reference,
            );
        }
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffers`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        unsafe {
            wgn::wgpu_render_pass_draw(
                self.id.as_mut().unwrap(),
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffers`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        unsafe {
            wgn::wgpu_render_pass_draw_indexed(
                self.id.as_mut().unwrap(),
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
        }
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffers`].
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
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffers`].
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
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            unsafe {
                wgn::wgpu_render_pass_end_pass(self.id);
            }
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
        unsafe {
            wgn::wgpu_compute_pass_set_bind_group(
                self.id.as_mut().unwrap(),
                index,
                bind_group.id,
                offsets.as_ptr(),
                offsets.len(),
            );
        }
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &'a ComputePipeline) {
        unsafe {
            wgn::wgpu_compute_pass_set_pipeline(
                self.id.as_mut().unwrap(),
                pipeline.id,
            );
        }
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            wgn::wgpu_compute_pass_dispatch(
                self.id.as_mut().unwrap(),
                x, y, z,
            );
        }
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    pub fn dispatch_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        unsafe {
            wgn::wgpu_compute_pass_dispatch_indirect(
                self.id.as_mut().unwrap(),
                indirect_buffer.id,
                indirect_offset,
            );
        }
    }
}

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            unsafe {
                wgn::wgpu_compute_pass_end_pass(self.id);
            }
        }
    }
}

impl Queue {
    /// Submits a series of finished command buffers for execution.
    pub fn submit(&self, command_buffers: &[CommandBuffer]) {
        let temp_command_buffers = command_buffers.iter()
            .map(|cb| cb.id)
            .collect::<SmallVec<[_; 4]>>();

        unsafe {
            wgn::wgpu_queue_submit(
                self.id,
                temp_command_buffers.as_ptr(),
                command_buffers.len(),
            )
        };
    }
}

impl Drop for SwapChainOutput {
    fn drop(&mut self) {
        if !thread::panicking() {
            wgn::wgpu_swap_chain_present(self.swap_chain_id);
        }
    }
}

impl SwapChain {
    /// Returns the next texture to be presented by the swapchain for drawing.
    ///
    /// When the [`SwapChainOutput`] returned by this method is dropped, the swapchain will present
    /// the texture to the associated [`Surface`].
    ///
    /// Returns an `Err` if the GPU timed out when attempting to acquire the next texture or if a
    /// previous output is still alive.
    pub fn get_next_texture(&mut self) -> Result<SwapChainOutput, ()> {
        let output = wgn::wgpu_swap_chain_get_next_texture(self.id);
        if output.view_id == wgc::id::Id::ERROR {
            Err(())
        } else {
            Ok(SwapChainOutput {
                view: TextureView {
                    id: output.view_id,
                    owned: false,
                },
                swap_chain_id: self.id,
            })
        }
    }
}

struct OwnedLabel(Option<CString>);

impl OwnedLabel {
    fn new(text: Option<&str>) -> Self {
        Self(text.map(|t| CString::new(t).expect("invalid label")))
    }

    fn as_ptr(&self) -> *const std::os::raw::c_char {
        match self.0 {
            Some(ref c_string) => c_string.as_ptr(),
            None => ptr::null(),
        }
    }
}
