//! A cross-platform graphics and compute library based on [WebGPU](https://gpuweb.github.io/gpuweb/).
//!
//! To start using the API, create an [`Instance`].

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu/master/logo.png")]
#![warn(missing_docs, unsafe_op_in_unsafe_fn)]

mod backend;
mod context;
pub mod util;
#[macro_use]
mod macros;

use std::{
    any::Any,
    borrow::Cow,
    error,
    fmt::{Debug, Display},
    future::Future,
    marker::PhantomData,
    num::NonZeroU32,
    ops::{Bound, Deref, DerefMut, Range, RangeBounds},
    sync::Arc,
    thread,
};

use context::{Context, DeviceRequest, DynContext, ObjectId};
use parking_lot::Mutex;

pub use wgt::{
    AdapterInfo, AddressMode, AstcBlock, AstcChannel, Backend, Backends, BindGroupLayoutEntry,
    BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferAddress,
    BufferBindingType, BufferSize, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandBufferDescriptor, CompareFunction, CompositeAlphaMode, DepthBiasState,
    DepthStencilState, DeviceType, DownlevelCapabilities, DownlevelFlags, Dx12Compiler,
    DynamicOffset, Extent3d, Face, Features, FilterMode, FrontFace, ImageDataLayout,
    ImageSubresourceRange, IndexFormat, InstanceDescriptor, Limits, MultisampleState, Origin2d,
    Origin3d, PipelineStatisticsTypes, PolygonMode, PowerPreference, PredefinedColorSpace,
    PresentMode, PresentationTimestamp, PrimitiveState, PrimitiveTopology, PushConstantRange,
    QueryType, RenderBundleDepthStencil, SamplerBindingType, SamplerBorderColor, ShaderLocation,
    ShaderModel, ShaderStages, StencilFaceState, StencilOperation, StencilState,
    StorageTextureAccess, SurfaceCapabilities, SurfaceStatus, TextureAspect, TextureDimension,
    TextureFormat, TextureFormatFeatureFlags, TextureFormatFeatures, TextureSampleType,
    TextureUsages, TextureViewDimension, VertexAttribute, VertexFormat, VertexStepMode,
    COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT, MAP_ALIGNMENT, PUSH_CONSTANT_ALIGNMENT,
    QUERY_RESOLVE_BUFFER_ALIGNMENT, QUERY_SET_MAX_QUERIES, QUERY_SIZE, VERTEX_STRIDE_ALIGNMENT,
};

// wasm-only types, we try to keep as many types non-platform
// specific, but these need to depend on web-sys.
#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
pub use wgt::{ExternalImageSource, ImageCopyExternalImage};
#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
static_assertions::assert_impl_all!(ExternalImageSource: Send, Sync);

/// Filter for error scopes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum ErrorFilter {
    /// Catch only out-of-memory errors.
    OutOfMemory,
    /// Catch only validation errors.
    Validation,
}
static_assertions::assert_impl_all!(ErrorFilter: Send, Sync);

type C = dyn DynContext;
type Data = dyn Any + Send + Sync;

/// Context for all other wgpu objects. Instance of wgpu.
///
/// This is the first thing you create when using wgpu.
/// Its primary use is to create [`Adapter`]s and [`Surface`]s.
///
/// Does not have to be kept alive.
///
/// Corresponds to [WebGPU `GPU`](https://gpuweb.github.io/gpuweb/#gpu-interface).
#[derive(Debug)]
pub struct Instance {
    context: Arc<C>,
}
static_assertions::assert_impl_all!(Instance: Send, Sync);

/// Handle to a physical graphics and/or compute device.
///
/// Adapters can be used to open a connection to the corresponding [`Device`]
/// on the host system by using [`Adapter::request_device`].
///
/// Does not have to be kept alive.
///
/// Corresponds to [WebGPU `GPUAdapter`](https://gpuweb.github.io/gpuweb/#gpu-adapter).
#[derive(Debug)]
pub struct Adapter {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(Adapter: Send, Sync);

impl Drop for Adapter {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.adapter_drop(&self.id, self.data.as_ref())
        }
    }
}

/// Open connection to a graphics and/or compute device.
///
/// Responsible for the creation of most rendering and compute resources.
/// These are then used in commands, which are submitted to a [`Queue`].
///
/// A device may be requested from an adapter with [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUDevice`](https://gpuweb.github.io/gpuweb/#gpu-device).
#[derive(Debug)]
pub struct Device {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(Device: Send, Sync);

/// Identifier for a particular call to [`Queue::submit`]. Can be used
/// as part of an argument to [`Device::poll`] to block for a particular
/// submission to finish.
///
/// This type is unique to the Rust API of `wgpu`.
/// There is no analogue in the WebGPU specification.
#[derive(Debug, Clone)]
pub struct SubmissionIndex(ObjectId, Arc<crate::Data>);
static_assertions::assert_impl_all!(SubmissionIndex: Send, Sync);

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
        Self {
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
                "Intersecting map range with {sub:?}"
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
/// Created with [`Device::create_buffer`] or
/// [`DeviceExt::create_buffer_init`](util::DeviceExt::create_buffer_init).
///
/// Corresponds to [WebGPU `GPUBuffer`](https://gpuweb.github.io/gpuweb/#buffer-interface).
#[derive(Debug)]
pub struct Buffer {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
    map_context: Mutex<MapContext>,
    size: wgt::BufferAddress,
    usage: BufferUsages,
    // Todo: missing map_state https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapstate
}
static_assertions::assert_impl_all!(Buffer: Send, Sync);

/// Slice into a [`Buffer`].
///
/// It can be created with [`Buffer::slice`]. To use the whole buffer, call with unbounded slice:
///
/// `buffer.slice(..)`
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// an offset and size are specified as arguments to each call working with the [`Buffer`], instead.
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: Option<BufferSize>,
}
static_assertions::assert_impl_all!(BufferSlice: Send, Sync);

/// Handle to a texture on the GPU.
///
/// It can be created with [`Device::create_texture`].
///
/// Corresponds to [WebGPU `GPUTexture`](https://gpuweb.github.io/gpuweb/#texture-interface).
#[derive(Debug)]
pub struct Texture {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
    owned: bool,
    descriptor: TextureDescriptor<'static>,
}
static_assertions::assert_impl_all!(Texture: Send, Sync);

/// Handle to a texture view.
///
/// A `TextureView` object describes a texture and associated metadata needed by a
/// [`RenderPipeline`] or [`BindGroup`].
///
/// Corresponds to [WebGPU `GPUTextureView`](https://gpuweb.github.io/gpuweb/#gputextureview).
#[derive(Debug)]
pub struct TextureView {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(TextureView: Send, Sync);

/// Handle to a sampler.
///
/// A `Sampler` object defines how a pipeline will sample from a [`TextureView`]. Samplers define
/// image filters (including anisotropy) and address (wrapping) modes, among other things. See
/// the documentation for [`SamplerDescriptor`] for more information.
///
/// It can be created with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSampler`](https://gpuweb.github.io/gpuweb/#sampler-interface).
#[derive(Debug)]
pub struct Sampler {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(Sampler: Send, Sync);

impl Drop for Sampler {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.sampler_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a [`Surface`].
///
/// For use with [`Surface::configure`].
///
/// Corresponds to [WebGPU `GPUCanvasConfiguration`](
/// https://gpuweb.github.io/gpuweb/#canvas-configuration).
pub type SurfaceConfiguration = wgt::SurfaceConfiguration<Vec<TextureFormat>>;
static_assertions::assert_impl_all!(SurfaceConfiguration: Send, Sync);

/// Handle to a presentable surface.
///
/// A `Surface` represents a platform-specific surface (e.g. a window) onto which rendered images may
/// be presented. A `Surface` may be created with the unsafe function [`Instance::create_surface`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// [`GPUCanvasContext`](https://gpuweb.github.io/gpuweb/#canvas-context)
/// serves a similar role.
#[derive(Debug)]
pub struct Surface {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
    // Stores the latest `SurfaceConfiguration` that was set using `Surface::configure`.
    // It is required to set the attributes of the `SurfaceTexture` in the
    // `Surface::get_current_texture` method.
    // Because the `Surface::configure` method operates on an immutable reference this type has to
    // be wrapped in a mutex and since the configuration is only supplied after the surface has
    // been created is is additionally wrapped in an option.
    config: Mutex<Option<SurfaceConfiguration>>,
}
static_assertions::assert_impl_all!(Surface: Send, Sync);

impl Drop for Surface {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.surface_drop(&self.id, self.data.as_ref())
        }
    }
}

/// Handle to a binding group layout.
///
/// A `BindGroupLayout` is a handle to the GPU-side layout of a binding group. It can be used to
/// create a [`BindGroupDescriptor`] object, which in turn can be used to create a [`BindGroup`]
/// object with [`Device::create_bind_group`]. A series of `BindGroupLayout`s can also be used to
/// create a [`PipelineLayoutDescriptor`], which can be used to create a [`PipelineLayout`].
///
/// It can be created with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayout`](
/// https://gpuweb.github.io/gpuweb/#gpubindgrouplayout).
#[derive(Debug)]
pub struct BindGroupLayout {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(BindGroupLayout: Send, Sync);

impl Drop for BindGroupLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .bind_group_layout_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Handle to a binding group.
///
/// A `BindGroup` represents the set of resources bound to the bindings described by a
/// [`BindGroupLayout`]. It can be created with [`Device::create_bind_group`]. A `BindGroup` can
/// be bound to a particular [`RenderPass`] with [`RenderPass::set_bind_group`], or to a
/// [`ComputePass`] with [`ComputePass::set_bind_group`].
///
/// Corresponds to [WebGPU `GPUBindGroup`](https://gpuweb.github.io/gpuweb/#gpubindgroup).
#[derive(Debug)]
pub struct BindGroup {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(BindGroup: Send, Sync);

impl Drop for BindGroup {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.bind_group_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Handle to a compiled shader module.
///
/// A `ShaderModule` represents a compiled shader module on the GPU. It can be created by passing
/// source code to [`Device::create_shader_module`] or valid SPIR-V binary to
/// [`Device::create_shader_module_spirv`]. Shader modules are used to define programmable stages
/// of a pipeline.
///
/// Corresponds to [WebGPU `GPUShaderModule`](https://gpuweb.github.io/gpuweb/#shader-module).
#[derive(Debug)]
pub struct ShaderModule {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(ShaderModule: Send, Sync);

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .shader_module_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Source of a shader module.
///
/// The source will be parsed and validated.
///
/// Any necessary shader translation (e.g. from WGSL to SPIR-V or vice versa)
/// will be done internally by wgpu.
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// only WGSL source code strings are accepted.
#[cfg_attr(feature = "naga", allow(clippy::large_enum_variant))]
#[derive(Clone)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// SPIR-V module represented as a slice of words.
    ///
    /// See also: [`util::make_spirv`], [`include_spirv`]
    #[cfg(feature = "spirv")]
    SpirV(Cow<'a, [u32]>),
    /// GLSL module as a string slice.
    ///
    /// Note: GLSL is not yet fully supported and must be a specific ShaderStage.
    #[cfg(feature = "glsl")]
    Glsl {
        /// The source code of the shader.
        shader: Cow<'a, str>,
        /// The shader stage that the shader targets. For example, `naga::ShaderStage::Vertex`
        stage: naga::ShaderStage,
        /// Defines to unlock configured shader features.
        defines: naga::FastHashMap<String, String>,
    },
    /// WGSL module as a string slice.
    #[cfg(feature = "wgsl")]
    Wgsl(Cow<'a, str>),
    /// Naga module.
    #[cfg(feature = "naga")]
    Naga(Cow<'static, naga::Module>),
    /// Dummy variant because `Naga` doesn't have a lifetime and without enough active features it
    /// could be the last one active.
    #[doc(hidden)]
    Dummy(PhantomData<&'a ()>),
}
static_assertions::assert_impl_all!(ShaderSource: Send, Sync);

/// Descriptor for use with [`Device::create_shader_module`].
///
/// Corresponds to [WebGPU `GPUShaderModuleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpushadermoduledescriptor).
#[derive(Clone)]
pub struct ShaderModuleDescriptor<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Source code for the shader.
    pub source: ShaderSource<'a>,
}
static_assertions::assert_impl_all!(ShaderModuleDescriptor: Send, Sync);

/// Descriptor for a shader module given by SPIR-V binary, for use with
/// [`Device::create_shader_module_spirv`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// only WGSL source code strings are accepted.
pub struct ShaderModuleDescriptorSpirV<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Binary SPIR-V data, in 4-byte words.
    pub source: Cow<'a, [u32]>,
}
static_assertions::assert_impl_all!(ShaderModuleDescriptorSpirV: Send, Sync);

/// Handle to a pipeline layout.
///
/// A `PipelineLayout` object describes the available binding groups of a pipeline.
/// It can be created with [`Device::create_pipeline_layout`].
///
/// Corresponds to [WebGPU `GPUPipelineLayout`](https://gpuweb.github.io/gpuweb/#gpupipelinelayout).
#[derive(Debug)]
pub struct PipelineLayout {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(PipelineLayout: Send, Sync);

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .pipeline_layout_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Handle to a rendering (graphics) pipeline.
///
/// A `RenderPipeline` object represents a graphics pipeline and its stages, bindings, vertex
/// buffers and targets. It can be created with [`Device::create_render_pipeline`].
///
/// Corresponds to [WebGPU `GPURenderPipeline`](https://gpuweb.github.io/gpuweb/#render-pipeline).
#[derive(Debug)]
pub struct RenderPipeline {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(RenderPipeline: Send, Sync);

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .render_pipeline_drop(&self.id, self.data.as_ref());
        }
    }
}

impl RenderPipeline {
    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        let (id, data) =
            self.context
                .render_pipeline_get_bind_group_layout(&self.id, self.data.as_ref(), index);
        BindGroupLayout { context, id, data }
    }
}

/// Handle to a compute pipeline.
///
/// A `ComputePipeline` object represents a compute pipeline and its single shader stage.
/// It can be created with [`Device::create_compute_pipeline`].
///
/// Corresponds to [WebGPU `GPUComputePipeline`](https://gpuweb.github.io/gpuweb/#compute-pipeline).
#[derive(Debug)]
pub struct ComputePipeline {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(ComputePipeline: Send, Sync);

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .compute_pipeline_drop(&self.id, self.data.as_ref());
        }
    }
}

impl ComputePipeline {
    /// Get an object representing the bind group layout at a given index.
    pub fn get_bind_group_layout(&self, index: u32) -> BindGroupLayout {
        let context = Arc::clone(&self.context);
        let (id, data) = self.context.compute_pipeline_get_bind_group_layout(
            &self.id,
            self.data.as_ref(),
            index,
        );
        BindGroupLayout { context, id, data }
    }
}

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
///
/// Corresponds to [WebGPU `GPUCommandBuffer`](https://gpuweb.github.io/gpuweb/#command-buffer).
#[derive(Debug)]
pub struct CommandBuffer {
    context: Arc<C>,
    id: Option<ObjectId>,
    data: Option<Box<Data>>,
}
static_assertions::assert_impl_all!(CommandBuffer: Send, Sync);

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            if let Some(ref id) = self.id {
                self.context.command_buffer_drop(id, &self.data.take());
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
///
/// Corresponds to [WebGPU `GPUCommandEncoder`](https://gpuweb.github.io/gpuweb/#command-encoder).
#[derive(Debug)]
pub struct CommandEncoder {
    context: Arc<C>,
    id: Option<ObjectId>,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(CommandEncoder: Send, Sync);

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if !thread::panicking() {
            if let Some(id) = self.id.take() {
                self.context.command_encoder_drop(&id, self.data.as_ref());
            }
        }
    }
}

/// In-progress recording of a render pass.
///
/// It can be created with [`CommandEncoder::begin_render_pass`].
///
/// Corresponds to [WebGPU `GPURenderPassEncoder`](
/// https://gpuweb.github.io/gpuweb/#render-pass-encoder).
#[derive(Debug)]
pub struct RenderPass<'a> {
    id: ObjectId,
    data: Box<Data>,
    parent: &'a mut CommandEncoder,
}

/// In-progress recording of a compute pass.
///
/// It can be created with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassEncoder`](
/// https://gpuweb.github.io/gpuweb/#compute-pass-encoder).
#[derive(Debug)]
pub struct ComputePass<'a> {
    id: ObjectId,
    data: Box<Data>,
    parent: &'a mut CommandEncoder,
}

/// Encodes a series of GPU operations into a reusable "render bundle".
///
/// It only supports a handful of render commands, but it makes them reusable.
/// It can be created with [`Device::create_render_bundle_encoder`].
/// It can be executed onto a [`CommandEncoder`] using [`RenderPass::execute_bundles`].
///
/// Executing a [`RenderBundle`] is often more efficient than issuing the underlying commands
/// manually.
///
/// Corresponds to [WebGPU `GPURenderBundleEncoder`](
/// https://gpuweb.github.io/gpuweb/#gpurenderbundleencoder).
#[derive(Debug)]
pub struct RenderBundleEncoder<'a> {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
    parent: &'a Device,
    /// This type should be !Send !Sync, because it represents an allocation on this thread's
    /// command buffer.
    _p: PhantomData<*const u8>,
}
static_assertions::assert_not_impl_any!(RenderBundleEncoder<'_>: Send, Sync);

/// Pre-prepared reusable bundle of GPU operations.
///
/// It only supports a handful of render commands, but it makes them reusable. Executing a
/// [`RenderBundle`] is often more efficient than issuing the underlying commands manually.
///
/// It can be created by use of a [`RenderBundleEncoder`], and executed onto a [`CommandEncoder`]
/// using [`RenderPass::execute_bundles`].
///
/// Corresponds to [WebGPU `GPURenderBundle`](https://gpuweb.github.io/gpuweb/#render-bundle).
#[derive(Debug)]
pub struct RenderBundle {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(RenderBundle: Send, Sync);

impl Drop for RenderBundle {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .render_bundle_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Handle to a query set.
///
/// It can be created with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySet`](https://gpuweb.github.io/gpuweb/#queryset).
pub struct QuerySet {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(QuerySet: Send, Sync);

impl Drop for QuerySet {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.query_set_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects and provides convenience methods
/// for writing to [buffers](Queue::write_buffer) and [textures](Queue::write_texture).
/// It can be created along with a [`Device`] by calling [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUQueue`](https://gpuweb.github.io/gpuweb/#gpu-queue).
#[derive(Debug)]
pub struct Queue {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
static_assertions::assert_impl_all!(Queue: Send, Sync);

/// Resource that can be bound to a pipeline.
///
/// Corresponds to [WebGPU `GPUBindingResource`](
/// https://gpuweb.github.io/gpuweb/#typedefdef-gpubindingresource).
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
    /// Binding is backed by an array of samplers.
    ///
    /// [`Features::TEXTURE_BINDING_ARRAY`] must be supported to use this feature.
    ///
    /// Corresponds to [`wgt::BindingType::Sampler`] with [`BindGroupLayoutEntry::count`] set
    /// to Some.
    SamplerArray(&'a [&'a Sampler]),
    /// Binding is backed by a texture.
    ///
    /// Corresponds to [`wgt::BindingType::Texture`] and [`wgt::BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to None.
    TextureView(&'a TextureView),
    /// Binding is backed by an array of textures.
    ///
    /// [`Features::TEXTURE_BINDING_ARRAY`] must be supported to use this feature.
    ///
    /// Corresponds to [`wgt::BindingType::Texture`] and [`wgt::BindingType::StorageTexture`] with
    /// [`BindGroupLayoutEntry::count`] set to Some.
    TextureViewArray(&'a [&'a TextureView]),
}
static_assertions::assert_impl_all!(BindingResource: Send, Sync);

/// Describes the segment of a buffer to bind.
///
/// Corresponds to [WebGPU `GPUBufferBinding`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbinding).
#[derive(Clone, Debug)]
pub struct BufferBinding<'a> {
    /// The buffer to bind.
    pub buffer: &'a Buffer,
    /// Base offset of the buffer. For bindings with `dynamic == true`, this offset
    /// will be added to the dynamic offset provided in [`RenderPass::set_bind_group`].
    ///
    /// The offset has to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub offset: BufferAddress,
    /// Size of the binding, or `None` for using the rest of the buffer.
    pub size: Option<BufferSize>,
}
static_assertions::assert_impl_all!(BufferBinding: Send, Sync);

/// Operation to perform to the output attachment at the start of a render pass.
///
/// The render target must be cleared at least once before its content is loaded.
///
/// Corresponds to [WebGPU `GPULoadOp`](https://gpuweb.github.io/gpuweb/#enumdef-gpuloadop).
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
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// separate `loadOp` and `storeOp` fields are used instead.
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
///
/// For use with [`RenderPassDescriptor`].
///
/// Corresponds to [WebGPU `GPURenderPassColorAttachment`](
/// https://gpuweb.github.io/gpuweb/#color-attachments).
#[derive(Clone, Debug)]
pub struct RenderPassColorAttachment<'tex> {
    /// The view to use as an attachment.
    pub view: &'tex TextureView,
    /// The view that will receive the resolved output if multisampling is used.
    pub resolve_target: Option<&'tex TextureView>,
    /// What operations will be performed on this color attachment.
    pub ops: Operations<Color>,
}
static_assertions::assert_impl_all!(RenderPassColorAttachment: Send, Sync);

/// Describes a depth/stencil attachment to a [`RenderPass`].
///
/// For use with [`RenderPassDescriptor`].
///
/// Corresponds to [WebGPU `GPURenderPassDepthStencilAttachment`](
/// https://gpuweb.github.io/gpuweb/#depth-stencil-attachments).
#[derive(Clone, Debug)]
pub struct RenderPassDepthStencilAttachment<'tex> {
    /// The view to use as an attachment.
    pub view: &'tex TextureView,
    /// What operations will be performed on the depth part of the attachment.
    pub depth_ops: Option<Operations<f32>>,
    /// What operations will be performed on the stencil part of the attachment.
    pub stencil_ops: Option<Operations<u32>>,
}
static_assertions::assert_impl_all!(RenderPassDepthStencilAttachment: Send, Sync);

// The underlying types are also exported so that documentation shows up for them

/// Object debugging label.
pub type Label<'a> = Option<&'a str>;
pub use wgt::RequestAdapterOptions as RequestAdapterOptionsBase;
/// Additional information required when requesting an adapter.
///
/// For use with [`Instance::request_adapter`].
///
/// Corresponds to [WebGPU `GPURequestAdapterOptions`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions).
pub type RequestAdapterOptions<'a> = RequestAdapterOptionsBase<&'a Surface>;
static_assertions::assert_impl_all!(RequestAdapterOptions: Send, Sync);
/// Describes a [`Device`].
///
/// For use with [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUDeviceDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudevicedescriptor).
pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(DeviceDescriptor: Send, Sync);
/// Describes a [`Buffer`].
///
/// For use with [`Device::create_buffer`].
///
/// Corresponds to [WebGPU `GPUBufferDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferdescriptor).
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(BufferDescriptor: Send, Sync);
/// Describes a [`CommandEncoder`].
///
/// For use with [`Device::create_command_encoder`].
///
/// Corresponds to [WebGPU `GPUCommandEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucommandencoderdescriptor).
pub type CommandEncoderDescriptor<'a> = wgt::CommandEncoderDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CommandEncoderDescriptor: Send, Sync);
/// Describes a [`RenderBundle`].
///
/// For use with [`RenderBundleEncoder::finish`].
///
/// Corresponds to [WebGPU `GPURenderBundleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundledescriptor).
pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(RenderBundleDescriptor: Send, Sync);
/// Describes a [`Texture`].
///
/// For use with [`Device::create_texture`].
///
/// Corresponds to [WebGPU `GPUTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturedescriptor).
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, &'a [TextureFormat]>;
static_assertions::assert_impl_all!(TextureDescriptor: Send, Sync);
/// Describes a [`QuerySet`].
///
/// For use with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySetDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuquerysetdescriptor).
pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(QuerySetDescriptor: Send, Sync);
pub use wgt::Maintain as MaintainBase;
/// Passed to [`Device::poll`] to control how and if it should block.
pub type Maintain = wgt::Maintain<SubmissionIndex>;
static_assertions::assert_impl_all!(Maintain: Send, Sync);

/// Describes a [`TextureView`].
///
/// For use with [`Texture::create_view`].
///
/// Corresponds to [WebGPU `GPUTextureViewDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputextureviewdescriptor).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view. Either must be the same as the texture format or in the list
    /// of `view_formats` in the texture's descriptor.
    pub format: Option<TextureFormat>,
    /// The dimension of the texture view. For 1D textures, this must be `D1`. For 2D textures it must be one of
    /// `D2`, `D2Array`, `Cube`, and `CubeArray`. For 3D textures it must be `D3`
    pub dimension: Option<TextureViewDimension>,
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`].
    pub aspect: TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<u32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<u32>,
}
static_assertions::assert_impl_all!(TextureViewDescriptor: Send, Sync);

/// Describes a [`PipelineLayout`].
///
/// For use with [`Device::create_pipeline_layout`].
///
/// Corresponds to [WebGPU `GPUPipelineLayoutDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpupipelinelayoutdescriptor).
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
static_assertions::assert_impl_all!(PipelineLayoutDescriptor: Send, Sync);

/// Describes a [`Sampler`].
///
/// For use with [`Device::create_sampler`].
///
/// Corresponds to [WebGPU `GPUSamplerDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpusamplerdescriptor).
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
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
    /// Border color to use when address_mode is [`AddressMode::ClampToBorder`]
    pub border_color: Option<SamplerBorderColor>,
}
static_assertions::assert_impl_all!(SamplerDescriptor: Send, Sync);

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
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

/// An element of a [`BindGroupDescriptor`], consisting of a bindable resource
/// and the slot to bind it to.
///
/// Corresponds to [WebGPU `GPUBindGroupEntry`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgroupentry).
#[derive(Clone, Debug)]
pub struct BindGroupEntry<'a> {
    /// Slot for which binding provides resource. Corresponds to an entry of the same
    /// binding index in the [`BindGroupLayoutDescriptor`].
    pub binding: u32,
    /// Resource to attach to the binding
    pub resource: BindingResource<'a>,
}
static_assertions::assert_impl_all!(BindGroupEntry: Send, Sync);

/// Describes a group of bindings and the resources to be bound.
///
/// For use with [`Device::create_bind_group`].
///
/// Corresponds to [WebGPU `GPUBindGroupDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgroupdescriptor).
#[derive(Clone, Debug)]
pub struct BindGroupDescriptor<'a> {
    /// Debug label of the bind group. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The [`BindGroupLayout`] that corresponds to this bind group.
    pub layout: &'a BindGroupLayout,
    /// The resources to bind to this bind group.
    pub entries: &'a [BindGroupEntry<'a>],
}
static_assertions::assert_impl_all!(BindGroupDescriptor: Send, Sync);

/// Describes the attachments of a render pass.
///
/// For use with [`CommandEncoder::begin_render_pass`].
///
/// Note: separate lifetimes are needed because the texture views (`'tex`)
/// have to live as long as the pass is recorded, while everything else (`'desc`) doesn't.
///
/// Corresponds to [WebGPU `GPURenderPassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdescriptor).
#[derive(Clone, Debug, Default)]
pub struct RenderPassDescriptor<'tex, 'desc> {
    /// Debug label of the render pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'desc>,
    /// The color attachments of the render pass.
    pub color_attachments: &'desc [Option<RenderPassColorAttachment<'tex>>],
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'tex>>,
}
static_assertions::assert_impl_all!(RenderPassDescriptor: Send, Sync);

/// Describes how the vertex buffer is interpreted.
///
/// For use in [`VertexState`].
///
/// Corresponds to [WebGPU `GPUVertexBufferLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdescriptor).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [VertexAttribute],
}
static_assertions::assert_impl_all!(VertexBufferLayout: Send, Sync);

/// Describes the vertex processing in a render pipeline.
///
/// For use in [`RenderPipelineDescriptor`].
///
/// Corresponds to [WebGPU `GPUVertexState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexstate).
#[derive(Clone, Debug)]
pub struct VertexState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function with this name
    /// in the shader.
    pub entry_point: &'a str,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}
static_assertions::assert_impl_all!(VertexState: Send, Sync);

/// Describes the fragment processing in a render pipeline.
///
/// For use in [`RenderPipelineDescriptor`].
///
/// Corresponds to [WebGPU `GPUFragmentState`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpufragmentstate).
#[derive(Clone, Debug)]
pub struct FragmentState<'a> {
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function with this name
    /// in the shader.
    pub entry_point: &'a str,
    /// The color state of the render targets.
    pub targets: &'a [Option<ColorTargetState>],
}
static_assertions::assert_impl_all!(FragmentState: Send, Sync);

/// Describes a render (graphics) pipeline.
///
/// For use with [`Device::create_render_pipeline`].
///
/// Corresponds to [WebGPU `GPURenderPipelineDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpipelinedescriptor).
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
    /// If the pipeline will be used with a multiview render pass, this indicates how many array
    /// layers the attachments will have.
    pub multiview: Option<NonZeroU32>,
}
static_assertions::assert_impl_all!(RenderPipelineDescriptor: Send, Sync);

/// Describes the attachments of a compute pass.
///
/// For use with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepassdescriptor).
#[derive(Clone, Debug, Default)]
pub struct ComputePassDescriptor<'a> {
    /// Debug label of the compute pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
}
static_assertions::assert_impl_all!(ComputePassDescriptor: Send, Sync);

/// Describes a compute pipeline.
///
/// For use with [`Device::create_compute_pipeline`].
///
/// Corresponds to [WebGPU `GPUComputePipelineDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepipelinedescriptor).
#[derive(Clone, Debug)]
pub struct ComputePipelineDescriptor<'a> {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The layout of bind groups for this pipeline.
    pub layout: Option<&'a PipelineLayout>,
    /// The compiled shader module for this stage.
    pub module: &'a ShaderModule,
    /// The name of the entry point in the compiled shader. There must be a function with this name
    /// and no return value in the shader.
    pub entry_point: &'a str,
}
static_assertions::assert_impl_all!(ComputePipelineDescriptor: Send, Sync);

pub use wgt::ImageCopyBuffer as ImageCopyBufferBase;
/// View of a buffer which can be used to copy to/from a texture.
///
/// Corresponds to [WebGPU `GPUImageCopyBuffer`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopybuffer).
pub type ImageCopyBuffer<'a> = ImageCopyBufferBase<&'a Buffer>;
static_assertions::assert_impl_all!(ImageCopyBuffer: Send, Sync);

pub use wgt::ImageCopyTexture as ImageCopyTextureBase;
/// View of a texture which can be used to copy to/from a buffer/texture.
///
/// Corresponds to [WebGPU `GPUImageCopyTexture`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexture).
pub type ImageCopyTexture<'a> = ImageCopyTextureBase<&'a Texture>;
static_assertions::assert_impl_all!(ImageCopyTexture: Send, Sync);

pub use wgt::ImageCopyTextureTagged as ImageCopyTextureTaggedBase;
/// View of a texture which can be used to copy to a texture, including
/// color space and alpha premultiplication information.
///
/// Corresponds to [WebGPU `GPUImageCopyTextureTagged`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexturetagged).
pub type ImageCopyTextureTagged<'a> = ImageCopyTextureTaggedBase<&'a Texture>;
static_assertions::assert_impl_all!(ImageCopyTexture: Send, Sync);

/// Describes a [`BindGroupLayout`].
///
/// For use with [`Device::create_bind_group_layout`].
///
/// Corresponds to [WebGPU `GPUBindGroupLayoutDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutdescriptor).
#[derive(Clone, Debug)]
pub struct BindGroupLayoutDescriptor<'a> {
    /// Debug label of the bind group layout. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,

    /// Array of entries in this BindGroupLayout
    pub entries: &'a [BindGroupLayoutEntry],
}
static_assertions::assert_impl_all!(BindGroupLayoutDescriptor: Send, Sync);

/// Describes a [`RenderBundleEncoder`].
///
/// For use with [`Device::create_render_bundle_encoder`].
///
/// Corresponds to [WebGPU `GPURenderBundleEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundleencoderdescriptor).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RenderBundleEncoderDescriptor<'a> {
    /// Debug label of the render bundle encoder. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The formats of the color attachments that this render bundle is capable to rendering to. This
    /// must match the formats of the color attachments in the render pass this render bundle is executed in.
    pub color_formats: &'a [Option<TextureFormat>],
    /// Information about the depth attachment that this render bundle is capable to rendering to. This
    /// must match the format of the depth attachments in the render pass this render bundle is executed in.
    pub depth_stencil: Option<RenderBundleDepthStencil>,
    /// Sample count this render bundle is capable of rendering to. This must match the pipelines and
    /// the render passes it is used in.
    pub sample_count: u32,
    /// If this render bundle will rendering to multiple array layers in the attachments at the same time.
    pub multiview: Option<NonZeroU32>,
}
static_assertions::assert_impl_all!(RenderBundleEncoderDescriptor: Send, Sync);

/// Surface texture that can be rendered to.
/// Result of a successful call to [`Surface::get_current_texture`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// the [`GPUCanvasContext`](https://gpuweb.github.io/gpuweb/#canvas-context) provides
/// a texture without any additional information.
#[derive(Debug)]
pub struct SurfaceTexture {
    /// Accessible view of the frame.
    pub texture: Texture,
    /// `true` if the acquired buffer can still be used for rendering,
    /// but should be recreated for maximum performance.
    pub suboptimal: bool,
    presented: bool,
    detail: Box<dyn Any + Send + Sync>,
}
static_assertions::assert_impl_all!(SurfaceTexture: Send, Sync);

/// Result of an unsuccessful call to [`Surface::get_current_texture`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SurfaceError {
    /// A timeout was encountered while trying to acquire the next frame.
    Timeout,
    /// The underlying surface has changed, and therefore the swap chain must be updated.
    Outdated,
    /// The swap chain has been lost and needs to be recreated.
    Lost,
    /// There is no more memory left to allocate a new frame.
    OutOfMemory,
}
static_assertions::assert_impl_all!(SurfaceError: Send, Sync);

impl Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::Timeout => "A timeout was encountered while trying to acquire the next frame",
            Self::Outdated => "The underlying surface has changed, and therefore the swap chain must be updated",
            Self::Lost =>  "The swap chain has been lost and needs to be recreated",
            Self::OutOfMemory => "There is no more memory left to allocate a new frame",
        })
    }
}

impl error::Error for SurfaceError {}

impl Default for Instance {
    /// Creates a new instance of wgpu with default options.
    ///
    /// Backends are set to `Backends::all()`, and FXC is chosen as the `dx12_shader_compiler`.
    fn default() -> Self {
        Self::new(InstanceDescriptor::default())
    }
}

impl Instance {
    /// Create an new instance of wgpu.
    ///
    /// # Arguments
    ///
    /// - `instance_desc` - Has fields for which [backends][Backends] wgpu will choose
    ///   during instantiation, and which [DX12 shader compiler][Dx12Compiler] wgpu will use.
    pub fn new(instance_desc: InstanceDescriptor) -> Self {
        Self {
            context: Arc::from(crate::backend::Context::init(instance_desc)),
        }
    }

    /// Create an new instance of wgpu from a wgpu-hal instance.
    ///
    /// # Arguments
    ///
    /// - `hal_instance` - wgpu-hal instance.
    ///
    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn from_hal<A: wgc::hub::HalApi>(hal_instance: A::Instance) -> Self {
        Self {
            context: Arc::new(unsafe {
                crate::backend::Context::from_hal_instance::<A>(hal_instance)
            }),
        }
    }

    /// Return a reference to a specific backend instance, if available.
    ///
    /// If this `Instance` has a wgpu-hal [`Instance`] for backend
    /// `A`, return a reference to it. Otherwise, return `None`.
    ///
    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    ///
    /// [`Instance`]: hal::Api::Instance
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn as_hal<A: wgc::hub::HalApi>(&self) -> Option<&A::Instance> {
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .instance_as_hal::<A>()
        }
    }

    /// Create an new instance of wgpu from a wgpu-core instance.
    ///
    /// # Arguments
    ///
    /// - `core_instance` - wgpu-core instance.
    ///
    /// # Safety
    ///
    /// Refer to the creation of wgpu-core Instance.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn from_core(core_instance: wgc::instance::Instance) -> Self {
        Self {
            context: Arc::new(unsafe {
                crate::backend::Context::from_core_instance(core_instance)
            }),
        }
    }

    /// Retrieves all available [`Adapter`]s that match the given [`Backends`].
    ///
    /// # Arguments
    ///
    /// - `backends` - Backends from which to enumerate adapters.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub fn enumerate_adapters(&self, backends: Backends) -> impl Iterator<Item = Adapter> {
        let context = Arc::clone(&self.context);
        self.context
            .as_any()
            .downcast_ref::<crate::backend::Context>()
            .unwrap()
            .enumerate_adapters(backends)
            .into_iter()
            .map(move |id| crate::Adapter {
                context: Arc::clone(&context),
                id: ObjectId::from(id),
                data: Box::new(()),
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
        async move {
            adapter
                .await
                .map(|(id, data)| Adapter { context, id, data })
        }
    }

    /// Converts a wgpu-hal `ExposedAdapter` to a wgpu [`Adapter`].
    ///
    /// # Safety
    ///
    /// `hal_adapter` must be created from this instance internal handle.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn create_adapter_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> Adapter {
        let context = Arc::clone(&self.context);
        let id = unsafe {
            context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_adapter_from_hal(hal_adapter)
                .into()
        };
        Adapter {
            context,
            id,
            data: Box::new(()),
        }
    }

    /// Creates a surface from a raw window handle.
    ///
    /// If the specified display and window handle are not supported by any of the backends, then the surface
    /// will not be supported by any adapters.
    ///
    /// # Safety
    ///
    /// - `raw_window_handle` must be a valid object to create a surface upon.
    /// - `raw_window_handle` must remain valid until after the returned [`Surface`] is
    ///   dropped.
    ///
    /// # Errors
    ///
    /// - On WebGL2: Will return an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    ///
    /// # Panics
    ///
    /// - On macOS/Metal: will panic if not called on the main thread.
    /// - On web: will panic if the `raw_window_handle` does not properly refer to a
    ///   canvas element.
    pub unsafe fn create_surface<
        W: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    >(
        &self,
        window: &W,
    ) -> Result<Surface, CreateSurfaceError> {
        let (id, data) = DynContext::instance_create_surface(
            &*self.context,
            raw_window_handle::HasRawDisplayHandle::raw_display_handle(window),
            raw_window_handle::HasRawWindowHandle::raw_window_handle(window),
        )?;
        Ok(Surface {
            context: Arc::clone(&self.context),
            id,
            data,
            config: Mutex::new(None),
        })
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
        let surface = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_surface_from_core_animation_layer(layer)
        };
        Surface {
            context: Arc::clone(&self.context),
            id: ObjectId::from(surface.id()),
            data: Box::new(surface),
            config: Mutex::new(None),
        }
    }

    /// Creates a surface from `IDCompositionVisual`.
    ///
    /// # Safety
    ///
    /// - visual must be a valid IDCompositionVisual to create a surface upon.
    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface_from_visual(&self, visual: *mut std::ffi::c_void) -> Surface {
        let surface = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_surface_from_visual(visual)
        };
        Surface {
            context: Arc::clone(&self.context),
            id: ObjectId::from(surface.id()),
            data: Box::new(surface),
            config: Mutex::new(None),
        }
    }

    /// Creates a surface from `SurfaceHandle`.
    ///
    /// # Safety
    ///
    /// - surface_handle must be a valid SurfaceHandle to create a surface upon.
    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface_from_surface_handle(
        &self,
        surface_handle: *mut std::ffi::c_void,
    ) -> Surface {
        let surface = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_surface_from_surface_handle(surface_handle)
        };
        Surface {
            context: Arc::clone(&self.context),
            id: ObjectId::from(surface.id()),
            data: Box::new(surface),
            config: Mutex::new(None),
        }
    }

    /// Creates a surface from a `web_sys::HtmlCanvasElement`.
    ///
    /// The `canvas` argument must be a valid `<canvas>` element to
    /// create a surface upon.
    ///
    /// # Errors
    ///
    /// - On WebGL2: Will return an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn create_surface_from_canvas(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<Surface, CreateSurfaceError> {
        let surface = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::Context>()
            .unwrap()
            .instance_create_surface_from_canvas(canvas)?;

        // TODO: This is ugly, a way to create things from a native context needs to be made nicer.
        Ok(Surface {
            context: Arc::clone(&self.context),
            #[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
            id: ObjectId::from(surface.id()),
            #[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
            data: Box::new(surface),
            #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
            id: ObjectId::from(surface),
            #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
            data: Box::new(()),
            config: Mutex::new(None),
        })
    }

    /// Creates a surface from a `web_sys::OffscreenCanvas`.
    ///
    /// The `canvas` argument must be a valid `OffscreenCanvas` object
    /// to create a surface upon.
    ///
    /// # Errors
    ///
    /// - On WebGL2: Will return an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn create_surface_from_offscreen_canvas(
        &self,
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<Surface, CreateSurfaceError> {
        let surface = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::Context>()
            .unwrap()
            .instance_create_surface_from_offscreen_canvas(canvas)?;

        // TODO: This is ugly, a way to create things from a native context needs to be made nicer.
        Ok(Surface {
            context: Arc::clone(&self.context),
            #[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
            id: ObjectId::from(surface.id()),
            #[cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]
            data: Box::new(surface),
            #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
            id: ObjectId::from(surface),
            #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
            data: Box::new(()),
            config: Mutex::new(None),
        })
    }

    /// Polls all devices.
    ///
    /// If `force_wait` is true and this is not running on the web, then this
    /// function will block until all in-flight buffers have been mapped and
    /// all submitted commands have finished execution.
    ///
    /// Return `true` if all devices' queues are empty, or `false` if there are
    /// queue submissions still in flight. (Note that, unless access to all
    /// [`Queue`s] associated with this [`Instance`] is coordinated somehow,
    /// this information could be out of date by the time the caller receives
    /// it. `Queue`s can be shared between threads, and other threads could
    /// submit new work at any time.)
    ///
    /// On the web, this is a no-op. `Device`s are automatically polled.
    ///
    /// [`Queue`s]: Queue
    pub fn poll_all(&self, force_wait: bool) -> bool {
        self.context.instance_poll_all_devices(force_wait)
    }

    /// Generates memory report.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub fn generate_report(&self) -> wgc::hub::GlobalReport {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::Context>()
            .unwrap()
            .generate_report()
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
        let device = DynContext::adapter_request_device(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
            trace_path,
        );
        async move {
            device.await.map(
                |DeviceRequest {
                     device_id,
                     device_data,
                     queue_id,
                     queue_data,
                 }| {
                    (
                        Device {
                            context: Arc::clone(&context),
                            id: device_id,
                            data: device_data,
                        },
                        Queue {
                            context,
                            id: queue_id,
                            data: queue_data,
                        },
                    )
                },
            )
        }
    }

    /// Create a wgpu [`Device`] and [`Queue`] from a wgpu-hal `OpenDevice`
    ///
    /// # Safety
    ///
    /// - `hal_device` must be created from this adapter internal handle.
    /// - `desc.features` must be a subset of `hal_device` features.
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn create_device_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_device: hal::OpenDevice<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), RequestDeviceError> {
        let context = Arc::clone(&self.context);
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_device_from_hal(&self.id.into(), hal_device, desc, trace_path)
        }
        .map(|(device, queue)| {
            (
                Device {
                    context: Arc::clone(&context),
                    id: device.id().into(),
                    data: Box::new(device),
                },
                Queue {
                    context,
                    id: queue.id().into(),
                    data: Box::new(queue),
                },
            )
        })
    }

    /// Apply a callback to this `Adapter`'s underlying backend adapter.
    ///
    /// If this `Adapter` is implemented by the backend API given by `A` (Vulkan,
    /// Dx12, etc.), then apply `hal_adapter_callback` to `Some(&adapter)`, where
    /// `adapter` is the underlying backend adapter type, [`A::Adapter`].
    ///
    /// If this `Adapter` uses a different backend, apply `hal_adapter_callback`
    /// to `None`.
    ///
    /// The adapter is locked for reading while `hal_adapter_callback` runs. If
    /// the callback attempts to perform any `wgpu` operations that require
    /// write access to the adapter, deadlock will occur. The locks are
    /// automatically released when the callback returns.
    ///
    /// # Safety
    ///
    /// - The raw handle passed to the callback must not be manually destroyed.
    ///
    /// [`A::Adapter`]: hal::Api::Adapter
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        hal_adapter_callback: F,
    ) -> R {
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .adapter_as_hal::<A, F, R>(self.id.into(), hal_adapter_callback)
        }
    }

    /// Returns whether this adapter may present to the passed surface.
    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        DynContext::adapter_is_surface_supported(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &surface.id,
            surface.data.as_ref(),
        )
    }

    /// List all features that are supported with this adapter.
    ///
    /// Features must be explicitly requested in [`Adapter::request_device`] in order
    /// to use them.
    pub fn features(&self) -> Features {
        DynContext::adapter_features(&*self.context, &self.id, self.data.as_ref())
    }

    /// List the "best" limits that are supported by this adapter.
    ///
    /// Limits must be explicitly requested in [`Adapter::request_device`] to set
    /// the values that you are allowed to use.
    pub fn limits(&self) -> Limits {
        DynContext::adapter_limits(&*self.context, &self.id, self.data.as_ref())
    }

    /// Get info about the adapter itself.
    pub fn get_info(&self) -> AdapterInfo {
        DynContext::adapter_get_info(&*self.context, &self.id, self.data.as_ref())
    }

    /// Get info about the adapter itself.
    pub fn get_downlevel_capabilities(&self) -> DownlevelCapabilities {
        DynContext::adapter_downlevel_capabilities(&*self.context, &self.id, self.data.as_ref())
    }

    /// Returns the features supported for a given texture format by this adapter.
    ///
    /// Note that the WebGPU spec further restricts the available usages/features.
    /// To disable these restrictions on a device, request the [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] feature.
    pub fn get_texture_format_features(&self, format: TextureFormat) -> TextureFormatFeatures {
        DynContext::adapter_get_texture_format_features(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            format,
        )
    }

    /// Generates a timestamp using the clock used by the presentation engine.
    ///
    /// When comparing completely opaque timestamp systems, we need a way of generating timestamps that signal
    /// the exact same time. You can do this by calling your own timestamp function immediately after a call to
    /// this function. This should result in timestamps that are 0.5 to 5 microseconds apart. There are locks
    /// that must be taken during the call, so don't call your function before.
    ///
    /// ```no_run
    /// # let adapter: wgpu::Adapter = panic!();
    /// # let some_code = || wgpu::PresentationTimestamp::INVALID_TIMESTAMP;
    /// use std::time::{Duration, Instant};
    /// let presentation = adapter.get_presentation_timestamp();
    /// let instant = Instant::now();
    ///
    /// // We can now turn a new presentation timestamp into an Instant.
    /// let some_pres_timestamp = some_code();
    /// let duration = Duration::from_nanos((some_pres_timestamp.0 - presentation.0) as u64);
    /// let new_instant: Instant = instant + duration;
    /// ```
    //
    /// [Instant]: std::time::Instant
    pub fn get_presentation_timestamp(&self) -> PresentationTimestamp {
        DynContext::adapter_get_presentation_timestamp(&*self.context, &self.id, self.data.as_ref())
    }
}

impl Device {
    /// Check for resource cleanups and mapping callbacks.
    ///
    /// Return `true` if the queue is empty, or `false` if there are more queue
    /// submissions still in flight. (Note that, unless access to the [`Queue`] is
    /// coordinated somehow, this information could be out of date by the time
    /// the caller receives it. `Queue`s can be shared between threads, so
    /// other threads could submit new work at any time.)
    ///
    /// On the web, this is a no-op. `Device`s are automatically polled.
    pub fn poll(&self, maintain: Maintain) -> bool {
        DynContext::device_poll(&*self.context, &self.id, self.data.as_ref(), maintain)
    }

    /// List all features that may be used with this device.
    ///
    /// Functions may panic if you use unsupported features.
    pub fn features(&self) -> Features {
        DynContext::device_features(&*self.context, &self.id, self.data.as_ref())
    }

    /// List all limits that were requested of this device.
    ///
    /// If any of these limits are exceeded, functions may panic.
    pub fn limits(&self) -> Limits {
        DynContext::device_limits(&*self.context, &self.id, self.data.as_ref())
    }

    /// Creates a shader module from either SPIR-V or WGSL source code.
    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor) -> ShaderModule {
        let (id, data) = DynContext::device_create_shader_module(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
            wgt::ShaderBoundChecks::new(),
        );
        ShaderModule {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a shader module from either SPIR-V or WGSL source code without runtime checks.
    ///
    /// # Safety
    /// In contrast with [`create_shader_module`](Self::create_shader_module) this function
    /// creates a shader module without runtime checks which allows shaders to perform
    /// operations which can lead to undefined behavior like indexing out of bounds, thus it's
    /// the caller responsibility to pass a shader which doesn't perform any of this
    /// operations.
    ///
    /// This has no effect on web.
    pub unsafe fn create_shader_module_unchecked(
        &self,
        desc: ShaderModuleDescriptor,
    ) -> ShaderModule {
        let (id, data) = DynContext::device_create_shader_module(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
            unsafe { wgt::ShaderBoundChecks::unchecked() },
        );
        ShaderModule {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a shader module from SPIR-V binary directly.
    ///
    /// # Safety
    ///
    /// This function passes binary data to the backend as-is and can potentially result in a
    /// driver crash or bogus behaviour. No attempt is made to ensure that data is valid SPIR-V.
    ///
    /// See also [`include_spirv_raw!`] and [`util::make_spirv_raw`].
    pub unsafe fn create_shader_module_spirv(
        &self,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> ShaderModule {
        let (id, data) = unsafe {
            DynContext::device_create_shader_module_spirv(
                &*self.context,
                &self.id,
                self.data.as_ref(),
                desc,
            )
        };
        ShaderModule {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates an empty [`CommandEncoder`].
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor) -> CommandEncoder {
        let (id, data) = DynContext::device_create_command_encoder(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        CommandEncoder {
            context: Arc::clone(&self.context),
            id: Some(id),
            data,
        }
    }

    /// Creates an empty [`RenderBundleEncoder`].
    pub fn create_render_bundle_encoder(
        &self,
        desc: &RenderBundleEncoderDescriptor,
    ) -> RenderBundleEncoder {
        let (id, data) = DynContext::device_create_render_bundle_encoder(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        RenderBundleEncoder {
            context: Arc::clone(&self.context),
            id,
            data,
            parent: self,
            _p: Default::default(),
        }
    }

    /// Creates a new [`BindGroup`].
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor) -> BindGroup {
        let (id, data) = DynContext::device_create_bind_group(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        BindGroup {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a [`BindGroupLayout`].
    pub fn create_bind_group_layout(&self, desc: &BindGroupLayoutDescriptor) -> BindGroupLayout {
        let (id, data) = DynContext::device_create_bind_group_layout(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        BindGroupLayout {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a [`PipelineLayout`].
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor) -> PipelineLayout {
        let (id, data) = DynContext::device_create_pipeline_layout(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        PipelineLayout {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a [`RenderPipeline`].
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor) -> RenderPipeline {
        let (id, data) = DynContext::device_create_render_pipeline(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        RenderPipeline {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a [`ComputePipeline`].
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor) -> ComputePipeline {
        let (id, data) = DynContext::device_create_compute_pipeline(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            desc,
        );
        ComputePipeline {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a [`Buffer`].
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> Buffer {
        let mut map_context = MapContext::new(desc.size);
        if desc.mapped_at_creation {
            map_context.initial_range = 0..desc.size;
        }

        let (id, data) =
            DynContext::device_create_buffer(&*self.context, &self.id, self.data.as_ref(), desc);

        Buffer {
            context: Arc::clone(&self.context),
            id,
            data,
            map_context: Mutex::new(map_context),
            size: desc.size,
            usage: desc.usage,
        }
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    pub fn create_texture(&self, desc: &TextureDescriptor) -> Texture {
        let (id, data) =
            DynContext::device_create_texture(&*self.context, &self.id, self.data.as_ref(), desc);
        Texture {
            context: Arc::clone(&self.context),
            id,
            data,
            owned: true,
            descriptor: TextureDescriptor {
                label: None,
                view_formats: &[],
                ..desc.clone()
            },
        }
    }

    /// Creates a [`Texture`] from a wgpu-hal Texture.
    ///
    /// # Safety
    ///
    /// - `hal_texture` must be created from this device internal handle
    /// - `hal_texture` must be created respecting `desc`
    /// - `hal_texture` must be initialized
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn create_texture_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_texture: A::Texture,
        desc: &TextureDescriptor,
    ) -> Texture {
        let texture = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .create_texture_from_hal::<A>(
                    hal_texture,
                    self.data.as_ref().downcast_ref().unwrap(),
                    desc,
                )
        };
        Texture {
            context: Arc::clone(&self.context),
            id: ObjectId::from(texture.id()),
            data: Box::new(texture),
            owned: true,
            descriptor: TextureDescriptor {
                label: None,
                view_formats: &[],
                ..desc.clone()
            },
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &SamplerDescriptor) -> Sampler {
        let (id, data) =
            DynContext::device_create_sampler(&*self.context, &self.id, self.data.as_ref(), desc);
        Sampler {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a new [`QuerySet`].
    pub fn create_query_set(&self, desc: &QuerySetDescriptor) -> QuerySet {
        let (id, data) =
            DynContext::device_create_query_set(&*self.context, &self.id, self.data.as_ref(), desc);
        QuerySet {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Set a callback for errors that are not handled in error scopes.
    pub fn on_uncaptured_error(&self, handler: Box<dyn UncapturedErrorHandler>) {
        self.context
            .device_on_uncaptured_error(&self.id, self.data.as_ref(), handler);
    }

    /// Push an error scope.
    pub fn push_error_scope(&self, filter: ErrorFilter) {
        self.context
            .device_push_error_scope(&self.id, self.data.as_ref(), filter);
    }

    /// Pop an error scope.
    pub fn pop_error_scope(&self) -> impl Future<Output = Option<Error>> + Send {
        self.context
            .device_pop_error_scope(&self.id, self.data.as_ref())
    }

    /// Starts frame capture.
    pub fn start_capture(&self) {
        DynContext::device_start_capture(&*self.context, &self.id, self.data.as_ref())
    }

    /// Stops frame capture.
    pub fn stop_capture(&self) {
        DynContext::device_stop_capture(&*self.context, &self.id, self.data.as_ref())
    }

    /// Apply a callback to this `Device`'s underlying backend device.
    ///
    /// If this `Device` is implemented by the backend API given by `A` (Vulkan,
    /// Dx12, etc.), then apply `hal_device_callback` to `Some(&device)`, where
    /// `device` is the underlying backend device type, [`A::Device`].
    ///
    /// If this `Device` uses a different backend, apply `hal_device_callback`
    /// to `None`.
    ///
    /// The device is locked for reading while `hal_device_callback` runs. If
    /// the callback attempts to perform any `wgpu` operations that require
    /// write access to the device (destroying a buffer, say), deadlock will
    /// occur. The locks are automatically released when the callback returns.
    ///
    /// # Safety
    ///
    /// - The raw handle passed to the callback must not be manually destroyed.
    ///
    /// [`A::Device`]: hal::Api::Device
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        hal_device_callback: F,
    ) -> R {
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .device_as_hal::<A, F, R>(
                    self.data.as_ref().downcast_ref().unwrap(),
                    hal_device_callback,
                )
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.device_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Requesting a device failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RequestDeviceError;
static_assertions::assert_impl_all!(RequestDeviceError: Send, Sync);

impl Display for RequestDeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Requesting a device failed")
    }
}

impl error::Error for RequestDeviceError {}

/// [`Instance::create_surface()`] or a related function failed.
#[derive(Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct CreateSurfaceError {
    // TODO: Report diagnostic clues
}
static_assertions::assert_impl_all!(CreateSurfaceError: Send, Sync);

impl Display for CreateSurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Creating a surface failed")
    }
}

impl error::Error for CreateSurfaceError {}

/// Error occurred when trying to async map a buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;
static_assertions::assert_impl_all!(BufferAsyncError: Send, Sync);

impl Display for BufferAsyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error occurred when trying to async map a buffer")
    }
}

impl error::Error for BufferAsyncError {}

/// Type of buffer mapping.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MapMode {
    /// Map only for reading
    Read,
    /// Map only for writing
    Write,
}
static_assertions::assert_impl_all!(MapMode: Send, Sync);

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

/// Read only view into a mapped buffer.
#[derive(Debug)]
pub struct BufferView<'a> {
    slice: BufferSlice<'a>,
    data: Box<dyn crate::context::BufferMappedRange>,
}

/// Write only view into mapped buffer.
///
/// It is possible to read the buffer using this view, but doing so is not
/// recommended, as it is likely to be slow.
#[derive(Debug)]
pub struct BufferViewMut<'a> {
    slice: BufferSlice<'a>,
    data: Box<dyn crate::context::BufferMappedRange>,
    readable: bool,
}

impl std::ops::Deref for BufferView<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.data.slice()
    }
}

impl AsRef<[u8]> for BufferView<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data.slice()
    }
}

impl AsMut<[u8]> for BufferViewMut<'_> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.slice_mut()
    }
}

impl Deref for BufferViewMut<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        if !self.readable {
            log::warn!("Reading from a BufferViewMut is slow and not recommended.");
        }

        self.data.slice()
    }
}

impl DerefMut for BufferViewMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
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
        DynContext::buffer_unmap(&*self.context, &self.id, self.data.as_ref());
    }

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::buffer_destroy(&*self.context, &self.id, self.data.as_ref());
    }

    /// Returns the length of the buffer allocation in bytes.
    ///
    /// This is always equal to the `size` that was specified when creating the buffer.
    pub fn size(&self) -> BufferAddress {
        self.size
    }

    /// Returns the allowed usages for this `Buffer`.
    ///
    /// This is always equal to the `usage` that was specified when creating the buffer.
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }
}

impl<'a> BufferSlice<'a> {
    /// Map the buffer. Buffer is ready to map once the callback is called.
    ///
    /// For the callback to complete, either `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)`
    /// must be called elsewhere in the runtime, possibly integrated into an event loop or run on a separate thread.
    ///
    /// The callback will be called on the thread that first calls the above functions after the gpu work
    /// has completed. There are no restrictions on the code you can run in the callback, however on native the
    /// call to the function will not complete until the callback returns, so prefer keeping callbacks short
    /// and used to set flags, send messages, etc.
    pub fn map_async(
        &self,
        mode: MapMode,
        callback: impl FnOnce(Result<(), BufferAsyncError>) + Send + 'static,
    ) {
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

        DynContext::buffer_map_async(
            &*self.buffer.context,
            &self.buffer.id,
            self.buffer.data.as_ref(),
            mode,
            self.offset..end,
            Box::new(callback),
        )
    }

    /// Synchronously and immediately map a buffer for reading. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will panic.
    pub fn get_mapped_range(&self) -> BufferView<'a> {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = DynContext::buffer_get_mapped_range(
            &*self.buffer.context,
            &self.buffer.id,
            self.buffer.data.as_ref(),
            self.offset..end,
        );
        BufferView { slice: *self, data }
    }

    /// Synchronously and immediately map a buffer for writing. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will panic.
    pub fn get_mapped_range_mut(&self) -> BufferViewMut<'a> {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = DynContext::buffer_get_mapped_range(
            &*self.buffer.context,
            &self.buffer.id,
            self.buffer.data.as_ref(),
            self.offset..end,
        );
        BufferViewMut {
            slice: *self,
            data,
            readable: self.buffer.usage.contains(BufferUsages::MAP_READ),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.buffer_drop(&self.id, self.data.as_ref());
        }
    }
}

impl Texture {
    /// Returns the inner hal Texture using a callback. The hal texture will be `None` if the
    /// backend type argument does not match with this wgpu Texture
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Texture must not be manually destroyed
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Texture>)>(
        &self,
        hal_texture_callback: F,
    ) {
        let texture = self.data.as_ref().downcast_ref().unwrap();
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .texture_as_hal::<A, F>(texture, hal_texture_callback)
        }
    }

    /// Creates a view of this texture.
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        let (id, data) =
            DynContext::texture_create_view(&*self.context, &self.id, self.data.as_ref(), desc);
        TextureView {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::texture_destroy(&*self.context, &self.id, self.data.as_ref());
    }

    /// Make an `ImageCopyTexture` representing the whole texture.
    pub fn as_image_copy(&self) -> ImageCopyTexture {
        ImageCopyTexture {
            texture: self,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        }
    }

    /// Returns the size of this `Texture`.
    ///
    /// This is always equal to the `size` that was specified when creating the texture.
    pub fn size(&self) -> Extent3d {
        self.descriptor.size
    }

    /// Returns the width of this `Texture`.
    ///
    /// This is always equal to the `size.width` that was specified when creating the texture.
    pub fn width(&self) -> u32 {
        self.descriptor.size.width
    }

    /// Returns the height of this `Texture`.
    ///
    /// This is always equal to the `size.height` that was specified when creating the texture.
    pub fn height(&self) -> u32 {
        self.descriptor.size.height
    }

    /// Returns the depth or layer count of this `Texture`.
    ///
    /// This is always equal to the `size.depth_or_array_layers` that was specified when creating the texture.
    pub fn depth_or_array_layers(&self) -> u32 {
        self.descriptor.size.depth_or_array_layers
    }

    /// Returns the mip_level_count of this `Texture`.
    ///
    /// This is always equal to the `mip_level_count` that was specified when creating the texture.
    pub fn mip_level_count(&self) -> u32 {
        self.descriptor.mip_level_count
    }

    /// Returns the sample_count of this `Texture`.
    ///
    /// This is always equal to the `sample_count` that was specified when creating the texture.
    pub fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
    }

    /// Returns the dimension of this `Texture`.
    ///
    /// This is always equal to the `dimension` that was specified when creating the texture.
    pub fn dimension(&self) -> TextureDimension {
        self.descriptor.dimension
    }

    /// Returns the format of this `Texture`.
    ///
    /// This is always equal to the `format` that was specified when creating the texture.
    pub fn format(&self) -> TextureFormat {
        self.descriptor.format
    }

    /// Returns the allowed usages of this `Texture`.
    ///
    /// This is always equal to the `usage` that was specified when creating the texture.
    pub fn usage(&self) -> TextureUsages {
        self.descriptor.usage
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if self.owned && !thread::panicking() {
            self.context.texture_drop(&self.id, self.data.as_ref());
        }
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.texture_view_drop(&self.id, self.data.as_ref());
        }
    }
}

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(mut self) -> CommandBuffer {
        let (id, data) = DynContext::command_encoder_finish(
            &*self.context,
            self.id.take().unwrap(),
            self.data.as_mut(),
        );
        CommandBuffer {
            context: Arc::clone(&self.context),
            id: Some(id),
            data: Some(data),
        }
    }

    /// Begins recording of a render pass.
    ///
    /// This function returns a [`RenderPass`] object which records a single render pass.
    pub fn begin_render_pass<'pass>(
        &'pass mut self,
        desc: &RenderPassDescriptor<'pass, '_>,
    ) -> RenderPass<'pass> {
        let id = self.id.as_ref().unwrap();
        let (id, data) = DynContext::command_encoder_begin_render_pass(
            &*self.context,
            id,
            self.data.as_ref(),
            desc,
        );
        RenderPass {
            id,
            data,
            parent: self,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    pub fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor) -> ComputePass {
        let id = self.id.as_ref().unwrap();
        let (id, data) = DynContext::command_encoder_begin_compute_pass(
            &*self.context,
            id,
            self.data.as_ref(),
            desc,
        );
        ComputePass {
            id,
            data,
            parent: self,
        }
    }

    /// Copy data from one buffer to another.
    ///
    /// # Panics
    ///
    /// - Buffer offsets or copy size not a multiple of [`COPY_BUFFER_ALIGNMENT`].
    /// - Copy would overrun buffer.
    /// - Copy within the same buffer.
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        DynContext::command_encoder_copy_buffer_to_buffer(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            &source.id,
            source.data.as_ref(),
            source_offset,
            &destination.id,
            destination.data.as_ref(),
            destination_offset,
            copy_size,
        );
    }

    /// Copy data from a buffer to a texture.
    pub fn copy_buffer_to_texture(
        &mut self,
        source: ImageCopyBuffer,
        destination: ImageCopyTexture,
        copy_size: Extent3d,
    ) {
        DynContext::command_encoder_copy_buffer_to_texture(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from a texture to a buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: ImageCopyTexture,
        destination: ImageCopyBuffer,
        copy_size: Extent3d,
    ) {
        DynContext::command_encoder_copy_texture_to_buffer(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
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
        DynContext::command_encoder_copy_texture_to_texture(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            source,
            destination,
            copy_size,
        );
    }

    /// Clears texture to zero.
    ///
    /// Note that unlike with clear_buffer, `COPY_DST` usage is not required.
    ///
    /// # Implementation notes
    ///
    /// - implemented either via buffer copies and render/depth target clear, path depends on texture usages
    /// - behaves like texture zero init, but is performed immediately (clearing is *not* delayed via marking it as uninitialized)
    ///
    /// # Panics
    ///
    /// - `CLEAR_TEXTURE` extension not enabled
    /// - Range is out of bounds
    pub fn clear_texture(&mut self, texture: &Texture, subresource_range: &ImageSubresourceRange) {
        DynContext::command_encoder_clear_texture(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            texture,
            subresource_range,
        );
    }

    /// Clears buffer to zero.
    ///
    /// # Panics
    ///
    /// - Buffer does not have `COPY_DST` usage.
    /// - Range it out of bounds
    pub fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        DynContext::command_encoder_clear_buffer(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            buffer,
            offset,
            size,
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        let id = self.id.as_ref().unwrap();
        DynContext::command_encoder_insert_debug_marker(
            &*self.context,
            id,
            self.data.as_ref(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        let id = self.id.as_ref().unwrap();
        DynContext::command_encoder_push_debug_group(&*self.context, id, self.data.as_ref(), label);
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        let id = self.id.as_ref().unwrap();
        DynContext::command_encoder_pop_debug_group(&*self.context, id, self.data.as_ref());
    }
}

/// [`Features::TIMESTAMP_QUERY`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Issue a timestamp command at this point in the queue.
    /// The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::command_encoder_write_timestamp(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        )
    }
}

/// [`Features::TIMESTAMP_QUERY`] or [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Resolve a query set, writing the results into the supplied destination buffer.
    ///
    /// Queries may be between 8 and 40 bytes each. See [`PipelineStatisticsTypes`] for more information.
    pub fn resolve_query_set(
        &mut self,
        query_set: &QuerySet,
        query_range: Range<u32>,
        destination: &Buffer,
        destination_offset: BufferAddress,
    ) {
        DynContext::command_encoder_resolve_query_set(
            &*self.context,
            self.id.as_ref().unwrap(),
            self.data.as_ref(),
            &query_set.id,
            query_set.data.as_ref(),
            query_range.start,
            query_range.end - query_range.start,
            &destination.id,
            destination.data.as_ref(),
            destination_offset,
        )
    }
}

impl<'a> RenderPass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        DynContext::render_pass_set_bind_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            index,
            &bind_group.id,
            bind_group.data.as_ref(),
            offsets,
        )
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        DynContext::render_pass_set_pipeline(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &pipeline.id,
            pipeline.data.as_ref(),
        )
    }

    /// Sets the blend color as used by some of the blending modes.
    ///
    /// Subsequent blending tests will test against this value.
    pub fn set_blend_constant(&mut self, color: Color) {
        DynContext::render_pass_set_blend_constant(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            color,
        )
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        DynContext::render_pass_set_index_buffer(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &buffer_slice.buffer.id,
            buffer_slice.buffer.data.as_ref(),
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
    /// [`VertexState::buffers`].
    ///
    /// [`draw`]: RenderPass::draw
    /// [`draw_indexed`]: RenderPass::draw_indexed
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        DynContext::render_pass_set_vertex_buffer(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.buffer.data.as_ref(),
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Sets the scissor region.
    ///
    /// Subsequent draw calls will discard any fragments that fall outside this region.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        DynContext::render_pass_set_scissor_rect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            x,
            y,
            width,
            height,
        );
    }

    /// Sets the viewport region.
    ///
    /// Subsequent draw calls will draw any fragments in this region.
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        DynContext::render_pass_set_viewport(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            x,
            y,
            w,
            h,
            min_depth,
            max_depth,
        );
    }

    /// Sets the stencil reference.
    ///
    /// Subsequent stencil tests will test against this value.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        DynContext::render_pass_set_stencil_reference(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            reference,
        );
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        DynContext::render_pass_draw(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            vertices,
            instances,
        )
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::render_pass_insert_debug_marker(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::render_pass_push_debug_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            label,
        );
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::render_pass_pop_debug_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
        );
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        DynContext::render_pass_draw_indexed(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            indices,
            base_vertex,
            instances,
        );
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirect`](crate::util::DrawIndirect).
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        DynContext::render_pass_draw_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirect`](crate::util::DrawIndexedIndirect).
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::render_pass_draw_indexed_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Execute a [render bundle][RenderBundle], which is a set of pre-recorded commands
    /// that can be run together.
    pub fn execute_bundles<I: IntoIterator<Item = &'a RenderBundle> + 'a>(
        &mut self,
        render_bundles: I,
    ) {
        DynContext::render_pass_execute_bundles(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            Box::new(
                render_bundles
                    .into_iter()
                    .map(|rb| (&rb.id, rb.data.as_ref())),
            ),
        )
    }
}

/// [`Features::MULTI_DRAW_INDIRECT`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Dispatches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// `count` draw calls are issued.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirect`](crate::util::DrawIndirect).
    ///
    /// These draw structures are expected to be tightly packed.
    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        DynContext::render_pass_multi_draw_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
            count,
        );
    }

    /// Dispatches multiple draw calls from the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`. `count` draw calls are issued.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirect`](crate::util::DrawIndexedIndirect).
    ///
    /// These draw structures are expected to be tightly packed.
    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        DynContext::render_pass_multi_draw_indexed_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
            count,
        );
    }
}

/// [`Features::MULTI_DRAW_INDIRECT_COUNT`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Dispatches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// The count buffer is read to determine how many draws to issue.
    ///
    /// The indirect buffer must be long enough to account for `max_count` draws, however only `count`
    /// draws will be read. If `count` is greater than `max_count`, `max_count` will be used.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirect`](crate::util::DrawIndirect).
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
        DynContext::render_pass_multi_draw_indirect_count(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
            &count_buffer.id,
            count_buffer.data.as_ref(),
            count_offset,
            max_count,
        );
    }

    /// Dispatches multiple draw calls from the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`. The count buffer is read to determine how many draws to issue.
    ///
    /// The indirect buffer must be long enough to account for `max_count` draws, however only `count`
    /// draws will be read. If `count` is greater than `max_count`, `max_count` will be used.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirect`](crate::util::DrawIndexedIndirect).
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
        DynContext::render_pass_multi_draw_indexed_indirect_count(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
            &count_buffer.id,
            count_buffer.data.as_ref(),
            count_offset,
            max_count,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Set push constant data for subsequent draw calls.
    ///
    /// Write the bytes in `data` at offset `offset` within push constant
    /// storage, all of which are accessible by all the pipeline stages in
    /// `stages`, and no others.  Both `offset` and the length of `data` must be
    /// multiples of [`PUSH_CONSTANT_ALIGNMENT`], which is always 4.
    ///
    /// For example, if `offset` is `4` and `data` is eight bytes long, this
    /// call will write `data` to bytes `4..12` of push constant storage.
    ///
    /// # Stage matching
    ///
    /// Every byte in the affected range of push constant storage must be
    /// accessible to exactly the same set of pipeline stages, which must match
    /// `stages`. If there are two bytes of storage that are accessible by
    /// different sets of pipeline stages - say, one is accessible by fragment
    /// shaders, and the other is accessible by both fragment shaders and vertex
    /// shaders - then no single `set_push_constants` call may affect both of
    /// them; to write both, you must make multiple calls, each with the
    /// appropriate `stages` value.
    ///
    /// Which pipeline stages may access a given byte is determined by the
    /// pipeline's [`PushConstant`] global variable and (if it is a struct) its
    /// members' offsets.
    ///
    /// For example, suppose you have twelve bytes of push constant storage,
    /// where bytes `0..8` are accessed by the vertex shader, and bytes `4..12`
    /// are accessed by the fragment shader. This means there are three byte
    /// ranges each accessed by a different set of stages:
    ///
    /// - Bytes `0..4` are accessed only by the fragment shader.
    ///
    /// - Bytes `4..8` are accessed by both the fragment shader and the vertex shader.
    ///
    /// - Bytes `8..12` are accessed only by the vertex shader.
    ///
    /// To write all twelve bytes requires three `set_push_constants` calls, one
    /// for each range, each passing the matching `stages` mask.
    ///
    /// [`PushConstant`]: https://docs.rs/naga/latest/naga/enum.StorageClass.html#variant.PushConstant
    pub fn set_push_constants(&mut self, stages: ShaderStages, offset: u32, data: &[u8]) {
        DynContext::render_pass_set_push_constants(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            stages,
            offset,
            data,
        );
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Issue a timestamp command at this point in the queue. The
    /// timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::render_pass_write_timestamp(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        )
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> RenderPass<'a> {
    /// Start a pipeline statistics query on this render pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::render_pass_begin_pipeline_statistics_query(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        );
    }

    /// End the pipeline statistics query on this render pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        DynContext::render_pass_end_pipeline_statistics_query(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
        );
    }
}

impl<'a> Drop for RenderPass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            let parent_id = self.parent.id.as_ref().unwrap();
            self.parent.context.command_encoder_end_render_pass(
                parent_id,
                self.parent.data.as_ref(),
                &mut self.id,
                self.data.as_mut(),
            );
        }
    }
}

impl<'a> ComputePass<'a> {
    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `dispatch()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        DynContext::compute_pass_set_bind_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            index,
            &bind_group.id,
            bind_group.data.as_ref(),
            offsets,
        );
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &'a ComputePipeline) {
        DynContext::compute_pass_set_pipeline(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &pipeline.id,
            pipeline.data.as_ref(),
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::compute_pass_insert_debug_marker(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::compute_pass_push_debug_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            label,
        );
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::compute_pass_pop_debug_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
        );
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        DynContext::compute_pass_dispatch_workgroups(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            x,
            y,
            z,
        );
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DispatchIndirect`](crate::util::DispatchIndirect).
    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::compute_pass_dispatch_workgroups_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Set push constant data for subsequent dispatch calls.
    ///
    /// Write the bytes in `data` at offset `offset` within push constant
    /// storage.  Both `offset` and the length of `data` must be
    /// multiples of [`PUSH_CONSTANT_ALIGNMENT`], which is always 4.
    ///
    /// For example, if `offset` is `4` and `data` is eight bytes long, this
    /// call will write `data` to bytes `4..12` of push constant storage.
    pub fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
        DynContext::compute_pass_set_push_constants(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            offset,
            data,
        );
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Issue a timestamp command at this point in the queue. The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_write_timestamp(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        )
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'a> ComputePass<'a> {
    /// Start a pipeline statistics query on this render pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_begin_pipeline_statistics_query(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        );
    }

    /// End the pipeline statistics query on this render pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        DynContext::compute_pass_end_pipeline_statistics_query(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
        );
    }
}

impl<'a> Drop for ComputePass<'a> {
    fn drop(&mut self) {
        if !thread::panicking() {
            let parent_id = self.parent.id.as_ref().unwrap();
            self.parent.context.command_encoder_end_compute_pass(
                parent_id,
                self.parent.data.as_ref(),
                &mut self.id,
                self.data.as_mut(),
            );
        }
    }
}

impl<'a> RenderBundleEncoder<'a> {
    /// Finishes recording and returns a [`RenderBundle`] that can be executed in other render passes.
    pub fn finish(self, desc: &RenderBundleDescriptor) -> RenderBundle {
        let (id, data) =
            DynContext::render_bundle_encoder_finish(&*self.context, self.id, self.data, desc);
        RenderBundle {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &'a BindGroup,
        offsets: &[DynamicOffset],
    ) {
        DynContext::render_bundle_encoder_set_bind_group(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            index,
            &bind_group.id,
            bind_group.data.as_ref(),
            offsets,
        )
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &'a RenderPipeline) {
        DynContext::render_bundle_encoder_set_pipeline(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &pipeline.id,
            pipeline.data.as_ref(),
        )
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderBundleEncoder::draw_indexed) on this [`RenderBundleEncoder`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'a>, index_format: IndexFormat) {
        DynContext::render_bundle_encoder_set_index_buffer(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &buffer_slice.buffer.id,
            buffer_slice.buffer.data.as_ref(),
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
    /// [`VertexState::buffers`].
    ///
    /// [`draw`]: RenderBundleEncoder::draw
    /// [`draw_indexed`]: RenderBundleEncoder::draw_indexed
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'a>) {
        DynContext::render_bundle_encoder_set_vertex_buffer(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.buffer.data.as_ref(),
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        DynContext::render_bundle_encoder_draw(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            vertices,
            instances,
        )
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        DynContext::render_bundle_encoder_draw_indexed(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            indices,
            base_vertex,
            instances,
        );
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// The active vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirect`](crate::util::DrawIndirect).
    pub fn draw_indirect(&mut self, indirect_buffer: &'a Buffer, indirect_offset: BufferAddress) {
        DynContext::render_bundle_encoder_draw_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`], while the active
    /// vertex buffers can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirect`](crate::util::DrawIndexedIndirect).
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &'a Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::render_bundle_encoder_draw_indexed_indirect(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'a> RenderBundleEncoder<'a> {
    /// Set push constant data.
    ///
    /// Offset is measured in bytes, but must be a multiple of [`PUSH_CONSTANT_ALIGNMENT`].
    ///
    /// Data size must be a multiple of 4 and must have an alignment of 4.
    /// For example, with an offset of 4 and an array of `[u8; 8]`, that will write to the range
    /// of 4..12.
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
    pub fn set_push_constants(&mut self, stages: ShaderStages, offset: u32, data: &[u8]) {
        DynContext::render_bundle_encoder_set_push_constants(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            stages,
            offset,
            data,
        );
    }
}

/// A read-only view into a staging buffer.
///
/// Reading into this buffer won't yield the contents of the buffer from the
/// GPU and is likely to be slow. Because of this, although [`AsMut`] is
/// implemented for this type, [`AsRef`] is not.
pub struct QueueWriteBufferView<'a> {
    queue: &'a Queue,
    buffer: &'a Buffer,
    offset: BufferAddress,
    inner: Box<dyn context::QueueWriteBuffer>,
}
static_assertions::assert_impl_all!(QueueWriteBufferView: Send, Sync);

impl Deref for QueueWriteBufferView<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        log::warn!("Reading from a QueueWriteBufferView won't yield the contents of the buffer and may be slow.");
        self.inner.slice()
    }
}

impl DerefMut for QueueWriteBufferView<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.slice_mut()
    }
}

impl<'a> AsMut<[u8]> for QueueWriteBufferView<'a> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.inner.slice_mut()
    }
}

impl<'a> Drop for QueueWriteBufferView<'a> {
    fn drop(&mut self) {
        DynContext::queue_write_staging_buffer(
            &*self.queue.context,
            &self.queue.id,
            self.queue.data.as_ref(),
            &self.buffer.id,
            self.buffer.data.as_ref(),
            self.offset,
            &*self.inner,
        );
    }
}

impl Queue {
    /// Schedule a data write into `buffer` starting at `offset`.
    ///
    /// This method is intended to have low performance costs.
    /// As such, the write is not immediately submitted, and instead enqueued
    /// internally to happen at the start of the next `submit()` call.
    ///
    /// This method fails if `data` overruns the size of `buffer` starting at `offset`.
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        DynContext::queue_write_buffer(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &buffer.id,
            buffer.data.as_ref(),
            offset,
            data,
        )
    }

    /// Schedule a data write into `buffer` starting at `offset` via the returned
    /// [`QueueWriteBufferView`].
    ///
    /// Reading from this buffer is slow and will not yield the actual contents of the buffer.
    ///
    /// This method is intended to have low performance costs.
    /// As such, the write is not immediately submitted, and instead enqueued
    /// internally to happen at the start of the next `submit()` call.
    ///
    /// This method fails if `size` is greater than the size of `buffer` starting at `offset`.
    #[must_use]
    pub fn write_buffer_with<'a>(
        &'a self,
        buffer: &'a Buffer,
        offset: BufferAddress,
        size: BufferSize,
    ) -> Option<QueueWriteBufferView<'a>> {
        profiling::scope!("Queue::write_buffer_with");
        DynContext::queue_validate_write_buffer(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &buffer.id,
            buffer.data.as_ref(),
            offset,
            size,
        )?;
        let staging_buffer = DynContext::queue_create_staging_buffer(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            size,
        )?;
        Some(QueueWriteBufferView {
            queue: self,
            buffer,
            offset,
            inner: staging_buffer,
        })
    }

    /// Schedule a write of some data into a texture.
    ///
    /// * `data` contains the texels to be written, which must be in
    ///   [the same format as the texture](TextureFormat).
    /// * `data_layout` describes the memory layout of `data`, which does not necessarily
    ///   have to have tightly packed rows.
    /// * `texture` specifies the texture to write into, and the location within the
    ///   texture (coordinate offset, mip level) that will be overwritten.
    /// * `size` is the size, in texels, of the region to be written.
    ///
    /// This method is intended to have low performance costs.
    /// As such, the write is not immediately submitted, and instead enqueued
    /// internally to happen at the start of the next `submit()` call.
    /// However, `data` will be immediately copied into staging memory; so the caller may
    /// discard it any time after this call completes.
    ///
    /// This method fails if `size` overruns the size of `texture`, or if `data` is too short.
    pub fn write_texture(
        &self,
        texture: ImageCopyTexture,
        data: &[u8],
        data_layout: ImageDataLayout,
        size: Extent3d,
    ) {
        DynContext::queue_write_texture(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            texture,
            data,
            data_layout,
            size,
        )
    }

    /// Schedule a copy of data from `image` into `texture`.
    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn copy_external_image_to_texture(
        &self,
        source: &wgt::ImageCopyExternalImage,
        dest: ImageCopyTextureTagged,
        size: Extent3d,
    ) {
        DynContext::queue_copy_external_image_to_texture(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            source,
            dest,
            size,
        )
    }

    /// Submits a series of finished command buffers for execution.
    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> SubmissionIndex {
        let (raw, data) = DynContext::queue_submit(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            Box::new(
                command_buffers
                    .into_iter()
                    .map(|mut comb| (comb.id.take().unwrap(), comb.data.take().unwrap())),
            ),
        );

        SubmissionIndex(raw, data)
    }

    /// Gets the amount of nanoseconds each tick of a timestamp query represents.
    ///
    /// Returns zero if timestamp queries are unsupported.
    pub fn get_timestamp_period(&self) -> f32 {
        DynContext::queue_get_timestamp_period(&*self.context, &self.id, self.data.as_ref())
    }

    /// Registers a callback when the previous call to submit finishes running on the gpu. This callback
    /// being called implies that all mapped buffer callbacks attached to the same submission have also
    /// been called.
    ///
    /// For the callback to complete, either `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)`
    /// must be called elsewhere in the runtime, possibly integrated into an event loop or run on a separate thread.
    ///
    /// The callback will be called on the thread that first calls the above functions after the gpu work
    /// has completed. There are no restrictions on the code you can run in the callback, however on native the
    /// call to the function will not complete until the callback returns, so prefer keeping callbacks short
    /// and used to set flags, send messages, etc.
    pub fn on_submitted_work_done(&self, callback: impl FnOnce() + Send + 'static) {
        DynContext::queue_on_submitted_work_done(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            Box::new(callback),
        )
    }
}

impl SurfaceTexture {
    /// Schedule this texture to be presented on the owning surface.
    ///
    /// Needs to be called after any work on the texture is scheduled via [`Queue::submit`].
    pub fn present(mut self) {
        self.presented = true;
        DynContext::surface_present(
            &*self.texture.context,
            &self.texture.id,
            // This call to as_ref is essential because we want the DynContext implementation to see the inner
            // value of the Box (T::SurfaceOutputDetail), not the Box itself.
            self.detail.as_ref(),
        );
    }
}

impl Drop for SurfaceTexture {
    fn drop(&mut self) {
        if !self.presented && !thread::panicking() {
            DynContext::surface_texture_discard(
                &*self.texture.context,
                &self.texture.id,
                // This call to as_ref is essential because we want the DynContext implementation to see the inner
                // value of the Box (T::SurfaceOutputDetail), not the Box itself.
                self.detail.as_ref(),
            );
        }
    }
}

impl Surface {
    /// Returns the capabilities of the surface when used with the given adapter.
    ///
    /// Returns specified values (see [`SurfaceCapabilities`]) if surface is incompatible with the adapter.
    pub fn get_capabilities(&self, adapter: &Adapter) -> SurfaceCapabilities {
        DynContext::surface_get_capabilities(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &adapter.id,
            adapter.data.as_ref(),
        )
    }

    /// Return a default `SurfaceConfiguration` from width and height to use for the [`Surface`] with this adapter.
    ///
    /// Returns None if the surface isn't supported by this adapter
    pub fn get_default_config(
        &self,
        adapter: &Adapter,
        width: u32,
        height: u32,
    ) -> Option<SurfaceConfiguration> {
        let caps = self.get_capabilities(adapter);
        Some(SurfaceConfiguration {
            usage: wgt::TextureUsages::RENDER_ATTACHMENT,
            format: *caps.formats.get(0)?,
            width,
            height,
            present_mode: *caps.present_modes.get(0)?,
            alpha_mode: wgt::CompositeAlphaMode::Auto,
            view_formats: vec![],
        })
    }

    /// Initializes [`Surface`] for presentation.
    ///
    /// # Panics
    ///
    /// - A old [`SurfaceTexture`] is still alive referencing an old surface.
    /// - Texture format requested is unsupported on the surface.
    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration) {
        DynContext::surface_configure(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &device.id,
            device.data.as_ref(),
            config,
        );

        let mut conf = self.config.lock();
        *conf = Some(config.clone());
    }

    /// Returns the next texture to be presented by the swapchain for drawing.
    ///
    /// In order to present the [`SurfaceTexture`] returned by this method,
    /// first a [`Queue::submit`] needs to be done with some work rendering to this texture.
    /// Then [`SurfaceTexture::present`] needs to be called.
    ///
    /// If a SurfaceTexture referencing this surface is alive when the swapchain is recreated,
    /// recreating the swapchain will panic.
    pub fn get_current_texture(&self) -> Result<SurfaceTexture, SurfaceError> {
        let (texture_id, texture_data, status, detail) =
            DynContext::surface_get_current_texture(&*self.context, &self.id, self.data.as_ref());

        let suboptimal = match status {
            SurfaceStatus::Good => false,
            SurfaceStatus::Suboptimal => true,
            SurfaceStatus::Timeout => return Err(SurfaceError::Timeout),
            SurfaceStatus::Outdated => return Err(SurfaceError::Outdated),
            SurfaceStatus::Lost => return Err(SurfaceError::Lost),
        };

        let guard = self.config.lock();
        let config = guard
            .as_ref()
            .expect("This surface has not been configured yet.");

        let descriptor = TextureDescriptor {
            label: None,
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            format: config.format,
            usage: config.usage,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            view_formats: &[],
        };

        texture_id
            .zip(texture_data)
            .map(|(id, data)| SurfaceTexture {
                texture: Texture {
                    context: Arc::clone(&self.context),
                    id,
                    data,
                    owned: false,
                    descriptor,
                },
                suboptimal,
                presented: false,
                detail,
            })
            .ok_or(SurfaceError::Lost)
    }

    /// Returns the inner hal Surface using a callback. The hal surface will be `None` if the
    /// backend type argument does not match with this wgpu Surface
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Surface must not be manually destroyed
    #[cfg(any(
        not(target_arch = "wasm32"),
        target_os = "emscripten",
        feature = "webgl"
    ))]
    pub unsafe fn as_hal_mut<A: wgc::hub::HalApi, F: FnOnce(Option<&mut A::Surface>) -> R, R>(
        &mut self,
        hal_surface_callback: F,
    ) -> R {
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::Context>()
                .unwrap()
                .surface_as_hal_mut::<A, F, R>(
                    self.data.downcast_ref().unwrap(),
                    hal_surface_callback,
                )
        }
    }
}

/// Opaque globally-unique identifier
#[cfg(feature = "expose-ids")]
#[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
#[repr(transparent)]
pub struct Id<T>(core::num::NonZeroU64, std::marker::PhantomData<*mut T>);

#[cfg(feature = "expose-ids")]
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "expose-ids")]
impl<T> Copy for Id<T> {}

#[cfg(feature = "expose-ids")]
impl<T> Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_tuple("Id").field(&self.0).finish()
    }
}

#[cfg(feature = "expose-ids")]
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Id<T>) -> bool {
        self.0 == other.0
    }
}

#[cfg(feature = "expose-ids")]
impl<T> Eq for Id<T> {}

#[cfg(feature = "expose-ids")]
impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[cfg(feature = "expose-ids")]
impl Adapter {
    /// Returns a globally-unique identifier for this `Adapter`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Adapter`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Adapter> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Device {
    /// Returns a globally-unique identifier for this `Device`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Device`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Device> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Queue {
    /// Returns a globally-unique identifier for this `Queue`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Queue`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Queue> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl ShaderModule {
    /// Returns a globally-unique identifier for this `ShaderModule`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `ShaderModule`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<ShaderModule> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl BindGroupLayout {
    /// Returns a globally-unique identifier for this `BindGroupLayout`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `BindGroupLayout`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<BindGroupLayout> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl BindGroup {
    /// Returns a globally-unique identifier for this `BindGroup`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `BindGroup`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<BindGroup> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl TextureView {
    /// Returns a globally-unique identifier for this `TextureView`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `TextureView`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<TextureView> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Sampler {
    /// Returns a globally-unique identifier for this `Sampler`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Sampler`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Sampler> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Buffer {
    /// Returns a globally-unique identifier for this `Buffer`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Buffer`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Buffer> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Texture {
    /// Returns a globally-unique identifier for this `Texture`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Texture`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Texture> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl QuerySet {
    /// Returns a globally-unique identifier for this `QuerySet`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `QuerySet`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<QuerySet> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl PipelineLayout {
    /// Returns a globally-unique identifier for this `PipelineLayout`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `PipelineLayout`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<PipelineLayout> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl RenderPipeline {
    /// Returns a globally-unique identifier for this `RenderPipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `RenderPipeline`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<RenderPipeline> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl ComputePipeline {
    /// Returns a globally-unique identifier for this `ComputePipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `ComputePipeline`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<ComputePipeline> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl RenderBundle {
    /// Returns a globally-unique identifier for this `RenderBundle`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `RenderBundle`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<RenderBundle> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

#[cfg(feature = "expose-ids")]
impl Surface {
    /// Returns a globally-unique identifier for this `Surface`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be unique among all `Surface`s created from the same
    /// `Instance`.
    #[cfg_attr(docsrs, doc(cfg(feature = "expose-ids")))]
    pub fn global_id(&self) -> Id<Surface> {
        Id(self.id.global_id(), std::marker::PhantomData)
    }
}

/// Type for the callback of uncaptured error handler
pub trait UncapturedErrorHandler: Fn(Error) + Send + 'static {}
impl<T> UncapturedErrorHandler for T where T: Fn(Error) + Send + 'static {}

/// Error type
#[derive(Debug)]
pub enum Error {
    /// Out of memory error
    OutOfMemory {
        /// Lower level source of the error.
        source: Box<dyn error::Error + Send + 'static>,
    },
    /// Validation error, signifying a bug in code or data
    Validation {
        /// Lower level source of the error.
        source: Box<dyn error::Error + Send + 'static>,
        /// Description of the validation error.
        description: String,
    },
}
static_assertions::assert_impl_all!(Error: Send);

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::OutOfMemory { source } => Some(source.as_ref()),
            Error::Validation { source, .. } => Some(source.as_ref()),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OutOfMemory { .. } => f.write_str("Out of Memory"),
            Error::Validation { description, .. } => f.write_str(description),
        }
    }
}
