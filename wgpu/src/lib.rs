//! A cross-platform graphics and compute library based on [WebGPU](https://gpuweb.github.io/gpuweb/).
//!
//! To start using the API, create an [`Instance`].
//!
//! ## Feature flags
#![doc = document_features::document_features!()]
//!
//! ### Feature Aliases
//!
//! These features aren't actually features on the crate itself, but a convenient shorthand for
//! complicated cases.
//!
//! - **`wgpu_core`** --- Enabled when there is any non-webgpu backend enabled on the platform.
//! - **`naga`** ---- Enabled when any non-wgsl shader input is enabled.
//!

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![doc(html_logo_url = "https://raw.githubusercontent.com/gfx-rs/wgpu/trunk/logo.png")]
#![warn(missing_docs, rust_2018_idioms, unsafe_op_in_unsafe_fn)]

mod backend;
mod context;
pub mod util;
#[macro_use]
mod macros;

use std::{
    any::Any,
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    error, fmt,
    future::Future,
    marker::PhantomData,
    num::{NonZeroU32, NonZeroU64},
    ops::{Bound, Deref, DerefMut, Range, RangeBounds},
    sync::Arc,
    thread,
};

#[allow(unused_imports)] // Unused if all backends are disabled.
use context::Context;

use context::{DeviceRequest, DynContext, ObjectId};
use parking_lot::Mutex;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
pub use wgt::{
    AdapterInfo, AddressMode, AstcBlock, AstcChannel, Backend, Backends, BindGroupLayoutEntry,
    BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferAddress,
    BufferBindingType, BufferSize, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandBufferDescriptor, CompareFunction, CompositeAlphaMode, DepthBiasState,
    DepthStencilState, DeviceLostReason, DeviceType, DownlevelCapabilities, DownlevelFlags,
    Dx12Compiler, DynamicOffset, Extent3d, Face, Features, FilterMode, FrontFace,
    Gles3MinorVersion, ImageDataLayout, ImageSubresourceRange, IndexFormat, InstanceDescriptor,
    InstanceFlags, Limits, MaintainResult, MemoryHints, MultisampleState, Origin2d, Origin3d,
    PipelineStatisticsTypes, PolygonMode, PowerPreference, PredefinedColorSpace, PresentMode,
    PresentationTimestamp, PrimitiveState, PrimitiveTopology, PushConstantRange, QueryType,
    RenderBundleDepthStencil, SamplerBindingType, SamplerBorderColor, ShaderLocation, ShaderModel,
    ShaderStages, StencilFaceState, StencilOperation, StencilState, StorageTextureAccess,
    SurfaceCapabilities, SurfaceStatus, TextureAspect, TextureDimension, TextureFormat,
    TextureFormatFeatureFlags, TextureFormatFeatures, TextureSampleType, TextureUsages,
    TextureViewDimension, VertexAttribute, VertexFormat, VertexStepMode, WasmNotSend,
    WasmNotSendSync, WasmNotSync, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
    MAP_ALIGNMENT, PUSH_CONSTANT_ALIGNMENT, QUERY_RESOLVE_BUFFER_ALIGNMENT, QUERY_SET_MAX_QUERIES,
    QUERY_SIZE, VERTEX_STRIDE_ALIGNMENT,
};

/// Re-export of our `wgpu-core` dependency.
///
#[cfg(wgpu_core)]
pub use ::wgc as core;

/// Re-export of our `wgpu-hal` dependency.
///
///
#[cfg(wgpu_core)]
pub use ::hal;

/// Re-export of our `naga` dependency.
///
#[cfg(wgpu_core)]
#[cfg_attr(docsrs, doc(cfg(any(wgpu_core, naga))))]
// We re-export wgpu-core's re-export of naga, as we may not have direct access to it.
pub use ::wgc::naga;
/// Re-export of our `naga` dependency.
///
#[cfg(all(not(wgpu_core), naga))]
#[cfg_attr(docsrs, doc(cfg(any(wgpu_core, naga))))]
// If that's not available, we re-export our own.
pub use naga;

/// Re-export of our `raw-window-handle` dependency.
///
pub use raw_window_handle as rwh;

/// Re-export of our `web-sys` dependency.
///
#[cfg(any(webgl, webgpu))]
pub use web_sys;

// wasm-only types, we try to keep as many types non-platform
// specific, but these need to depend on web-sys.
#[cfg(any(webgpu, webgl))]
pub use wgt::{ExternalImageSource, ImageCopyExternalImage};

/// Filter for error scopes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd)]
pub enum ErrorFilter {
    /// Catch only out-of-memory errors.
    OutOfMemory,
    /// Catch only validation errors.
    Validation,
    /// Catch only internal errors.
    Internal,
}
static_assertions::assert_impl_all!(ErrorFilter: Send, Sync);

type C = dyn DynContext;
#[cfg(send_sync)]
type Data = dyn Any + Send + Sync;
#[cfg(not(send_sync))]
type Data = dyn Any;

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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Device: Send, Sync);

/// Identifier for a particular call to [`Queue::submit`]. Can be used
/// as part of an argument to [`Device::poll`] to block for a particular
/// submission to finish.
///
/// This type is unique to the Rust API of `wgpu`.
/// There is no analogue in the WebGPU specification.
#[derive(Debug, Clone)]
pub struct SubmissionIndex(Arc<crate::Data>);
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubmissionIndex: Send, Sync);

/// The mapped portion of a buffer, if any, and its outstanding views.
///
/// This ensures that views fall within the mapped range and don't overlap, and
/// also takes care of turning `Option<BufferSize>` sizes into actual buffer
/// offsets.
#[derive(Debug)]
struct MapContext {
    /// The overall size of the buffer.
    ///
    /// This is just a convenient copy of [`Buffer::size`].
    total_size: BufferAddress,

    /// The range of the buffer that is mapped.
    ///
    /// This is `0..0` if the buffer is not mapped. This becomes non-empty when
    /// the buffer is mapped at creation time, and when you call `map_async` on
    /// some [`BufferSlice`] (so technically, it indicates the portion that is
    /// *or has been requested to be* mapped.)
    ///
    /// All [`BufferView`]s and [`BufferViewMut`]s must fall within this range.
    initial_range: Range<BufferAddress>,

    /// The ranges covered by all outstanding [`BufferView`]s and
    /// [`BufferViewMut`]s. These are non-overlapping, and are all contained
    /// within `initial_range`.
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

    /// Record that the buffer is no longer mapped.
    fn reset(&mut self) {
        self.initial_range = 0..0;

        assert!(
            self.sub_ranges.is_empty(),
            "You cannot unmap a buffer that still has accessible mapped views"
        );
    }

    /// Record that the `size` bytes of the buffer at `offset` are now viewed.
    ///
    /// Return the byte offset within the buffer of the end of the viewed range.
    ///
    /// # Panics
    ///
    /// This panics if the given range overlaps with any existing range.
    fn add(&mut self, offset: BufferAddress, size: Option<BufferSize>) -> BufferAddress {
        let end = match size {
            Some(s) => offset + s.get(),
            None => self.initial_range.end,
        };
        assert!(self.initial_range.start <= offset && end <= self.initial_range.end);
        // This check is essential for avoiding undefined behavior: it is the
        // only thing that ensures that `&mut` references to the buffer's
        // contents don't alias anything else.
        for sub in self.sub_ranges.iter() {
            assert!(
                end <= sub.start || offset >= sub.end,
                "Intersecting map range with {sub:?}"
            );
        }
        self.sub_ranges.push(offset..end);
        end
    }

    /// Record that the `size` bytes of the buffer at `offset` are no longer viewed.
    ///
    /// # Panics
    ///
    /// This panics if the given range does not exactly match one previously
    /// passed to [`add`].
    ///
    /// [`add]`: MapContext::add
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
///
/// A `Buffer`'s bytes have "interior mutability": functions like
/// [`Queue::write_buffer`] or [mapping] a buffer for writing only require a
/// `&Buffer`, not a `&mut Buffer`, even though they modify its contents. `wgpu`
/// prevents simultaneous reads and writes of buffer contents using run-time
/// checks.
///
/// [mapping]: Buffer#mapping-buffers
///
/// # Mapping buffers
///
/// If a `Buffer` is created with the appropriate [`usage`], it can be *mapped*:
/// you can make its contents accessible to the CPU as an ordinary `&[u8]` or
/// `&mut [u8]` slice of bytes. Buffers created with the
/// [`mapped_at_creation`][mac] flag set are also mapped initially.
///
/// Depending on the hardware, the buffer could be memory shared between CPU and
/// GPU, so that the CPU has direct access to the same bytes the GPU will
/// consult; or it may be ordinary CPU memory, whose contents the system must
/// copy to/from the GPU as needed. This crate's API is designed to work the
/// same way in either case: at any given time, a buffer is either mapped and
/// available to the CPU, or unmapped and ready for use by the GPU, but never
/// both. This makes it impossible for either side to observe changes by the
/// other immediately, and any necessary transfers can be carried out when the
/// buffer transitions from one state to the other.
///
/// There are two ways to map a buffer:
///
/// - If [`BufferDescriptor::mapped_at_creation`] is `true`, then the entire
///   buffer is mapped when it is created. This is the easiest way to initialize
///   a new buffer. You can set `mapped_at_creation` on any kind of buffer,
///   regardless of its [`usage`] flags.
///
/// - If the buffer's [`usage`] includes the [`MAP_READ`] or [`MAP_WRITE`]
///   flags, then you can call `buffer.slice(range).map_async(mode, callback)`
///   to map the portion of `buffer` given by `range`. This waits for the GPU to
///   finish using the buffer, and invokes `callback` as soon as the buffer is
///   safe for the CPU to access.
///
/// Once a buffer is mapped:
///
/// - You can call `buffer.slice(range).get_mapped_range()` to obtain a
///   [`BufferView`], which dereferences to a `&[u8]` that you can use to read
///   the buffer's contents.
///
/// - Or, you can call `buffer.slice(range).get_mapped_range_mut()` to obtain a
///   [`BufferViewMut`], which dereferences to a `&mut [u8]` that you can use to
///   read and write the buffer's contents.
///
/// The given `range` must fall within the mapped portion of the buffer. If you
/// attempt to access overlapping ranges, even for shared access only, these
/// methods panic.
///
/// While a buffer is mapped, you may not submit any commands to the GPU that
/// access it. You may record command buffers that use the buffer, but if you
/// submit them while the buffer is mapped, submission will panic.
///
/// When you are done using the buffer on the CPU, you must call
/// [`Buffer::unmap`] to make it available for use by the GPU again. All
/// [`BufferView`] and [`BufferViewMut`] views referring to the buffer must be
/// dropped before you unmap it; otherwise, [`Buffer::unmap`] will panic.
///
/// # Example
///
/// If `buffer` was created with [`BufferUsages::MAP_WRITE`], we could fill it
/// with `f32` values like this:
///
/// ```no_run
/// # mod bytemuck {
/// #     pub fn cast_slice_mut(bytes: &mut [u8]) -> &mut [f32] { todo!() }
/// # }
/// # let device: wgpu::Device = todo!();
/// # let buffer: wgpu::Buffer = todo!();
/// let buffer = std::sync::Arc::new(buffer);
/// let capturable = buffer.clone();
/// buffer.slice(..).map_async(wgpu::MapMode::Write, move |result| {
///     if result.is_ok() {
///         let mut view = capturable.slice(..).get_mapped_range_mut();
///         let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
///         floats.fill(42.0);
///         drop(view);
///         capturable.unmap();
///     }
/// });
/// ```
///
/// This code takes the following steps:
///
/// - First, it moves `buffer` into an [`Arc`], and makes a clone for capture by
///   the callback passed to [`map_async`]. Since a [`map_async`] callback may be
///   invoked from another thread, interaction between the callback and the
///   thread calling [`map_async`] generally requires some sort of shared heap
///   data like this. In real code, the [`Arc`] would probably own some larger
///   structure that itself owns `buffer`.
///
/// - Then, it calls [`Buffer::slice`] to make a [`BufferSlice`] referring to
///   the buffer's entire contents.
///
/// - Next, it calls [`BufferSlice::map_async`] to request that the bytes to
///   which the slice refers be made accessible to the CPU ("mapped"). This may
///   entail waiting for previously enqueued operations on `buffer` to finish.
///   Although [`map_async`] itself always returns immediately, it saves the
///   callback function to be invoked later.
///
/// - When some later call to [`Device::poll`] or [`Instance::poll_all`] (not
///   shown in this example) determines that the buffer is mapped and ready for
///   the CPU to use, it invokes the callback function.
///
/// - The callback function calls [`Buffer::slice`] and then
///   [`BufferSlice::get_mapped_range_mut`] to obtain a [`BufferViewMut`], which
///   dereferences to a `&mut [u8]` slice referring to the buffer's bytes.
///
/// - It then uses the [`bytemuck`] crate to turn the `&mut [u8]` into a `&mut
///   [f32]`, and calls the slice [`fill`] method to fill the buffer with a
///   useful value.
///
/// - Finally, the callback drops the view and calls [`Buffer::unmap`] to unmap
///   the buffer. In real code, the callback would also need to do some sort of
///   synchronization to let the rest of the program know that it has completed
///   its work.
///
/// If using [`map_async`] directly is awkward, you may find it more convenient to
/// use [`Queue::write_buffer`] and [`util::DownloadBuffer::read_buffer`].
/// However, those each have their own tradeoffs; the asynchronous nature of GPU
/// execution makes it hard to avoid friction altogether.
///
/// [`Arc`]: std::sync::Arc
/// [`map_async`]: BufferSlice::map_async
/// [`bytemuck`]: https://crates.io/crates/bytemuck
/// [`fill`]: slice::fill
///
/// ## Mapping buffers on the web
///
/// When compiled to WebAssembly and running in a browser content process,
/// `wgpu` implements its API in terms of the browser's WebGPU implementation.
/// In this context, `wgpu` is further isolated from the GPU:
///
/// - Depending on the browser's WebGPU implementation, mapping and unmapping
///   buffers probably entails copies between WebAssembly linear memory and the
///   graphics driver's buffers.
///
/// - All modern web browsers isolate web content in its own sandboxed process,
///   which can only interact with the GPU via interprocess communication (IPC).
///   Although most browsers' IPC systems use shared memory for large data
///   transfers, there will still probably need to be copies into and out of the
///   shared memory buffers.
///
/// All of these copies contribute to the cost of buffer mapping in this
/// configuration.
///
/// [`usage`]: BufferDescriptor::usage
/// [mac]: BufferDescriptor::mapped_at_creation
/// [`MAP_READ`]: BufferUsages::MAP_READ
/// [`MAP_WRITE`]: BufferUsages::MAP_WRITE
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Buffer: Send, Sync);

/// A slice of a [`Buffer`], to be mapped, used for vertex or index data, or the like.
///
/// You can create a `BufferSlice` by calling [`Buffer::slice`]:
///
/// ```no_run
/// # let buffer: wgpu::Buffer = todo!();
/// let slice = buffer.slice(10..20);
/// ```
///
/// This returns a slice referring to the second ten bytes of `buffer`. To get a
/// slice of the entire `Buffer`:
///
/// ```no_run
/// # let buffer: wgpu::Buffer = todo!();
/// let whole_buffer_slice = buffer.slice(..);
/// ```
///
/// You can pass buffer slices to methods like [`RenderPass::set_vertex_buffer`]
/// and [`RenderPass::set_index_buffer`] to indicate which portion of the buffer
/// a draw call should consult.
///
/// To access the slice's contents on the CPU, you must first [map] the buffer,
/// and then call [`BufferSlice::get_mapped_range`] or
/// [`BufferSlice::get_mapped_range_mut`] to obtain a view of the slice's
/// contents. See the documentation on [mapping][map] for more details,
/// including example code.
///
/// Unlike a Rust shared slice `&[T]`, whose existence guarantees that
/// nobody else is modifying the `T` values to which it refers, a
/// [`BufferSlice`] doesn't guarantee that the buffer's contents aren't
/// changing. You can still record and submit commands operating on the
/// buffer while holding a [`BufferSlice`]. A [`BufferSlice`] simply
/// represents a certain range of the buffer's bytes.
///
/// The `BufferSlice` type is unique to the Rust API of `wgpu`. In the WebGPU
/// specification, an offset and size are specified as arguments to each call
/// working with the [`Buffer`], instead.
///
/// [map]: Buffer#mapping-buffers
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a> {
    buffer: &'a Buffer,
    offset: BufferAddress,
    size: Option<BufferSize>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BufferSlice<'_>: Send, Sync);

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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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
/// be presented. A `Surface` may be created with the function [`Instance::create_surface`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// [`GPUCanvasContext`](https://gpuweb.github.io/gpuweb/#canvas-context)
/// serves a similar role.
pub struct Surface<'window> {
    context: Arc<C>,

    /// Optionally, keep the source of the handle used for the surface alive.
    ///
    /// This is useful for platforms where the surface is created from a window and the surface
    /// would become invalid when the window is dropped.
    _handle_source: Option<Box<dyn WindowHandle + 'window>>,

    /// Wgpu-core surface id.
    id: ObjectId,

    /// Additional surface data returned by [`DynContext::instance_create_surface`].
    surface_data: Box<Data>,

    // Stores the latest `SurfaceConfiguration` that was set using `Surface::configure`.
    // It is required to set the attributes of the `SurfaceTexture` in the
    // `Surface::get_current_texture` method.
    // Because the `Surface::configure` method operates on an immutable reference this type has to
    // be wrapped in a mutex and since the configuration is only supplied after the surface has
    // been created is is additionally wrapped in an option.
    config: Mutex<Option<SurfaceConfiguration>>,
}

// This custom implementation is required because [`Surface::_surface`] doesn't
// require [`Debug`](fmt::Debug), which we should not require from the user.
impl<'window> fmt::Debug for Surface<'window> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Surface")
            .field("context", &self.context)
            .field(
                "_handle_source",
                &if self._handle_source.is_some() {
                    "Some"
                } else {
                    "None"
                },
            )
            .field("id", &self.id)
            .field("data", &self.surface_data)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(Surface<'_>: Send, Sync);

impl Drop for Surface<'_> {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .surface_drop(&self.id, self.surface_data.as_ref())
        }
    }
}

/// Super trait for window handles as used in [`SurfaceTarget`].
pub trait WindowHandle: HasWindowHandle + HasDisplayHandle + WasmNotSendSync {}

impl<T> WindowHandle for T where T: HasWindowHandle + HasDisplayHandle + WasmNotSendSync {}

/// The window/canvas/surface/swap-chain/etc. a surface is attached to, for use with safe surface creation.
///
/// This is either a window or an actual web canvas depending on the platform and
/// enabled features.
/// Refer to the individual variants for more information.
///
/// See also [`SurfaceTargetUnsafe`] for unsafe variants.
#[non_exhaustive]
pub enum SurfaceTarget<'window> {
    /// Window handle producer.
    ///
    /// If the specified display and window handle are not supported by any of the backends, then the surface
    /// will not be supported by any adapters.
    ///
    /// # Errors
    ///
    /// - On WebGL2: surface creation returns an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    ///
    /// # Panics
    ///
    /// - On macOS/Metal: will panic if not called on the main thread.
    /// - On web: will panic if the `raw_window_handle` does not properly refer to a
    ///   canvas element.
    Window(Box<dyn WindowHandle + 'window>),

    /// Surface from a `web_sys::HtmlCanvasElement`.
    ///
    /// The `canvas` argument must be a valid `<canvas>` element to
    /// create a surface upon.
    ///
    /// # Errors
    ///
    /// - On WebGL2: surface creation will return an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    #[cfg(any(webgpu, webgl))]
    Canvas(web_sys::HtmlCanvasElement),

    /// Surface from a `web_sys::OffscreenCanvas`.
    ///
    /// The `canvas` argument must be a valid `OffscreenCanvas` object
    /// to create a surface upon.
    ///
    /// # Errors
    ///
    /// - On WebGL2: surface creation will return an error if the browser does not support WebGL2,
    ///   or declines to provide GPU access (such as due to a resource shortage).
    #[cfg(any(webgpu, webgl))]
    OffscreenCanvas(web_sys::OffscreenCanvas),
}

impl<'a, T> From<T> for SurfaceTarget<'a>
where
    T: WindowHandle + 'a,
{
    fn from(window: T) -> Self {
        Self::Window(Box::new(window))
    }
}

/// The window/canvas/surface/swap-chain/etc. a surface is attached to, for use with unsafe surface creation.
///
/// This is either a window or an actual web canvas depending on the platform and
/// enabled features.
/// Refer to the individual variants for more information.
///
/// See also [`SurfaceTarget`] for safe variants.
#[non_exhaustive]
pub enum SurfaceTargetUnsafe {
    /// Raw window & display handle.
    ///
    /// If the specified display and window handle are not supported by any of the backends, then the surface
    /// will not be supported by any adapters.
    ///
    /// # Safety
    ///
    /// - `raw_window_handle` & `raw_display_handle` must be valid objects to create a surface upon.
    /// - `raw_window_handle` & `raw_display_handle` must remain valid until after the returned
    ///    [`Surface`] is  dropped.
    RawHandle {
        /// Raw display handle, underlying display must outlive the surface created from this.
        raw_display_handle: raw_window_handle::RawDisplayHandle,

        /// Raw display handle, underlying window must outlive the surface created from this.
        raw_window_handle: raw_window_handle::RawWindowHandle,
    },

    /// Surface from `CoreAnimationLayer`.
    ///
    /// # Safety
    ///
    /// - layer must be a valid object to create a surface upon.
    #[cfg(metal)]
    CoreAnimationLayer(*mut std::ffi::c_void),

    /// Surface from `IDCompositionVisual`.
    ///
    /// # Safety
    ///
    /// - visual must be a valid IDCompositionVisual to create a surface upon.
    #[cfg(dx12)]
    CompositionVisual(*mut std::ffi::c_void),

    /// Surface from DX12 `SurfaceHandle`.
    ///
    /// # Safety
    ///
    /// - surface_handle must be a valid SurfaceHandle to create a surface upon.
    #[cfg(dx12)]
    SurfaceHandle(*mut std::ffi::c_void),

    /// Surface from DX12 `SwapChainPanel`.
    ///
    /// # Safety
    ///
    /// - visual must be a valid SwapChainPanel to create a surface upon.
    #[cfg(dx12)]
    SwapChainPanel(*mut std::ffi::c_void),
}

impl SurfaceTargetUnsafe {
    /// Creates a [`SurfaceTargetUnsafe::RawHandle`] from a window.
    ///
    /// # Safety
    ///
    /// - `window` must outlive the resulting surface target
    ///   (and subsequently the surface created for this target).
    pub unsafe fn from_window<T>(window: &T) -> Result<Self, raw_window_handle::HandleError>
    where
        T: HasDisplayHandle + HasWindowHandle,
    {
        Ok(Self::RawHandle {
            raw_display_handle: window.display_handle()?.as_raw(),
            raw_window_handle: window.window_handle()?.as_raw(),
        })
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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ShaderModule: Send, Sync);

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .shader_module_drop(&self.id, self.data.as_ref());
        }
    }
}

impl ShaderModule {
    /// Get the compilation info for the shader module.
    pub fn get_compilation_info(&self) -> impl Future<Output = CompilationInfo> + WasmNotSend {
        self.context
            .shader_get_compilation_info(&self.id, self.data.as_ref())
    }
}

/// Compilation information for a shader module.
///
/// Corresponds to [WebGPU `GPUCompilationInfo`](https://gpuweb.github.io/gpuweb/#gpucompilationinfo).
/// The source locations use bytes, and index a UTF-8 encoded string.
#[derive(Debug, Clone)]
pub struct CompilationInfo {
    /// The messages from the shader compilation process.
    pub messages: Vec<CompilationMessage>,
}

/// A single message from the shader compilation process.
///
/// Roughly corresponds to [`GPUCompilationMessage`](https://www.w3.org/TR/webgpu/#gpucompilationmessage),
/// except that the location uses UTF-8 for all positions.
#[derive(Debug, Clone)]
pub struct CompilationMessage {
    /// The text of the message.
    pub message: String,
    /// The type of the message.
    pub message_type: CompilationMessageType,
    /// Where in the source code the message points at.
    pub location: Option<SourceLocation>,
}

/// The type of a compilation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationMessageType {
    /// An error message.
    Error,
    /// A warning message.
    Warning,
    /// An informational message.
    Info,
}

/// A human-readable representation for a span, tailored for text source.
///
/// Roughly corresponds to the positional members of [`GPUCompilationMessage`][gcm] from
/// the WebGPU specification, except
/// - `offset` and `length` are in bytes (UTF-8 code units), instead of UTF-16 code units.
/// - `line_position` is in bytes (UTF-8 code units), and is usually not directly intended for humans.
///
/// [gcm]: https://www.w3.org/TR/webgpu/#gpucompilationmessage
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SourceLocation {
    /// 1-based line number.
    pub line_number: u32,
    /// 1-based column in code units (in bytes) of the start of the span.
    /// Remember to convert accordingly when displaying to the user.
    pub line_position: u32,
    /// 0-based Offset in code units (in bytes) of the start of the span.
    pub offset: u32,
    /// Length in code units (in bytes) of the span.
    pub length: u32,
}

#[cfg(all(feature = "wgsl", wgpu_core))]
impl From<naga::error::ShaderError<naga::front::wgsl::ParseError>> for CompilationInfo {
    fn from(value: naga::error::ShaderError<naga::front::wgsl::ParseError>) -> Self {
        CompilationInfo {
            messages: vec![CompilationMessage {
                message: value.to_string(),
                message_type: CompilationMessageType::Error,
                location: value.inner.location(&value.source).map(Into::into),
            }],
        }
    }
}
#[cfg(feature = "glsl")]
impl From<naga::error::ShaderError<naga::front::glsl::ParseErrors>> for CompilationInfo {
    fn from(value: naga::error::ShaderError<naga::front::glsl::ParseErrors>) -> Self {
        let messages = value
            .inner
            .errors
            .into_iter()
            .map(|err| CompilationMessage {
                message: err.to_string(),
                message_type: CompilationMessageType::Error,
                location: err.location(&value.source).map(Into::into),
            })
            .collect();
        CompilationInfo { messages }
    }
}

#[cfg(feature = "spirv")]
impl From<naga::error::ShaderError<naga::front::spv::Error>> for CompilationInfo {
    fn from(value: naga::error::ShaderError<naga::front::spv::Error>) -> Self {
        CompilationInfo {
            messages: vec![CompilationMessage {
                message: value.to_string(),
                message_type: CompilationMessageType::Error,
                location: None,
            }],
        }
    }
}

#[cfg(any(wgpu_core, naga))]
impl From<naga::error::ShaderError<naga::WithSpan<naga::valid::ValidationError>>>
    for CompilationInfo
{
    fn from(value: naga::error::ShaderError<naga::WithSpan<naga::valid::ValidationError>>) -> Self {
        CompilationInfo {
            messages: vec![CompilationMessage {
                message: value.to_string(),
                message_type: CompilationMessageType::Error,
                location: value.inner.location(&value.source).map(Into::into),
            }],
        }
    }
}

#[cfg(any(wgpu_core, naga))]
impl From<naga::SourceLocation> for SourceLocation {
    fn from(value: naga::SourceLocation) -> Self {
        SourceLocation {
            length: value.length,
            offset: value.offset,
            line_number: value.line_number,
            line_position: value.line_position,
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
#[cfg_attr(feature = "naga-ir", allow(clippy::large_enum_variant))]
#[derive(Clone, Debug)]
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
    #[cfg(feature = "naga-ir")]
    Naga(Cow<'static, naga::Module>),
    /// Dummy variant because `Naga` doesn't have a lifetime and without enough active features it
    /// could be the last one active.
    #[doc(hidden)]
    Dummy(PhantomData<&'a ()>),
}
static_assertions::assert_impl_all!(ShaderSource<'_>: Send, Sync);

/// Descriptor for use with [`Device::create_shader_module`].
///
/// Corresponds to [WebGPU `GPUShaderModuleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpushadermoduledescriptor).
#[derive(Clone, Debug)]
pub struct ShaderModuleDescriptor<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Source code for the shader.
    pub source: ShaderSource<'a>,
}
static_assertions::assert_impl_all!(ShaderModuleDescriptor<'_>: Send, Sync);

/// Descriptor for a shader module given by SPIR-V binary, for use with
/// [`Device::create_shader_module_spirv`].
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// only WGSL source code strings are accepted.
#[derive(Debug)]
pub struct ShaderModuleDescriptorSpirV<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Binary SPIR-V data, in 4-byte words.
    pub source: Cow<'a, [u32]>,
}
static_assertions::assert_impl_all!(ShaderModuleDescriptorSpirV<'_>: Send, Sync);

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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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
#[cfg(send_sync)]
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

/// Handle to a pipeline cache, which is used to accelerate
/// creating [`RenderPipeline`]s and [`ComputePipeline`]s
/// in subsequent executions
///
/// This reuse is only applicable for the same or similar devices.
/// See [`util::pipeline_cache_key`] for some details.
///
/// # Background
///
/// In most GPU drivers, shader code must be converted into a machine code
/// which can be executed on the GPU.
/// Generating this machine code can require a lot of computation.
/// Pipeline caches allow this computation to be reused between executions
/// of the program.
/// This can be very useful for reducing program startup time.
///
/// Note that most desktop GPU drivers will manage their own caches,
/// meaning that little advantage can be gained from this on those platforms.
/// However, on some platforms, especially Android, drivers leave this to the
/// application to implement.
///
/// Unfortunately, drivers do not expose whether they manage their own caches.
/// Some reasonable policies for applications to use are:
/// - Manage their own pipeline cache on all platforms
/// - Only manage pipeline caches on Android
///
/// # Usage
///
/// It is valid to use this resource when creating multiple pipelines, in
/// which case it will likely cache each of those pipelines.
/// It is also valid to create a new cache for each pipeline.
///
/// This resource is most useful when the data produced from it (using
/// [`PipelineCache::get_data`]) is persisted.
/// Care should be taken that pipeline caches are only used for the same device,
/// as pipeline caches from compatible devices are unlikely to provide any advantage.
/// `util::pipeline_cache_key` can be used as a file/directory name to help ensure that.
///
/// It is recommended to store pipeline caches atomically. If persisting to disk,
/// this can usually be achieved by creating a temporary file, then moving/[renaming]
/// the temporary file over the existing cache
///
/// # Storage Usage
///
/// There is not currently an API available to reduce the size of a cache.
/// This is due to limitations in the underlying graphics APIs used.
/// This is especially impactful if your application is being updated, so
/// previous caches are no longer being used.
///
/// One option to work around this is to regenerate the cache.
/// That is, creating the pipelines which your program runs using
/// with the stored cached data, then recreating the *same* pipelines
/// using a new cache, which your application then store.
///
/// # Implementations
///
/// This resource currently only works on the following backends:
///  - Vulkan
///
/// This type is unique to the Rust API of `wgpu`.
///
/// [renaming]: std::fs::rename
#[derive(Debug)]
pub struct PipelineCache {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineCache: Send, Sync);

impl PipelineCache {
    /// Get the data associated with this pipeline cache.
    /// The data format is an implementation detail of `wgpu`.
    /// The only defined operation on this data setting it as the `data` field
    /// on [`PipelineCacheDescriptor`], then to [`Device::create_pipeline_cache`].
    ///
    /// This function is unique to the Rust API of `wgpu`.
    pub fn get_data(&self) -> Option<Vec<u8>> {
        self.context
            .pipeline_cache_get_data(&self.id, self.data.as_ref())
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .pipeline_cache_drop(&self.id, self.data.as_ref());
        }
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(CommandBuffer: Send, Sync);

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            if let Some(id) = self.id.take() {
                self.context
                    .command_buffer_drop(&id, self.data.take().unwrap().as_ref());
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
#[cfg(send_sync)]
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

/// In-progress recording of a render pass: a list of render commands in a [`CommandEncoder`].
///
/// It can be created with [`CommandEncoder::begin_render_pass()`], whose [`RenderPassDescriptor`]
/// specifies the attachments (textures) that will be rendered to.
///
/// Most of the methods on `RenderPass` serve one of two purposes, identifiable by their names:
///
/// * `draw_*()`: Drawing (that is, encoding a render command, which, when executed by the GPU, will
///   rasterize something and execute shaders).
/// * `set_*()`: Setting part of the [render state](https://gpuweb.github.io/gpuweb/#renderstate)
///   for future drawing commands.
///
/// A render pass may contain any number of drawing commands, and before/between each command the
/// render state may be updated however you wish; each drawing command will be executed using the
/// render state that has been set when the `draw_*()` function is called.
///
/// Corresponds to [WebGPU `GPURenderPassEncoder`](
/// https://gpuweb.github.io/gpuweb/#render-pass-encoder).
#[derive(Debug)]
pub struct RenderPass<'encoder> {
    /// The inner data of the render pass, separated out so it's easy to replace the lifetime with 'static if desired.
    inner: RenderPassInner,

    /// This lifetime is used to protect the [`CommandEncoder`] from being used
    /// while the pass is alive.
    encoder_guard: PhantomData<&'encoder ()>,
}

#[derive(Debug)]
struct RenderPassInner {
    id: ObjectId,
    data: Box<Data>,
    context: Arc<C>,
}

/// In-progress recording of a compute pass.
///
/// It can be created with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassEncoder`](
/// https://gpuweb.github.io/gpuweb/#compute-pass-encoder).
#[derive(Debug)]
pub struct ComputePass<'encoder> {
    /// The inner data of the compute pass, separated out so it's easy to replace the lifetime with 'static if desired.
    inner: ComputePassInner,

    /// This lifetime is used to protect the [`CommandEncoder`] from being used
    /// while the pass is alive.
    encoder_guard: PhantomData<&'encoder ()>,
}

#[derive(Debug)]
struct ComputePassInner {
    id: ObjectId,
    data: Box<Data>,
    context: Arc<C>,
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
#[cfg(send_sync)]
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
#[derive(Debug)]
pub struct QuerySet {
    context: Arc<C>,
    id: ObjectId,
    data: Box<Data>,
}
#[cfg(send_sync)]
#[cfg(send_sync)]
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Queue: Send, Sync);

impl Drop for Queue {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.queue_drop(&self.id, self.data.as_ref());
        }
    }
}

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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BindingResource<'_>: Send, Sync);

/// Describes the segment of a buffer to bind.
///
/// Corresponds to [WebGPU `GPUBufferBinding`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferbinding).
#[derive(Clone, Debug)]
pub struct BufferBinding<'a> {
    /// The buffer to bind.
    pub buffer: &'a Buffer,

    /// Base offset of the buffer, in bytes.
    ///
    /// If the [`has_dynamic_offset`] field of this buffer's layout entry is
    /// `true`, the offset here will be added to the dynamic offset passed to
    /// [`RenderPass::set_bind_group`] or [`ComputePass::set_bind_group`].
    ///
    /// If the buffer was created with [`BufferUsages::UNIFORM`], then this
    /// offset must be a multiple of
    /// [`Limits::min_uniform_buffer_offset_alignment`].
    ///
    /// If the buffer was created with [`BufferUsages::STORAGE`], then this
    /// offset must be a multiple of
    /// [`Limits::min_storage_buffer_offset_alignment`].
    ///
    /// [`has_dynamic_offset`]: BindingType::Buffer::has_dynamic_offset
    pub offset: BufferAddress,

    /// Size of the binding in bytes, or `None` for using the rest of the buffer.
    pub size: Option<BufferSize>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BufferBinding<'_>: Send, Sync);

/// Operation to perform to the output attachment at the start of a render pass.
///
/// Corresponds to [WebGPU `GPULoadOp`](https://gpuweb.github.io/gpuweb/#enumdef-gpuloadop),
/// plus the corresponding clearValue.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoadOp<V> {
    /// Loads the specified value for this attachment into the render pass.
    ///
    /// On some GPU hardware (primarily mobile), "clear" is significantly cheaper
    /// because it avoids loading data from main memory into tile-local memory.
    ///
    /// On other GPU hardware, there isn’t a significant difference.
    ///
    /// As a result, it is recommended to use "clear" rather than "load" in cases
    /// where the initial value doesn’t matter
    /// (e.g. the render target will be cleared using a skybox).
    Clear(V),
    /// Loads the existing value for this attachment into the render pass.
    Load,
}

impl<V: Default> Default for LoadOp<V> {
    fn default() -> Self {
        Self::Clear(Default::default())
    }
}

/// Operation to perform to the output attachment at the end of a render pass.
///
/// Corresponds to [WebGPU `GPUStoreOp`](https://gpuweb.github.io/gpuweb/#enumdef-gpustoreop).
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StoreOp {
    /// Stores the resulting value of the render pass for this attachment.
    #[default]
    Store,
    /// Discards the resulting value of the render pass for this attachment.
    ///
    /// The attachment will be treated as uninitialized afterwards.
    /// (If only either Depth or Stencil texture-aspects is set to `Discard`,
    /// the respective other texture-aspect will be preserved.)
    ///
    /// This can be significantly faster on tile-based render hardware.
    ///
    /// Prefer this if the attachment is not read by subsequent passes.
    Discard,
}

/// Pair of load and store operations for an attachment aspect.
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// separate `loadOp` and `storeOp` fields are used instead.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Operations<V> {
    /// How data should be read through this attachment.
    pub load: LoadOp<V>,
    /// Whether data will be written to through this attachment.
    ///
    /// Note that resolve textures (if specified) are always written to,
    /// regardless of this setting.
    pub store: StoreOp,
}

impl<V: Default> Default for Operations<V> {
    #[inline]
    fn default() -> Self {
        Self {
            load: LoadOp::<V>::default(),
            store: StoreOp::default(),
        }
    }
}

/// Describes the timestamp writes of a render pass.
///
/// For use with [`RenderPassDescriptor`].
/// At least one of `beginning_of_pass_write_index` and `end_of_pass_write_index` must be `Some`.
///
/// Corresponds to [WebGPU `GPURenderPassTimestampWrite`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpasstimestampwrites).
#[derive(Clone, Debug)]
pub struct RenderPassTimestampWrites<'a> {
    /// The query set to write to.
    pub query_set: &'a QuerySet,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPassTimestampWrites<'_>: Send, Sync);

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
    ///
    /// If set, it is always written to, regardless of how [`Self::ops`] is configured.
    pub resolve_target: Option<&'tex TextureView>,
    /// What operations will be performed on this color attachment.
    pub ops: Operations<Color>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPassColorAttachment<'_>: Send, Sync);

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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPassDepthStencilAttachment<'_>: Send, Sync);

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
pub type RequestAdapterOptions<'a, 'b> = RequestAdapterOptionsBase<&'a Surface<'b>>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RequestAdapterOptions<'_, '_>: Send, Sync);
/// Describes a [`Device`].
///
/// For use with [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUDeviceDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudevicedescriptor).
pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(DeviceDescriptor<'_>: Send, Sync);
/// Describes a [`Buffer`].
///
/// For use with [`Device::create_buffer`].
///
/// Corresponds to [WebGPU `GPUBufferDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferdescriptor).
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(BufferDescriptor<'_>: Send, Sync);
/// Describes a [`CommandEncoder`].
///
/// For use with [`Device::create_command_encoder`].
///
/// Corresponds to [WebGPU `GPUCommandEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucommandencoderdescriptor).
pub type CommandEncoderDescriptor<'a> = wgt::CommandEncoderDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CommandEncoderDescriptor<'_>: Send, Sync);
/// Describes a [`RenderBundle`].
///
/// For use with [`RenderBundleEncoder::finish`].
///
/// Corresponds to [WebGPU `GPURenderBundleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderbundledescriptor).
pub type RenderBundleDescriptor<'a> = wgt::RenderBundleDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(RenderBundleDescriptor<'_>: Send, Sync);
/// Describes a [`Texture`].
///
/// For use with [`Device::create_texture`].
///
/// Corresponds to [WebGPU `GPUTextureDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gputexturedescriptor).
pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, &'a [TextureFormat]>;
static_assertions::assert_impl_all!(TextureDescriptor<'_>: Send, Sync);
/// Describes a [`QuerySet`].
///
/// For use with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySetDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuquerysetdescriptor).
pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(QuerySetDescriptor<'_>: Send, Sync);
pub use wgt::Maintain as MaintainBase;
/// Passed to [`Device::poll`] to control how and if it should block.
pub type Maintain = wgt::Maintain<SubmissionIndex>;
#[cfg(send_sync)]
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
static_assertions::assert_impl_all!(TextureViewDescriptor<'_>: Send, Sync);

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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineLayoutDescriptor<'_>: Send, Sync);

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
static_assertions::assert_impl_all!(SamplerDescriptor<'_>: Send, Sync);

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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BindGroupEntry<'_>: Send, Sync);

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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BindGroupDescriptor<'_>: Send, Sync);

/// Describes the attachments of a render pass.
///
/// For use with [`CommandEncoder::begin_render_pass`].
///
/// Corresponds to [WebGPU `GPURenderPassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurenderpassdescriptor).
#[derive(Clone, Debug, Default)]
pub struct RenderPassDescriptor<'a> {
    /// Debug label of the render pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// The color attachments of the render pass.
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
    /// The depth and stencil attachment of the render pass, if any.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    /// Defines which timestamp values will be written for this pass, and where to write them to.
    ///
    /// Requires [`Features::TIMESTAMP_QUERY`] to be enabled.
    pub timestamp_writes: Option<RenderPassTimestampWrites<'a>>,
    /// Defines where the occlusion query results will be stored for this pass.
    pub occlusion_query_set: Option<&'a QuerySet>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPassDescriptor<'_>: Send, Sync);

/// Describes how the vertex buffer is interpreted.
///
/// For use in [`VertexState`].
///
/// Corresponds to [WebGPU `GPUVertexBufferLayout`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuvertexbufferlayout).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct VertexBufferLayout<'a> {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: BufferAddress,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: &'a [VertexAttribute],
}
static_assertions::assert_impl_all!(VertexBufferLayout<'_>: Send, Sync);

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
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(VertexState<'_>: Send, Sync);

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
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The color state of the render targets.
    pub targets: &'a [Option<ColorTargetState>],
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(FragmentState<'_>: Send, Sync);

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
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RenderPipelineDescriptor<'_>: Send, Sync);

/// Describes the timestamp writes of a compute pass.
///
/// For use with [`ComputePassDescriptor`].
/// At least one of `beginning_of_pass_write_index` and `end_of_pass_write_index` must be `Some`.
///
/// Corresponds to [WebGPU `GPUComputePassTimestampWrites`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepasstimestampwrites).
#[derive(Clone, Debug)]
pub struct ComputePassTimestampWrites<'a> {
    /// The query set to write to.
    pub query_set: &'a QuerySet,
    /// The index of the query set at which a start timestamp of this pass is written, if any.
    pub beginning_of_pass_write_index: Option<u32>,
    /// The index of the query set at which an end timestamp of this pass is written, if any.
    pub end_of_pass_write_index: Option<u32>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePassTimestampWrites<'_>: Send, Sync);

/// Describes the attachments of a compute pass.
///
/// For use with [`CommandEncoder::begin_compute_pass`].
///
/// Corresponds to [WebGPU `GPUComputePassDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucomputepassdescriptor).
#[derive(Clone, Default, Debug)]
pub struct ComputePassDescriptor<'a> {
    /// Debug label of the compute pass. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Defines which timestamp values will be written for this pass, and where to write them to.
    ///
    /// Requires [`Features::TIMESTAMP_QUERY`] to be enabled.
    pub timestamp_writes: Option<ComputePassTimestampWrites<'a>>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePassDescriptor<'_>: Send, Sync);

#[derive(Clone, Debug)]
/// Advanced options for use when a pipeline is compiled
///
/// This implements `Default`, and for most users can be set to `Default::default()`
pub struct PipelineCompilationOptions<'a> {
    /// Specifies the values of pipeline-overridable constants in the shader module.
    ///
    /// If an `@id` attribute was specified on the declaration,
    /// the key must be the pipeline constant ID as a decimal ASCII number; if not,
    /// the key must be the constant's identifier name.
    ///
    /// The value may represent any of WGSL's concrete scalar types.
    pub constants: &'a HashMap<String, f64>,
    /// Whether workgroup scoped memory will be initialized with zero values for this stage.
    ///
    /// This is required by the WebGPU spec, but may have overhead which can be avoided
    /// for cross-platform applications
    pub zero_initialize_workgroup_memory: bool,
    /// Should the pipeline attempt to transform vertex shaders to use vertex pulling.
    pub vertex_pulling_transform: bool,
}

impl<'a> Default for PipelineCompilationOptions<'a> {
    fn default() -> Self {
        // HashMap doesn't have a const constructor, due to the use of RandomState
        // This does introduce some synchronisation costs, but these should be minor,
        // and might be cheaper than the alternative of getting new random state
        static DEFAULT_CONSTANTS: std::sync::OnceLock<HashMap<String, f64>> =
            std::sync::OnceLock::new();
        let constants = DEFAULT_CONSTANTS.get_or_init(Default::default);
        Self {
            constants,
            zero_initialize_workgroup_memory: true,
            vertex_pulling_transform: false,
        }
    }
}

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
    /// Advanced options for when this pipeline is compiled
    ///
    /// This implements `Default`, and for most users can be set to `Default::default()`
    pub compilation_options: PipelineCompilationOptions<'a>,
    /// The pipeline cache to use when creating this pipeline.
    pub cache: Option<&'a PipelineCache>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ComputePipelineDescriptor<'_>: Send, Sync);

/// Describes a pipeline cache, which allows reusing compilation work
/// between program runs.
///
/// For use with [`Device::create_pipeline_cache`]
///
/// This type is unique to the Rust API of `wgpu`.
#[derive(Clone, Debug)]
pub struct PipelineCacheDescriptor<'a> {
    /// Debug label of the pipeline cache. This might show up in some logs from `wgpu`
    pub label: Label<'a>,
    /// The data used to initialise the cache initialise
    ///
    /// # Safety
    ///
    /// This data must have been provided from a previous call to
    /// [`PipelineCache::get_data`], if not `None`
    pub data: Option<&'a [u8]>,
    /// Whether to create a cache without data when the provided data
    /// is invalid.
    ///
    /// Recommended to set to true
    pub fallback: bool,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(PipelineCacheDescriptor<'_>: Send, Sync);

pub use wgt::ImageCopyBuffer as ImageCopyBufferBase;
/// View of a buffer which can be used to copy to/from a texture.
///
/// Corresponds to [WebGPU `GPUImageCopyBuffer`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopybuffer).
pub type ImageCopyBuffer<'a> = ImageCopyBufferBase<&'a Buffer>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyBuffer<'_>: Send, Sync);

pub use wgt::ImageCopyTexture as ImageCopyTextureBase;
/// View of a texture which can be used to copy to/from a buffer/texture.
///
/// Corresponds to [WebGPU `GPUImageCopyTexture`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexture).
pub type ImageCopyTexture<'a> = ImageCopyTextureBase<&'a Texture>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyTexture<'_>: Send, Sync);

pub use wgt::ImageCopyTextureTagged as ImageCopyTextureTaggedBase;
/// View of a texture which can be used to copy to a texture, including
/// color space and alpha premultiplication information.
///
/// Corresponds to [WebGPU `GPUImageCopyTextureTagged`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexturetagged).
pub type ImageCopyTextureTagged<'a> = ImageCopyTextureTaggedBase<&'a Texture>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyTexture<'_>: Send, Sync);

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
static_assertions::assert_impl_all!(BindGroupLayoutDescriptor<'_>: Send, Sync);

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
static_assertions::assert_impl_all!(RenderBundleEncoderDescriptor<'_>: Send, Sync);

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
    detail: Box<dyn AnyWasmNotSendSync>,
}
#[cfg(send_sync)]
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

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
    ///
    /// # Panics
    ///
    /// If no backend feature for the active target platform is enabled,
    /// this method will panic, see [`Instance::enabled_backend_features()`].
    fn default() -> Self {
        Self::new(InstanceDescriptor::default())
    }
}

impl Instance {
    /// Returns which backends can be picked for the current build configuration.
    ///
    /// The returned set depends on a combination of target platform and enabled features.
    /// This does *not* do any runtime checks and is exclusively based on compile time information.
    ///
    /// `InstanceDescriptor::backends` does not need to be a subset of this,
    /// but any backend that is not in this set, will not be picked.
    ///
    /// TODO: Right now it's otherwise not possible yet to opt-out of all features on some platforms.
    /// See <https://github.com/gfx-rs/wgpu/issues/3514>
    /// * Windows/Linux/Android: always enables Vulkan and GLES with no way to opt out
    pub const fn enabled_backend_features() -> Backends {
        let mut backends = Backends::empty();

        if cfg!(native) {
            if cfg!(metal) {
                backends = backends.union(Backends::METAL);
            }
            if cfg!(dx12) {
                backends = backends.union(Backends::DX12);
            }

            // Windows, Android, Linux currently always enable Vulkan and OpenGL.
            // See <https://github.com/gfx-rs/wgpu/issues/3514>
            if cfg!(target_os = "windows") || cfg!(unix) {
                backends = backends.union(Backends::VULKAN).union(Backends::GL);
            }

            // Vulkan on Mac/iOS is only available through vulkan-portability.
            if (cfg!(target_os = "ios") || cfg!(target_os = "macos"))
                && cfg!(feature = "vulkan-portability")
            {
                backends = backends.union(Backends::VULKAN);
            }

            // GL on Mac is only available through angle.
            if cfg!(target_os = "macos") && cfg!(feature = "angle") {
                backends = backends.union(Backends::GL);
            }
        } else {
            if cfg!(webgpu) {
                backends = backends.union(Backends::BROWSER_WEBGPU);
            }
            if cfg!(webgl) {
                backends = backends.union(Backends::GL);
            }
        }

        backends
    }

    /// Create an new instance of wgpu.
    ///
    /// # Arguments
    ///
    /// - `instance_desc` - Has fields for which [backends][Backends] wgpu will choose
    ///   during instantiation, and which [DX12 shader compiler][Dx12Compiler] wgpu will use.
    ///
    ///   [`Backends::BROWSER_WEBGPU`] takes a special role:
    ///   If it is set and WebGPU support is detected, this instance will *only* be able to create
    ///   WebGPU adapters. If you instead want to force use of WebGL, either
    ///   disable the `webgpu` compile-time feature or do add the [`Backends::BROWSER_WEBGPU`]
    ///   flag to the the `instance_desc`'s `backends` field.
    ///   If it is set and WebGPU support is *not* detected, the instance will use wgpu-core
    ///   to create adapters. Meaning that if the `webgl` feature is enabled, it is able to create
    ///   a WebGL adapter.
    ///
    /// # Panics
    ///
    /// If no backend feature for the active target platform is enabled,
    /// this method will panic, see [`Instance::enabled_backend_features()`].
    #[allow(unreachable_code)]
    pub fn new(_instance_desc: InstanceDescriptor) -> Self {
        if Self::enabled_backend_features().is_empty() {
            panic!(
                "No wgpu backend feature that is implemented for the target platform was enabled. \
                 See `wgpu::Instance::enabled_backend_features()` for more information."
            );
        }

        #[cfg(webgpu)]
        {
            let is_only_available_backend = !cfg!(wgpu_core);
            let requested_webgpu = _instance_desc.backends.contains(Backends::BROWSER_WEBGPU);
            let support_webgpu =
                crate::backend::get_browser_gpu_property().map_or(false, |gpu| !gpu.is_undefined());

            if is_only_available_backend || (requested_webgpu && support_webgpu) {
                return Self {
                    context: Arc::from(crate::backend::ContextWebGpu::init(_instance_desc)),
                };
            }
        }

        #[cfg(wgpu_core)]
        {
            return Self {
                context: Arc::from(crate::backend::ContextWgpuCore::init(_instance_desc)),
            };
        }

        unreachable!(
            "Earlier check of `enabled_backend_features` should have prevented getting here!"
        );
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
    #[cfg(wgpu_core)]
    pub unsafe fn from_hal<A: wgc::hal_api::HalApi>(hal_instance: A::Instance) -> Self {
        Self {
            context: Arc::new(unsafe {
                crate::backend::ContextWgpuCore::from_hal_instance::<A>(hal_instance)
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
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi>(&self) -> Option<&A::Instance> {
        self.context
            .as_any()
            // If we don't have a wgpu-core instance, we don't have a hal instance either.
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .and_then(|ctx| unsafe { ctx.instance_as_hal::<A>() })
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
    #[cfg(wgpu_core)]
    pub unsafe fn from_core(core_instance: wgc::instance::Instance) -> Self {
        Self {
            context: Arc::new(unsafe {
                crate::backend::ContextWgpuCore::from_core_instance(core_instance)
            }),
        }
    }

    /// Retrieves all available [`Adapter`]s that match the given [`Backends`].
    ///
    /// # Arguments
    ///
    /// - `backends` - Backends from which to enumerate adapters.
    #[cfg(native)]
    pub fn enumerate_adapters(&self, backends: Backends) -> Vec<Adapter> {
        let context = Arc::clone(&self.context);
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| {
                ctx.enumerate_adapters(backends)
                    .into_iter()
                    .map(move |id| crate::Adapter {
                        context: Arc::clone(&context),
                        id: ObjectId::from(id),
                        data: Box::new(()),
                    })
                    .collect()
            })
            .unwrap()
    }

    /// Retrieves an [`Adapter`] which matches the given [`RequestAdapterOptions`].
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that suffice all the "hard" options, `None` is returned.
    ///
    /// A `compatible_surface` is required when targeting WebGL2.
    pub fn request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> impl Future<Output = Option<Adapter>> + WasmNotSend {
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
    #[cfg(wgpu_core)]
    pub unsafe fn create_adapter_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> Adapter {
        let context = Arc::clone(&self.context);
        let id = unsafe {
            context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
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

    /// Creates a new surface targeting a given window/canvas/surface/etc..
    ///
    /// Internally, this creates surfaces for all backends that are enabled for this instance.
    ///
    /// See [`SurfaceTarget`] for what targets are supported.
    /// See [`Instance::create_surface_unsafe`] for surface creation with unsafe target variants.
    ///
    /// Most commonly used are window handles (or provider of windows handles)
    /// which can be passed directly as they're automatically converted to [`SurfaceTarget`].
    pub fn create_surface<'window>(
        &self,
        target: impl Into<SurfaceTarget<'window>>,
    ) -> Result<Surface<'window>, CreateSurfaceError> {
        // Handle origin (i.e. window) to optionally take ownership of to make the surface outlast the window.
        let handle_source;

        let target = target.into();
        let mut surface = match target {
            SurfaceTarget::Window(window) => unsafe {
                let surface = self.create_surface_unsafe(
                    SurfaceTargetUnsafe::from_window(&window).map_err(|e| CreateSurfaceError {
                        inner: CreateSurfaceErrorKind::RawHandle(e),
                    })?,
                );
                handle_source = Some(window);

                surface
            }?,

            #[cfg(any(webgpu, webgl))]
            SurfaceTarget::Canvas(canvas) => {
                handle_source = None;

                let value: &wasm_bindgen::JsValue = &canvas;
                let obj = std::ptr::NonNull::from(value).cast();
                let raw_window_handle = raw_window_handle::WebCanvasWindowHandle::new(obj).into();
                let raw_display_handle = raw_window_handle::WebDisplayHandle::new().into();

                // Note that we need to call this while we still have `value` around.
                // This is safe without storing canvas to `handle_origin` since the surface will create a copy internally.
                unsafe {
                    self.create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle,
                        raw_window_handle,
                    })
                }?
            }

            #[cfg(any(webgpu, webgl))]
            SurfaceTarget::OffscreenCanvas(canvas) => {
                handle_source = None;

                let value: &wasm_bindgen::JsValue = &canvas;
                let obj = std::ptr::NonNull::from(value).cast();
                let raw_window_handle =
                    raw_window_handle::WebOffscreenCanvasWindowHandle::new(obj).into();
                let raw_display_handle = raw_window_handle::WebDisplayHandle::new().into();

                // Note that we need to call this while we still have `value` around.
                // This is safe without storing canvas to `handle_origin` since the surface will create a copy internally.
                unsafe {
                    self.create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                        raw_display_handle,
                        raw_window_handle,
                    })
                }?
            }
        };

        surface._handle_source = handle_source;

        Ok(surface)
    }

    /// Creates a new surface targeting a given window/canvas/surface/etc. using an unsafe target.
    ///
    /// Internally, this creates surfaces for all backends that are enabled for this instance.
    ///
    /// See [`SurfaceTargetUnsafe`] for what targets are supported.
    /// See [`Instance::create_surface`] for surface creation with safe target variants.
    ///
    /// # Safety
    ///
    /// - See respective [`SurfaceTargetUnsafe`] variants for safety requirements.
    pub unsafe fn create_surface_unsafe<'window>(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Surface<'window>, CreateSurfaceError> {
        let (id, data) = unsafe { self.context.instance_create_surface(target) }?;

        Ok(Surface {
            context: Arc::clone(&self.context),
            _handle_source: None,
            id,
            surface_data: data,
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
    ///
    /// Returns `None` if the feature is not supported by the backend
    /// which happens only when WebGPU is pre-selected by the instance creation.
    #[cfg(wgpu_core)]
    pub fn generate_report(&self) -> Option<wgc::global::GlobalReport> {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| ctx.generate_report())
    }
}

impl Adapter {
    /// Requests a connection to a physical device, creating a logical device.
    ///
    /// Returns the [`Device`] together with a [`Queue`] that executes command buffers.
    ///
    /// [Per the WebGPU specification], an [`Adapter`] may only be used once to create a device.
    /// If another device is wanted, call [`Instance::request_adapter()`] again to get a fresh
    /// [`Adapter`].
    /// However, `wgpu` does not currently enforce this restriction.
    ///
    /// # Arguments
    ///
    /// - `desc` - Description of the features and limits requested from the given device.
    /// - `trace_path` - Can be used for API call tracing, if that feature is
    ///   enabled in `wgpu-core`.
    ///
    /// # Panics
    ///
    /// - `request_device()` was already called on this `Adapter`.
    /// - Features specified by `desc` are not supported by this adapter.
    /// - Unsafe features were requested but not enabled when requesting the adapter.
    /// - Limits requested exceed the values provided by the adapter.
    /// - Adapter does not support all features wgpu requires to safely operate.
    ///
    /// [Per the WebGPU specification]: https://www.w3.org/TR/webgpu/#dom-gpuadapter-requestdevice
    pub fn request_device(
        &self,
        desc: &DeviceDescriptor<'_>,
        trace_path: Option<&std::path::Path>,
    ) -> impl Future<Output = Result<(Device, Queue), RequestDeviceError>> + WasmNotSend {
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
    #[cfg(wgpu_core)]
    pub unsafe fn create_device_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_device: hal::OpenDevice<A>,
        desc: &DeviceDescriptor<'_>,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), RequestDeviceError> {
        let context = Arc::clone(&self.context);
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                // Part of the safety requirements is that the device was generated from the same adapter.
                // Therefore, unwrap is fine here since only WgpuCoreContext based adapters have the ability to create hal devices.
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
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        hal_adapter_callback: F,
    ) -> R {
        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe { ctx.adapter_as_hal::<A, F, R>(self.id.into(), hal_adapter_callback) }
        } else {
            hal_adapter_callback(None)
        }
    }

    /// Returns whether this adapter may present to the passed surface.
    pub fn is_surface_supported(&self, surface: &Surface<'_>) -> bool {
        DynContext::adapter_is_surface_supported(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &surface.id,
            surface.surface_data.as_ref(),
        )
    }

    /// The features which can be used to create devices on this adapter.
    pub fn features(&self) -> Features {
        DynContext::adapter_features(&*self.context, &self.id, self.data.as_ref())
    }

    /// The best limits which can be used to create devices on this adapter.
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
    /// Check for resource cleanups and mapping callbacks. Will block if [`Maintain::Wait`] is passed.
    ///
    /// Return `true` if the queue is empty, or `false` if there are more queue
    /// submissions still in flight. (Note that, unless access to the [`Queue`] is
    /// coordinated somehow, this information could be out of date by the time
    /// the caller receives it. `Queue`s can be shared between threads, so
    /// other threads could submit new work at any time.)
    ///
    /// When running on WebGPU, this is a no-op. `Device`s are automatically polled.
    pub fn poll(&self, maintain: Maintain) -> MaintainResult {
        DynContext::device_poll(&*self.context, &self.id, self.data.as_ref(), maintain)
    }

    /// The features which can be used on this device.
    ///
    /// No additional features can be used, even if the underlying adapter can support them.
    pub fn features(&self) -> Features {
        DynContext::device_features(&*self.context, &self.id, self.data.as_ref())
    }

    /// The limits which can be used on this device.
    ///
    /// No better limits can be used, even if the underlying adapter can support them.
    pub fn limits(&self) -> Limits {
        DynContext::device_limits(&*self.context, &self.id, self.data.as_ref())
    }

    /// Creates a shader module from either SPIR-V or WGSL source code.
    ///
    /// <div class="warning">
    // NOTE: Keep this in sync with `naga::front::wgsl::parse_str`!
    // NOTE: Keep this in sync with `wgpu_core::Global::device_create_shader_module`!
    ///
    /// This function may consume a lot of stack space. Compiler-enforced limits for parsing
    /// recursion exist; if shader compilation runs into them, it will return an error gracefully.
    /// However, on some build profiles and platforms, the default stack size for a thread may be
    /// exceeded before this limit is reached during parsing. Callers should ensure that there is
    /// enough stack space for this, particularly if calls to this method are exposed to user
    /// input.
    ///
    /// </div>
    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor<'_>) -> ShaderModule {
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
        desc: ShaderModuleDescriptor<'_>,
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
        desc: &ShaderModuleDescriptorSpirV<'_>,
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
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor<'_>) -> CommandEncoder {
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
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> RenderBundleEncoder<'_> {
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
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
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
    pub fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> BindGroupLayout {
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
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor<'_>) -> PipelineLayout {
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
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> RenderPipeline {
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
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline {
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
    pub fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Buffer {
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
    pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture {
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
    #[cfg(wgpu_core)]
    pub unsafe fn create_texture_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_texture: A::Texture,
        desc: &TextureDescriptor<'_>,
    ) -> Texture {
        let texture = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                // Part of the safety requirements is that the texture was generated from the same hal device.
                // Therefore, unwrap is fine here since only WgpuCoreContext has the ability to create hal textures.
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

    /// Creates a [`Buffer`] from a wgpu-hal Buffer.
    ///
    /// # Safety
    ///
    /// - `hal_buffer` must be created from this device internal handle
    /// - `hal_buffer` must be created respecting `desc`
    /// - `hal_buffer` must be initialized
    #[cfg(wgpu_core)]
    pub unsafe fn create_buffer_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_buffer: A::Buffer,
        desc: &BufferDescriptor<'_>,
    ) -> Buffer {
        let mut map_context = MapContext::new(desc.size);
        if desc.mapped_at_creation {
            map_context.initial_range = 0..desc.size;
        }

        let (id, buffer) = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                // Part of the safety requirements is that the buffer was generated from the same hal device.
                // Therefore, unwrap is fine here since only WgpuCoreContext has the ability to create hal buffers.
                .unwrap()
                .create_buffer_from_hal::<A>(
                    hal_buffer,
                    self.data.as_ref().downcast_ref().unwrap(),
                    desc,
                )
        };

        Buffer {
            context: Arc::clone(&self.context),
            id: ObjectId::from(id),
            data: Box::new(buffer),
            map_context: Mutex::new(map_context),
            size: desc.size,
            usage: desc.usage,
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    pub fn create_sampler(&self, desc: &SamplerDescriptor<'_>) -> Sampler {
        let (id, data) =
            DynContext::device_create_sampler(&*self.context, &self.id, self.data.as_ref(), desc);
        Sampler {
            context: Arc::clone(&self.context),
            id,
            data,
        }
    }

    /// Creates a new [`QuerySet`].
    pub fn create_query_set(&self, desc: &QuerySetDescriptor<'_>) -> QuerySet {
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
    pub fn pop_error_scope(&self) -> impl Future<Output = Option<Error>> + WasmNotSend {
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

    /// Query internal counters from the native backend for debugging purposes.
    ///
    /// Some backends may not set all counters, or may not set any counter at all.
    /// The `counters` cargo feature must be enabled for any counter to be set.
    ///
    /// If a counter is not set, its contains its default value (zero).
    pub fn get_internal_counters(&self) -> wgt::InternalCounters {
        DynContext::device_get_internal_counters(&*self.context, &self.id, self.data.as_ref())
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
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        hal_device_callback: F,
    ) -> Option<R> {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| unsafe {
                ctx.device_as_hal::<A, F, R>(
                    self.data.as_ref().downcast_ref().unwrap(),
                    hal_device_callback,
                )
            })
    }

    /// Destroy this device.
    pub fn destroy(&self) {
        DynContext::device_destroy(&*self.context, &self.id, self.data.as_ref())
    }

    /// Set a DeviceLostCallback on this device.
    pub fn set_device_lost_callback(
        &self,
        callback: impl Fn(DeviceLostReason, String) + Send + 'static,
    ) {
        DynContext::device_set_device_lost_callback(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            Box::new(callback),
        )
    }

    /// Test-only function to make this device invalid.
    #[doc(hidden)]
    pub fn make_invalid(&self) {
        DynContext::device_make_invalid(&*self.context, &self.id, self.data.as_ref())
    }

    /// Create a [`PipelineCache`] with initial data
    ///
    /// This can be passed to [`Device::create_compute_pipeline`]
    /// and [`Device::create_render_pipeline`] to either accelerate these
    /// or add the cache results from those.
    ///
    /// # Safety
    ///
    /// If the `data` field of `desc` is set, it must have previously been returned from a call
    /// to [`PipelineCache::get_data`][^saving]. This `data` will only be used if it came
    /// from an adapter with the same [`util::pipeline_cache_key`].
    /// This *is* compatible across wgpu versions, as any data format change will
    /// be accounted for.
    ///
    /// It is *not* supported to bring caches from previous direct uses of backend APIs
    /// into this method.
    ///
    /// # Errors
    ///
    /// Returns an error value if:
    ///  * the [`PIPELINE_CACHE`](wgt::Features::PIPELINE_CACHE) feature is not enabled
    ///  * this device is invalid; or
    ///  * the device is out of memory
    ///
    /// This method also returns an error value if:
    ///  * The `fallback` field on `desc` is false; and
    ///  * the `data` provided would not be used[^data_not_used]
    ///
    /// If an error value is used in subsequent calls, default caching will be used.
    ///
    /// [^saving]: We do recognise that saving this data to disk means this condition
    /// is impossible to fully prove. Consider the risks for your own application in this case.
    ///
    /// [^data_not_used]: This data may be not used if: the data was produced by a prior
    /// version of wgpu; or was created for an incompatible adapter, or there was a GPU driver
    /// update. In some cases, the data might not be used and a real value is returned,
    /// this is left to the discretion of GPU drivers.
    pub unsafe fn create_pipeline_cache(
        &self,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> PipelineCache {
        let (id, data) = unsafe {
            DynContext::device_create_pipeline_cache(
                &*self.context,
                &self.id,
                self.data.as_ref(),
                desc,
            )
        };
        PipelineCache {
            context: Arc::clone(&self.context),
            id,
            data,
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

/// Requesting a device from an [`Adapter`] failed.
#[derive(Clone, Debug)]
pub struct RequestDeviceError {
    inner: RequestDeviceErrorKind,
}
#[derive(Clone, Debug)]
enum RequestDeviceErrorKind {
    /// Error from [`wgpu_core`].
    // must match dependency cfg
    #[cfg(wgpu_core)]
    Core(wgc::instance::RequestDeviceError),

    /// Error from web API that was called by `wgpu` to request a device.
    ///
    /// (This is currently never used by the webgl backend, but it could be.)
    #[cfg(webgpu)]
    WebGpu(wasm_bindgen::JsValue),
}

#[cfg(send_sync)]
unsafe impl Send for RequestDeviceErrorKind {}
#[cfg(send_sync)]
unsafe impl Sync for RequestDeviceErrorKind {}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(RequestDeviceError: Send, Sync);

impl fmt::Display for RequestDeviceError {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            #[cfg(wgpu_core)]
            RequestDeviceErrorKind::Core(error) => error.fmt(_f),
            #[cfg(webgpu)]
            RequestDeviceErrorKind::WebGpu(error_js_value) => {
                // wasm-bindgen provides a reasonable error stringification via `Debug` impl
                write!(_f, "{error_js_value:?}")
            }
            #[cfg(not(any(webgpu, wgpu_core)))]
            _ => unimplemented!("unknown `RequestDeviceErrorKind`"),
        }
    }
}

impl error::Error for RequestDeviceError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.inner {
            #[cfg(wgpu_core)]
            RequestDeviceErrorKind::Core(error) => error.source(),
            #[cfg(webgpu)]
            RequestDeviceErrorKind::WebGpu(_) => None,
            #[cfg(not(any(webgpu, wgpu_core)))]
            _ => unimplemented!("unknown `RequestDeviceErrorKind`"),
        }
    }
}

#[cfg(wgpu_core)]
impl From<wgc::instance::RequestDeviceError> for RequestDeviceError {
    fn from(error: wgc::instance::RequestDeviceError) -> Self {
        Self {
            inner: RequestDeviceErrorKind::Core(error),
        }
    }
}

/// [`Instance::create_surface()`] or a related function failed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CreateSurfaceError {
    inner: CreateSurfaceErrorKind,
}
#[derive(Clone, Debug)]
enum CreateSurfaceErrorKind {
    /// Error from [`wgpu_hal`].
    #[cfg(wgpu_core)]
    Hal(wgc::instance::CreateSurfaceError),

    /// Error from WebGPU surface creation.
    #[allow(dead_code)] // may be unused depending on target and features
    Web(String),

    /// Error when trying to get a [`DisplayHandle`] or a [`WindowHandle`] from
    /// `raw_window_handle`.
    RawHandle(raw_window_handle::HandleError),
}
static_assertions::assert_impl_all!(CreateSurfaceError: Send, Sync);

impl fmt::Display for CreateSurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            #[cfg(wgpu_core)]
            CreateSurfaceErrorKind::Hal(e) => e.fmt(f),
            CreateSurfaceErrorKind::Web(e) => e.fmt(f),
            CreateSurfaceErrorKind::RawHandle(e) => e.fmt(f),
        }
    }
}

impl error::Error for CreateSurfaceError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.inner {
            #[cfg(wgpu_core)]
            CreateSurfaceErrorKind::Hal(e) => e.source(),
            CreateSurfaceErrorKind::Web(_) => None,
            CreateSurfaceErrorKind::RawHandle(e) => e.source(),
        }
    }
}

#[cfg(wgpu_core)]
impl From<wgc::instance::CreateSurfaceError> for CreateSurfaceError {
    fn from(e: wgc::instance::CreateSurfaceError) -> Self {
        Self {
            inner: CreateSurfaceErrorKind::Hal(e),
        }
    }
}

/// Error occurred when trying to async map a buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;
static_assertions::assert_impl_all!(BufferAsyncError: Send, Sync);

impl fmt::Display for BufferAsyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// A read-only view of a mapped buffer's bytes.
///
/// To get a `BufferView`, first [map] the buffer, and then
/// call `buffer.slice(range).get_mapped_range()`.
///
/// `BufferView` dereferences to `&[u8]`, so you can use all the usual Rust
/// slice methods to access the buffer's contents. It also implements
/// `AsRef<[u8]>`, if that's more convenient.
///
/// Before the buffer can be unmapped, all `BufferView`s observing it
/// must be dropped. Otherwise, the call to [`Buffer::unmap`] will panic.
///
/// For example code, see the documentation on [mapping buffers][map].
///
/// [map]: Buffer#mapping-buffers
/// [`map_async`]: BufferSlice::map_async
#[derive(Debug)]
pub struct BufferView<'a> {
    slice: BufferSlice<'a>,
    data: Box<dyn crate::context::BufferMappedRange>,
}

/// A write-only view of a mapped buffer's bytes.
///
/// To get a `BufferViewMut`, first [map] the buffer, and then
/// call `buffer.slice(range).get_mapped_range_mut()`.
///
/// `BufferViewMut` dereferences to `&mut [u8]`, so you can use all the usual
/// Rust slice methods to access the buffer's contents. It also implements
/// `AsMut<[u8]>`, if that's more convenient.
///
/// It is possible to read the buffer using this view, but doing so is not
/// recommended, as it is likely to be slow.
///
/// Before the buffer can be unmapped, all `BufferViewMut`s observing it
/// must be dropped. Otherwise, the call to [`Buffer::unmap`] will panic.
///
/// For example code, see the documentation on [mapping buffers][map].
///
/// [map]: Buffer#mapping-buffers
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
    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(self.as_entire_buffer_binding())
    }

    /// Return the binding view of the entire buffer.
    pub fn as_entire_buffer_binding(&self) -> BufferBinding<'_> {
        BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        }
    }

    /// Returns the inner hal Buffer using a callback. The hal buffer will be `None` if the
    /// backend type argument does not match with this wgpu Buffer
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Buffer must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        hal_buffer_callback: F,
    ) -> R {
        let id = self.id;

        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe { ctx.buffer_as_hal::<A, F, R>(id.into(), hal_buffer_callback) }
        } else {
            hal_buffer_callback(None)
        }
    }

    /// Return a slice of a [`Buffer`]'s bytes.
    ///
    /// Return a [`BufferSlice`] referring to the portion of `self`'s contents
    /// indicated by `bounds`. Regardless of what sort of data `self` stores,
    /// `bounds` start and end are given in bytes.
    ///
    /// A [`BufferSlice`] can be used to supply vertex and index data, or to map
    /// buffer contents for access from the CPU. See the [`BufferSlice`]
    /// documentation for details.
    ///
    /// The `range` argument can be half or fully unbounded: for example,
    /// `buffer.slice(..)` refers to the entire buffer, and `buffer.slice(n..)`
    /// refers to the portion starting at the `n`th byte and extending to the
    /// end of the buffer.
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice<'_> {
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
        callback: impl FnOnce(Result<(), BufferAsyncError>) + WasmNotSend + 'static,
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

    /// Gain read-only access to the bytes of a [mapped] [`Buffer`].
    ///
    /// Return a [`BufferView`] referring to the buffer range represented by
    /// `self`. See the documentation for [`BufferView`] for details.
    ///
    /// # Panics
    ///
    /// - This panics if the buffer to which `self` refers is not currently
    ///   [mapped].
    ///
    /// - If you try to create overlapping views of a buffer, mutable or
    ///   otherwise, `get_mapped_range` will panic.
    ///
    /// [mapped]: Buffer#mapping-buffers
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

    /// Synchronously and immediately map a buffer for reading. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will fail.
    ///
    /// This is useful when targeting WebGPU and you want to pass mapped data directly to js.
    /// Unlike `get_mapped_range` which unconditionally copies mapped data into the wasm heap,
    /// this function directly hands you the ArrayBuffer that we mapped the data into in js.
    ///
    /// This is only available on WebGPU, on any other backends this will return `None`.
    #[cfg(webgpu)]
    pub fn get_mapped_range_as_array_buffer(&self) -> Option<js_sys::ArrayBuffer> {
        self.buffer
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWebGpu>()
            .map(|ctx| {
                let buffer_data = crate::context::downcast_ref(self.buffer.data.as_ref());
                let end = self.buffer.map_context.lock().add(self.offset, self.size);
                ctx.buffer_get_mapped_range_as_array_buffer(buffer_data, self.offset..end)
            })
    }

    /// Gain write access to the bytes of a [mapped] [`Buffer`].
    ///
    /// Return a [`BufferViewMut`] referring to the buffer range represented by
    /// `self`. See the documentation for [`BufferViewMut`] for more details.
    ///
    /// # Panics
    ///
    /// - This panics if the buffer to which `self` refers is not currently
    ///   [mapped].
    ///
    /// - If you try to create overlapping views of a buffer, mutable or
    ///   otherwise, `get_mapped_range_mut` will panic.
    ///
    /// [mapped]: Buffer#mapping-buffers
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
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Texture>) -> R, R>(
        &self,
        hal_texture_callback: F,
    ) -> R {
        let texture = self.data.as_ref().downcast_ref().unwrap();

        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe { ctx.texture_as_hal::<A, F, R>(texture, hal_texture_callback) }
        } else {
            hal_texture_callback(None)
        }
    }

    /// Creates a view of this texture.
    pub fn create_view(&self, desc: &TextureViewDescriptor<'_>) -> TextureView {
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
    pub fn as_image_copy(&self) -> ImageCopyTexture<'_> {
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
    ///
    /// As long as the returned  [`RenderPass`] has not ended,
    /// any mutating operation on this command encoder causes an error and invalidates it.
    /// Note that the `'encoder` lifetime relationship protects against this,
    /// but it is possible to opt out of it by calling [`RenderPass::forget_lifetime`].
    /// This can be useful for runtime handling of the encoder->pass
    /// dependency e.g. when pass and encoder are stored in the same data structure.
    pub fn begin_render_pass<'encoder>(
        &'encoder mut self,
        desc: &RenderPassDescriptor<'_>,
    ) -> RenderPass<'encoder> {
        let id = self.id.as_ref().unwrap();
        let (id, data) = DynContext::command_encoder_begin_render_pass(
            &*self.context,
            id,
            self.data.as_ref(),
            desc,
        );
        RenderPass {
            inner: RenderPassInner {
                id,
                data,
                context: self.context.clone(),
            },
            encoder_guard: PhantomData,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    ///
    /// As long as the returned  [`ComputePass`] has not ended,
    /// any mutating operation on this command encoder causes an error and invalidates it.
    /// Note that the `'encoder` lifetime relationship protects against this,
    /// but it is possible to opt out of it by calling [`ComputePass::forget_lifetime`].
    /// This can be useful for runtime handling of the encoder->pass
    /// dependency e.g. when pass and encoder are stored in the same data structure.
    pub fn begin_compute_pass<'encoder>(
        &'encoder mut self,
        desc: &ComputePassDescriptor<'_>,
    ) -> ComputePass<'encoder> {
        let id = self.id.as_ref().unwrap();
        let (id, data) = DynContext::command_encoder_begin_compute_pass(
            &*self.context,
            id,
            self.data.as_ref(),
            desc,
        );
        ComputePass {
            inner: ComputePassInner {
                id,
                data,
                context: self.context.clone(),
            },
            encoder_guard: PhantomData,
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
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
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
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
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
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
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
    /// - Range is out of bounds
    pub fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferAddress>,
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

    /// Resolves a query set, writing the results into the supplied destination buffer.
    ///
    /// Occlusion and timestamp queries are 8 bytes each (see [`crate::QUERY_SIZE`]). For pipeline statistics queries,
    /// see [`PipelineStatisticsTypes`] for more information.
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

    /// Returns the inner hal CommandEncoder using a callback. The hal command encoder will be `None` if the
    /// backend type argument does not match with this wgpu CommandEncoder
    ///
    /// This method will start the wgpu_core level command recording.
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal CommandEncoder must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal_mut<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &mut self,
        hal_command_encoder_callback: F,
    ) -> Option<R> {
        use core::id::CommandEncoderId;

        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| unsafe {
                ctx.command_encoder_as_hal_mut::<A, F, R>(
                    CommandEncoderId::from(self.id.unwrap()),
                    hal_command_encoder_callback,
                )
            })
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Issue a timestamp command at this point in the queue.
    /// The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    ///
    /// Attention: Since commands within a command recorder may be reordered,
    /// there is no strict guarantee that timestamps are taken after all commands
    /// recorded so far and all before all commands recorded after.
    /// This may depend both on the backend and the driver.
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

impl<'encoder> RenderPass<'encoder> {
    /// Drops the lifetime relationship to the parent command encoder, making usage of
    /// the encoder while this pass is recorded a run-time error instead.
    ///
    /// Attention: As long as the render pass has not been ended, any mutating operation on the parent
    /// command encoder will cause a run-time error and invalidate it!
    /// By default, the lifetime constraint prevents this, but it can be useful
    /// to handle this at run time, such as when storing the pass and encoder in the same
    /// data structure.
    ///
    /// This operation has no effect on pass recording.
    /// It's a safe operation, since [`CommandEncoder`] is in a locked state as long as the pass is active
    /// regardless of the lifetime constraint or its absence.
    pub fn forget_lifetime(self) -> RenderPass<'static> {
        RenderPass {
            inner: self.inner,
            encoder_guard: PhantomData,
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when any `draw_*()` method is called must match the layout of
    /// this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    ///
    /// Subsequent draw calls’ shader executions will be able to access data in these bind groups.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &BindGroup,
        offsets: &[DynamicOffset],
    ) {
        DynContext::render_pass_set_bind_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            index,
            &bind_group.id,
            bind_group.data.as_ref(),
            offsets,
        )
    }

    /// Sets the active render pipeline.
    ///
    /// Subsequent draw calls will exhibit the behavior defined by `pipeline`.
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        DynContext::render_pass_set_pipeline(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &pipeline.id,
            pipeline.data.as_ref(),
        )
    }

    /// Sets the blend color as used by some of the blending modes.
    ///
    /// Subsequent blending tests will test against this value.
    /// If this method has not been called, the blend constant defaults to [`Color::TRANSPARENT`]
    /// (all components zero).
    pub fn set_blend_constant(&mut self, color: Color) {
        DynContext::render_pass_set_blend_constant(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            color,
        )
    }

    /// Sets the active index buffer.
    ///
    /// Subsequent calls to [`draw_indexed`](RenderPass::draw_indexed) on this [`RenderPass`] will
    /// use `buffer` as the source index buffer.
    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice<'_>, index_format: IndexFormat) {
        DynContext::render_pass_set_index_buffer(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
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
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer_slice: BufferSlice<'_>) {
        DynContext::render_pass_set_vertex_buffer(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            slot,
            &buffer_slice.buffer.id,
            buffer_slice.buffer.data.as_ref(),
            buffer_slice.offset,
            buffer_slice.size,
        )
    }

    /// Sets the scissor rectangle used during the rasterization stage.
    /// After transformation into [viewport coordinates](https://www.w3.org/TR/webgpu/#viewport-coordinates).
    ///
    /// Subsequent draw calls will discard any fragments which fall outside the scissor rectangle.
    /// If this method has not been called, the scissor rectangle defaults to the entire bounds of
    /// the render targets.
    ///
    /// The function of the scissor rectangle resembles [`set_viewport()`](Self::set_viewport),
    /// but it does not affect the coordinate system, only which fragments are discarded.
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        DynContext::render_pass_set_scissor_rect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            x,
            y,
            width,
            height,
        );
    }

    /// Sets the viewport used during the rasterization stage to linearly map
    /// from [normalized device coordinates](https://www.w3.org/TR/webgpu/#ndc) to [viewport coordinates](https://www.w3.org/TR/webgpu/#viewport-coordinates).
    ///
    /// Subsequent draw calls will only draw within this region.
    /// If this method has not been called, the viewport defaults to the entire bounds of the render
    /// targets.
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        DynContext::render_pass_set_viewport(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
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
    /// If this method has not been called, the stencil reference value defaults to `0`.
    pub fn set_stencil_reference(&mut self, reference: u32) {
        DynContext::render_pass_set_stencil_reference(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            reference,
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::render_pass_insert_debug_marker(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::render_pass_push_debug_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::render_pass_pop_debug_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
        );
    }

    /// Draws primitives from the active vertex buffer(s).
    ///
    /// The active vertex buffer(s) can be set with [`RenderPass::set_vertex_buffer`].
    /// Does not use an Index Buffer. If you need this see [`RenderPass::draw_indexed`]
    ///
    /// Panics if vertices Range is outside of the range of the vertices range of any set vertex buffer.
    ///
    /// vertices: The range of vertices to draw.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for vertex_id in vertex_range {
    ///         let vertex = vertex[vertex_id];
    ///         vertex_shader(vertex, vertex_id, instance_id);
    ///     }
    /// }
    /// ```
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        DynContext::render_pass_draw(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            vertices,
            instances,
        )
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers.
    ///
    /// The active index buffer can be set with [`RenderPass::set_index_buffer`]
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// Panics if indices Range is outside of the range of the indices range of any set index buffer.
    ///
    /// indices: The range of indices to draw.
    /// base_vertex: value added to each index value before indexing into the vertex buffers.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for index_index in index_range {
    ///         let vertex_id = index_buffer[index_index];
    ///         let adjusted_vertex_id = vertex_id + base_vertex;
    ///         let vertex = vertex[adjusted_vertex_id];
    ///         vertex_shader(vertex, adjusted_vertex_id, instance_id);
    ///     }
    /// }
    /// ```
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        DynContext::render_pass_draw_indexed(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            indices,
            base_vertex,
            instances,
        );
    }

    /// Draws primitives from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    ///
    /// This is like calling [`RenderPass::draw`] but the contents of the call are specified in the `indirect_buffer`.
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
    ///
    /// Indirect drawing has some caveats depending on the features available. We are not currently able to validate
    /// these and issue an error.
    /// - If [`Features::INDIRECT_FIRST_INSTANCE`] is not present on the adapter,
    ///   [`DrawIndirect::first_instance`](crate::util::DrawIndirectArgs::first_instance) will be ignored.
    /// - If [`DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW`] is not present on the adapter,
    ///   any use of `@builtin(vertex_index)` or `@builtin(instance_index)` in the vertex shader will have different values.
    ///
    /// See details on the individual flags for more information.
    pub fn draw_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress) {
        DynContext::render_pass_draw_indirect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffers,
    /// based on the contents of the `indirect_buffer`.
    ///
    /// This is like calling [`RenderPass::draw_indexed`] but the contents of the call are specified in the `indirect_buffer`.
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
    ///
    /// Indirect drawing has some caveats depending on the features available. We are not currently able to validate
    /// these and issue an error.
    /// - If [`Features::INDIRECT_FIRST_INSTANCE`] is not present on the adapter,
    ///   [`DrawIndexedIndirect::first_instance`](crate::util::DrawIndexedIndirectArgs::first_instance) will be ignored.
    /// - If [`DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW`] is not present on the adapter,
    ///   any use of `@builtin(vertex_index)` or `@builtin(instance_index)` in the vertex shader will have different values.
    ///
    /// See details on the individual flags for more information.
    pub fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::render_pass_draw_indexed_indirect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }

    /// Execute a [render bundle][RenderBundle], which is a set of pre-recorded commands
    /// that can be run together.
    ///
    /// Commands in the bundle do not inherit this render pass's current render state, and after the
    /// bundle has executed, the state is **cleared** (reset to defaults, not the previous state).
    pub fn execute_bundles<'a, I: IntoIterator<Item = &'a RenderBundle>>(
        &mut self,
        render_bundles: I,
    ) {
        let mut render_bundles = render_bundles
            .into_iter()
            .map(|rb| (&rb.id, rb.data.as_ref()));

        DynContext::render_pass_execute_bundles(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &mut render_bundles,
        )
    }
}

/// [`Features::MULTI_DRAW_INDIRECT`] must be enabled on the device in order to call these functions.
impl<'encoder> RenderPass<'encoder> {
    /// Dispatches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// `count` draw calls are issued.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
    /// These draw structures are expected to be tightly packed.
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        DynContext::render_pass_multi_draw_indirect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
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
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
    /// These draw structures are expected to be tightly packed.
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    ) {
        DynContext::render_pass_multi_draw_indexed_indirect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
            count,
        );
    }
}

/// [`Features::MULTI_DRAW_INDIRECT_COUNT`] must be enabled on the device in order to call these functions.
impl<'encoder> RenderPass<'encoder> {
    /// Dispatches multiple draw calls from the active vertex buffer(s) based on the contents of the `indirect_buffer`.
    /// The count buffer is read to determine how many draws to issue.
    ///
    /// The indirect buffer must be long enough to account for `max_count` draws, however only `count`
    /// draws will be read. If `count` is greater than `max_count`, `max_count` will be used.
    ///
    /// The active vertex buffers can be set with [`RenderPass::set_vertex_buffer`].
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
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
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    ) {
        DynContext::render_pass_multi_draw_indirect_count(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
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
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
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
    ///
    /// This drawing command uses the current render state, as set by preceding `set_*()` methods.
    /// It is not affected by changes to the state that are performed after it is called.
    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    ) {
        DynContext::render_pass_multi_draw_indexed_indirect_count(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
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
impl<'encoder> RenderPass<'encoder> {
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
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            stages,
            offset,
            data,
        );
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`] must be enabled on the device in order to call these functions.
impl<'encoder> RenderPass<'encoder> {
    /// Issue a timestamp command at this point in the queue. The
    /// timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::render_pass_write_timestamp(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        )
    }
}

impl<'encoder> RenderPass<'encoder> {
    /// Start a occlusion query on this render pass. It can be ended with
    /// `end_occlusion_query`. Occlusion queries may not be nested.
    pub fn begin_occlusion_query(&mut self, query_index: u32) {
        DynContext::render_pass_begin_occlusion_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            query_index,
        );
    }

    /// End the occlusion query on this render pass. It can be started with
    /// `begin_occlusion_query`. Occlusion queries may not be nested.
    pub fn end_occlusion_query(&mut self) {
        DynContext::render_pass_end_occlusion_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
        );
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'encoder> RenderPass<'encoder> {
    /// Start a pipeline statistics query on this render pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::render_pass_begin_pipeline_statistics_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        );
    }

    /// End the pipeline statistics query on this render pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        DynContext::render_pass_end_pipeline_statistics_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
        );
    }
}

impl Drop for RenderPassInner {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .render_pass_end(&mut self.id, self.data.as_mut());
        }
    }
}

impl<'encoder> ComputePass<'encoder> {
    /// Drops the lifetime relationship to the parent command encoder, making usage of
    /// the encoder while this pass is recorded a run-time error instead.
    ///
    /// Attention: As long as the compute pass has not been ended, any mutating operation on the parent
    /// command encoder will cause a run-time error and invalidate it!
    /// By default, the lifetime constraint prevents this, but it can be useful
    /// to handle this at run time, such as when storing the pass and encoder in the same
    /// data structure.
    ///
    /// This operation has no effect on pass recording.
    /// It's a safe operation, since [`CommandEncoder`] is in a locked state as long as the pass is active
    /// regardless of the lifetime constraint or its absence.
    pub fn forget_lifetime(self) -> ComputePass<'static> {
        ComputePass {
            inner: self.inner,
            encoder_guard: PhantomData,
        }
    }

    /// Sets the active bind group for a given bind group index. The bind group layout
    /// in the active pipeline when the `dispatch()` function is called must match the layout of this bind group.
    ///
    /// If the bind group have dynamic offsets, provide them in the binding order.
    /// These offsets have to be aligned to [`Limits::min_uniform_buffer_offset_alignment`]
    /// or [`Limits::min_storage_buffer_offset_alignment`] appropriately.
    pub fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: &BindGroup,
        offsets: &[DynamicOffset],
    ) {
        DynContext::compute_pass_set_bind_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            index,
            &bind_group.id,
            bind_group.data.as_ref(),
            offsets,
        );
    }

    /// Sets the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        DynContext::compute_pass_set_pipeline(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &pipeline.id,
            pipeline.data.as_ref(),
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::compute_pass_insert_debug_marker(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::compute_pass_push_debug_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            label,
        );
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::compute_pass_pop_debug_group(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
        );
    }

    /// Dispatches compute work operations.
    ///
    /// `x`, `y` and `z` denote the number of work groups to dispatch in each dimension.
    pub fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        DynContext::compute_pass_dispatch_workgroups(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            x,
            y,
            z,
        );
    }

    /// Dispatches compute work operations, based on the contents of the `indirect_buffer`.
    ///
    /// The structure expected in `indirect_buffer` must conform to [`DispatchIndirectArgs`](crate::util::DispatchIndirectArgs).
    pub fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
    ) {
        DynContext::compute_pass_dispatch_workgroups_indirect(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &indirect_buffer.id,
            indirect_buffer.data.as_ref(),
            indirect_offset,
        );
    }
}

/// [`Features::PUSH_CONSTANTS`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
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
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            offset,
            data,
        );
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_PASSES`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
    /// Issue a timestamp command at this point in the queue. The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_write_timestamp(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        )
    }
}

/// [`Features::PIPELINE_STATISTICS_QUERY`] must be enabled on the device in order to call these functions.
impl<'encoder> ComputePass<'encoder> {
    /// Start a pipeline statistics query on this compute pass. It can be ended with
    /// `end_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn begin_pipeline_statistics_query(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::compute_pass_begin_pipeline_statistics_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
            &query_set.id,
            query_set.data.as_ref(),
            query_index,
        );
    }

    /// End the pipeline statistics query on this compute pass. It can be started with
    /// `begin_pipeline_statistics_query`. Pipeline statistics queries may not be nested.
    pub fn end_pipeline_statistics_query(&mut self) {
        DynContext::compute_pass_end_pipeline_statistics_query(
            &*self.inner.context,
            &mut self.inner.id,
            self.inner.data.as_mut(),
        );
    }
}

impl Drop for ComputePassInner {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context
                .compute_pass_end(&mut self.id, self.data.as_mut());
        }
    }
}

impl<'a> RenderBundleEncoder<'a> {
    /// Finishes recording and returns a [`RenderBundle`] that can be executed in other render passes.
    pub fn finish(self, desc: &RenderBundleDescriptor<'_>) -> RenderBundle {
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
    /// Does not use an Index Buffer. If you need this see [`RenderBundleEncoder::draw_indexed`]
    ///
    /// Panics if vertices Range is outside of the range of the vertices range of any set vertex buffer.
    ///
    /// vertices: The range of vertices to draw.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for vertex_id in vertex_range {
    ///         let vertex = vertex[vertex_id];
    ///         vertex_shader(vertex, vertex_id, instance_id);
    ///     }
    /// }
    /// ```
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        DynContext::render_bundle_encoder_draw(
            &*self.parent.context,
            &mut self.id,
            self.data.as_mut(),
            vertices,
            instances,
        )
    }

    /// Draws indexed primitives using the active index buffer and the active vertex buffer(s).
    ///
    /// The active index buffer can be set with [`RenderBundleEncoder::set_index_buffer`].
    /// The active vertex buffer(s) can be set with [`RenderBundleEncoder::set_vertex_buffer`].
    ///
    /// Panics if indices Range is outside of the range of the indices range of any set index buffer.
    ///
    /// indices: The range of indices to draw.
    /// base_vertex: value added to each index value before indexing into the vertex buffers.
    /// instances: Range of Instances to draw. Use 0..1 if instance buffers are not used.
    /// E.g.of how its used internally
    /// ```rust ignore
    /// for instance_id in instance_range {
    ///     for index_index in index_range {
    ///         let vertex_id = index_buffer[index_index];
    ///         let adjusted_vertex_id = vertex_id + base_vertex;
    ///         let vertex = vertex[adjusted_vertex_id];
    ///         vertex_shader(vertex, adjusted_vertex_id, instance_id);
    ///     }
    /// }
    /// ```
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
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndirectArgs`](crate::util::DrawIndirectArgs).
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
    /// The structure expected in `indirect_buffer` must conform to [`DrawIndexedIndirectArgs`](crate::util::DrawIndexedIndirectArgs).
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

/// A write-only view into a staging buffer.
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
#[cfg(send_sync)]
static_assertions::assert_impl_all!(QueueWriteBufferView<'_>: Send, Sync);

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
    /// This method fails if `data` overruns the size of `buffer` starting at `offset`.
    ///
    /// This does *not* submit the transfer to the GPU immediately. Calls to
    /// `write_buffer` begin execution only on the next call to
    /// [`Queue::submit`]. To get a set of scheduled transfers started
    /// immediately, it's fine to call `submit` with no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
    ///
    /// However, `data` will be immediately copied into staging memory, so the
    /// caller may discard it any time after this call completes.
    ///
    /// If possible, consider using [`Queue::write_buffer_with`] instead. That
    /// method avoids an intermediate copy and is often able to transfer data
    /// more efficiently than this one.
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

    /// Write to a buffer via a directly mapped staging buffer.
    ///
    /// Return a [`QueueWriteBufferView`] which, when dropped, schedules a copy
    /// of its contents into `buffer` at `offset`. The returned view
    /// dereferences to a `size`-byte long `&mut [u8]`, in which you should
    /// store the data you would like written to `buffer`.
    ///
    /// This method may perform transfers faster than [`Queue::write_buffer`],
    /// because the returned [`QueueWriteBufferView`] is actually the staging
    /// buffer for the write, mapped into the caller's address space. Writing
    /// your data directly into this staging buffer avoids the temporary
    /// CPU-side buffer needed by `write_buffer`.
    ///
    /// Reading from the returned view is slow, and will not yield the current
    /// contents of `buffer`.
    ///
    /// Note that dropping the [`QueueWriteBufferView`] does *not* submit the
    /// transfer to the GPU immediately. The transfer begins only on the next
    /// call to [`Queue::submit`] after the view is dropped. To get a set of
    /// scheduled transfers started immediately, it's fine to call `submit` with
    /// no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
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
    /// This method fails if `size` overruns the size of `texture`, or if `data` is too short.
    ///
    /// This does *not* submit the transfer to the GPU immediately. Calls to
    /// `write_texture` begin execution only on the next call to
    /// [`Queue::submit`]. To get a set of scheduled transfers started
    /// immediately, it's fine to call `submit` with no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
    ///
    /// However, `data` will be immediately copied into staging memory, so the
    /// caller may discard it any time after this call completes.
    pub fn write_texture(
        &self,
        texture: ImageCopyTexture<'_>,
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
    #[cfg(any(webgpu, webgl))]
    pub fn copy_external_image_to_texture(
        &self,
        source: &wgt::ImageCopyExternalImage,
        dest: ImageCopyTextureTagged<'_>,
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
        let mut command_buffers = command_buffers
            .into_iter()
            .map(|mut comb| (comb.id.take().unwrap(), comb.data.take().unwrap()));

        let data = DynContext::queue_submit(
            &*self.context,
            &self.id,
            self.data.as_ref(),
            &mut command_buffers,
        );

        SubmissionIndex(data)
    }

    /// Gets the amount of nanoseconds each tick of a timestamp query represents.
    ///
    /// Returns zero if timestamp queries are unsupported.
    ///
    /// Timestamp values are represented in nanosecond values on WebGPU, see `<https://gpuweb.github.io/gpuweb/#timestamp>`
    /// Therefore, this is always 1.0 on the web, but on wgpu-core a manual conversion is required.
    pub fn get_timestamp_period(&self) -> f32 {
        DynContext::queue_get_timestamp_period(&*self.context, &self.id, self.data.as_ref())
    }

    /// Registers a callback when the previous call to submit finishes running on the gpu. This callback
    /// being called implies that all mapped buffer callbacks which were registered before this call will
    /// have been called.
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
    ///
    /// # Platform dependent behavior
    ///
    /// On Wayland, `present` will attach a `wl_buffer` to the underlying `wl_surface` and commit the new surface
    /// state. If it is desired to do things such as request a frame callback, scale the surface using the viewporter
    /// or synchronize other double buffered state, then these operations should be done before the call to `present`.
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

impl Surface<'_> {
    /// Returns the capabilities of the surface when used with the given adapter.
    ///
    /// Returns specified values (see [`SurfaceCapabilities`]) if surface is incompatible with the adapter.
    pub fn get_capabilities(&self, adapter: &Adapter) -> SurfaceCapabilities {
        DynContext::surface_get_capabilities(
            &*self.context,
            &self.id,
            self.surface_data.as_ref(),
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
            format: *caps.formats.first()?,
            width,
            height,
            desired_maximum_frame_latency: 2,
            present_mode: *caps.present_modes.first()?,
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
    /// - `config.width` or `config.height` is zero.
    pub fn configure(&self, device: &Device, config: &SurfaceConfiguration) {
        DynContext::surface_configure(
            &*self.context,
            &self.id,
            self.surface_data.as_ref(),
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
        let (texture_id, texture_data, status, detail) = DynContext::surface_get_current_texture(
            &*self.context,
            &self.id,
            self.surface_data.as_ref(),
        );

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
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Surface>) -> R, R>(
        &mut self,
        hal_surface_callback: F,
    ) -> Option<R> {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| unsafe {
                ctx.surface_as_hal::<A, F, R>(
                    self.surface_data.downcast_ref().unwrap(),
                    hal_surface_callback,
                )
            })
    }
}

/// Opaque globally-unique identifier
#[repr(transparent)]
pub struct Id<T>(NonZeroU64, PhantomData<*mut T>);

impl<T> Id<T> {
    /// For testing use only. We provide no guarantees about the actual value of the ids.
    #[doc(hidden)]
    pub fn inner(&self) -> u64 {
        self.0.get()
    }
}

// SAFETY: `Id` is a bare `NonZeroU64`, the type parameter is a marker purely to avoid confusing Ids
// returned for different types , so `Id` can safely implement Send and Sync.
unsafe impl<T> Send for Id<T> {}

// SAFETY: See the implementation for `Send`.
unsafe impl<T> Sync for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Id").field(&self.0).finish()
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Id<T>) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Id<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Id<T>) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

impl Adapter {
    /// Returns a globally-unique identifier for this `Adapter`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl Device {
    /// Returns a globally-unique identifier for this `Device`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl Queue {
    /// Returns a globally-unique identifier for this `Queue`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl ShaderModule {
    /// Returns a globally-unique identifier for this `ShaderModule`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl BindGroupLayout {
    /// Returns a globally-unique identifier for this `BindGroupLayout`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl BindGroup {
    /// Returns a globally-unique identifier for this `BindGroup`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl TextureView {
    /// Returns a globally-unique identifier for this `TextureView`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }

    /// Returns the inner hal TextureView using a callback. The hal texture will be `None` if the
    /// backend type argument does not match with this wgpu Texture
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal TextureView must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::TextureView>) -> R, R>(
        &self,
        hal_texture_view_callback: F,
    ) -> R {
        use core::id::TextureViewId;

        let texture_view_id = TextureViewId::from(self.id);

        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe {
                ctx.texture_view_as_hal::<A, F, R>(texture_view_id, hal_texture_view_callback)
            }
        } else {
            hal_texture_view_callback(None)
        }
    }
}

impl Sampler {
    /// Returns a globally-unique identifier for this `Sampler`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl Buffer {
    /// Returns a globally-unique identifier for this `Buffer`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl Texture {
    /// Returns a globally-unique identifier for this `Texture`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl QuerySet {
    /// Returns a globally-unique identifier for this `QuerySet`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl PipelineLayout {
    /// Returns a globally-unique identifier for this `PipelineLayout`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl RenderPipeline {
    /// Returns a globally-unique identifier for this `RenderPipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl ComputePipeline {
    /// Returns a globally-unique identifier for this `ComputePipeline`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl RenderBundle {
    /// Returns a globally-unique identifier for this `RenderBundle`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id(self.id.global_id(), PhantomData)
    }
}

impl Surface<'_> {
    /// Returns a globally-unique identifier for this `Surface`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Surface<'_>> {
        Id(self.id.global_id(), PhantomData)
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
        #[cfg(send_sync)]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + Send + Sync + 'static>,
        /// Lower level source of the error.
        #[cfg(not(send_sync))]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + 'static>,
    },
    /// Validation error, signifying a bug in code or data
    Validation {
        /// Lower level source of the error.
        #[cfg(send_sync)]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + Send + Sync + 'static>,
        /// Lower level source of the error.
        #[cfg(not(send_sync))]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + 'static>,
        /// Description of the validation error.
        description: String,
    },
    /// Internal error. Used for signalling any failures not explicitly expected by WebGPU.
    ///
    /// These could be due to internal implementation or system limits being reached.
    Internal {
        /// Lower level source of the error.
        #[cfg(send_sync)]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + Send + Sync + 'static>,
        /// Lower level source of the error.
        #[cfg(not(send_sync))]
        #[cfg_attr(docsrs, doc(cfg(all())))]
        source: Box<dyn error::Error + 'static>,
        /// Description of the internal GPU error.
        description: String,
    },
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Error: Send, Sync);

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Error::OutOfMemory { source } => Some(source.as_ref()),
            Error::Validation { source, .. } => Some(source.as_ref()),
            Error::Internal { source, .. } => Some(source.as_ref()),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OutOfMemory { .. } => f.write_str("Out of Memory"),
            Error::Validation { description, .. } => f.write_str(description),
            Error::Internal { description, .. } => f.write_str(description),
        }
    }
}

use send_sync::*;

mod send_sync {
    use std::any::Any;
    use std::fmt;

    use wgt::WasmNotSendSync;

    pub trait AnyWasmNotSendSync: Any + WasmNotSendSync {
        fn upcast_any_ref(&self) -> &dyn Any;
    }
    impl<T: Any + WasmNotSendSync> AnyWasmNotSendSync for T {
        #[inline]
        fn upcast_any_ref(&self) -> &dyn Any {
            self
        }
    }

    impl dyn AnyWasmNotSendSync + 'static {
        #[inline]
        pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
            self.upcast_any_ref().downcast_ref::<T>()
        }
    }

    impl fmt::Debug for dyn AnyWasmNotSendSync {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Any").finish_non_exhaustive()
        }
    }
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
