use std::{error, fmt, future::Future, sync::Arc, thread};

use parking_lot::Mutex;

use crate::context::DynContext;
use crate::*;

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
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Device: Send, Sync);

/// Describes a [`Device`].
///
/// For use with [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUDeviceDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpudevicedescriptor).
pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(DeviceDescriptor<'_>: Send, Sync);

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
        DynContext::device_poll(&*self.context, self.data.as_ref(), maintain)
    }

    /// The features which can be used on this device.
    ///
    /// No additional features can be used, even if the underlying adapter can support them.
    #[must_use]
    pub fn features(&self) -> Features {
        DynContext::device_features(&*self.context, self.data.as_ref())
    }

    /// The limits which can be used on this device.
    ///
    /// No better limits can be used, even if the underlying adapter can support them.
    #[must_use]
    pub fn limits(&self) -> Limits {
        DynContext::device_limits(&*self.context, self.data.as_ref())
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
    #[must_use]
    pub fn create_shader_module(&self, desc: ShaderModuleDescriptor<'_>) -> ShaderModule {
        let data = DynContext::device_create_shader_module(
            &*self.context,
            self.data.as_ref(),
            desc,
            wgt::ShaderBoundChecks::new(),
        );
        ShaderModule {
            context: Arc::clone(&self.context),
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
    #[must_use]
    pub unsafe fn create_shader_module_unchecked(
        &self,
        desc: ShaderModuleDescriptor<'_>,
    ) -> ShaderModule {
        let data = DynContext::device_create_shader_module(
            &*self.context,
            self.data.as_ref(),
            desc,
            unsafe { wgt::ShaderBoundChecks::unchecked() },
        );
        ShaderModule {
            context: Arc::clone(&self.context),
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
    #[must_use]
    pub unsafe fn create_shader_module_spirv(
        &self,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> ShaderModule {
        let data = unsafe {
            DynContext::device_create_shader_module_spirv(&*self.context, self.data.as_ref(), desc)
        };
        ShaderModule {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates an empty [`CommandEncoder`].
    #[must_use]
    pub fn create_command_encoder(&self, desc: &CommandEncoderDescriptor<'_>) -> CommandEncoder {
        let data =
            DynContext::device_create_command_encoder(&*self.context, self.data.as_ref(), desc);
        CommandEncoder {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates an empty [`RenderBundleEncoder`].
    #[must_use]
    pub fn create_render_bundle_encoder(
        &self,
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> RenderBundleEncoder<'_> {
        let data = DynContext::device_create_render_bundle_encoder(
            &*self.context,
            self.data.as_ref(),
            desc,
        );
        RenderBundleEncoder {
            context: Arc::clone(&self.context),
            data,
            parent: self,
            _p: Default::default(),
        }
    }

    /// Creates a new [`BindGroup`].
    #[must_use]
    pub fn create_bind_group(&self, desc: &BindGroupDescriptor<'_>) -> BindGroup {
        let data = DynContext::device_create_bind_group(&*self.context, self.data.as_ref(), desc);
        BindGroup {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a [`BindGroupLayout`].
    #[must_use]
    pub fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> BindGroupLayout {
        let data =
            DynContext::device_create_bind_group_layout(&*self.context, self.data.as_ref(), desc);
        BindGroupLayout {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a [`PipelineLayout`].
    #[must_use]
    pub fn create_pipeline_layout(&self, desc: &PipelineLayoutDescriptor<'_>) -> PipelineLayout {
        let data =
            DynContext::device_create_pipeline_layout(&*self.context, self.data.as_ref(), desc);
        PipelineLayout {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a [`RenderPipeline`].
    #[must_use]
    pub fn create_render_pipeline(&self, desc: &RenderPipelineDescriptor<'_>) -> RenderPipeline {
        let data =
            DynContext::device_create_render_pipeline(&*self.context, self.data.as_ref(), desc);
        RenderPipeline {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a [`ComputePipeline`].
    #[must_use]
    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDescriptor<'_>) -> ComputePipeline {
        let data =
            DynContext::device_create_compute_pipeline(&*self.context, self.data.as_ref(), desc);
        ComputePipeline {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a [`Buffer`].
    #[must_use]
    pub fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Buffer {
        let mut map_context = MapContext::new(desc.size);
        if desc.mapped_at_creation {
            map_context.initial_range = 0..desc.size;
        }

        let data = DynContext::device_create_buffer(&*self.context, self.data.as_ref(), desc);

        Buffer {
            context: Arc::clone(&self.context),
            data,
            map_context: Mutex::new(map_context),
            size: desc.size,
            usage: desc.usage,
        }
    }

    /// Creates a new [`Texture`].
    ///
    /// `desc` specifies the general format of the texture.
    #[must_use]
    pub fn create_texture(&self, desc: &TextureDescriptor<'_>) -> Texture {
        let data = DynContext::device_create_texture(&*self.context, self.data.as_ref(), desc);
        Texture {
            context: Arc::clone(&self.context),
            data,
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
    #[must_use]
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
                    crate::context::downcast_ref(self.data.as_ref()),
                    desc,
                )
        };
        Texture {
            context: Arc::clone(&self.context),
            data: Box::new(texture),
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
    #[must_use]
    pub unsafe fn create_buffer_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_buffer: A::Buffer,
        desc: &BufferDescriptor<'_>,
    ) -> Buffer {
        let mut map_context = MapContext::new(desc.size);
        if desc.mapped_at_creation {
            map_context.initial_range = 0..desc.size;
        }

        let buffer = unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                // Part of the safety requirements is that the buffer was generated from the same hal device.
                // Therefore, unwrap is fine here since only WgpuCoreContext has the ability to create hal buffers.
                .unwrap()
                .create_buffer_from_hal::<A>(
                    hal_buffer,
                    crate::context::downcast_ref(self.data.as_ref()),
                    desc,
                )
        };

        Buffer {
            context: Arc::clone(&self.context),
            data: Box::new(buffer),
            map_context: Mutex::new(map_context),
            size: desc.size,
            usage: desc.usage,
        }
    }

    /// Creates a new [`Sampler`].
    ///
    /// `desc` specifies the behavior of the sampler.
    #[must_use]
    pub fn create_sampler(&self, desc: &SamplerDescriptor<'_>) -> Sampler {
        let data = DynContext::device_create_sampler(&*self.context, self.data.as_ref(), desc);
        Sampler {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Creates a new [`QuerySet`].
    #[must_use]
    pub fn create_query_set(&self, desc: &QuerySetDescriptor<'_>) -> QuerySet {
        let data = DynContext::device_create_query_set(&*self.context, self.data.as_ref(), desc);
        QuerySet {
            context: Arc::clone(&self.context),
            data,
        }
    }

    /// Set a callback for errors that are not handled in error scopes.
    pub fn on_uncaptured_error(&self, handler: Box<dyn UncapturedErrorHandler>) {
        self.context
            .device_on_uncaptured_error(self.data.as_ref(), handler);
    }

    /// Push an error scope.
    pub fn push_error_scope(&self, filter: ErrorFilter) {
        self.context
            .device_push_error_scope(self.data.as_ref(), filter);
    }

    /// Pop an error scope.
    pub fn pop_error_scope(&self) -> impl Future<Output = Option<Error>> + WasmNotSend {
        self.context.device_pop_error_scope(self.data.as_ref())
    }

    /// Starts frame capture.
    pub fn start_capture(&self) {
        DynContext::device_start_capture(&*self.context, self.data.as_ref())
    }

    /// Stops frame capture.
    pub fn stop_capture(&self) {
        DynContext::device_stop_capture(&*self.context, self.data.as_ref())
    }

    /// Query internal counters from the native backend for debugging purposes.
    ///
    /// Some backends may not set all counters, or may not set any counter at all.
    /// The `counters` cargo feature must be enabled for any counter to be set.
    ///
    /// If a counter is not set, its contains its default value (zero).
    #[must_use]
    pub fn get_internal_counters(&self) -> wgt::InternalCounters {
        DynContext::device_get_internal_counters(&*self.context, self.data.as_ref())
    }

    /// Generate an GPU memory allocation report if the underlying backend supports it.
    ///
    /// Backends that do not support producing these reports return `None`. A backend may
    /// Support it and still return `None` if it is not using performing sub-allocation,
    /// for example as a workaround for driver issues.
    #[must_use]
    pub fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport> {
        DynContext::generate_allocator_report(&*self.context, self.data.as_ref())
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
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_device_callback,
                )
            })
    }

    /// Destroy this device.
    pub fn destroy(&self) {
        DynContext::device_destroy(&*self.context, self.data.as_ref())
    }

    /// Set a DeviceLostCallback on this device.
    pub fn set_device_lost_callback(
        &self,
        callback: impl Fn(DeviceLostReason, String) + Send + 'static,
    ) {
        DynContext::device_set_device_lost_callback(
            &*self.context,
            self.data.as_ref(),
            Box::new(callback),
        )
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
    #[must_use]
    pub unsafe fn create_pipeline_cache(
        &self,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> PipelineCache {
        let data = unsafe {
            DynContext::device_create_pipeline_cache(&*self.context, self.data.as_ref(), desc)
        };
        PipelineCache {
            context: Arc::clone(&self.context),
            data,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.device_drop(self.data.as_ref());
        }
    }
}

/// Requesting a device from an [`Adapter`] failed.
#[derive(Clone, Debug)]
pub struct RequestDeviceError {
    pub(crate) inner: RequestDeviceErrorKind,
}
#[derive(Clone, Debug)]
pub(crate) enum RequestDeviceErrorKind {
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

/// Type for the callback of uncaptured error handler
pub trait UncapturedErrorHandler: Fn(Error) + Send + 'static {}
impl<T> UncapturedErrorHandler for T where T: Fn(Error) + Send + 'static {}

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

/// Lower level source of the error.
///
/// `Send + Sync` varies depending on configuration.
#[cfg(send_sync)]
#[cfg_attr(docsrs, doc(cfg(all())))]
pub type ErrorSource = Box<dyn error::Error + Send + Sync + 'static>;
/// Lower level source of the error.
///
/// `Send + Sync` varies depending on configuration.
#[cfg(not(send_sync))]
#[cfg_attr(docsrs, doc(cfg(all())))]
pub type ErrorSource = Box<dyn error::Error + 'static>;

/// Error type
#[derive(Debug)]
pub enum Error {
    /// Out of memory error
    OutOfMemory {
        /// Lower level source of the error.
        source: ErrorSource,
    },
    /// Validation error, signifying a bug in code or data
    Validation {
        /// Lower level source of the error.
        source: ErrorSource,
        /// Description of the validation error.
        description: String,
    },
    /// Internal error. Used for signalling any failures not explicitly expected by WebGPU.
    ///
    /// These could be due to internal implementation or system limits being reached.
    Internal {
        /// Lower level source of the error.
        source: ErrorSource,
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
