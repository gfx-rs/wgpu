#[cfg(wgpu_core)]
use alloc::vec::Vec;
use core::future::Future;

use crate::{dispatch::InstanceInterface, util::Mutex, *};

bitflags::bitflags! {
    /// WGSL language extensions.
    ///
    /// WGSL spec.: <https://www.w3.org/TR/WGSL/#language-extensions-sec>
    #[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
    pub struct WgslLanguageFeatures: u32 {
        /// <https://www.w3.org/TR/WGSL/#language_extension-readonly_and_readwrite_storage_textures>
        const ReadOnlyAndReadWriteStorageTextures = 1 << 0;
        /// <https://www.w3.org/TR/WGSL/#language_extension-packed_4x8_integer_dot_product>
        const Packed4x8IntegerDotProduct = 1 << 1;
        /// <https://www.w3.org/TR/WGSL/#language_extension-unrestricted_pointer_parameters>
        const UnrestrictedPointerParameters = 1 << 2;
        /// <https://www.w3.org/TR/WGSL/#language_extension-pointer_composite_access>
        const PointerCompositeAccess = 1 << 3;
    }
}

/// Context for all other wgpu objects. Instance of wgpu.
///
/// This is the first thing you create when using wgpu.
/// Its primary use is to create [`Adapter`]s and [`Surface`]s.
///
/// Does not have to be kept alive.
///
/// Corresponds to [WebGPU `GPU`](https://gpuweb.github.io/gpuweb/#gpu-interface).
#[derive(Debug, Clone)]
pub struct Instance {
    inner: dispatch::DispatchInstance,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Instance: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(Instance => .inner);

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
        Self::new(&InstanceDescriptor::default())
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
    pub const fn enabled_backend_features() -> Backends {
        let mut backends = Backends::empty();
        // `.set` and `|=` don't work in a `const` context.
        if cfg!(noop) {
            backends = backends.union(Backends::NOOP);
        }
        if cfg!(vulkan) {
            backends = backends.union(Backends::VULKAN);
        }
        if cfg!(any(gles, webgl)) {
            backends = backends.union(Backends::GL);
        }
        if cfg!(metal) {
            backends = backends.union(Backends::METAL);
        }
        if cfg!(dx12) {
            backends = backends.union(Backends::DX12);
        }
        if cfg!(webgpu) {
            backends = backends.union(Backends::BROWSER_WEBGPU);
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
    ///   If it is set and a [`navigator.gpu`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/gpu)
    ///   object is present, this instance will *only* be able to create WebGPU adapters.
    ///
    ///   ⚠️ On some browsers this check is insufficient to determine whether WebGPU is supported,
    ///   as the browser may define the `navigator.gpu` object, but be unable to create any WebGPU adapters.
    ///   For targeting _both_ WebGPU & WebGL is recommended to use [`crate::util::new_instance_with_webgpu_detection`].
    ///
    ///   If you instead want to force use of WebGL, either disable the `webgpu` compile-time feature
    ///   or don't add the [`Backends::BROWSER_WEBGPU`] flag to the the `instance_desc`'s `backends` field.
    ///   If it is set and WebGPU support is *not* detected, the instance will use wgpu-core
    ///   to create adapters. Meaning that if the `webgl` feature is enabled, it is able to create
    ///   a WebGL adapter.
    ///
    /// # Panics
    ///
    /// If no backend feature for the active target platform is enabled,
    /// this method will panic, see [`Instance::enabled_backend_features()`].
    #[allow(clippy::allow_attributes, unreachable_code)]
    pub fn new(_instance_desc: &InstanceDescriptor) -> Self {
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
            let support_webgpu = crate::backend::get_browser_gpu_property()
                .map(|maybe_gpu| maybe_gpu.is_some())
                .unwrap_or(false);

            if is_only_available_backend || (requested_webgpu && support_webgpu) {
                return Self {
                    inner: crate::backend::ContextWebGpu::new(_instance_desc).into(),
                };
            }
        }

        #[cfg(wgpu_core)]
        {
            return Self {
                inner: crate::backend::ContextWgpuCore::new(_instance_desc).into(),
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
            inner: unsafe {
                crate::backend::ContextWgpuCore::from_hal_instance::<A>(hal_instance).into()
            },
        }
    }

    /// Get the [`wgpu_hal`] instance from this `Instance`.
    ///
    /// Find the Api struct corresponding to the active backend in [`wgpu_hal::api`],
    /// and pass that struct to the to the `A` type parameter.
    ///
    /// Returns a guard that dereferences to the type of the hal backend
    /// which implements [`A::Instance`].
    ///
    /// # Errors
    ///
    /// This method will return None if:
    /// - The instance is not from the backend specified by `A`.
    /// - The instance is from the `webgpu` or `custom` backend.
    ///
    /// # Safety
    ///
    /// - The returned resource must not be destroyed unless the guard
    ///   is the last reference to it and it is not in use by the GPU.
    ///   The guard and handle may be dropped at any time however.
    /// - All the safety requirements of wgpu-hal must be upheld.
    ///
    /// [`A::Instance`]: hal::Api::Instance
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi>(&self) -> Option<&A::Instance> {
        self.inner
            .as_core_opt()
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
            inner: unsafe {
                crate::backend::ContextWgpuCore::from_core_instance(core_instance).into()
            },
        }
    }

    #[cfg(custom)]
    /// Creates instance from custom context implementation
    pub fn from_custom<T: InstanceInterface>(instance: T) -> Self {
        Self {
            inner: dispatch::DispatchInstance::Custom(backend::custom::DynContext::new(instance)),
        }
    }

    #[cfg(custom)]
    /// Returns custom implementation of Instance (if custom backend and is internally T)
    pub fn as_custom<T: custom::InstanceInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }

    /// Retrieves all available [`Adapter`]s that match the given [`Backends`].
    ///
    /// # Arguments
    ///
    /// - `backends` - Backends from which to enumerate adapters.
    #[cfg(wgpu_core)]
    pub fn enumerate_adapters(&self, backends: Backends) -> Vec<Adapter> {
        let Some(core_instance) = self.inner.as_core_opt() else {
            return Vec::new();
        };

        core_instance
            .enumerate_adapters(backends)
            .into_iter()
            .map(|adapter| {
                let core = backend::wgpu_core::CoreAdapter {
                    context: core_instance.clone(),
                    id: adapter,
                };
                crate::Adapter { inner: core.into() }
            })
            .collect()
    }

    /// Retrieves an [`Adapter`] which matches the given [`RequestAdapterOptions`].
    ///
    /// Some options are "soft", so treated as non-mandatory. Others are "hard".
    ///
    /// If no adapters are found that satisfy all the "hard" options, an error is returned.
    ///
    /// When targeting WebGL2, a [`compatible_surface`](RequestAdapterOptions::compatible_surface)
    /// must be specified; using `RequestAdapterOptions::default()` will not succeed.
    pub fn request_adapter(
        &self,
        options: &RequestAdapterOptions<'_, '_>,
    ) -> impl Future<Output = Result<Adapter, RequestAdapterError>> + WasmNotSend {
        let future = self.inner.request_adapter(options);
        async move { future.await.map(|adapter| Adapter { inner: adapter }) }
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
        let core_instance = self.inner.as_core();
        let adapter = unsafe { core_instance.create_adapter_from_hal(hal_adapter) };
        let core = backend::wgpu_core::CoreAdapter {
            context: core_instance.clone(),
            id: adapter,
        };

        Adapter { inner: core.into() }
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

            #[cfg(web)]
            SurfaceTarget::Canvas(canvas) => {
                handle_source = None;

                let value: &wasm_bindgen::JsValue = &canvas;
                let obj = core::ptr::NonNull::from(value).cast();
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

            #[cfg(web)]
            SurfaceTarget::OffscreenCanvas(canvas) => {
                handle_source = None;

                let value: &wasm_bindgen::JsValue = &canvas;
                let obj = core::ptr::NonNull::from(value).cast();
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
        let surface = unsafe { self.inner.create_surface(target)? };

        Ok(Surface {
            _handle_source: None,
            inner: surface,
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
        self.inner.poll_all_devices(force_wait)
    }

    /// Generates memory report.
    ///
    /// Returns `None` if the feature is not supported by the backend
    /// which happens only when WebGPU is pre-selected by the instance creation.
    #[cfg(wgpu_core)]
    pub fn generate_report(&self) -> Option<wgc::global::GlobalReport> {
        self.inner.as_core_opt().map(|ctx| ctx.generate_report())
    }

    /// Returns set of supported WGSL language extensions supported by this instance.
    ///
    /// <https://www.w3.org/TR/webgpu/#gpuwgsllanguagefeatures>
    #[cfg(feature = "wgsl")]
    pub fn wgsl_language_features(&self) -> WgslLanguageFeatures {
        self.inner.wgsl_language_features()
    }
}
