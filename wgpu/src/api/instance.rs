use parking_lot::Mutex;

use crate::*;

use std::{future::Future, sync::Arc};

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
            let support_webgpu = crate::backend::get_browser_gpu_property()
                .map(|maybe_gpu| maybe_gpu.is_some())
                .unwrap_or(false);

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
                    .map(move |adapter| crate::Adapter {
                        context: Arc::clone(&context),
                        data: Box::new(adapter),
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
        async move { adapter.await.map(|data| Adapter { context, data }) }
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
        let adapter = unsafe {
            context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                .unwrap()
                .create_adapter_from_hal(hal_adapter)
        };
        Adapter {
            context,
            data: Box::new(adapter),
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
        let data = unsafe { self.context.instance_create_surface(target) }?;

        Ok(Surface {
            context: Arc::clone(&self.context),
            _handle_source: None,
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
