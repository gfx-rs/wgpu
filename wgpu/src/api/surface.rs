use std::{error, fmt, sync::Arc, thread};

use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::context::DynContext;
use crate::*;

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
    pub(crate) context: Arc<C>,

    /// Optionally, keep the source of the handle used for the surface alive.
    ///
    /// This is useful for platforms where the surface is created from a window and the surface
    /// would become invalid when the window is dropped.
    pub(crate) _handle_source: Option<Box<dyn WindowHandle + 'window>>,

    /// Additional surface data returned by [`DynContext::instance_create_surface`].
    pub(crate) surface_data: Box<Data>,

    // Stores the latest `SurfaceConfiguration` that was set using `Surface::configure`.
    // It is required to set the attributes of the `SurfaceTexture` in the
    // `Surface::get_current_texture` method.
    // Because the `Surface::configure` method operates on an immutable reference this type has to
    // be wrapped in a mutex and since the configuration is only supplied after the surface has
    // been created is is additionally wrapped in an option.
    pub(crate) config: Mutex<Option<SurfaceConfiguration>>,
}

impl Surface<'_> {
    /// Returns the capabilities of the surface when used with the given adapter.
    ///
    /// Returns specified values (see [`SurfaceCapabilities`]) if surface is incompatible with the adapter.
    pub fn get_capabilities(&self, adapter: &Adapter) -> SurfaceCapabilities {
        DynContext::surface_get_capabilities(
            &*self.context,
            self.surface_data.as_ref(),
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
            self.surface_data.as_ref(),
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
        let (texture_data, status, detail) =
            DynContext::surface_get_current_texture(&*self.context, self.surface_data.as_ref());

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

        texture_data
            .map(|data| SurfaceTexture {
                texture: Texture {
                    context: Arc::clone(&self.context),
                    data,
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
        &self,
        hal_surface_callback: F,
    ) -> Option<R> {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| unsafe {
                ctx.surface_as_hal::<A, F, R>(
                    crate::context::downcast_ref(self.surface_data.as_ref()),
                    hal_surface_callback,
                )
            })
    }
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
            self.context.surface_drop(self.surface_data.as_ref())
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
    /// - visual must be a valid `IDCompositionVisual` to create a surface upon.  Its refcount will be incremented internally and kept live as long as the resulting [`Surface`] is live.
    #[cfg(dx12)]
    CompositionVisual(*mut std::ffi::c_void),

    /// Surface from DX12 `DirectComposition` handle.
    ///
    /// <https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nf-dxgi1_3-idxgifactorymedia-createswapchainforcompositionsurfacehandle>
    ///
    /// # Safety
    ///
    /// - surface_handle must be a valid `DirectComposition` handle to create a surface upon.   Its lifetime **will not** be internally managed: this handle **should not** be freed before
    ///   the resulting [`Surface`] is destroyed.
    #[cfg(dx12)]
    SurfaceHandle(*mut std::ffi::c_void),

    /// Surface from DX12 `SwapChainPanel`.
    ///
    /// # Safety
    ///
    /// - visual must be a valid SwapChainPanel to create a surface upon.  Its refcount will be incremented internally and kept live as long as the resulting [`Surface`] is live.
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

/// [`Instance::create_surface()`] or a related function failed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CreateSurfaceError {
    pub(crate) inner: CreateSurfaceErrorKind,
}
#[derive(Clone, Debug)]
pub(crate) enum CreateSurfaceErrorKind {
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
