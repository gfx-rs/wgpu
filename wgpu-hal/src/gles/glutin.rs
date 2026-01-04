use alloc::{string::String, sync::Arc, vec::Vec};
use core::{ffi, mem::ManuallyDrop, num::NonZeroU32, ptr, time::Duration};
use glutin::{context::AsRawContext, prelude::*};

use glow::HasContext;
use parking_lot::{Mutex, MutexGuard, RwLock};

/// The amount of time to wait while trying to obtain a lock to the adapter context
const CONTEXT_LOCK_TIMEOUT_SECS: u64 = 6;

struct GlutinContext {
    current_context: Option<glutin::context::PossiblyCurrentContext>,
    not_current_context: Option<glutin::context::NotCurrentContext>,
}

type GlutinSurface<T> = glutin::surface::Surface<T>;
type GlutinWindowSurfaceAttributesBuilder =
    glutin::surface::SurfaceAttributesBuilder<glutin::surface::WindowSurface>;
type GlutinWindowSurface = GlutinSurface<glutin::surface::WindowSurface>;

type GlutinDisplay = glutin::display::Display;

impl GlutinContext {
    fn make_current<T: glutin::surface::SurfaceTypeTrait>(
        &mut self,
        surface: &GlutinSurface<T>,
    ) -> Result<(), glutin::error::Error> {
        if let Some(not_current) = self.not_current_context.take() {
            let current = not_current.make_current(surface)?;
            self.current_context = Some(current);
        }

        Ok(())
    }

    fn unmake_current(&mut self) -> Result<(), glutin::error::Error> {
        if let Some(current) = self.current_context.take() {
            let not_current = current.make_not_current()?;
            self.not_current_context = Some(not_current);
        }

        Ok(())
    }

    fn raw_context(&self) -> *mut ffi::c_void {
        let raw_context = if let Some(current) = &self.current_context {
            current.raw_context()
        } else if let Some(not_current) = &self.not_current_context {
            not_current.raw_context()
        } else {
            return ptr::null_mut();
        };

        match raw_context {
            #[cfg(gles_egl_backend)]
            glutin::context::RawContext::Egl(ctx) => ctx as *mut ffi::c_void,
            // #[cfg(gles_glx_backend)]
            // glutin::context::RawContext::Glx(ctx) => ctx as *mut ffi::c_void,
            #[cfg(gles_wgl_backend)]
            glutin::context::RawContext::Wgl(ctx) => ctx as *mut ffi::c_void,
            #[cfg(gles_cgl_backend)]
            glutin::context::RawContext::Cgl(ctx) => ctx as *mut ffi::c_void,
        }
    }
}

pub struct AdapterContext {
    inner: Arc<Mutex<Inner>>,
}

unsafe impl Sync for AdapterContext {}
unsafe impl Send for AdapterContext {}

impl AdapterContext {
    pub fn is_owned(&self) -> bool {
        true
    }

    pub fn raw_context(&self) -> *mut ffi::c_void {
        self.inner.lock().context.raw_context()
    }

    pub fn lock(&self) -> AdapterContextLock<'_> {
        let mut inner = self
            .inner
            // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
            // deadlock and should panic to show where we got stuck
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
            .expect("Could not lock adapter context. This is most-likely a deadlock.");

        inner.lock_surface().unwrap();

        AdapterContextLock { inner }
    }

    fn lock_with_surface(
        &self,
        surface: &GlutinWindowSurface,
    ) -> Result<AdapterContextLock<'_>, glutin::error::Error> {
        let mut inner = self
            .inner
            // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
            // deadlock and should panic to show where we got stuck
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
            .expect("Could not lock adapter context. This is most-likely a deadlock.");

        inner.context.make_current(surface)?;

        Ok(AdapterContextLock { inner })
    }
}

pub struct AdapterContextLock<'a> {
    inner: MutexGuard<'a, Inner>,
}

impl<'a> core::ops::Deref for AdapterContextLock<'a> {
    type Target = glow::Context;

    fn deref(&self) -> &Self::Target {
        &self.inner.gl
    }
}

impl<'a> Drop for AdapterContextLock<'a> {
    fn drop(&mut self) {
        self.inner.context.unmake_current().unwrap();
    }
}

struct Inner {
    gl: ManuallyDrop<glow::Context>,
    context: GlutinContext,
    surface: Arc<Mutex<Option<GlutinWindowSurface>>>,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn lock_surface(&mut self) -> Result<(), glutin::error::Error> {
        if let Some(surface) = self.surface.lock().as_ref() {
            self.context.make_current(surface)?;
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        struct Guard<'a> {
            context: &'a mut GlutinContext,
        }

        impl<'a> Drop for Guard<'a> {
            fn drop(&mut self) {
                self.context.unmake_current().unwrap();
            }
        }

        let surface = self.surface.lock();
        let _guard = surface.as_ref().map(|surface| {
            self.context.make_current(surface).unwrap();
            Guard {
                context: &mut self.context,
            }
        });
        unsafe { ManuallyDrop::drop(&mut self.gl) };
    }
}

#[cfg(windows)]
fn preference_default(
    window_handle: raw_window_handle::RawWindowHandle,
) -> glutin::display::DisplayApiPreference {
    #[cfg(all(gles_wgl_backend, gles_egl_backend))]
    let preference = glutin::display::DisplayApiPreference::WglThenEgl(Some(window_handle));
    #[cfg(all(gles_wgl_backend, not(gles_egl_backend)))]
    let preference = glutin::display::DisplayApiPreference::Wgl(Some(window_handle.as_raw()));
    #[cfg(all(not(gles_wgl_backend), gles_egl_backend))]
    let preference = glutin::display::DisplayApiPreference::Egl;

    preference
}

#[cfg(free_unix)]
fn preference_default(
    _window_handle: raw_window_handle::RawWindowHandle,
) -> glutin::display::DisplayApiPreference {
    // TODO: Add Surport for x11 gl
    // #[cfg(all(gles_egl_backend, gles_glx_backend))]
    // let preference = glutin::display::DisplayApiPreference::GlxThenEgl()
    // #[cfg(all(gles_glx_backend, not(gles_egl_backend)))]
    // let preference = glutin::display::DisplayApiPreference::Glx
    #[cfg(all(gles_egl_backend, not(gles_glx_backend)))]
    let preference = glutin::display::DisplayApiPreference::Egl;

    preference
}

#[cfg(apple)]
fn preference_default(
    _window_handle: raw_window_handle::RawWindowHandle,
) -> glutin::display::DisplayApiPreference {
    #[cfg(all(gles_cgl_backend))]
    let preference = glutin::display::DisplayApiPreference::Cgl;

    preference
}

pub struct Instance {
    inner: Arc<Mutex<Inner>>,
    display: GlutinDisplay,
    window: raw_window_handle::RawWindowHandle,
    config: glutin::config::Config,
    options: wgt::GlBackendOptions,
    srgb_capable: bool,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl crate::Instance for Instance {
    type A = super::Api;

    unsafe fn init(desc: &crate::InstanceDescriptor<'_>) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init OpenGL Backend");

        let window_handle = desc.window.ok_or(crate::InstanceError::new(String::from(
            "RawWindowHandle is required to create OpenGL Instance",
        )))?;
        let display_handle = desc.display.ok_or(crate::InstanceError::new(String::from(
            "RawDisplayHandle is required to create OpenGL Instance",
        )))?;

        let preference = preference_default(window_handle.as_raw());
        let display = unsafe {
            GlutinDisplay::new(display_handle.as_raw(), preference).map_err(|e| {
                crate::InstanceError::with_source(
                    format!("Failed to create glutin display: {}", e),
                    e,
                )
            })?
        };

        let config_template = glutin::config::ConfigTemplateBuilder::new().build();
        let configs = unsafe {
            display
                .find_configs(config_template)
                .map_err(|e| {
                    crate::InstanceError::with_source(
                        String::from("Failed to find suitable glutin config"),
                        e,
                    )
                })?
                .collect::<Vec<_>>()
        };

        if log::log_enabled!(log::Level::Trace) {
            for (i, config) in configs.iter().enumerate() {
                log::trace!("GL Config {}: {:#?}", i, config);
            }
        }

        // TODO: Surface attributes can be customized later
        let config = configs
            .first()
            .ok_or_else(|| {
                crate::InstanceError::new(String::from(
                    "No suitable glutin config found for the display",
                ))
            })?
            .clone();

        let context_attributes = glutin::context::ContextAttributesBuilder::new()
            .with_debug(desc.flags.contains(wgt::InstanceFlags::DEBUG))
            .build(desc.window.map(|w| w.as_raw()));

        let context = unsafe {
            display
                .create_context(&config, &context_attributes)
                .map_err(|e| {
                    crate::InstanceError::with_source(
                        String::from("Failed to create glutin context"),
                        e,
                    )
                })?
        };

        let mut context = GlutinContext {
            current_context: None,
            not_current_context: Some(context),
        };

        let surface_attributes = GlutinWindowSurfaceAttributesBuilder::new().build(
            window_handle.as_raw(),
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(1).unwrap(),
        );

        let surface = unsafe {
            display
                .create_window_surface(&config, &surface_attributes)
                .map_err(|e| {
                    crate::InstanceError::with_source(
                        String::from("Failed to create glutin pbuffer surface"),
                        e,
                    )
                })?
        };

        context.make_current(&surface).map_err(|e| {
            crate::InstanceError::with_source(
                String::from("Failed to make glutin context current with pbuffer"),
                e,
            )
        })?;
        let mut gl = unsafe {
            glow::Context::from_loader_function(|name| {
                display.get_proc_address(ffi::CStr::from_bytes_with_nul_unchecked(name.as_bytes()))
                    as _
            })
        };

        // check for sRGB capability
        let srgb_capable = config.srgb_capable();

        // In contrast to OpenGL ES, OpenGL requires explicitly enabling sRGB conversions,
        // as otherwise the user has to do the sRGB conversion.
        if srgb_capable {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        if desc.flags.contains(wgt::InstanceFlags::VALIDATION) && gl.supports_debug() {
            log::debug!("Enabling GL debug output");
            unsafe { gl.enable(glow::DEBUG_OUTPUT) };
            unsafe { gl.debug_message_callback(super::gl_debug_message_callback) };
        }

        let gl = ManuallyDrop::new(gl);
        context.unmake_current().map_err(|e| {
            crate::InstanceError::with_source(
                String::from("Failed to unmake glutin context current after initialization"),
                e,
            )
        })?;

        let inner = Inner {
            gl,
            context,
            surface: Arc::new(Mutex::new(Some(surface))),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            display,
            window: window_handle.as_raw(),
            config,
            options: desc.backend_options.gl.clone(),
            srgb_capable,
        })
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        Ok(Surface {
            display: self.display.clone(),
            window: window_handle,
            parent: self.window,
            config: self.config.clone(),
            presentable: true,
            swapchain: RwLock::new(None),
            srgb_capable: self.srgb_capable,
        })
    }

    unsafe fn enumerate_adapters(
        &self,
        _surface_hint: Option<&Surface>,
    ) -> Vec<crate::ExposedAdapter<super::Api>> {
        unsafe {
            super::Adapter::expose(
                AdapterContext {
                    inner: self.inner.clone(),
                },
                self.options.clone(),
            )
        }
        .into_iter()
        .collect()
    }
}

impl super::Adapter {
    /// Creates a new external adapter using the specified loader function.
    ///
    /// # Safety
    ///
    /// - The underlying OpenGL ES context must be current.
    /// - The underlying OpenGL ES context must be current when interfacing with any objects returned by
    ///   wgpu-hal from this adapter.
    /// - The underlying OpenGL ES context must be current when dropping this adapter and when
    ///   dropping any objects returned from this adapter.
    pub unsafe fn new_external(
        fun: impl FnMut(&str) -> *const ffi::c_void,
        options: wgt::GlBackendOptions,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        let context = unsafe { glow::Context::from_loader_function(fun) };
        unsafe {
            Self::expose(
                AdapterContext {
                    inner: Arc::new(Mutex::new(Inner {
                        gl: ManuallyDrop::new(context),
                        context: GlutinContext {
                            current_context: None,
                            not_current_context: None,
                        },
                        surface: Arc::new(Mutex::new(None)),
                    })),
                },
                options,
            )
        }
    }

    pub fn adapter_context(&self) -> &AdapterContext {
        &self.shared.context
    }
}

impl super::Device {
    pub fn context(&self) -> &AdapterContext {
        &self.shared.context
    }
}

#[derive(Debug)]
pub struct SwapchainInner {
    framebuffer: glow::Framebuffer,
    renderbuffer: glow::Renderbuffer,
    /// Extent because the window lies
    extent: wgt::Extent3d,
    format: wgt::TextureFormat,
    format_desc: super::TextureFormatDesc,
    #[allow(unused)]
    sample_type: wgt::TextureSampleType,
}

pub enum Swapchain {
    Parent(SwapchainInner),
    Other(GlutinWindowSurface, SwapchainInner),
}

pub struct Surface {
    display: glutin::display::Display,
    window: raw_window_handle::RawWindowHandle,
    parent: raw_window_handle::RawWindowHandle,
    config: glutin::config::Config,
    pub(crate) presentable: bool,
    swapchain: RwLock<Option<Swapchain>>,
    srgb_capable: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub(super) unsafe fn present(
        &self,
        _suf_texture: super::Texture,
        context: &AdapterContext,
    ) -> Result<(), crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = swapchain.as_ref().ok_or(crate::SurfaceError::Other(
            "Surface has no swap-chain configured",
        ))?;

        let (gl, sci) = match sc {
            Swapchain::Other(sc_surface, sc) => {
                let gl = context
                    .lock_with_surface(sc_surface)
                    .map_err(|_| crate::SurfaceError::Other("Failed to lock adapter context"))?;
                (gl, sc)
            }
            Swapchain::Parent(sc) => {
                let gl = context.lock();
                (gl, sc)
            }
        };

        // Need ?
        // unsafe { gl.disable(glow::SCISSOR_TEST) };
        // unsafe { gl.color_mask(true, true, true, true) };

        unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sci.framebuffer)) };

        if self.srgb_capable {
            // Disable sRGB conversions for `glBlitFramebuffer` as behavior does diverge between
            // drivers and formats otherwise and we want to ensure no sRGB conversions happen.
            unsafe { gl.disable(glow::FRAMEBUFFER_SRGB) };
        }

        // Note the Y-flipping here. GL's presentation is not flipped,
        // but main rendering is. Therefore, we Y-flip the output positions
        // in the shader, and also this blit.
        unsafe {
            gl.blit_framebuffer(
                0,
                sci.extent.height as i32,
                sci.extent.width as i32,
                0,
                0,
                0,
                sci.extent.width as i32,
                sci.extent.height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            )
        };

        if self.srgb_capable {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        match sc {
            Swapchain::Other(sc_surface, _) => {
                sc_surface
                    .swap_buffers(gl.inner.context.current_context.as_ref().unwrap())
                    .map_err(|_| {
                        crate::SurfaceError::Other("Failed to swap buffers on glutin surface")
                    })?;
            }
            Swapchain::Parent(_) => {
                let surface = gl.inner.surface.lock();
                let parent_surface = surface.as_ref().ok_or(crate::SurfaceError::Other(
                    "Parent surface is not available for buffer swap",
                ))?;
                parent_surface
                    .swap_buffers(gl.inner.context.current_context.as_ref().unwrap())
                    .map_err(|_| {
                        crate::SurfaceError::Other(
                            "Failed to swap buffers on parent glutin surface",
                        )
                    })?;
            }
        }

        Ok(())
    }

    pub fn supports_srgb(&self) -> bool {
        self.srgb_capable
    }

    fn create_swapchain(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
        gl: &AdapterContextLock<'_>,
    ) -> Result<SwapchainInner, crate::SurfaceError> {
        let format_desc = device.shared.describe_texture_format(config.format);

        let renderbuffer = unsafe { gl.create_renderbuffer() }.map_err(|error| {
            log::error!("Internal swapchain renderbuffer creation failed: {error}");
            crate::DeviceError::OutOfMemory
        })?;
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer)) };
        unsafe {
            gl.renderbuffer_storage(
                glow::RENDERBUFFER,
                format_desc.internal,
                config.extent.width as _,
                config.extent.height as _,
            )
        };

        let framebuffer = unsafe { gl.create_framebuffer() }.map_err(|error| {
            log::error!("Internal swapchain framebuffer creation failed: {error}");
            crate::DeviceError::OutOfMemory
        })?;
        unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer)) };
        unsafe {
            gl.framebuffer_renderbuffer(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            )
        };

        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        Ok(SwapchainInner {
            renderbuffer,
            framebuffer,
            extent: config.extent,
            format: config.format,
            format_desc,
            sample_type: wgt::TextureSampleType::Float { filterable: false },
        })
    }
}

impl crate::Surface for Surface {
    type A = super::Api;

    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let swapchain = match self.swapchain.write().take() {
            Some(Swapchain::Other(sc_surface, sc)) => {
                let gl = &device
                    .shared
                    .context
                    .lock_with_surface(&sc_surface)
                    .map_err(|_| {
                        crate::SurfaceError::Other(
                            "Failed to lock adapter context for swapchain re-configuration",
                        )
                    })?;
                unsafe { gl.delete_framebuffer(sc.framebuffer) };
                unsafe { gl.delete_renderbuffer(sc.renderbuffer) };

                sc_surface.resize(
                    gl.inner.context.current_context.as_ref().unwrap(),
                    unsafe { NonZeroU32::new_unchecked(config.extent.width) },
                    unsafe { NonZeroU32::new_unchecked(config.extent.height) },
                );

                self.create_swapchain(device, config, &gl)
                    .map(|sc_inner| Swapchain::Other(sc_surface, sc_inner))?
            }
            Some(Swapchain::Parent(sc)) => {
                let gl = &device.shared.context.lock();
                unsafe { gl.delete_framebuffer(sc.framebuffer) };
                unsafe { gl.delete_renderbuffer(sc.renderbuffer) };

                let surface = gl.inner.surface.lock();
                let parent_surface = surface.as_ref().ok_or(crate::SurfaceError::Other(
                    "Parent surface is not available for swapchain re-configuration",
                ))?;
                parent_surface.resize(
                    gl.inner.context.current_context.as_ref().unwrap(),
                    unsafe { NonZeroU32::new_unchecked(config.extent.width) },
                    unsafe { NonZeroU32::new_unchecked(config.extent.height) },
                );

                self.create_swapchain(device, config, &gl)
                    .map(Swapchain::Parent)?
            }
            None if self.window == self.parent => {
                let gl = &device.shared.context.lock();
                let surface = gl.inner.surface.lock();
                if let Some(surface) = surface.as_ref() {
                    surface.resize(
                        gl.inner.context.current_context.as_ref().unwrap(),
                        unsafe { NonZeroU32::new_unchecked(config.extent.width) },
                        unsafe { NonZeroU32::new_unchecked(config.extent.height) },
                    );
                } else {
                    return Err(crate::SurfaceError::Other(
                        "Parent surface is not available for swapchain configuration",
                    ));
                }
                self.create_swapchain(device, config, &gl)
                    .map(Swapchain::Parent)?
            }
            None => {
                let surface_attributes = GlutinWindowSurfaceAttributesBuilder::new()
                    .with_single_buffer(false)
                    // .with_srgb(config.format.describe().srgb)
                    .build(
                        self.window,
                        NonZeroU32::new(config.extent.width).unwrap(),
                        NonZeroU32::new(config.extent.height).unwrap(),
                    );

                let surface = unsafe {
                    self.display
                        .create_window_surface(&self.config, &surface_attributes)
                        // TODO: map error properly
                        .map_err(|_| {
                            crate::SurfaceError::Other("Failed to create glutin surface")
                        })?
                };

                let gl = &device
                    .shared
                    .context
                    .lock_with_surface(&surface)
                    .map_err(|_| {
                        crate::SurfaceError::Other(
                            "Failed to lock adapter context for swapchain configuration",
                        )
                    })?;

                self.create_swapchain(device, config, gl)
                    .map(|sc_inner| Swapchain::Other(surface, sc_inner))?
            }
        };

        let mut swapchain_write = self.swapchain.write();
        *swapchain_write = Some(swapchain);
        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        match self.swapchain.write().take() {
            Some(Swapchain::Other(sc_surface, sc)) => {
                let gl = &device
                    .shared
                    .context
                    .lock_with_surface(&sc_surface)
                    .unwrap();
                unsafe { gl.delete_framebuffer(sc.framebuffer) };
                unsafe { gl.delete_renderbuffer(sc.renderbuffer) };
            }
            Some(Swapchain::Parent(sc)) => {
                let gl = &device.shared.context.lock();
                unsafe { gl.delete_framebuffer(sc.framebuffer) };
                unsafe { gl.delete_renderbuffer(sc.renderbuffer) };
            }
            None => {}
        }
    }

    unsafe fn acquire_texture(
        &self,
        _timeout_ms: Option<Duration>,
        _fence: &super::Fence,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = match swapchain.as_ref().ok_or(crate::SurfaceError::Other(
            "Surface has no swap-chain configured",
        ))? {
            Swapchain::Other(_, sc) => sc,
            Swapchain::Parent(sc) => sc,
        };

        let texture = super::Texture {
            inner: super::TextureInner::Renderbuffer {
                raw: sc.renderbuffer,
            },
            drop_guard: None,
            array_layer_count: 1,
            mip_level_count: 1,
            format: sc.format,
            format_desc: sc.format_desc.clone(),
            copy_size: crate::CopyExtent {
                width: sc.extent.width,
                height: sc.extent.height,
                depth: 1,
            },
        };
        Ok(Some(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal: false,
        }))
    }

    unsafe fn discard_texture(&self, _texture: super::Texture) {}
}
