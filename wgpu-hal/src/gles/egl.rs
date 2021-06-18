use glow::HasContext;
use parking_lot::Mutex;

use std::{os::raw, ptr, sync::Arc};

const EGL_PLATFORM_WAYLAND_KHR: u32 = 0x31D8;
const EGL_PLATFORM_X11_KHR: u32 = 0x31D5;

type XOpenDisplayFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;

type WlDisplayConnectFun =
    unsafe extern "system" fn(display_name: *const raw::c_char) -> *mut raw::c_void;

type WlDisplayDisconnectFun = unsafe extern "system" fn(display: *const raw::c_void);

#[cfg(not(any(target_os = "android", target_os = "macos")))]
type WlEglWindowCreateFun = unsafe extern "system" fn(
    surface: *const raw::c_void,
    width: raw::c_int,
    height: raw::c_int,
) -> *mut raw::c_void;

type WlEglWindowResizeFun = unsafe extern "system" fn(
    window: *const raw::c_void,
    width: raw::c_int,
    height: raw::c_int,
    dx: raw::c_int,
    dy: raw::c_int,
);

type WlEglWindowDestroyFun = unsafe extern "system" fn(window: *const raw::c_void);

#[cfg(target_os = "android")]
extern "C" {
    pub fn ANativeWindow_setBuffersGeometry(
        window: *mut raw::c_void,
        width: i32,
        height: i32,
        format: i32,
    ) -> i32;
}

fn open_x_display() -> Option<(ptr::NonNull<raw::c_void>, libloading::Library)> {
    log::info!("Loading X11 library to get the current display");
    unsafe {
        let library = libloading::Library::new("libX11.so").ok()?;
        let func: libloading::Symbol<XOpenDisplayFun> = library.get(b"XOpenDisplay").unwrap();
        let result = func(ptr::null());
        ptr::NonNull::new(result).map(|ptr| (ptr, library))
    }
}

fn test_wayland_display() -> Option<libloading::Library> {
    /* We try to connect and disconnect here to simply ensure there
     * is an active wayland display available.
     */
    log::info!("Loading Wayland library to get the current display");
    let library = unsafe {
        let client_library = libloading::Library::new("libwayland-client.so").ok()?;
        let wl_display_connect: libloading::Symbol<WlDisplayConnectFun> =
            client_library.get(b"wl_display_connect").unwrap();
        let wl_display_disconnect: libloading::Symbol<WlDisplayDisconnectFun> =
            client_library.get(b"wl_display_disconnect").unwrap();
        let display = ptr::NonNull::new(wl_display_connect(ptr::null()))?;
        wl_display_disconnect(display.as_ptr());
        libloading::Library::new("libwayland-egl.so").ok()?
    };
    Some(library)
}

/// Choose GLES framebuffer configuration.
fn choose_config(
    egl: &egl::DynamicInstance<egl::EGL1_4>,
    display: egl::Display,
) -> Result<(egl::Config, bool), crate::InstanceError> {
    //TODO: EGL_SLOW_CONFIG
    let tiers = [
        (
            "off-screen",
            &[egl::RENDERABLE_TYPE, egl::OPENGL_ES2_BIT][..],
        ),
        ("presentation", &[egl::SURFACE_TYPE, egl::WINDOW_BIT]),
        #[cfg(not(target_os = "android"))]
        ("native-render", &[egl::NATIVE_RENDERABLE, egl::TRUE as _]),
    ];

    let mut attributes = Vec::with_capacity(7);
    for tier_max in (0..tiers.len()).rev() {
        let name = tiers[tier_max].0;
        log::info!("Trying {}", name);

        attributes.clear();
        for &(_, tier_attr) in tiers[..=tier_max].iter() {
            attributes.extend_from_slice(tier_attr);
        }
        attributes.push(egl::NONE);

        match egl.choose_first_config(display, &attributes) {
            Ok(Some(config)) => {
                return Ok((config, tier_max >= 1));
            }
            Ok(None) => {
                log::warn!("No config found!");
            }
            Err(e) => {
                log::error!("error in choose_first_config: {:?}", e);
            }
        }
    }

    Err(crate::InstanceError)
}

#[derive(Debug)]
struct Inner {
    egl: Arc<egl::DynamicInstance<egl::EGL1_4>>,
    version: (i32, i32),
    supports_native_window: bool,
    display: egl::Display,
    config: egl::Config,
    context: egl::Context,
    /// Dummy pbuffer (1x1).
    /// Required for `eglMakeCurrent` on platforms that doesn't supports `EGL_KHR_surfaceless_context`.
    pbuffer: Option<egl::Surface>,
    wl_display: Option<*mut raw::c_void>,
}

impl Inner {
    fn create(
        flags: crate::InstanceFlag,
        egl: Arc<egl::DynamicInstance<egl::EGL1_4>>,
        display: egl::Display,
        wsi_library: Option<&libloading::Library>,
    ) -> Result<Self, crate::InstanceError> {
        let version = egl.initialize(display).map_err(|_| crate::InstanceError)?;
        let vendor = egl.query_string(Some(display), egl::VENDOR).unwrap();
        let display_extensions = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .unwrap()
            .to_string_lossy();
        log::info!(
            "Display vendor {:?}, version {:?}, extensions: {:?}",
            vendor,
            version,
            display_extensions
        );

        if log::max_level() >= log::LevelFilter::Trace {
            log::trace!("Configurations:");
            let config_count = egl.get_config_count(display).unwrap();
            let mut configurations = Vec::with_capacity(config_count);
            egl.get_configs(display, &mut configurations).unwrap();
            for &config in configurations.iter() {
                log::trace!("\tCONFORMANT=0x{:X}, RENDERABLE=0x{:X}, NATIVE_RENDERABLE=0x{:X}, SURFACE_TYPE=0x{:X}",
                    egl.get_config_attrib(display, config, egl::CONFORMANT).unwrap(),
                    egl.get_config_attrib(display, config, egl::RENDERABLE_TYPE).unwrap(),
                    egl.get_config_attrib(display, config, egl::NATIVE_RENDERABLE).unwrap(),
                    egl.get_config_attrib(display, config, egl::SURFACE_TYPE).unwrap(),
                );
            }
        }

        let (config, supports_native_window) = choose_config(&egl, display)?;
        egl.bind_api(egl::OPENGL_ES_API).unwrap();

        //TODO: make it so `Device` == EGL Context
        let mut context_attributes = vec![
            egl::CONTEXT_CLIENT_VERSION,
            3, // Request GLES 3.0 or higher
        ];
        if flags.contains(crate::InstanceFlag::VALIDATION)
            && wsi_library.is_none()
            && !cfg!(target_os = "android")
        {
            //TODO: figure out why this is needed
            context_attributes.push(egl::CONTEXT_OPENGL_DEBUG);
            context_attributes.push(egl::TRUE as _);
        }
        context_attributes.push(egl::NONE);
        let context = match egl.create_context(display, config, None, &context_attributes) {
            Ok(context) => context,
            Err(e) => {
                log::warn!("unable to create GLES 3.x context: {:?}", e);
                return Err(crate::InstanceError);
            }
        };

        // Testing if context can be binded without surface
        // and creating dummy pbuffer surface if not.
        let pbuffer = if version < (1, 5)
            || !display_extensions.contains("EGL_KHR_surfaceless_context")
        {
            let attributes = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
            egl.create_pbuffer_surface(display, config, &attributes)
                .map(Some)
                .map_err(|e| {
                    log::warn!("Error in create_pbuffer_surface: {:?}", e);
                    crate::InstanceError
                })?
        } else {
            log::info!("EGL_KHR_surfaceless_context is present. No need to create a dummy pbuffer");
            None
        };

        Ok(Self {
            egl,
            display,
            version,
            supports_native_window,
            config,
            context,
            pbuffer,
            wl_display: None,
        })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Err(e) = self.egl.destroy_context(self.display, self.context) {
            log::warn!("Error in destroy_context: {:?}", e);
        }
        if let Err(e) = self.egl.terminate(self.display) {
            log::warn!("Error in terminate: {:?}", e);
        }
    }
}

pub struct Instance {
    wsi_library: Option<libloading::Library>,
    flags: crate::InstanceFlag,
    inner: Mutex<Inner>,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl crate::Instance<super::Api> for Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let egl = match unsafe { egl::DynamicInstance::<egl::EGL1_4>::load_required() } {
            Ok(egl) => Arc::new(egl),
            Err(e) => {
                log::warn!("Unable to open libEGL.so: {:?}", e);
                return Err(crate::InstanceError);
            }
        };

        let client_extensions = egl.query_string(None, egl::EXTENSIONS);

        let client_ext_str = match client_extensions {
            Ok(ext) => ext.to_string_lossy().into_owned(),
            Err(_) => String::new(),
        };
        log::info!("Client extensions: {:?}", client_ext_str);

        let mut wsi_library = None;

        let wayland_library = if client_ext_str.contains(&"EGL_EXT_platform_wayland") {
            test_wayland_display()
        } else {
            None
        };

        let x11_display_library = if client_ext_str.contains(&"EGL_EXT_platform_x11") {
            open_x_display()
        } else {
            None
        };

        let display = if let (Some(library), Some(egl)) =
            (wayland_library, egl.upcast::<egl::EGL1_5>())
        {
            log::info!("Using Wayland platform");
            let display_attributes = [egl::ATTRIB_NONE];
            wsi_library = Some(library);
            egl.get_platform_display(
                EGL_PLATFORM_WAYLAND_KHR,
                egl::DEFAULT_DISPLAY,
                &display_attributes,
            )
            .unwrap()
        } else if let (Some((display, library)), Some(egl)) =
            (x11_display_library, egl.upcast::<egl::EGL1_5>())
        {
            log::info!("Using X11 platform");
            let display_attributes = [egl::ATTRIB_NONE];
            wsi_library = Some(library);
            egl.get_platform_display(EGL_PLATFORM_X11_KHR, display.as_ptr(), &display_attributes)
                .unwrap()
        } else {
            log::info!("Using default platform");
            egl.get_display(egl::DEFAULT_DISPLAY).unwrap()
        };

        let inner = Inner::create(desc.flags, egl, display, wsi_library.as_ref())?;

        Ok(Instance {
            wsi_library,
            flags: desc.flags,
            inner: Mutex::new(inner),
        })
    }

    #[cfg_attr(target_os = "macos", allow(unused, unused_mut, unreachable_code))]
    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        use raw_window_handle::RawWindowHandle as Rwh;

        let mut inner = self.inner.lock();
        let mut wl_window = None;
        #[cfg(not(any(target_os = "android", target_os = "macos")))]
        let (mut temp_xlib_handle, mut temp_xcb_handle);

        let native_window_ptr = match has_handle.raw_window_handle() {
            #[cfg(not(any(target_os = "android", target_os = "macos")))]
            Rwh::Xlib(handle) => {
                temp_xlib_handle = handle.window;
                &mut temp_xlib_handle as *mut _ as *mut std::ffi::c_void
            }
            #[cfg(not(any(target_os = "android", target_os = "macos")))]
            Rwh::Xcb(handle) => {
                temp_xcb_handle = handle.window;
                &mut temp_xcb_handle as *mut _ as *mut std::ffi::c_void
            }
            #[cfg(target_os = "android")]
            Rwh::Android(handle) => handle.a_native_window as *mut _ as *mut std::ffi::c_void,
            #[cfg(not(any(target_os = "android", target_os = "macos")))]
            Rwh::Wayland(handle) => {
                /* Wayland displays are not sharable between surfaces so if the
                 * surface we receive from this handle is from a different
                 * display, we must re-initialize the context.
                 *
                 * See gfx-rs/gfx#3545
                 */
                if inner
                    .wl_display
                    .map(|ptr| ptr != handle.display)
                    .unwrap_or(true)
                {
                    use std::ops::DerefMut;
                    let display_attributes = [egl::ATTRIB_NONE];
                    let display = inner
                        .egl
                        .upcast::<egl::EGL1_5>()
                        .unwrap()
                        .get_platform_display(
                            EGL_PLATFORM_WAYLAND_KHR,
                            handle.display,
                            &display_attributes,
                        )
                        .unwrap();

                    let new_inner =
                        Inner::create(inner.egl.clone(), display, self.wsi_library.as_ref())
                            .map_err(|_| w::InitError::UnsupportedWindowHandle)?;

                    let old_inner = std::mem::replace(inner.deref_mut(), new_inner);
                    inner.wl_display = Some(handle.display);
                    drop(old_inner);
                }

                let wl_egl_window_create: libloading::Symbol<WlEglWindowCreateFun> = self
                    .wsi_library
                    .as_ref()
                    .expect("unsupported window")
                    .get(b"wl_egl_window_create")
                    .unwrap();
                let result = wl_egl_window_create(handle.surface, 640, 480) as *mut _
                    as *mut std::ffi::c_void;
                wl_window = Some(result);
                result
            }
            other => {
                log::error!("Unsupported window: {:?}", other);
                return Err(crate::InstanceError);
            }
        };

        let mut attributes = vec![
            egl::RENDER_BUFFER as usize,
            if cfg!(target_os = "android") {
                egl::BACK_BUFFER as usize
            } else {
                egl::SINGLE_BUFFER as usize
            },
        ];
        if inner.version >= (1, 5) {
            // Always enable sRGB in EGL 1.5
            attributes.push(egl::GL_COLORSPACE as usize);
            attributes.push(egl::GL_COLORSPACE_SRGB as usize);
        }
        attributes.push(egl::ATTRIB_NONE);

        let raw = if let Some(egl) = inner.egl.upcast::<egl::EGL1_5>() {
            egl.create_platform_window_surface(
                inner.display,
                inner.config,
                native_window_ptr,
                &attributes,
            )
            .map_err(|e| {
                log::warn!("Error in create_platform_window_surface: {:?}", e);
                crate::InstanceError
            })
        } else {
            let attributes_i32: Vec<i32> = attributes.iter().map(|a| (*a as i32).into()).collect();
            inner
                .egl
                .create_window_surface(
                    inner.display,
                    inner.config,
                    native_window_ptr,
                    Some(&attributes_i32),
                )
                .map_err(|e| {
                    log::warn!("Error in create_platform_window_surface: {:?}", e);
                    crate::InstanceError
                })
        }?;

        #[cfg(target_os = "android")]
        {
            let format = inner
                .egl
                .get_config_attrib(inner.display, inner.config, egl::NATIVE_VISUAL_ID)
                .unwrap();

            let ret = ANativeWindow_setBuffersGeometry(native_window_ptr, 0, 0, format);

            if ret != 0 {
                log::error!("Error returned from ANativeWindow_setBuffersGeometry");
                return Err(w::InitError::UnsupportedWindowHandle);
            }
        }

        Ok(Surface {
            egl: Arc::clone(&inner.egl),
            raw,
            display: inner.display,
            context: inner.context,
            presentable: inner.supports_native_window,
            pbuffer: inner.pbuffer,
            wl_window,
            swapchain: None,
        })
    }
    unsafe fn destroy_surface(&self, surface: Surface) {}

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Swapchain {
    framebuffer: glow::Framebuffer,
    renderbuffer: glow::Renderbuffer,
    /// Extent because the window lies
    extent: wgt::Extent3d,
    format: super::TextureFormat,
    sample_type: wgt::TextureSampleType,
}

#[derive(Debug)]
pub struct Surface {
    egl: Arc<egl::DynamicInstance<egl::EGL1_4>>,
    raw: egl::Surface,
    display: egl::Display,
    context: egl::Context,
    pbuffer: Option<egl::Surface>,
    presentable: bool,
    wl_window: Option<*mut raw::c_void>,
    swapchain: Option<Swapchain>,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl crate::Surface<super::Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &super::Context,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        self.unconfigure(device);

        if let Some(window) = self.wl_window {
            let library = libloading::Library::new("libwayland-egl.so").unwrap();
            let wl_egl_window_resize: libloading::Symbol<WlEglWindowResizeFun> =
                library.get(b"wl_egl_window_resize").unwrap();
            wl_egl_window_resize(
                window,
                config.extent.width as i32,
                config.extent.height as i32,
                0,
                0,
            );
        }

        //let desc = conv::describe_format(config.format).unwrap();
        let desc: super::FormatDescription = unimplemented!();

        let gl: glow::Context = unimplemented!(); //&device.share.context;
        let renderbuffer = gl.create_renderbuffer().unwrap();
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer));
        gl.renderbuffer_storage(
            glow::RENDERBUFFER,
            desc.tex_internal,
            config.extent.width as _,
            config.extent.height as _,
        );
        let framebuffer = gl.create_framebuffer().unwrap();
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_renderbuffer(
            glow::READ_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::RENDERBUFFER,
            Some(renderbuffer),
        );
        gl.bind_renderbuffer(glow::RENDERBUFFER, None);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);

        self.swapchain = Some(Swapchain {
            renderbuffer,
            framebuffer,
            extent: config.extent,
            format: desc.tex_internal,
            sample_type: wgt::TextureSampleType::Float { filterable: false },
        });

        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &super::Context) {
        /*
        let gl = &device.share.context;
        if let Some(sc) = self.swapchain.take() {
            gl.delete_renderbuffer(sc.renderbuffer);
            gl.delete_framebuffer(sc.framebuffer);
        }*/
    }

    unsafe fn acquire_texture(
        &mut self,
        timeout_ms: u32,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let sc = self.swapchain.as_ref().unwrap();
        //let sc_image =
        //    native::SwapchainImage::new(sc.renderbuffer, sc.format, sc.extent, sc.channel);
        Ok(None)
    }
    unsafe fn discard_texture(&mut self, texture: super::Resource) {}
}
