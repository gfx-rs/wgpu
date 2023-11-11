use glow::HasContext;
use glutin_wgl_sys::wgl_extra::{
    Wgl, CONTEXT_CORE_PROFILE_BIT_ARB, CONTEXT_DEBUG_BIT_ARB, CONTEXT_FLAGS_ARB,
    CONTEXT_PROFILE_MASK_ARB,
};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::{
    collections::HashSet,
    ffi::{c_void, CStr, CString},
    io::Error,
    mem,
    os::raw::c_int,
    ptr,
    sync::{
        mpsc::{sync_channel, SyncSender},
        Arc,
    },
    thread,
    time::Duration,
};
use wgt::InstanceFlags;
use winapi::{
    shared::{
        minwindef::{FALSE, HMODULE, LPARAM, LRESULT, UINT, WPARAM},
        windef::{HDC, HGLRC, HWND},
    },
    um::{
        libloaderapi::{GetModuleHandleA, GetProcAddress, LoadLibraryA},
        wingdi::{
            wglCreateContext, wglDeleteContext, wglGetCurrentContext, wglGetProcAddress,
            wglMakeCurrent, ChoosePixelFormat, DescribePixelFormat, GetPixelFormat, SetPixelFormat,
            SwapBuffers, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_SUPPORT_OPENGL, PFD_TYPE_RGBA,
            PIXELFORMATDESCRIPTOR,
        },
        winuser::{
            CreateWindowExA, DefWindowProcA, DestroyWindow, GetDC, RegisterClassExA, ReleaseDC,
            CS_OWNDC, WNDCLASSEXA,
        },
    },
};

/// The amount of time to wait while trying to obtain a lock to the adapter context
const CONTEXT_LOCK_TIMEOUT_SECS: u64 = 1;

/// A wrapper around a `[`glow::Context`]` and the required WGL context that uses locking to
/// guarantee exclusive access when shared with multiple threads.
pub struct AdapterContext {
    inner: Arc<Mutex<Inner>>,
}

unsafe impl Sync for AdapterContext {}
unsafe impl Send for AdapterContext {}

impl AdapterContext {
    pub fn is_owned(&self) -> bool {
        true
    }

    pub fn raw_context(&self) -> *mut c_void {
        self.inner.lock().context.context as *mut _
    }

    /// Obtain a lock to the WGL context and get handle to the [`glow::Context`] that can be used to
    /// do rendering.
    #[track_caller]
    pub fn lock(&self) -> AdapterContextLock<'_> {
        let inner = self
            .inner
            // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
            // deadlock and should panic to show where we got stuck
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
            .expect("Could not lock adapter context. This is most-likely a deadlock.");

        inner.context.make_current(inner.device.dc).unwrap();

        AdapterContextLock { inner }
    }
}

/// A guard containing a lock to an [`AdapterContext`]
pub struct AdapterContextLock<'a> {
    inner: MutexGuard<'a, Inner>,
}

impl<'a> std::ops::Deref for AdapterContextLock<'a> {
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

struct WglContext {
    context: HGLRC,
}

impl WglContext {
    fn make_current(&self, device: HDC) -> Result<(), Error> {
        if unsafe { wglMakeCurrent(device, self.context) } == FALSE {
            Err(Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn unmake_current(&self) -> Result<(), Error> {
        if unsafe { wglGetCurrentContext().is_null() } {
            return Ok(());
        }
        if unsafe { wglMakeCurrent(ptr::null_mut(), ptr::null_mut()) } == FALSE {
            Err(Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

impl Drop for WglContext {
    fn drop(&mut self) {
        unsafe {
            if wglDeleteContext(self.context) == FALSE {
                log::error!("failed to delete WGL context {}", Error::last_os_error());
            }
        };
    }
}

unsafe impl Send for WglContext {}
unsafe impl Sync for WglContext {}

struct Inner {
    gl: glow::Context,
    device: InstanceDevice,
    context: WglContext,
}

pub struct Instance {
    srgb_capable: bool,
    inner: Arc<Mutex<Inner>>,
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

fn load_gl_func(name: &str, module: Option<HMODULE>) -> *const c_void {
    let addr = CString::new(name.as_bytes()).unwrap();
    let mut ptr = unsafe { wglGetProcAddress(addr.as_ptr()) };
    if ptr.is_null() {
        if let Some(module) = module {
            ptr = unsafe { GetProcAddress(module, addr.as_ptr()) };
        }
    }
    ptr.cast()
}

fn extensions(extra: &Wgl, dc: HDC) -> HashSet<String> {
    if extra.GetExtensionsStringARB.is_loaded() {
        unsafe { CStr::from_ptr(extra.GetExtensionsStringARB(dc as *const _)) }
            .to_str()
            .unwrap_or("")
    } else {
        ""
    }
    .split(' ')
    .map(|s| s.to_owned())
    .collect()
}

unsafe fn setup_pixel_format(dc: HDC) -> Result<(), crate::InstanceError> {
    let mut format: PIXELFORMATDESCRIPTOR = unsafe { mem::zeroed() };
    format.nVersion = 1;
    format.nSize = mem::size_of_val(&format) as u16;
    format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    format.iPixelType = PFD_TYPE_RGBA;
    format.cColorBits = 8;

    let index = unsafe { ChoosePixelFormat(dc, &format) };
    if index == 0 {
        return Err(crate::InstanceError::with_source(
            String::from("unable to choose pixel format"),
            Error::last_os_error(),
        ));
    }

    let current = unsafe { GetPixelFormat(dc) };

    if index != current && unsafe { SetPixelFormat(dc, index, &format) } == FALSE {
        return Err(crate::InstanceError::with_source(
            String::from("unable to set pixel format"),
            Error::last_os_error(),
        ));
    }

    let index = unsafe { GetPixelFormat(dc) };
    if index == 0 {
        return Err(crate::InstanceError::with_source(
            String::from("unable to get pixel format index"),
            Error::last_os_error(),
        ));
    }
    if unsafe { DescribePixelFormat(dc, index, mem::size_of_val(&format) as UINT, &mut format) }
        == 0
    {
        return Err(crate::InstanceError::with_source(
            String::from("unable to read pixel format"),
            Error::last_os_error(),
        ));
    }

    if format.dwFlags & PFD_SUPPORT_OPENGL == 0 || format.iPixelType != PFD_TYPE_RGBA {
        return Err(crate::InstanceError::new(String::from(
            "unsuitable pixel format",
        )));
    }
    Ok(())
}

fn create_global_window_class() -> Result<CString, crate::InstanceError> {
    let instance = unsafe { GetModuleHandleA(ptr::null()) };
    if instance.is_null() {
        return Err(crate::InstanceError::with_source(
            String::from("unable to get executable instance"),
            Error::last_os_error(),
        ));
    }

    // Use the address of `UNIQUE` as part of the window class name to ensure different
    // `wgpu` versions use different names.
    static UNIQUE: Mutex<u8> = Mutex::new(0);
    let class_addr: *const _ = &UNIQUE;
    let name = format!("wgpu Device Class {:x}\0", class_addr as usize);
    let name = CString::from_vec_with_nul(name.into_bytes()).unwrap();

    // Use a wrapper function for compatibility with `windows-rs`.
    unsafe extern "system" fn wnd_proc(
        window: HWND,
        msg: UINT,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        unsafe { DefWindowProcA(window, msg, wparam, lparam) }
    }

    let window_class = WNDCLASSEXA {
        cbSize: mem::size_of::<WNDCLASSEXA>() as u32,
        style: CS_OWNDC,
        lpfnWndProc: Some(wnd_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: instance,
        hIcon: ptr::null_mut(),
        hCursor: ptr::null_mut(),
        hbrBackground: ptr::null_mut(),
        lpszMenuName: ptr::null_mut(),
        lpszClassName: name.as_ptr(),
        hIconSm: ptr::null_mut(),
    };

    let atom = unsafe { RegisterClassExA(&window_class) };

    if atom == 0 {
        return Err(crate::InstanceError::with_source(
            String::from("unable to register window class"),
            Error::last_os_error(),
        ));
    }

    // We intentionally leak the window class as we only need one per process.

    Ok(name)
}

fn get_global_window_class() -> Result<CString, crate::InstanceError> {
    static GLOBAL: Lazy<Result<CString, crate::InstanceError>> =
        Lazy::new(create_global_window_class);
    GLOBAL.clone()
}

struct InstanceDevice {
    dc: HDC,

    /// This is used to keep the thread owning `dc` alive until this struct is dropped.
    _tx: SyncSender<()>,
}

fn create_instance_device() -> Result<InstanceDevice, crate::InstanceError> {
    #[derive(Clone, Copy)]
    struct SendDc(HDC);
    unsafe impl Sync for SendDc {}
    unsafe impl Send for SendDc {}

    struct Window {
        window: HWND,
    }
    impl Drop for Window {
        fn drop(&mut self) {
            unsafe {
                if DestroyWindow(self.window) == FALSE {
                    log::error!("failed to destroy window {}", Error::last_os_error());
                }
            };
        }
    }
    struct DeviceContextHandle {
        dc: HDC,
        window: HWND,
    }
    impl Drop for DeviceContextHandle {
        fn drop(&mut self) {
            unsafe {
                ReleaseDC(self.window, self.dc);
            };
        }
    }

    let window_class = get_global_window_class()?;

    let (drop_tx, drop_rx) = sync_channel(0);
    let (setup_tx, setup_rx) = sync_channel(0);

    // We spawn a thread which owns the hidden window for this instance.
    thread::Builder::new()
        .stack_size(256 * 1024)
        .name("wgpu-hal WGL Instance Thread".to_owned())
        .spawn(move || {
            let setup = (|| {
                let instance = unsafe { GetModuleHandleA(ptr::null()) };
                if instance.is_null() {
                    return Err(crate::InstanceError::with_source(
                        String::from("unable to get executable instance"),
                        Error::last_os_error(),
                    ));
                }

                // Create a hidden window since we don't pass `WS_VISIBLE`.
                let window = unsafe {
                    CreateWindowExA(
                        0,
                        window_class.as_ptr(),
                        window_class.as_ptr(),
                        0,
                        0,
                        0,
                        1,
                        1,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        instance,
                        ptr::null_mut(),
                    )
                };
                if window.is_null() {
                    return Err(crate::InstanceError::with_source(
                        String::from("unable to create hidden instance window"),
                        Error::last_os_error(),
                    ));
                }
                let window = Window { window };

                let dc = unsafe { GetDC(window.window) };
                if dc.is_null() {
                    return Err(crate::InstanceError::with_source(
                        String::from("unable to create memory device"),
                        Error::last_os_error(),
                    ));
                }
                let dc = DeviceContextHandle {
                    dc,
                    window: window.window,
                };
                unsafe { setup_pixel_format(dc.dc)? };

                Ok((window, dc))
            })();

            match setup {
                Ok((_window, dc)) => {
                    setup_tx.send(Ok(SendDc(dc.dc))).unwrap();
                    // Wait for the shutdown event to free the window and device context handle.
                    drop_rx.recv().ok();
                }
                Err(err) => {
                    setup_tx.send(Err(err)).unwrap();
                }
            }
        })
        .map_err(|e| {
            crate::InstanceError::with_source(String::from("unable to create instance thread"), e)
        })?;

    let dc = setup_rx.recv().unwrap()?.0;

    Ok(InstanceDevice { dc, _tx: drop_tx })
}

impl crate::Instance<super::Api> for Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init OpenGL (WGL) Backend");
        let opengl_module = unsafe { LoadLibraryA("opengl32.dll\0".as_ptr() as *const _) };
        if opengl_module.is_null() {
            return Err(crate::InstanceError::with_source(
                String::from("unable to load the OpenGL library"),
                Error::last_os_error(),
            ));
        }

        let device = create_instance_device()?;
        let dc = device.dc;

        let context = unsafe { wglCreateContext(dc) };
        if context.is_null() {
            return Err(crate::InstanceError::with_source(
                String::from("unable to create initial OpenGL context"),
                Error::last_os_error(),
            ));
        }
        let context = WglContext { context };
        context.make_current(dc).map_err(|e| {
            crate::InstanceError::with_source(
                String::from("unable to set initial OpenGL context as current"),
                e,
            )
        })?;

        let extra = Wgl::load_with(|name| load_gl_func(name, None));
        let extentions = extensions(&extra, dc);

        let can_use_profile = extentions.contains("WGL_ARB_create_context_profile")
            && extra.CreateContextAttribsARB.is_loaded();

        let context = if can_use_profile {
            let attributes = [
                CONTEXT_PROFILE_MASK_ARB as c_int,
                CONTEXT_CORE_PROFILE_BIT_ARB as c_int,
                CONTEXT_FLAGS_ARB as c_int,
                if desc.flags.contains(InstanceFlags::DEBUG) {
                    CONTEXT_DEBUG_BIT_ARB as c_int
                } else {
                    0
                },
                0, // End of list
            ];
            let context = unsafe {
                extra.CreateContextAttribsARB(dc as *const _, ptr::null(), attributes.as_ptr())
            };
            if context.is_null() {
                return Err(crate::InstanceError::with_source(
                    String::from("unable to create OpenGL context"),
                    Error::last_os_error(),
                ));
            }
            WglContext {
                context: context as *mut _,
            }
        } else {
            context
        };

        context.make_current(dc).map_err(|e| {
            crate::InstanceError::with_source(
                String::from("unable to set OpenGL context as current"),
                e,
            )
        })?;

        let mut gl = unsafe {
            glow::Context::from_loader_function(|name| load_gl_func(name, Some(opengl_module)))
        };

        let extra = Wgl::load_with(|name| load_gl_func(name, None));
        let extentions = extensions(&extra, dc);

        let srgb_capable = extentions.contains("WGL_EXT_framebuffer_sRGB")
            || extentions.contains("WGL_ARB_framebuffer_sRGB")
            || gl
                .supported_extensions()
                .contains("GL_ARB_framebuffer_sRGB");

        if srgb_capable {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        if desc.flags.contains(InstanceFlags::VALIDATION) && gl.supports_debug() {
            log::debug!("Enabling GL debug output");
            unsafe { gl.enable(glow::DEBUG_OUTPUT) };
            unsafe { gl.debug_message_callback(super::gl_debug_message_callback) };
        }

        context.unmake_current().map_err(|e| {
            crate::InstanceError::with_source(
                String::from("unable to unset the current WGL context"),
                e,
            )
        })?;

        Ok(Instance {
            inner: Arc::new(Mutex::new(Inner {
                device,
                gl,
                context,
            })),
            srgb_capable,
        })
    }

    #[cfg_attr(target_os = "macos", allow(unused, unused_mut, unreachable_code))]
    unsafe fn create_surface(
        &self,
        _display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        let window = if let RawWindowHandle::Win32(handle) = window_handle {
            handle
        } else {
            return Err(crate::InstanceError::new(format!(
                "unsupported window: {window_handle:?}"
            )));
        };
        Ok(Surface {
            window: window.hwnd.get() as *mut _,
            presentable: true,
            swapchain: None,
            srgb_capable: self.srgb_capable,
        })
    }
    unsafe fn destroy_surface(&self, _surface: Surface) {}

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        unsafe {
            super::Adapter::expose(AdapterContext {
                inner: self.inner.clone(),
            })
        }
        .into_iter()
        .collect()
    }
}

struct DeviceContextHandle {
    device: HDC,
    window: HWND,
}

impl Drop for DeviceContextHandle {
    fn drop(&mut self) {
        unsafe {
            ReleaseDC(self.window, self.device);
        };
    }
}

pub struct Swapchain {
    framebuffer: glow::Framebuffer,
    renderbuffer: glow::Renderbuffer,

    /// Extent because the window lies
    extent: wgt::Extent3d,

    format: wgt::TextureFormat,
    format_desc: super::TextureFormatDesc,
    #[allow(unused)]
    sample_type: wgt::TextureSampleType,
}

pub struct Surface {
    window: HWND,
    pub(super) presentable: bool,
    swapchain: Option<Swapchain>,
    srgb_capable: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub(super) unsafe fn present(
        &mut self,
        _suf_texture: super::Texture,
        context: &AdapterContext,
    ) -> Result<(), crate::SurfaceError> {
        let sc = self.swapchain.as_ref().unwrap();
        let dc = unsafe { GetDC(self.window) };
        if dc.is_null() {
            log::error!(
                "unable to get the device context from window: {}",
                Error::last_os_error()
            );
            return Err(crate::SurfaceError::Other(
                "unable to get the device context from window",
            ));
        }
        let dc = DeviceContextHandle {
            device: dc,
            window: self.window,
        };

        let inner = context.inner.lock();

        if let Err(e) = inner.context.make_current(dc.device) {
            log::error!("unable to make the OpenGL context current for surface: {e}",);
            return Err(crate::SurfaceError::Other(
                "unable to make the OpenGL context current for surface",
            ));
        }

        let gl = &inner.gl;

        unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sc.framebuffer)) };

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
                sc.extent.height as i32,
                sc.extent.width as i32,
                0,
                0,
                0,
                sc.extent.width as i32,
                sc.extent.height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            )
        };

        if self.srgb_capable {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        if unsafe { SwapBuffers(dc.device) } == FALSE {
            log::error!("unable to swap buffers: {}", Error::last_os_error());
            return Err(crate::SurfaceError::Other("unable to swap buffers"));
        }

        Ok(())
    }

    pub fn supports_srgb(&self) -> bool {
        self.srgb_capable
    }
}

impl crate::Surface<super::Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        // Remove the old configuration.
        unsafe { self.unconfigure(device) };

        let dc = unsafe { GetDC(self.window) };
        if dc.is_null() {
            log::error!(
                "unable to get the device context from window: {}",
                Error::last_os_error()
            );
            return Err(crate::SurfaceError::Other(
                "unable to get the device context from window",
            ));
        }
        let dc = DeviceContextHandle {
            device: dc,
            window: self.window,
        };

        if let Err(e) = unsafe { setup_pixel_format(dc.device) } {
            log::error!("unable to setup surface pixel format: {e}",);
            return Err(crate::SurfaceError::Other(
                "unable to setup surface pixel format",
            ));
        }

        let format_desc = device.shared.describe_texture_format(config.format);
        let inner = &device.shared.context.inner.lock();

        if let Err(e) = inner.context.make_current(dc.device) {
            log::error!("unable to make the OpenGL context current for surface: {e}",);
            return Err(crate::SurfaceError::Other(
                "unable to make the OpenGL context current for surface",
            ));
        }

        let gl = &inner.gl;
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
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer)) };
        unsafe {
            gl.framebuffer_renderbuffer(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            )
        };
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        // Setup presentation mode
        let extra = Wgl::load_with(|name| load_gl_func(name, None));
        let extentions = extensions(&extra, dc.device);
        if !(extentions.contains("WGL_EXT_swap_control") && extra.SwapIntervalEXT.is_loaded()) {
            log::error!("WGL_EXT_swap_control is unsupported");
            return Err(crate::SurfaceError::Other(
                "WGL_EXT_swap_control is unsupported",
            ));
        }

        let vsync = match config.present_mode {
            wgt::PresentMode::Mailbox => false,
            wgt::PresentMode::Fifo => true,
            _ => {
                log::error!("unsupported present mode: {:?}", config.present_mode);
                return Err(crate::SurfaceError::Other("unsupported present mode"));
            }
        };

        if unsafe { extra.SwapIntervalEXT(if vsync { 1 } else { 0 }) } == FALSE {
            log::error!("unable to set swap interval: {}", Error::last_os_error());
            return Err(crate::SurfaceError::Other("unable to set swap interval"));
        }

        self.swapchain = Some(Swapchain {
            renderbuffer,
            framebuffer,
            extent: config.extent,
            format: config.format,
            format_desc,
            sample_type: wgt::TextureSampleType::Float { filterable: false },
        });

        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &super::Device) {
        let gl = &device.shared.context.lock();
        if let Some(sc) = self.swapchain.take() {
            unsafe {
                gl.delete_renderbuffer(sc.renderbuffer);
                gl.delete_framebuffer(sc.framebuffer)
            };
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        _timeout_ms: Option<Duration>,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let sc = self.swapchain.as_ref().unwrap();
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
    unsafe fn discard_texture(&mut self, _texture: super::Texture) {}
}
