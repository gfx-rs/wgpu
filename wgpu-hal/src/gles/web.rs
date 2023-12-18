use glow::HasContext;
use parking_lot::{Mutex, RwLock};
use wasm_bindgen::{JsCast, JsValue};

use super::TextureFormatDesc;

/// A wrapper around a [`glow::Context`] to provide a fake `lock()` api that makes it compatible
/// with the `AdapterContext` API from the EGL implementation.
pub struct AdapterContext {
    pub glow_context: glow::Context,
}

impl AdapterContext {
    pub fn is_owned(&self) -> bool {
        false
    }

    /// Obtain a lock to the EGL context and get handle to the [`glow::Context`] that can be used to
    /// do rendering.
    #[track_caller]
    pub fn lock(&self) -> &glow::Context {
        &self.glow_context
    }
}

#[derive(Debug)]
pub struct Instance {
    /// Set when a canvas is provided, and used to implement [`Instance::enumerate_adapters()`].
    webgl2_context: Mutex<Option<web_sys::WebGl2RenderingContext>>,
}

impl Instance {
    pub fn create_surface_from_canvas(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<Surface, crate::InstanceError> {
        let result =
            canvas.get_context_with_context_options("webgl2", &Self::create_context_options());
        self.create_surface_from_context(Canvas::Canvas(canvas), result)
    }

    pub fn create_surface_from_offscreen_canvas(
        &self,
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<Surface, crate::InstanceError> {
        let result =
            canvas.get_context_with_context_options("webgl2", &Self::create_context_options());
        self.create_surface_from_context(Canvas::Offscreen(canvas), result)
    }

    /// Common portion of public `create_surface_from_*` functions.
    ///
    /// Note: Analogous code also exists in the WebGPU backend at
    /// `wgpu::backend::web::Context`.
    fn create_surface_from_context(
        &self,
        canvas: Canvas,
        context_result: Result<Option<js_sys::Object>, JsValue>,
    ) -> Result<Surface, crate::InstanceError> {
        let context_object: js_sys::Object = match context_result {
            Ok(Some(context)) => context,
            Ok(None) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext-dev>
                // A getContext() call “returns null if contextId is not supported, or if the
                // canvas has already been initialized with another context type”. Additionally,
                // “not supported” could include “insufficient GPU resources” or “the GPU process
                // previously crashed”. So, we must return it as an `Err` since it could occur
                // for circumstances outside the application author's control.
                return Err(crate::InstanceError::new(String::from(
                    "canvas.getContext() returned null; webgl2 not available or canvas already in use"
                )));
            }
            Err(js_error) => {
                // <https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext>
                // A thrown exception indicates misuse of the canvas state.
                return Err(crate::InstanceError::new(format!(
                    "canvas.getContext() threw exception {js_error:?}",
                )));
            }
        };

        // Not returning this error because it is a type error that shouldn't happen unless
        // the browser, JS builtin objects, or wasm bindings are misbehaving somehow.
        let webgl2_context: web_sys::WebGl2RenderingContext = context_object
            .dyn_into()
            .expect("canvas context is not a WebGl2RenderingContext");

        // It is not inconsistent to overwrite an existing context, because the only thing that
        // `self.webgl2_context` is used for is producing the response to `enumerate_adapters()`.
        *self.webgl2_context.lock() = Some(webgl2_context.clone());

        Ok(Surface {
            canvas,
            webgl2_context,
            srgb_present_program: Mutex::new(None),
            swapchain: RwLock::new(None),
            texture: Mutex::new(None),
            presentable: true,
        })
    }

    fn create_context_options() -> js_sys::Object {
        let context_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &context_options,
            &"antialias".into(),
            &wasm_bindgen::JsValue::FALSE,
        )
        .expect("Cannot create context options");
        context_options
    }
}

#[cfg(all(
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Sync for Instance {}
#[cfg(all(
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Send for Instance {}

impl crate::Instance<super::Api> for Instance {
    unsafe fn init(_desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init OpenGL (WebGL) Backend");
        Ok(Instance {
            webgl2_context: Mutex::new(None),
        })
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        let context_guard = self.webgl2_context.lock();
        let gl = match *context_guard {
            Some(ref webgl2_context) => glow::Context::from_webgl2_context(webgl2_context.clone()),
            None => return Vec::new(),
        };

        unsafe { super::Adapter::expose(AdapterContext { glow_context: gl }) }
            .into_iter()
            .collect()
    }

    unsafe fn create_surface(
        &self,
        _display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        let canvas: web_sys::HtmlCanvasElement = match window_handle {
            raw_window_handle::RawWindowHandle::Web(handle) => web_sys::window()
                .and_then(|win| win.document())
                .expect("Cannot get document")
                .query_selector(&format!("canvas[data-raw-handle=\"{}\"]", handle.id))
                .expect("Cannot query for canvas")
                .expect("Canvas is not found")
                .dyn_into()
                .expect("Failed to downcast to canvas type"),
            raw_window_handle::RawWindowHandle::WebCanvas(handle) => {
                let value: &JsValue = unsafe { handle.obj.cast().as_ref() };
                value.clone().unchecked_into()
            }
            raw_window_handle::RawWindowHandle::WebOffscreenCanvas(handle) => {
                let value: &JsValue = unsafe { handle.obj.cast().as_ref() };
                let canvas: web_sys::OffscreenCanvas = value.clone().unchecked_into();

                return self.create_surface_from_offscreen_canvas(canvas);
            }
            _ => {
                return Err(crate::InstanceError::new(format!(
                    "window handle {window_handle:?} is not a web handle"
                )))
            }
        };

        self.create_surface_from_canvas(canvas)
    }

    unsafe fn destroy_surface(&self, surface: Surface) {
        let mut context_option_ref = self.webgl2_context.lock();

        if let Some(context) = context_option_ref.as_ref() {
            if context == &surface.webgl2_context {
                *context_option_ref = None;
            }
        }
    }
}

#[derive(Debug)]
pub struct Surface {
    canvas: Canvas,
    webgl2_context: web_sys::WebGl2RenderingContext,
    pub(super) swapchain: RwLock<Option<Swapchain>>,
    texture: Mutex<Option<glow::Texture>>,
    pub(super) presentable: bool,
    srgb_present_program: Mutex<Option<glow::Program>>,
}

impl Clone for Surface {
    fn clone(&self) -> Self {
        Self {
            canvas: self.canvas.clone(),
            webgl2_context: self.webgl2_context.clone(),
            swapchain: RwLock::new(self.swapchain.read().clone()),
            texture: Mutex::new(*self.texture.lock()),
            presentable: self.presentable,
            srgb_present_program: Mutex::new(*self.srgb_present_program.lock()),
        }
    }
}

#[cfg(all(
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Sync for Surface {}
#[cfg(all(
    feature = "fragile-send-sync-non-atomic-wasm",
    not(target_feature = "atomics")
))]
unsafe impl Send for Surface {}

#[derive(Clone, Debug)]
enum Canvas {
    Canvas(web_sys::HtmlCanvasElement),
    Offscreen(web_sys::OffscreenCanvas),
}

#[derive(Clone, Debug)]
pub struct Swapchain {
    pub(crate) extent: wgt::Extent3d,
    // pub(crate) channel: f::ChannelType,
    pub(super) format: wgt::TextureFormat,
    pub(super) framebuffer: glow::Framebuffer,
    pub(super) format_desc: TextureFormatDesc,
}

impl Surface {
    pub(super) unsafe fn present(
        &self,
        _suf_texture: super::Texture,
        context: &AdapterContext,
    ) -> Result<(), crate::SurfaceError> {
        let gl = &context.glow_context;
        let swapchain = self.swapchain.read();
        let swapchain = swapchain.as_ref().ok_or(crate::SurfaceError::Other(
            "need to configure surface before presenting",
        ))?;

        if swapchain.format.is_srgb() {
            // Important to set the viewport since we don't know in what state the user left it.
            unsafe {
                gl.viewport(
                    0,
                    0,
                    swapchain.extent.width as _,
                    swapchain.extent.height as _,
                )
            };
            unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
            unsafe { gl.bind_sampler(0, None) };
            unsafe { gl.active_texture(glow::TEXTURE0) };
            unsafe { gl.bind_texture(glow::TEXTURE_2D, *self.texture.lock()) };
            unsafe { gl.use_program(*self.srgb_present_program.lock()) };
            unsafe { gl.disable(glow::DEPTH_TEST) };
            unsafe { gl.disable(glow::STENCIL_TEST) };
            unsafe { gl.disable(glow::SCISSOR_TEST) };
            unsafe { gl.disable(glow::BLEND) };
            unsafe { gl.disable(glow::CULL_FACE) };
            unsafe { gl.draw_buffers(&[glow::BACK]) };
            unsafe { gl.draw_arrays(glow::TRIANGLES, 0, 3) };
        } else {
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(swapchain.framebuffer)) };
            unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
            // Note the Y-flipping here. GL's presentation is not flipped,
            // but main rendering is. Therefore, we Y-flip the output positions
            // in the shader, and also this blit.
            unsafe {
                gl.blit_framebuffer(
                    0,
                    swapchain.extent.height as i32,
                    swapchain.extent.width as i32,
                    0,
                    0,
                    0,
                    swapchain.extent.width as i32,
                    swapchain.extent.height as i32,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                )
            };
        }

        Ok(())
    }

    unsafe fn create_srgb_present_program(gl: &glow::Context) -> glow::Program {
        let program = unsafe { gl.create_program() }.expect("Could not create shader program");
        let vertex =
            unsafe { gl.create_shader(glow::VERTEX_SHADER) }.expect("Could not create shader");
        unsafe { gl.shader_source(vertex, include_str!("./shaders/srgb_present.vert")) };
        unsafe { gl.compile_shader(vertex) };
        let fragment =
            unsafe { gl.create_shader(glow::FRAGMENT_SHADER) }.expect("Could not create shader");
        unsafe { gl.shader_source(fragment, include_str!("./shaders/srgb_present.frag")) };
        unsafe { gl.compile_shader(fragment) };
        unsafe { gl.attach_shader(program, vertex) };
        unsafe { gl.attach_shader(program, fragment) };
        unsafe { gl.link_program(program) };
        unsafe { gl.delete_shader(vertex) };
        unsafe { gl.delete_shader(fragment) };
        unsafe { gl.bind_texture(glow::TEXTURE_2D, None) };

        program
    }

    pub fn supports_srgb(&self) -> bool {
        // present.frag takes care of handling srgb conversion
        true
    }
}

impl crate::Surface<super::Api> for Surface {
    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        match self.canvas {
            Canvas::Canvas(ref canvas) => {
                canvas.set_width(config.extent.width);
                canvas.set_height(config.extent.height);
            }
            Canvas::Offscreen(ref canvas) => {
                canvas.set_width(config.extent.width);
                canvas.set_height(config.extent.height);
            }
        }

        let gl = &device.shared.context.lock();

        {
            let mut swapchain = self.swapchain.write();
            if let Some(swapchain) = swapchain.take() {
                // delete all frame buffers already allocated
                unsafe { gl.delete_framebuffer(swapchain.framebuffer) };
            }
        }
        {
            let mut srgb_present_program = self.srgb_present_program.lock();
            if srgb_present_program.is_none() && config.format.is_srgb() {
                *srgb_present_program = Some(unsafe { Self::create_srgb_present_program(gl) });
            }
        }
        {
            let mut texture = self.texture.lock();
            if let Some(texture) = texture.take() {
                unsafe { gl.delete_texture(texture) };
            }

            *texture = Some(unsafe { gl.create_texture() }.map_err(|error| {
                log::error!("Internal swapchain texture creation failed: {error}");
                crate::DeviceError::OutOfMemory
            })?);

            let desc = device.shared.describe_texture_format(config.format);
            unsafe { gl.bind_texture(glow::TEXTURE_2D, *texture) };
            unsafe {
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    glow::NEAREST as _,
                )
            };
            unsafe {
                gl.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    glow::NEAREST as _,
                )
            };
            unsafe {
                gl.tex_storage_2d(
                    glow::TEXTURE_2D,
                    1,
                    desc.internal,
                    config.extent.width as i32,
                    config.extent.height as i32,
                )
            };

            let framebuffer = unsafe { gl.create_framebuffer() }.map_err(|error| {
                log::error!("Internal swapchain framebuffer creation failed: {error}");
                crate::DeviceError::OutOfMemory
            })?;
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer)) };
            unsafe {
                gl.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    *texture,
                    0,
                )
            };
            unsafe { gl.bind_texture(glow::TEXTURE_2D, None) };

            let mut swapchain = self.swapchain.write();
            *swapchain = Some(Swapchain {
                extent: config.extent,
                // channel: config.format.base_format().1,
                format: config.format,
                format_desc: desc,
                framebuffer,
            });
        }

        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        let gl = device.shared.context.lock();
        {
            let mut swapchain = self.swapchain.write();
            if let Some(swapchain) = swapchain.take() {
                unsafe { gl.delete_framebuffer(swapchain.framebuffer) };
            }
        }
        if let Some(renderbuffer) = self.texture.lock().take() {
            unsafe { gl.delete_texture(renderbuffer) };
        }
    }

    unsafe fn acquire_texture(
        &self,
        _timeout_ms: Option<std::time::Duration>, //TODO
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = swapchain.as_ref().unwrap();
        let texture = super::Texture {
            inner: super::TextureInner::Texture {
                raw: self.texture.lock().unwrap(),
                target: glow::TEXTURE_2D,
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
