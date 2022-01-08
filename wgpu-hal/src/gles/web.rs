use glow::HasContext;
use parking_lot::Mutex;
use wasm_bindgen::JsCast;

use super::TextureFormatDesc;

/// A wrapper around a [`glow::Context`] to provide a fake `lock()` api that makes it compatible
/// with the `AdapterContext` API fromt the EGL implementation.
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
    canvas: Mutex<Option<web_sys::HtmlCanvasElement>>,
}

// SAFE: WASM doesn't have threads
unsafe impl Sync for Instance {}
unsafe impl Send for Instance {}

impl crate::Instance<super::Api> for Instance {
    unsafe fn init(_desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        Ok(Instance {
            canvas: Mutex::new(None),
        })
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        let canvas_guard = self.canvas.lock();
        let gl = match *canvas_guard {
            Some(ref canvas) => {
                let context_options = js_sys::Object::new();
                js_sys::Reflect::set(
                    &context_options,
                    &"antialias".into(),
                    &wasm_bindgen::JsValue::FALSE,
                )
                .expect("Cannot create context options");
                let webgl2_context = canvas
                    .get_context_with_context_options("webgl2", &context_options)
                    .expect("Cannot create WebGL2 context")
                    .and_then(|context| context.dyn_into::<web_sys::WebGl2RenderingContext>().ok())
                    .expect("Cannot convert into WebGL2 context");
                glow::Context::from_webgl2_context(webgl2_context)
            }
            None => return Vec::new(),
        };

        super::Adapter::expose(AdapterContext { glow_context: gl })
            .into_iter()
            .collect()
    }

    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        if let raw_window_handle::RawWindowHandle::Web(handle) = has_handle.raw_window_handle() {
            let canvas: web_sys::HtmlCanvasElement = web_sys::window()
                .and_then(|win| win.document())
                .expect("Cannot get document")
                .query_selector(&format!("canvas[data-raw-handle=\"{}\"]", handle.id))
                .expect("Cannot query for canvas")
                .expect("Canvas is not found")
                .dyn_into()
                .expect("Failed to downcast to canvas type");

            *self.canvas.lock() = Some(canvas.clone());

            Ok(Surface {
                canvas,
                present_program: None,
                swapchain: None,
                texture: None,
                presentable: true,
            })
        } else {
            unreachable!()
        }
    }

    unsafe fn destroy_surface(&self, surface: Surface) {
        let mut canvas_option_ref = self.canvas.lock();

        if let Some(canvas) = canvas_option_ref.as_ref() {
            if canvas == &surface.canvas {
                *canvas_option_ref = None;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Surface {
    canvas: web_sys::HtmlCanvasElement,
    pub(super) swapchain: Option<Swapchain>,
    texture: Option<glow::Texture>,
    pub(super) presentable: bool,
    present_program: Option<glow::Program>,
}

// SAFE: Because web doesn't have threads ( yet )
unsafe impl Sync for Surface {}
unsafe impl Send for Surface {}

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
        &mut self,
        _suf_texture: super::Texture,
        gl: &glow::Context,
    ) -> Result<(), crate::SurfaceError> {
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        gl.bind_sampler(0, None);
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, self.texture);
        gl.use_program(self.present_program);
        gl.disable(glow::DEPTH_TEST);
        gl.disable(glow::STENCIL_TEST);
        gl.disable(glow::SCISSOR_TEST);
        gl.disable(glow::BLEND);
        gl.disable(glow::CULL_FACE);
        gl.draw_buffers(&[glow::BACK]);
        gl.draw_arrays(glow::TRIANGLES, 0, 3);

        Ok(())
    }

    unsafe fn create_present_program(gl: &glow::Context) -> glow::Program {
        let program = gl
            .create_program()
            .expect("Could not create shader program");
        let vertex = gl
            .create_shader(glow::VERTEX_SHADER)
            .expect("Could not create shader");
        gl.shader_source(vertex, include_str!("./shaders/present.vert"));
        gl.compile_shader(vertex);
        let fragment = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .expect("Could not create shader");
        gl.shader_source(fragment, include_str!("./shaders/present.frag"));
        gl.compile_shader(fragment);
        gl.attach_shader(program, vertex);
        gl.attach_shader(program, fragment);
        gl.link_program(program);
        gl.delete_shader(vertex);
        gl.delete_shader(fragment);
        gl.bind_texture(glow::TEXTURE_2D, None);

        program
    }

    pub fn supports_srgb(&self) -> bool {
        true // WebGL only supports sRGB
    }
}

impl crate::Surface<super::Api> for Surface {
    unsafe fn configure(
        &mut self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let gl = &device.shared.context.lock();

        if let Some(swapchain) = self.swapchain.take() {
            // delete all frame buffers already allocated
            gl.delete_framebuffer(swapchain.framebuffer);
        }

        if self.present_program.is_none() {
            self.present_program = Some(Self::create_present_program(gl));
        }

        if let Some(texture) = self.texture.take() {
            gl.delete_texture(texture);
        }

        self.texture = Some(gl.create_texture().unwrap());

        let desc = device.shared.describe_texture_format(config.format);
        gl.bind_texture(glow::TEXTURE_2D, self.texture);
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as _,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as _,
        );
        gl.tex_storage_2d(
            glow::TEXTURE_2D,
            1,
            desc.internal,
            config.extent.width as i32,
            config.extent.height as i32,
        );

        let framebuffer = gl.create_framebuffer().unwrap();
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_texture_2d(
            glow::READ_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            self.texture,
            0,
        );
        gl.bind_texture(glow::TEXTURE_2D, None);

        self.swapchain = Some(Swapchain {
            extent: config.extent,
            // channel: config.format.base_format().1,
            format: config.format,
            format_desc: desc,
            framebuffer,
        });
        Ok(())
    }

    unsafe fn unconfigure(&mut self, device: &super::Device) {
        let gl = device.shared.context.lock();
        if let Some(swapchain) = self.swapchain.take() {
            gl.delete_framebuffer(swapchain.framebuffer);
        }
        if let Some(renderbuffer) = self.texture.take() {
            gl.delete_texture(renderbuffer);
        }
    }

    unsafe fn acquire_texture(
        &mut self,
        _timeout_ms: u32,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let sc = self.swapchain.as_ref().unwrap();
        let texture = super::Texture {
            inner: super::TextureInner::Texture {
                raw: self.texture.unwrap(),
                target: glow::TEXTURE_2D,
            },
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
