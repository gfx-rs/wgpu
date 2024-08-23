//! This example shows interop with raw GLES contexts -
//! the ability to hook up wgpu-hal to an existing context and draw into it.
//!
//! Emscripten build:
//! 1. install emsdk
//! 2. build this example with cargo:
//!    EMCC_CFLAGS="-g -s ERROR_ON_UNDEFINED_SYMBOLS=0 --no-entry -s FULL_ES3=1" cargo build --example raw-gles --target wasm32-unknown-emscripten
//! 3. copy raw-gles.em.html into target directory and open it in browser:
//!    cp wgpu-hal/examples/raw-gles.em.html target/wasm32-unknown-emscripten/debug/examples

extern crate wgpu_hal as hal;

#[cfg(not(any(windows, target_arch = "wasm32", target_os = "ios")))]
fn main() {
    use std::ffi::CString;

    use glutin::{
        config::GlConfig as _,
        context::{NotCurrentGlContext as _, Version},
        display::{GetGlDisplay as _, GlDisplay as _},
        surface::GlSurface as _,
    };
    use glutin_winit::GlWindow as _;
    use rwh_05::HasRawWindowHandle as _;

    env_logger::init();
    println!("Initializing external GL context");

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    // Only Windows requires the window to be present before creating the display.
    // Other platforms don't really need one.
    let window_builder = cfg!(windows).then(|| {
        winit::window::WindowBuilder::new()
            .with_title("WGPU raw GLES example (press Escape to exit)")
    });

    // The template will match only the configurations supporting rendering
    // to Windows.
    let template = glutin::config::ConfigTemplateBuilder::new();

    let display_builder = glutin_winit::DisplayBuilder::new().with_window_builder(window_builder);

    // Find the config with the maximum number of samples, so our triangle will be
    // smooth.
    pub fn gl_config_picker(
        configs: Box<dyn Iterator<Item = glutin::config::Config> + '_>,
    ) -> glutin::config::Config {
        configs
            .reduce(|accum, config| {
                if config.num_samples() > accum.num_samples() {
                    config
                } else {
                    accum
                }
            })
            .expect("Failed to find a matching config")
    }

    let (mut window, gl_config) = display_builder
        .build(&event_loop, template, gl_config_picker)
        .expect("Failed to build window and config from display");

    println!("Picked a config with {} samples", gl_config.num_samples());

    let raw_window_handle = window.as_ref().map(|window| window.raw_window_handle());

    // XXX The display could be obtained from any object created by it, so we can
    // query it from the config.
    let gl_display = gl_config.display();

    // Glutin tries to create an OpenGL context by default.  Force it to use any version of GLES.
    let context_attributes = glutin::context::ContextAttributesBuilder::new()
        // WGPU expects GLES 3.0+.
        .with_context_api(glutin::context::ContextApi::Gles(Some(Version::new(3, 0))))
        .build(raw_window_handle);

    // TODO: Wrap in Option
    let mut not_current_gl_context = unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .expect("failed to create context")
    };

    // TODO: For Android support this should happen inside Event::Resumed, and the inverse inside ::Suspended.
    let window = window.take().unwrap_or_else(|| {
        let window_builder = winit::window::WindowBuilder::new()
            .with_title("WGPU raw GLES example (press Escape to exit)");
        glutin_winit::finalize_window(&event_loop, window_builder, &gl_config).unwrap()
    });

    let attrs = window.build_surface_attributes(Default::default());
    let gl_surface = unsafe {
        gl_config
            .display()
            .create_window_surface(&gl_config, &attrs)
            .expect("Cannot create GL WindowSurface")
    };

    // Make it current.
    // let gl_context = not_current_gl_context
    //     .take()
    //     .unwrap()
    //     .make_current(&gl_surface)
    //     .unwrap();
    let gl_context = not_current_gl_context
        .make_current(&gl_surface)
        .expect("GL context cannot be made current with WindowSurface");

    println!("Hooking up to wgpu-hal");
    let exposed = unsafe {
        <hal::api::Gles as hal::Api>::Adapter::new_external(|name| {
            // XXX: On WGL this should only be called after the context was made current
            gl_config
                .display()
                .get_proc_address(&CString::new(name).expect(name))
        })
    }
    .expect("GL adapter can't be initialized");

    let inner_size = window.inner_size();

    fill_screen(&exposed, inner_size.width, inner_size.height);

    // TODO: Swap in RedrawRequested
    println!("Showing the window");
    gl_surface
        .swap_buffers(&gl_context)
        .expect("Failed to swap buffers");

    event_loop
        .run(move |event, window_target| {
            use winit::{
                event::{Event, KeyEvent, WindowEvent},
                event_loop::ControlFlow,
                keyboard::{Key, NamedKey},
            };
            window_target.set_control_flow(ControlFlow::Wait);

            match event {
                // Event::LoopExiting => (),
                Event::WindowEvent {
                    window_id: _,
                    event:
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key: Key::Named(NamedKey::Escape),
                                    ..
                                },
                            ..
                        },
                } => window_target.exit(),
                _ => (),
            }
        })
        .expect("Couldn't run event loop");
}

#[cfg(target_os = "emscripten")]
fn main() {
    env_logger::init();

    println!("Initializing external GL context");
    let egl = khronos_egl::Instance::new(khronos_egl::Static);
    let display = unsafe { egl.get_display(khronos_egl::DEFAULT_DISPLAY) }.unwrap();
    egl.initialize(display)
        .expect("unable to initialize display");

    let attributes = [
        khronos_egl::RED_SIZE,
        8,
        khronos_egl::GREEN_SIZE,
        8,
        khronos_egl::BLUE_SIZE,
        8,
        khronos_egl::NONE,
    ];

    let config = egl
        .choose_first_config(display, &attributes)
        .unwrap()
        .expect("unable to choose config");
    let surface = unsafe {
        let window = std::ptr::null_mut::<std::ffi::c_void>();
        egl.create_window_surface(display, config, window, None)
    }
    .expect("unable to create surface");

    let context_attributes = [khronos_egl::CONTEXT_CLIENT_VERSION, 3, khronos_egl::NONE];

    let gl_context = egl
        .create_context(display, config, None, &context_attributes)
        .expect("unable to create context");
    egl.make_current(display, Some(surface), Some(surface), Some(gl_context))
        .expect("can't make context current");

    println!("Hooking up to wgpu-hal");
    let exposed = unsafe {
        <hal::api::Gles as hal::Api>::Adapter::new_external(|name| {
            egl.get_proc_address(name)
                .map_or(std::ptr::null(), |p| p as *const _)
        })
    }
    .expect("GL adapter can't be initialized");

    fill_screen(&exposed, 640, 400);
}

#[cfg(any(
    windows,
    all(target_arch = "wasm32", not(target_os = "emscripten")),
    target_os = "ios"
))]
fn main() {
    eprintln!("This example is not supported on Windows and non-emscripten wasm32")
}

#[cfg(not(any(
    windows,
    all(target_arch = "wasm32", not(target_os = "emscripten")),
    target_os = "ios"
)))]
fn fill_screen(exposed: &hal::ExposedAdapter<hal::api::Gles>, width: u32, height: u32) {
    use hal::{Adapter as _, CommandEncoder as _, Device as _, Queue as _};

    let od = unsafe {
        exposed.adapter.open(
            wgt::Features::empty(),
            &wgt::Limits::downlevel_defaults(),
            &wgt::MemoryHints::default(),
        )
    }
    .unwrap();

    let format = wgt::TextureFormat::Rgba8UnormSrgb;
    let texture = <hal::api::Gles as hal::Api>::Texture::default_framebuffer(format);
    let view = unsafe {
        od.device
            .create_texture_view(
                &texture,
                &hal::TextureViewDescriptor {
                    label: None,
                    format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: hal::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                },
            )
            .unwrap()
    };

    println!("Filling the screen");
    let mut encoder = unsafe {
        od.device
            .create_command_encoder(&hal::CommandEncoderDescriptor {
                label: None,
                queue: &od.queue,
            })
            .unwrap()
    };
    let mut fence = unsafe { od.device.create_fence().unwrap() };
    let rp_desc = hal::RenderPassDescriptor {
        label: None,
        extent: wgt::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        sample_count: 1,
        color_attachments: &[Some(hal::ColorAttachment {
            target: hal::Attachment {
                view: &view,
                usage: hal::TextureUses::COLOR_TARGET,
            },
            resolve_target: None,
            ops: hal::AttachmentOps::STORE,
            clear_value: wgt::Color::BLUE,
        })],
        depth_stencil_attachment: None,
        multiview: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    };
    unsafe {
        encoder.begin_encoding(None).unwrap();
        encoder.begin_render_pass(&rp_desc);
        encoder.end_render_pass();
        let cmd_buf = encoder.end_encoding().unwrap();
        od.queue.submit(&[&cmd_buf], &[], (&mut fence, 0)).unwrap();
    }
}
