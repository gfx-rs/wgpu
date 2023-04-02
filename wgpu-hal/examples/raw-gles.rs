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

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    println!("Initializing external GL context");

    let event_loop = glutin::event_loop::EventLoop::new();
    let window_builder = glutin::window::WindowBuilder::new();
    let gl_context = unsafe {
        glutin::ContextBuilder::new()
            .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGlEs, (3, 0)))
            .build_windowed(window_builder, &event_loop)
            .unwrap()
            .make_current()
            .unwrap()
    };
    let inner_size = gl_context.window().inner_size();

    println!("Hooking up to wgpu-hal");
    let exposed = unsafe {
        <hal::api::Gles as hal::Api>::Adapter::new_external(|name| {
            gl_context.get_proc_address(name)
        })
    }
    .expect("GL adapter can't be initialized");

    fill_screen(&exposed, inner_size.width, inner_size.height);

    println!("Showing the window");
    gl_context.swap_buffers().unwrap();

    event_loop.run(move |event, _, control_flow| {
        use glutin::{
            event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Wait;

        match event {
            Event::LoopDestroyed => (),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            _ => (),
        }
    });
}

#[cfg(target_os = "emscripten")]
fn main() {
    env_logger::init();

    println!("Initializing external GL context");
    let egl = khronos_egl::Instance::new(khronos_egl::Static);
    let display = egl.get_display(khronos_egl::DEFAULT_DISPLAY).unwrap();
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

#[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
fn main() {}

fn fill_screen(exposed: &hal::ExposedAdapter<hal::api::Gles>, width: u32, height: u32) {
    use hal::{Adapter as _, CommandEncoder as _, Device as _, Queue as _};

    let mut od = unsafe {
        exposed
            .adapter
            .open(wgt::Features::empty(), &wgt::Limits::downlevel_defaults())
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
        timestamp_writes: &[],
    };
    unsafe {
        encoder.begin_encoding(None).unwrap();
        encoder.begin_render_pass(&rp_desc);
        encoder.end_render_pass();
        let cmd_buf = encoder.end_encoding().unwrap();
        od.queue.submit(&[&cmd_buf], None).unwrap();
    }
}
