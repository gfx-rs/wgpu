//! This example shows interop with raw GLES contexts -
//! the ability to hook up wgpu-hal to an existing context and draw into it.

extern crate wgpu_hal as hal;

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use hal::{Adapter as _, CommandEncoder as _, Device as _, Queue as _};

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
            width: inner_size.width,
            height: inner_size.height,
            depth_or_array_layers: 1,
        },
        sample_count: 1,
        color_attachments: &[hal::ColorAttachment {
            target: hal::Attachment {
                view: &view,
                usage: hal::TextureUses::COLOR_TARGET,
            },
            resolve_target: None,
            ops: hal::AttachmentOps::STORE,
            clear_value: wgt::Color::BLUE,
        }],
        depth_stencil_attachment: None,
        multiview: None,
    };
    unsafe {
        encoder.begin_encoding(None).unwrap();
        encoder.begin_render_pass(&rp_desc);
        encoder.end_render_pass();
        let cmd_buf = encoder.end_encoding().unwrap();
        od.queue.submit(&[&cmd_buf], None).unwrap();
    }

    println!("Showing the window");
    gl_context.swap_buffers().unwrap();

    event_loop.run(move |event, _, control_flow| {
        use glutin::{
            event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
            event_loop::ControlFlow,
        };
        *control_flow = ControlFlow::Wait;

        match event {
            Event::LoopDestroyed => return,
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
