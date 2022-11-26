use std::borrow::Cow;

use wasm_bindgen_test::*;
use winit::{
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    platform::web::{EventLoopExtWebSys, WindowExtWebSys},
    window::Window,
};

use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;
use wgpu::{Adapter, Device, Instance, Queue, Surface};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_triangle_rendering() {
    render_triangle(|window| {
        let size = window.inner_size();

        // fetch triangle pixel
        let result = read_pixel(
            window.canvas(),
            (size.width as f32 * 0.5) as i32,
            (size.height as f32 * 0.5) as i32,
        );
        let red = [255, 0, 0, 255_u8];
        assert_eq!(result, red);

        // fetch background pixel
        let result = read_pixel(
            window.canvas(),
            (size.width as f32 * 0.1) as i32,
            (size.height as f32 * 0.9) as i32,
        );
        let green = [0, 255, 0, 255_u8];
        assert_eq!(result, green);
    })
    .await;
}

async fn render_triangle<F>(assert_rendering_result: F)
where
    F: Fn(&Window) + 'static,
{
    let (window, event_loop, surface, adapter, device, queue): (
        Window,
        EventLoop<()>,
        Surface,
        Adapter,
        Device,
        Queue,
    ) = init().await;

    let size = window.inner_size();

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let swapchain_format = surface.get_supported_formats(&adapter)[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
    };

    surface.configure(&device, &config);

    event_loop.spawn(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.draw(0..3, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();

                assert_rendering_result(&window);

                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        }
    });
}

fn read_pixel(canvas: HtmlCanvasElement, x: i32, y: i32) -> [u8; 4] {
    let mut result = [0_u8; 4];
    let context = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .unwrap();

    context.read_pixels_with_u8_array_and_dst_offset(
            x, y, 1, 1,
            web_sys::WebGl2RenderingContext::RGBA,
            web_sys::WebGl2RenderingContext::UNSIGNED_BYTE,
            &mut result,
            0,
        )
        .unwrap();
    result
}

#[cfg(target_arch = "wasm32")]
async fn init() -> (
    Window,
    EventLoop<()>,
    Surface,
    Adapter,
    Device,
    Queue,
) {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    // On wasm, append the canvas to the document body
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| doc.body())
        .and_then(|body| {
            body.append_child(&web_sys::Element::from(window.canvas()))
                .ok()
        })
        .expect("couldn't append canvas to document body");

    let instance = wgpu::Instance::new(wgpu::Backends::GL);
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    (window, event_loop, surface, adapter, device, queue)
}
