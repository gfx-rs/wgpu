extern crate env_logger;
extern crate log;
extern crate wgpu_native;

pub use self::wgpu_native::winit;

use self::log::info;


pub const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::B8g8r8a8Unorm;

pub trait Example {
    fn init(device: &wgpu::Device) -> Self;
    fn update(&mut self, event: winit::WindowEvent);
    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &wgpu::Device);
}

pub fn run<E: Example>() {
    use self::wgpu_native::winit::{
        Event, ElementState, EventsLoop, KeyboardInput, Window, WindowEvent, VirtualKeyCode
    };

    info!("Initializing the device...");
    env_logger::init();
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });
    let device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    info!("Initializing the example...");
    let mut example = E::init(&device);

    info!("Initializing the window...");
    let mut events_loop = EventsLoop::new();
    let window = Window::new(&events_loop).unwrap();
    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let surface = instance.create_surface(&window);
    let mut swap_chain = device.create_swap_chain(&surface, &wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
        format: SWAP_CHAIN_FORMAT,
        width: size.width as u32,
        height: size.height as u32,
    });

    info!("Entering render loop...");
    let mut running = true;
    while running {
        events_loop.poll_events(|event| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    let physical = size.to_physical(window.get_hidpi_factor());
                    info!("Resized to {:?}", physical);
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    } |
                    WindowEvent::CloseRequested => {
                        running = false;
                    }
                    _ => {
                        example.update(event);
                    }
                }
                _ => ()
            }
        });

        let frame = swap_chain.get_next_texture();
        example.render(&frame, &device);
    }
}
