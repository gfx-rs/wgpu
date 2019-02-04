extern crate env_logger;
extern crate glsl_to_spirv;
extern crate log;
extern crate wgpu_native;

pub use self::wgpu_native::winit;

use self::log::info;


pub const SWAP_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::B8g8r8a8Unorm;

pub fn cast_slice<T>(data: &[T]) -> &[u8] {
    use std::mem::size_of;
    use std::slice::from_raw_parts;

    unsafe {
        from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>())
    }
}

pub fn load_glsl_pair(name: &str) -> (Vec<u8>, Vec<u8>) {
    use self::glsl_to_spirv::{ShaderType, compile};
    use std::fs::read_to_string;
    use std::io::Read;
    use std::path::PathBuf;

    let base_path = PathBuf::from("data").join(name);
    let code_vs = read_to_string(base_path.with_extension("vert")).unwrap();
    let code_fs = read_to_string(base_path.with_extension("frag")).unwrap();

    let mut output_vs = compile(&code_vs, ShaderType::Vertex).unwrap();
    let mut output_fs = compile(&code_fs, ShaderType::Fragment).unwrap();

    let (mut spv_vs, mut spv_fs) = (Vec::new(), Vec::new());
    output_vs.read_to_end(&mut spv_vs).unwrap();
    output_fs.read_to_end(&mut spv_fs).unwrap();
    (spv_vs, spv_fs)
}

pub trait Example {
    fn init(device: &mut wgpu::Device) -> Self;
    fn update(&mut self, event: winit::WindowEvent);
    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device);
}

pub fn run<E: Example>(title: &str) {
    use self::wgpu_native::winit::{
        Event, ElementState, EventsLoop, KeyboardInput, Window, WindowEvent, VirtualKeyCode
    };

    info!("Initializing the device...");
    env_logger::init();
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });
    let mut device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    info!("Initializing the example...");
    let mut example = E::init(&mut device);

    info!("Initializing the window...");
    let mut events_loop = EventsLoop::new();
    let window = Window::new(&events_loop).unwrap();
    window.set_title(title);
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
        example.render(&frame, &mut device);
    }
}
