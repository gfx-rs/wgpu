//! This is a player for WebGPU traces.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    extern crate wgpu_core as wgc;
    extern crate wgpu_types as wgt;

    use player::Player;
    use wgc::device::trace;
    use wgpu_core::command::PointerReferences;

    use std::{
        fs,
        path::{Path, PathBuf},
        process::exit,
        sync::Arc,
    };

    #[cfg(feature = "winit")]
    use raw_window_handle::HasWindowHandle;
    #[cfg(feature = "winit")]
    use winit::{
        application::ApplicationHandler,
        event::{ElementState, KeyEvent, WindowEvent},
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        keyboard::{Key, NamedKey},
        window::Window,
    };

    env_logger::init();

    //TODO: setting for the backend bits
    //TODO: setting for the target frame, or controls

    const HELP: &str = "\
    Usage: play <trace directory> | <trace file>\n\
    \n\
    Play a wgpu trace from the specified file or directory. If the trace contains\n\
    buffers, textures, or shaders, the directory form must be used.\n";

    let (dir, trace) = match std::env::args().nth(1) {
        Some(arg) if Path::new(&arg).is_dir() => (
            PathBuf::from(arg.clone()),
            PathBuf::from(arg).join(trace::FILE_NAME),
        ),
        Some(arg) if Path::new(&arg).is_file() => {
            (PathBuf::from("/nonexistent"), PathBuf::from(arg))
        }
        _ => {
            eprintln!("{HELP}");
            exit(1);
        }
    };

    log::info!("Loading trace '{trace:?}'");
    let file = fs::File::open(trace).unwrap();
    let mut actions: Vec<trace::Action<PointerReferences>> = ron::de::from_reader(file).unwrap();
    actions.reverse(); // allows us to pop from the top
    log::info!("Found {} actions", actions.len());

    #[cfg(feature = "winit")]
    let event_loop = {
        log::info!("Creating a window");
        EventLoop::new().unwrap()
    };

    let instance_desc = wgt::InstanceDescriptor::new_without_display_handle_from_env();
    #[cfg(feature = "winit")]
    let instance_desc =
        instance_desc.with_display_handle(Box::new(event_loop.owned_display_handle()));
    let instance_flags = instance_desc.flags;
    let instance = wgc::instance::Instance::new("player", instance_desc, None);

    let (backends, device_desc) =
        match actions.pop_if(|action| matches!(action, trace::Action::Init { .. })) {
            Some(trace::Action::Init { desc, backend }) => {
                log::info!("Initializing the device for backend: {backend:?}");
                (wgt::Backends::from(backend), desc)
            }
            Some(_) => unreachable!(),
            None => (wgt::Backends::all(), wgt::DeviceDescriptor::default()),
        };

    let adapter = Arc::new(
        instance
            .request_adapter(
                &wgt::RequestAdapterOptions {
                    compatible_surface: None,
                    ..Default::default()
                },
                backends,
            )
            .expect("Unable to obtain an adapter"),
    );

    let info = adapter.get_info();
    log::info!("Using '{}'", info.name);

    let (device, queue) = adapter
        .create_device_and_queue(&device_desc, instance_flags)
        .unwrap();

    let mut player = Player::default();

    log::info!("Executing actions");
    #[cfg(not(feature = "winit"))]
    {
        unsafe { device.start_graphics_debugger_capture() };

        while let Some(action) = actions.pop() {
            player.process(&device, &queue, action, trace::DiskTraceLoader::new(&dir));
        }

        unsafe { device.stop_graphics_debugger_capture() };
        device.poll(wgt::PollType::wait_indefinitely()).unwrap();
    }
    #[cfg(feature = "winit")]
    {
        struct App<'a> {
            window: Option<Arc<Window>>,
            surface: Option<wgc::instance::Surface>,
            configured_surface_id: Option<wgc::id::PointerId<wgc::id::markers::Surface>>,
            instance: &'a wgc::instance::Instance,
            device: &'a Arc<wgc::device::Device>,
            queue: &'a Arc<wgc::device::queue::Queue>,
            player: &'a mut Player,
            actions: &'a mut Vec<trace::Action<'a, PointerReferences>>,
            dir: &'a Path,
            resize_config: Option<wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>>,
            frame_count: usize,
            done: bool,
        }

        impl ApplicationHandler for App<'_> {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                if self.window.is_some() {
                    return;
                }
                let window = Arc::new(
                    event_loop
                        .create_window(
                            Window::default_attributes()
                                .with_title("wgpu player")
                                .with_resizable(true),
                        )
                        .unwrap(),
                );
                let surface = unsafe {
                    self.instance
                        .create_surface(None, window.window_handle().unwrap().into())
                }
                .unwrap();
                self.window = Some(window);
                self.surface = Some(surface);
            }

            fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
                log::info!("Closing");
                self.device
                    .poll(wgt::PollType::wait_indefinitely())
                    .unwrap();
            }

            fn window_event(
                &mut self,
                event_loop: &ActiveEventLoop,
                _window_id: winit::window::WindowId,
                event: WindowEvent,
            ) {
                event_loop.set_control_flow(ControlFlow::Poll);

                let window = self.window.as_ref().unwrap();
                let surface = self.surface.as_ref().unwrap();

                match event {
                    WindowEvent::RedrawRequested if self.resize_config.is_none() => loop {
                        match self.actions.pop() {
                            Some(trace::Action::ConfigureSurface(surface_id, config)) => {
                                log::info!("Configuring the surface");
                                let current_size: (u32, u32) = window.inner_size().into();
                                let size = (config.width, config.height);
                                if current_size != size {
                                    let _ = window.request_inner_size(
                                        winit::dpi::PhysicalSize::new(config.width, config.height),
                                    );
                                    self.resize_config = Some(config);
                                    break;
                                } else {
                                    let error = self.device.configure_surface(surface, &config);
                                    self.configured_surface_id = Some(surface_id);
                                    if let Some(e) = error {
                                        panic!("{e:?}");
                                    }
                                }
                            }
                            Some(trace::Action::GetSurfaceTexture { id, parent }) => {
                                log::debug!("Get surface texture for frame {}", self.frame_count);
                                assert!(
                                    self.configured_surface_id == Some(parent),
                                    "rendering to an unexpected surface"
                                );
                                self.player.get_surface_texture(id, surface);
                            }
                            Some(trace::Action::Present(_id)) => {
                                self.frame_count += 1;
                                log::debug!("Presenting frame {}", self.frame_count);
                                surface.present().unwrap();
                                break;
                            }
                            Some(trace::Action::DiscardSurfaceTexture(_id)) => {
                                log::debug!("Discarding frame {}", self.frame_count);
                                surface.discard().unwrap();
                                break;
                            }
                            Some(action) => {
                                self.player.process(
                                    self.device,
                                    self.queue,
                                    action,
                                    trace::DiskTraceLoader::new(self.dir),
                                );
                            }
                            None => {
                                if !self.done {
                                    println!("Finished the end at frame {}", self.frame_count);
                                    self.done = true;
                                }
                                break;
                            }
                        }
                    },
                    WindowEvent::Resized(_) => {
                        if let Some(config) = self.resize_config.take() {
                            let error = self.device.configure_surface(surface, &config);
                            if let Some(e) = error {
                                panic!("{e:?}");
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => event_loop.exit(),
                    _ => {}
                }
            }
        }

        let mut app = App {
            window: None,
            surface: None,
            configured_surface_id: None,
            instance: &instance,
            device: &device,
            queue: &queue,
            player: &mut player,
            actions: &mut actions,
            dir: &dir,
            resize_config: None,
            frame_count: 0,
            done: false,
        };
        event_loop.run_app(&mut app).unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
