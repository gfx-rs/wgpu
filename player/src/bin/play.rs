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
        event::KeyEvent,
        event_loop::EventLoop,
        keyboard::{Key, NamedKey},
        window::WindowBuilder,
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
    #[cfg(feature = "winit")]
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("wgpu player")
            .with_resizable(true)
            .build(&event_loop)
            .unwrap(),
    );

    let instance_desc = wgt::InstanceDescriptor::new_without_display_handle_from_env();
    #[cfg(feature = "winit")]
    // TODO: Use event_loop.owned_display_handle() with winit 0.30
    let instance_desc = instance_desc.with_display_handle(Box::new(window.clone()));
    let instance_flags = instance_desc.flags;
    let instance = wgc::instance::Instance::new("player", instance_desc, None);

    #[cfg(feature = "winit")]
    let surface =
        unsafe { instance.create_surface(None, window.window_handle().unwrap().into()) }.unwrap();
    #[cfg(feature = "winit")]
    let mut configured_surface_id = None;

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
                    #[cfg(feature = "winit")]
                    compatible_surface: Some(&surface),
                    #[cfg(not(feature = "winit"))]
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
        use winit::{
            event::{ElementState, Event, WindowEvent},
            event_loop::ControlFlow,
        };

        let mut resize_config = None;
        let mut frame_count = 0;
        let mut done = false;
        event_loop
            .run(move |event, target| {
                target.set_control_flow(ControlFlow::Poll);

                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::RedrawRequested if resize_config.is_none() => loop {
                            match actions.pop() {
                                Some(trace::Action::ConfigureSurface(surface_id, config)) => {
                                    log::info!("Configuring the surface");
                                    let current_size: (u32, u32) = window.inner_size().into();
                                    let size = (config.width, config.height);
                                    if current_size != size {
                                        let _ = window.request_inner_size(
                                            winit::dpi::PhysicalSize::new(
                                                config.width,
                                                config.height,
                                            ),
                                        );
                                        resize_config = Some(config);
                                        break;
                                    } else {
                                        let error = device.configure_surface(&surface, &config);
                                        configured_surface_id = Some(surface_id);
                                        if let Some(e) = error {
                                            panic!("{e:?}");
                                        }
                                    }
                                }
                                Some(trace::Action::GetSurfaceTexture { id, parent }) => {
                                    log::debug!("Get surface texture for frame {frame_count}");
                                    assert!(
                                        configured_surface_id == Some(parent),
                                        "rendering to an unexpected surface"
                                    );
                                    player.get_surface_texture(id, &surface);
                                }
                                Some(trace::Action::Present(_id)) => {
                                    frame_count += 1;
                                    log::debug!("Presenting frame {frame_count}");
                                    surface.present().unwrap();
                                    break;
                                }
                                Some(trace::Action::DiscardSurfaceTexture(_id)) => {
                                    log::debug!("Discarding frame {frame_count}");
                                    surface.discard().unwrap();
                                    break;
                                }
                                Some(action) => {
                                    player.process(
                                        &device,
                                        &queue,
                                        action,
                                        trace::DiskTraceLoader::new(&dir),
                                    );
                                }
                                None => {
                                    if !done {
                                        println!("Finished the end at frame {frame_count}");
                                        done = true;
                                    }
                                    break;
                                }
                            }
                        },
                        WindowEvent::Resized(_) => {
                            if let Some(config) = resize_config.take() {
                                let error = device.configure_surface(&surface, &config);
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
                        | WindowEvent::CloseRequested => target.exit(),
                        _ => {}
                    },
                    Event::LoopExiting => {
                        log::info!("Closing");
                        device.poll(wgt::PollType::wait_indefinitely()).unwrap();
                    }
                    _ => {}
                }
            })
            .unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
