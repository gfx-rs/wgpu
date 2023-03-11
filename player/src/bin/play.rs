/*! This is a player for WebGPU traces.
!*/

use player::{GlobalPlay as _, IdentityPassThroughFactory};
use wgc::{device::trace, gfx_select};

use std::{
    fs,
    path::{Path, PathBuf},
};

fn main() {
    #[cfg(feature = "winit")]
    use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
    #[cfg(feature = "winit")]
    use winit::{event_loop::EventLoop, window::WindowBuilder};

    env_logger::init();

    //TODO: setting for the backend bits
    //TODO: setting for the target frame, or controls

    let dir = match std::env::args().nth(1) {
        Some(arg) if Path::new(&arg).is_dir() => PathBuf::from(arg),
        _ => panic!("Provide the dir path as the parameter"),
    };

    log::info!("Loading trace '{:?}'", dir);
    let file = fs::File::open(dir.join(trace::FILE_NAME)).unwrap();
    let mut actions: Vec<trace::Action> = ron::de::from_reader(file).unwrap();
    actions.reverse(); // allows us to pop from the top
    log::info!("Found {} actions", actions.len());

    #[cfg(feature = "winit")]
    let event_loop = {
        log::info!("Creating a window");
        EventLoop::new()
    };
    #[cfg(feature = "winit")]
    let window = WindowBuilder::new()
        .with_title("wgpu player")
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();

    let global = wgc::hub::Global::new(
        "player",
        IdentityPassThroughFactory,
        wgt::InstanceDescriptor::default(),
    );
    let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();

    #[cfg(feature = "winit")]
    let surface = global.instance_create_surface(
        window.raw_display_handle(),
        window.raw_window_handle(),
        wgc::id::TypedId::zip(0, 1, wgt::Backend::Empty),
    );

    let device = match actions.pop() {
        Some(trace::Action::Init { desc, backend }) => {
            log::info!("Initializing the device for backend: {:?}", backend);
            let adapter = global
                .request_adapter(
                    &wgc::instance::RequestAdapterOptions {
                        power_preference: wgt::PowerPreference::LowPower,
                        force_fallback_adapter: false,
                        #[cfg(feature = "winit")]
                        compatible_surface: Some(surface),
                        #[cfg(not(feature = "winit"))]
                        compatible_surface: None,
                    },
                    wgc::instance::AdapterInputs::IdSet(
                        &[wgc::id::TypedId::zip(0, 0, backend)],
                        |id| id.backend(),
                    ),
                )
                .expect("Unable to find an adapter for selected backend");

            let info = gfx_select!(adapter => global.adapter_get_info(adapter)).unwrap();
            log::info!("Picked '{}'", info.name);
            let id = wgc::id::TypedId::zip(1, 0, backend);
            let (_, error) = gfx_select!(adapter => global.adapter_request_device(
                adapter,
                &desc,
                None,
                id
            ));
            if let Some(e) = error {
                panic!("{:?}", e);
            }
            id
        }
        _ => panic!("Expected Action::Init"),
    };

    log::info!("Executing actions");
    #[cfg(not(feature = "winit"))]
    {
        gfx_select!(device => global.device_start_capture(device));

        while let Some(action) = actions.pop() {
            gfx_select!(device => global.process(device, action, &dir, &mut command_buffer_id_manager));
        }

        gfx_select!(device => global.device_stop_capture(device));
        gfx_select!(device => global.device_poll(device, wgt::Maintain::Wait)).unwrap();
    }
    #[cfg(feature = "winit")]
    {
        use winit::{
            event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
            event_loop::ControlFlow,
        };

        let mut resize_config = None;
        let mut frame_count = 0;
        let mut done = false;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) if resize_config.is_none() => loop {
                    match actions.pop() {
                        Some(trace::Action::ConfigureSurface(_device_id, config)) => {
                            log::info!("Configuring the surface");
                            let current_size: (u32, u32) = window.inner_size().into();
                            let size = (config.width, config.height);
                            if current_size != size {
                                window.set_inner_size(winit::dpi::PhysicalSize::new(
                                    config.width,
                                    config.height,
                                ));
                                resize_config = Some(config);
                                break;
                            } else {
                                let error = gfx_select!(device => global.surface_configure(surface, device, &config));
                                if let Some(e) = error {
                                    panic!("{:?}", e);
                                }
                            }
                        }
                        Some(trace::Action::Present(id)) => {
                            frame_count += 1;
                            log::debug!("Presenting frame {}", frame_count);
                            gfx_select!(device => global.surface_present(id)).unwrap();
                            break;
                        }
                        Some(trace::Action::DiscardSurfaceTexture(id)) => {
                            log::debug!("Discarding frame {}", frame_count);
                            gfx_select!(device => global.surface_texture_discard(id)).unwrap();
                            break;
                        }
                        Some(action) => {
                            gfx_select!(device => global.process(device, action, &dir, &mut command_buffer_id_manager));
                        }
                        None => {
                            if !done {
                                println!("Finished the end at frame {}", frame_count);
                                done = true;
                            }
                            break;
                        }
                    }
                },
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::Resized(_) => {
                        if let Some(config) = resize_config.take() {
                            let error = gfx_select!(device => global.surface_configure(surface, device, &config));
                            if let Some(e) = error {
                                panic!("{:?}", e);
                            }
                        }
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                },
                Event::LoopDestroyed => {
                    log::info!("Closing");
                    gfx_select!(device => global.device_poll(device, wgt::Maintain::Wait)).unwrap();
                }
                _ => {}
            }
        });
    }
}
