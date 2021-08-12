/*! This is a player for WebGPU traces.
!*/

use player::{Action, GlobalPlay as _, IdentityPassThroughFactory};
use wgc::{device::trace, gfx_select2};

use std::{
    io::Read,
    fs,
    path::{Path, PathBuf},
};

fn main() {
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
    let mut bytes = Vec::new();
    fs::File::open(dir.join(trace::FILE_NAME)).unwrap().read_to_end(&mut bytes).unwrap();
    // NOTE: Required since the closure to winit has a 'static lifetime, for some reason... most
    // likely this is not necessary.
    let actions: Vec<trace::Action> = ron::de::from_bytes(bytes.leak()).unwrap();
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

    let global = wgc::hub::Global::new("player", IdentityPassThroughFactory, wgt::Backends::all());
    let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();

    // NOTE: Returns reversed action list, so we can pop from the top.
    let mut actions = global.init_actions(actions);

    #[cfg(feature = "winit")]
    let surface =
        global.instance_create_surface(&window, wgc::id::TypedId::zip(0, 1, wgt::Backend::Empty));

    let device_ = match actions.pop() {
        Some(Action::Trace(trace::Action::Init { desc, backend })) => {
            log::info!("Initializing the device for backend: {:?}", backend);
            let adapter_ = global
                .request_adapter(
                    &wgc::instance::RequestAdapterOptions {
                        power_preference: wgt::PowerPreference::LowPower,
                        #[cfg(feature = "winit")]
                        compatible_surface: Some(surface),
                        #[cfg(not(feature = "winit"))]
                        compatible_surface: None,
                    },
                    /*wgc::instance::AdapterInputs::IdSet(
                        &[wgc::id::TypedId::zip(0, 0, backend)],
                        |id| id.backend(),
                    )*/backend.into(),
                )
                .expect("Unable to find an adapter for selected backend");

            let adapter = &adapter_;
            let info = gfx_select2!(&Box adapter => wgc::hub::Global::<IdentityPassThroughFactory>::adapter_get_info(&adapter))/*.unwrap()*/;
            log::info!("Picked '{}'", info.name);
            // let id = wgc::id::TypedId::zip(1, 0, backend);
            /*let (_, error) = */gfx_select2!(Box adapter_ => wgc::hub::Global::<IdentityPassThroughFactory>::adapter_request_device(
                *adapter_,
                &desc,
                None,
                // id
            )).unwrap()
            /* if let Some(e) = error {
                panic!("{:?}", e);
            } */
            // id
        }
        _ => panic!("Expected Action::Init"),
    };

    log::info!("Executing actions");
    #[cfg(not(feature = "winit"))]
    {
        let device = &device_;
        gfx_select2!(&Arc device => global.device_start_capture(device));

        let mut trace_cache = wgc::id::IdMap::default();
        let mut cache = wgc::id::IdCache2::default();

        while let Some(action) = actions.pop() {
            gfx_select2!(&Arc device =>
                         global.process(device, action, &dir, &mut trace_cache, &mut cache, &mut command_buffer_id_manager));
        }

        gfx_select2!(&Arc device => global.device_stop_capture(device));
        gfx_select2!(&Arc device => global.device_poll(device, true)).unwrap();
    }
    #[cfg(feature = "winit")]
    {
        use wgc::gfx_select;
        use winit::{
            event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
            event_loop::ControlFlow,
        };

        let mut trace_cache = wgc::id::IdMap::default();
        let mut cache = wgc::id::IdCache2::default();

        let mut resize_config = None;
        let mut frame_count = 0;
        let mut done = false;
        event_loop.run(move |event, _, control_flow| {
            let device = &device_;
            *control_flow = ControlFlow::Poll;
            match event {
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) if resize_config.is_none() => loop {
                    match actions.pop() {
                        Some(Action::Trace(trace::Action::ConfigureSurface(_device_id, config))) => {
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
                                let device = device.clone();
                                let error = gfx_select2!(Arc device => global.surface_configure(surface, device, &config));
                                if let Some(e) = error {
                                    panic!("{:?}", e);
                                }
                            }
                        }
                        Some(Action::Trace(trace::Action::Present(id))) => {
                            frame_count += 1;
                            log::debug!("Presenting frame {}", frame_count);
                            gfx_select!(device => global.surface_present(id)).unwrap();
                            break;
                        }
                        Some(action) => {
                            gfx_select2!(&Arc device =>
                                         global.process(device, action, &dir, &mut trace_cache, &mut cache,
                                                        &mut command_buffer_id_manager));
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
                            let device = device.clone();
                            let error = gfx_select2!(Arc device => global.surface_configure(surface, device, &config));
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
                    gfx_select2!(&Arc device => global.device_poll(device, true)).unwrap();
                }
                _ => {}
            }
        });
    }
}
