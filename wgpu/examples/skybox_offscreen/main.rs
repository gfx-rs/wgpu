#![cfg(target_arch = "wasm32")]

mod framework {
    use std::future::Future;
    use winit::{
        event::{self, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
    };

    #[allow(dead_code)]
    pub enum ShaderStage {
        Vertex,
        Fragment,
        Compute,
    }

    pub trait Example: 'static + Sized {
        fn optional_features() -> wgpu::Features {
            wgpu::Features::empty()
        }
        fn required_features() -> wgpu::Features {
            wgpu::Features::empty()
        }
        fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
            wgpu::DownlevelCapabilities {
                flags: wgpu::DownlevelFlags::empty(),
                shader_model: wgpu::ShaderModel::Sm5,
                ..wgpu::DownlevelCapabilities::default()
            }
        }
        fn required_limits() -> wgpu::Limits {
            wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
        }
        fn init(
            config: &wgpu::SurfaceConfiguration,
            adapter: &wgpu::Adapter,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
        ) -> Self;
        fn resize(
            &mut self,
            config: &wgpu::SurfaceConfiguration,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
        );
        fn update(&mut self, event: WindowEvent);
        fn render(
            &mut self,
            view: &wgpu::TextureView,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            spawner: &Spawner,
        );
    }

    struct Setup {
        window: winit::window::Window,
        event_loop: EventLoop<()>,
        instance: wgpu::Instance,
        size: winit::dpi::PhysicalSize<u32>,
        surface: wgpu::Surface,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    async fn setup<E: Example>(title: &str) -> Setup {
        let event_loop = EventLoop::new();
        let mut builder = winit::window::WindowBuilder::new();
        builder = builder.with_title(title);

        let window = builder.build(&event_loop).unwrap();

        use winit::platform::web::WindowExtWebSys;
        let query_string = web_sys::window().unwrap().location().search().unwrap();
        let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
            .and_then(|x| x.parse().ok())
            .unwrap_or(log::Level::Error);
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");

        log::info!("Initializing the surface...");

        let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);

        let instance = wgpu::Instance::new(backend);
        let (size, surface) = unsafe {
            let size = window.inner_size();
            let surface = instance.create_surface(&window);
            (size, surface)
        };
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, Some(&surface))
                .await
                .expect("No suitable GPU adapters found on the system!");

        let optional_features = E::optional_features();
        let required_features = E::required_features();
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this example: {:?}",
            required_features - adapter_features
        );

        let required_downlevel_capabilities = E::required_downlevel_capabilities();
        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required to run this example: {:?}",
            required_downlevel_capabilities.shader_model
        );
        assert!(
            downlevel_capabilities
                .flags
                .contains(required_downlevel_capabilities.flags),
            "Adapter does not support the downlevel capabilities required to run this example: {:?}",
            required_downlevel_capabilities.flags - downlevel_capabilities.flags
        );

        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = E::required_limits().using_resolution(adapter.limits());

        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (optional_features & adapter_features) | required_features,
                    limits: needed_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        Setup {
            window,
            event_loop,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
        }
    }

    fn start<E: Example>(
        Setup {
            window,
            event_loop,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
        }: Setup,
    ) {
        let spawner = Spawner::new();
        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        surface.configure(&device, &config);

        log::info!("Initializing the example...");
        let mut example = E::init(&config, &adapter, &device, &queue);

        log::info!("Entering render loop...");
        event_loop.run(move |event, _, control_flow| {
            let _ = (&instance, &adapter); // force ownership by the closure
            *control_flow = if cfg!(feature = "metal-auto-capture") {
                ControlFlow::Exit
            } else {
                ControlFlow::Poll
            };
            match event {
                event::Event::RedrawEventsCleared => {
                    window.request_redraw();
                }
                event::Event::WindowEvent {
                    event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                    ..
                } => {
                    log::info!("Resizing to {:?}", size);
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    example.resize(&config, &device, &queue);
                    surface.configure(&device, &config);
                }
                event::Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {
                        example.update(event);
                    }
                },
                event::Event::RedrawRequested(_) => {
                    let frame = match surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(_) => {
                            surface.configure(&device, &config);
                            surface
                                .get_current_texture()
                                .expect("Failed to acquire next surface texture!")
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    example.render(&view, &device, &queue, &spawner);

                    frame.present();
                }
                _ => {}
            }
        });
    }

    pub struct Spawner {}

    impl Spawner {
        fn new() -> Self {
            Self {}
        }

        #[allow(dead_code)]
        pub fn spawn_local(&self, future: impl Future<Output = ()> + 'static) {
            wasm_bindgen_futures::spawn_local(future);
        }
    }

    pub fn run<E: Example>(title: &str) {
        use wasm_bindgen::{prelude::*, JsCast};

        let title = title.to_owned();
        wasm_bindgen_futures::spawn_local(async move {
            let setup = setup::<E>(&title).await;
            let start_closure = Closure::once_into_js(move || start::<E>(setup));

            // make sure to handle JS exceptions thrown inside start.
            // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
            // This is required, because winit uses JS exception for control flow to escape from `run`.
            if let Err(error) = call_catch(&start_closure) {
                let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                    e.message().includes("Using exceptions for control flow", 0)
                });

                if !is_control_flow_exception {
                    web_sys::console::error_1(&error);
                }
            }

            #[wasm_bindgen]
            extern "C" {
                #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
                fn call_catch(this: &JsValue) -> Result<(), JsValue>;
            }
        });
    }

    /// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
    /// specific key out of it.
    pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
        let query_string = query.strip_prefix('?')?;

        for pair in query_string.split('&') {
            let mut pair = pair.split('=');
            let key = pair.next()?;
            let value = pair.next()?;

            if key == search_key {
                return Some(value);
            }
        }

        None
    }

    // This allows treating the framework as a standalone example,
// thus avoiding listing the example names in `Cargo.toml`.
    #[allow(dead_code)]
    fn main() {}
}

use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, f32::consts};
use wgpu::{util::DeviceExt, AstcBlock, AstcChannel};

const IMAGE_SIZE: u32 = 128;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

struct Entity {
    vertex_count: u32,
    vertex_buf: wgpu::Buffer,
}

// Note: we use the Y=up coordinate space in this example.
struct Camera {
    screen_size: (u32, u32),
    angle_y: f32,
    angle_xz: f32,
    dist: f32,
}

const MODEL_CENTER_Y: f32 = 2.0;

impl Camera {
    fn to_uniform_data(&self) -> [f32; 16 * 3 + 4] {
        let aspect = self.screen_size.0 as f32 / self.screen_size.1 as f32;
        let proj = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect, 1.0, 50.0);
        let cam_pos = glam::Vec3::new(
            self.angle_xz.cos() * self.angle_y.sin() * self.dist,
            self.angle_xz.sin() * self.dist + MODEL_CENTER_Y,
            self.angle_xz.cos() * self.angle_y.cos() * self.dist,
        );
        let view = glam::Mat4::look_at_rh(
            cam_pos,
            glam::Vec3::new(0f32, MODEL_CENTER_Y, 0.0),
            glam::Vec3::Y,
        );
        let proj_inv = proj.inverse();

        let mut raw = [0f32; 16 * 3 + 4];
        raw[..16].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj)[..]);
        raw[16..32].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&proj_inv)[..]);
        raw[32..48].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&view)[..]);
        raw[48..51].copy_from_slice(AsRef::<[f32; 3]>::as_ref(&cam_pos));
        raw[51] = 1.0;
        raw
    }
}

pub struct Skybox {
    camera: Camera,
    sky_pipeline: wgpu::RenderPipeline,
    entity_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    entities: Vec<Entity>,
    depth_view: wgpu::TextureView,
    staging_belt: wgpu::util::StagingBelt,
}

impl Skybox {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    fn create_depth_texture(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl framework::Example for Skybox {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu::Features::TEXTURE_COMPRESSION_BC
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut entities = Vec::new();
        {
            let source = include_bytes!("models/teslacyberv3.0.obj");
            let data = obj::ObjData::load_buf(&source[..]).unwrap();
            let mut vertices = Vec::new();
            for object in data.objects {
                for group in object.groups {
                    vertices.clear();
                    for poly in group.polys {
                        for end_index in 2..poly.0.len() {
                            for &index in &[0, end_index - 1, end_index] {
                                let obj::IndexTuple(position_id, _texture_id, normal_id) =
                                    poly.0[index];
                                vertices.push(Vertex {
                                    pos: data.position[position_id],
                                    normal: data.normal[normal_id.unwrap()],
                                })
                            }
                        }
                    }
                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    entities.push(Entity {
                        vertex_count: vertices.len() as u32,
                        vertex_buf,
                    });
                }
            }
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::Cube,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create the render pipeline
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let camera = Camera {
            screen_size: (config.width, config.height),
            angle_xz: 0.2,
            angle_y: 0.2,
            dist: 20.0,
        };
        let raw_uniforms = camera.to_uniform_data();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Buffer"),
            contents: bytemuck::cast_slice(&raw_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the render pipelines
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                targets: &[config.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let entity_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Entity"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_entity",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_entity",
                targets: &[config.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let device_features = device.features();

        let skybox_format =
            if device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR) {
                log::info!("Using ASTC_LDR");
                wgpu::TextureFormat::Astc {
                    block: AstcBlock::B4x4,
                    channel: AstcChannel::UnormSrgb,
                }
            } else if device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
                log::info!("Using ETC2");
                wgpu::TextureFormat::Etc2Rgb8UnormSrgb
            } else if device_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                log::info!("Using BC");
                wgpu::TextureFormat::Bc1RgbaUnormSrgb
            } else {
                log::info!("Using plain");
                wgpu::TextureFormat::Bgra8UnormSrgb
            };

        let size = wgpu::Extent3d {
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
            depth_or_array_layers: 6,
        };

        let layer_size = wgpu::Extent3d {
            depth_or_array_layers: 1,
            ..size
        };
        let max_mips = layer_size.max_mips(wgpu::TextureDimension::D2);

        log::debug!(
            "Copying {:?} skybox images of size {}, {}, 6 with {} mips to gpu",
            skybox_format,
            IMAGE_SIZE,
            IMAGE_SIZE,
            max_mips,
        );

        let bytes = match skybox_format {
            wgpu::TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::UnormSrgb,
            } => &include_bytes!("images/astc.dds")[..],
            wgpu::TextureFormat::Etc2Rgb8UnormSrgb => &include_bytes!("images/etc2.dds")[..],
            wgpu::TextureFormat::Bc1RgbaUnormSrgb => &include_bytes!("images/bc1.dds")[..],
            wgpu::TextureFormat::Bgra8UnormSrgb => &include_bytes!("images/bgra.dds")[..],
            _ => unreachable!(),
        };

        let image = ddsfile::Dds::read(&mut std::io::Cursor::new(&bytes)).unwrap();

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                size,
                mip_level_count: max_mips as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: skybox_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: None,
            },
            &image.data,
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let depth_view = Self::create_depth_texture(config, device);

        Skybox {
            camera,
            sky_pipeline,
            entity_pipeline,
            bind_group,
            uniform_buf,
            entities,
            depth_view,
            staging_belt: wgpu::util::StagingBelt::new(0x100),
        }
    }

    #[allow(clippy::single_match)]
    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                let norm_x = position.x as f32 / self.camera.screen_size.0 as f32 - 0.5;
                let norm_y = position.y as f32 / self.camera.screen_size.1 as f32 - 0.5;
                self.camera.angle_y = norm_x * 5.0;
                self.camera.angle_xz = norm_y;
            }
            _ => {}
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.depth_view = Self::create_depth_texture(config, device);
        self.camera.screen_size = (config.width, config.height);
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &framework::Spawner,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // update rotation
        let raw_uniforms = self.camera.to_uniform_data();
        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.uniform_buf,
                0,
                wgpu::BufferSize::new((raw_uniforms.len() * 4) as wgpu::BufferAddress).unwrap(),
                device,
            )
            .copy_from_slice(bytemuck::cast_slice(&raw_uniforms));

        self.staging_belt.finish();

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });

            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_pipeline(&self.entity_pipeline);

            for entity in self.entities.iter() {
                rpass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                rpass.draw(0..entity.vertex_count, 0..1);
            }

            rpass.set_pipeline(&self.sky_pipeline);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let belt_future = self.staging_belt.recall();
        spawner.spawn_local(belt_future);
    }
}

fn main() {
    framework::run::<Skybox>("skybox");
}
