use glam::Vec2;
use std::borrow::Cow;
use wgpu::{
    BindGroup, BindGroupLayout, ColorTargetState, CommandEncoder, Device, Extent3d, Queue,
    RenderPassColorAttachment, RenderPipeline, Sampler, ShaderModule, SurfaceConfiguration,
    Texture, TextureAspect, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

/// Renderer that draws its outputs to two output texture targets at the same time.
struct MultiTargetRenderer {
    pipeline: RenderPipeline,
    texture_view: TextureView,
    texture: Texture,
    bindgroup: BindGroup,
}

impl MultiTargetRenderer {
    fn create_image_texture(device: &Device, queue: &Queue) -> (Texture, TextureView) {
        fn create_halo_texture_data(width: usize, height: usize) -> Vec<u8> {
            // Creates black and white pixel data for the texture to sample.
            let mut img_data = vec![];
            let center: Vec2 = Vec2::new(width as f32 * 0.5, height as f32 * 0.5);
            let max_rad = width as f32 * 0.5;
            for y in 0..width {
                for x in 0..height {
                    let cp = Vec2::new(x as f32, y as f32);
                    let mag = (cp - center).length();
                    let p = max_rad - mag;
                    let boom = p * p.log(2.0);
                    let val: u8 = boom as u8;
                    img_data.push(val)
                }
            }
            img_data
        }

        let width = 64;
        let height = 64;
        let size = Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("data texture"),
            size: size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm, // we need only the red channel for black/white image,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &create_halo_texture_data(width, height),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some((1 * width) as u32),
                rows_per_image: Some(height as u32),
            },
            size,
        );

        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("view"),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        (texture, view)
    }

    fn init(
        device: &Device,
        queue: &Queue,
        shader: &ShaderModule,
        target_states: &[ColorTargetState],
    ) -> MultiTargetRenderer {
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: None,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (texture, texture_view) = Self::create_image_texture(device, queue);

        let bindgroup = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let ts = target_states
            .iter()
            .map(|x| Some(x.clone()))
            .collect::<Vec<_>>();
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
                entry_point: "fs_multi_main",
                // IMPORTANT: specify the color states for the outputs:
                targets: ts.as_slice(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline: render_pipeline,
            texture: texture,
            texture_view: texture_view,
            bindgroup: bindgroup,
        }
    }

    fn draw<'tex>(
        &self,
        encoder: &mut CommandEncoder,
        targets: &[Option<RenderPassColorAttachment<'tex>>],
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: targets,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bindgroup, &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn create_target_textures(
    device: &Device,
    format: TextureFormat,
    width: u32,
    height: u32,
) -> TextureTargets {
    let size = Extent3d {
        width: width as u32,
        height: height as u32,
        depth_or_array_layers: 1,
    };

    let a = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: format,
        usage: TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[format],
    });
    let b = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: format,
        usage: TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[format],
    });
    let a_view = a.create_view(&TextureViewDescriptor {
        format: Some(format),
        dimension: Some(TextureViewDimension::D2),
        ..TextureViewDescriptor::default()
    });
    let b_view = b.create_view(&TextureViewDescriptor {
        format: Some(format),
        dimension: Some(TextureViewDimension::D2),
        ..TextureViewDescriptor::default()
    });
    TextureTargets {
        a: a,
        a_view: a_view,
        b: b,
        b_view: b_view,
    }
}

/// Renderer that displays results on the screen.
struct TargetRenderer {
    pipeline: RenderPipeline,
    bindgroup_layout: BindGroupLayout,
    bindgroup_a: BindGroup,
    bindgroup_b: BindGroup,
    sampler: Sampler,
}

impl TargetRenderer {
    fn init(
        device: &Device,
        shader: &ShaderModule,
        format: TextureFormat,
        targets: &TextureTargets,
    ) -> TargetRenderer {
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: None,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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
                entry_point: "fs_display_main",
                targets: &[Some(ColorTargetState {
                    format: format,
                    blend: None,
                    write_mask: Default::default(),
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let (ba, bb) =
            Self::create_bindgroups(device, &texture_bind_group_layout, &targets, &sampler);
        Self {
            pipeline: render_pipeline,
            bindgroup_layout: texture_bind_group_layout,
            bindgroup_a: ba,
            bindgroup_b: bb,
            sampler: sampler,
        }
    }
    fn create_bindgroups(
        device: &Device,
        layout: &BindGroupLayout,
        texture_targets: &TextureTargets,
        sampler: &Sampler,
    ) -> (BindGroup, BindGroup) {
        let a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_targets.a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        let b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_targets.b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        (a, b)
    }

    fn draw(
        &self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        config: &SurfaceConfiguration,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bindgroup_a, &[]);

        let height = (config.height as f32);
        let half_w = (config.width as f32 * 0.5);

        // draw results in two separate viewports that split the screen:

        rpass.set_viewport(0.0, 0.0, half_w, height, 0.0, 1.0);
        rpass.draw(0..3, 0..1);

        rpass.set_viewport(half_w, 0.0, half_w, height, 0.0, 1.0);
        rpass.set_bind_group(0, &self.bindgroup_b, &[]);
        rpass.draw(0..3, 0..1);
    }

    fn rebuild_resources(&mut self, device: &Device, texture_targets: &TextureTargets) {
        (self.bindgroup_a, self.bindgroup_b) = Self::create_bindgroups(
            device,
            &self.bindgroup_layout,
            texture_targets,
            &self.sampler,
        )
    }
}

struct TextureTargets {
    a: Texture,
    a_view: TextureView,
    b: Texture,
    b_view: TextureView,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
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
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // Renderer that draws to 2 textures at the same time:
    let mtr = MultiTargetRenderer::init(
        &device,
        &queue,
        &shader,
        // ColorTargetStates specify how the data will be written to the
        // output textures:
        &[
            ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: Default::default(),
            },
            ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: Default::default(),
            },
        ],
    );

    // create our target textures that will receive the simultaneous rendering:
    let mut texture_targets =
        create_target_textures(&device, config.format, config.width, config.height);

    // Helper renderer that displays the results in 2 separate viewports:
    let mut drawer = TargetRenderer::init(&device, &shader, config.format, &texture_targets);

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter);

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);

                        texture_targets = create_target_textures(
                            &device,
                            config.format,
                            config.width,
                            config.height,
                        );
                        drawer.rebuild_resources(&device, &texture_targets);

                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame.texture.create_view(&TextureViewDescriptor::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        // draw to 2 textures at the same time:
                        mtr.draw(
                            &mut encoder,
                            &[
                                Some(RenderPassColorAttachment {
                                    view: &texture_targets.a_view,
                                    resolve_target: None,
                                    ops: Default::default(),
                                }),
                                Some(RenderPassColorAttachment {
                                    view: &texture_targets.b_view,
                                    resolve_target: None,
                                    ops: Default::default(),
                                }),
                            ],
                        );

                        // display results of the both drawn textures on screen:
                        drawer.draw(&mut encoder, &view, &config);

                        queue.submit(Some(encoder.finish()));
                        frame.present();
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
