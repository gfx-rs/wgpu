extern crate env_logger;
extern crate wgpu;
extern crate wgpu_native;

fn main() {
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

    let vs_bytes = include_bytes!("./../data/hello_triangle.vert.spv");
    let vs_module = device.create_shader_module(vs_bytes);
    let fs_bytes = include_bytes!("./../data/hello_triangle.frag.spv");
    let fs_module = device.create_shader_module(fs_bytes);

    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let blend_state0 = device.create_blend_state(&wgpu::BlendStateDescriptor::REPLACE);
    let depth_stencil_state =
        device.create_depth_stencil_state(&wgpu::DepthStencilStateDescriptor::IGNORE);

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        stages: &[
            wgpu::PipelineStageDescriptor {
                module: &vs_module,
                stage: wgpu::ShaderStage::Vertex,
                entry_point: "main",
            },
            wgpu::PipelineStageDescriptor {
                module: &fs_module,
                stage: wgpu::ShaderStage::Fragment,
                entry_point: "main",
            },
        ],
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        attachments_state: wgpu::AttachmentsState {
            color_attachments: &[wgpu::Attachment {
                format: wgpu::TextureFormat::B8g8r8a8Unorm,
                samples: 1,
            }],
            depth_stencil_attachment: None,
        },
        blend_states: &[&blend_state0],
        depth_stencil_state: &depth_stencil_state,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[],
    });

    #[cfg(feature = "winit")]
    {
        use wgpu_native::winit::{ControlFlow, Event, ElementState, EventsLoop, KeyboardInput, Window, WindowEvent, VirtualKeyCode};

        let mut events_loop = EventsLoop::new();
        let window = Window::new(&events_loop).unwrap();
        let size = window
            .get_inner_size()
            .unwrap()
            .to_physical(window.get_hidpi_factor());

        let surface = instance.create_surface(&window);
        let mut swap_chain = device.create_swap_chain(&surface, &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::B8g8r8a8Unorm,
            width: size.width as u32,
            height: size.height as u32,
        });

        events_loop.run_forever(|event| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input: KeyboardInput { virtual_keycode: Some(code), state: ElementState::Pressed, .. },
                        ..
                    } => match code {
                        VirtualKeyCode::Escape => {
                            return ControlFlow::Break
                        }
                        _ => {}
                    }
                    WindowEvent::CloseRequested => {
                        return ControlFlow::Break
                    }
                    _ => {}
                }
                _ => {}
            }

            let frame = swap_chain.get_next_texture();
            let mut cmd_buf = device.create_command_buffer(&wgpu::CommandBufferDescriptor { todo: 0 });
            {
                let mut rpass = cmd_buf.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color::GREEN,
                    }],
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(&render_pipeline);
                rpass.draw(0..3, 0..1);
                rpass.end_pass();
            }

            device
                .get_queue()
                .submit(&[cmd_buf]);

            ControlFlow::Continue
        });
    }
}
