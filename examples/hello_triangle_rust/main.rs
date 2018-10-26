extern crate wgpu;
fn main() {
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });
    let device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth: 1,
        },
        array_size: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8g8b8a8Unorm,
        usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
    });
    let color_view = texture.create_default_texture_view();

    let vs_bytes = include_bytes!("./../data/hello_triangle.vert.spv");
    let vs_module = device.create_shader_module(vs_bytes);
    let fs_bytes = include_bytes!("./../data/hello_triangle.frag.spv");
    let fs_module = device.create_shader_module(fs_bytes);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let blend_state0 = device.create_blend_state(&wgpu::BlendStateDescriptor::REPLACE);
    let depth_stencil_state = device.create_depth_stencil_state(&wgpu::DepthStencilStateDescriptor::IGNORE);
    let attachment_state = device.create_attachment_state(&wgpu::AttachmentStateDescriptor {
        formats: &[wgpu::TextureFormat::R8g8b8a8Unorm],
    });

    let _render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        blend_states: &[
            &blend_state0,
        ],
        depth_stencil_state: &depth_stencil_state,
        attachment_state: &attachment_state,
    });

    let mut cmd_buf = device.create_command_buffer(&wgpu::CommandBufferDescriptor {});

    {
        let rpass = cmd_buf.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &color_view,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::GREEN,
                },
            ],
            depth_stencil_attachment: None,
        });
        rpass.end_pass();
    }


    let queue = device.get_queue();
    queue.submit(&[cmd_buf]);
}
