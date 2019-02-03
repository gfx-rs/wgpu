extern crate wgpu;

#[path="framework.rs"]
mod fw;


#[derive(Clone)]
struct Vertex {
    pos: [f32; 4],
    tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1,  1], [0, 0]),
        vertex([ 1, -1,  1], [1, 0]),
        vertex([ 1,  1,  1], [1, 1]),
        vertex([-1,  1,  1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1,  1, -1], [1, 0]),
        vertex([ 1,  1, -1], [0, 0]),
        vertex([ 1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([ 1, -1, -1], [0, 0]),
        vertex([ 1,  1, -1], [1, 0]),
        vertex([ 1,  1,  1], [1, 1]),
        vertex([ 1, -1,  1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1,  1], [1, 0]),
        vertex([-1,  1,  1], [0, 0]),
        vertex([-1,  1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([ 1,  1, -1], [1, 0]),
        vertex([-1,  1, -1], [0, 0]),
        vertex([-1,  1,  1], [0, 1]),
        vertex([ 1,  1,  1], [1, 1]),
        // back (0, -1, 0)
        vertex([ 1, -1,  1], [0, 0]),
        vertex([-1, -1,  1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([ 1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
         0,  1,  2,  2,  3,  0, // top
         4,  5,  6,  6,  7,  4, // bottom
         8,  9, 10, 10, 11,  8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

struct Cube {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl fw::Example for Cube {
    fn init(device: &mut wgpu::Device) -> Self {
        use std::mem;

        let mut init_command_buf = device.create_command_buffer(&wgpu::CommandBufferDescriptor {
            todo: 0,
        });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (vertex_data.len() * vertex_size) as u32,
            usage: wgpu::BufferUsageFlags::VERTEX | wgpu::BufferUsageFlags::TRANSFER_DST,
        });
        vertex_buf.set_sub_data(0, unsafe { mem::transmute(&vertex_data[..]) });
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (index_data.len() * 2) as u32,
            usage: wgpu::BufferUsageFlags::INDEX | wgpu::BufferUsageFlags::TRANSFER_DST,
        });
        index_buf.set_sub_data(0, unsafe { mem::transmute(&index_data[..]) });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStageFlags::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStageFlags::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        // Create the texture
        let texels = [0x20u8, 0xA0, 0xC0, 0xFF];
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
            array_size: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8g8b8a8Unorm,
            usage: wgpu::TextureUsageFlags::SAMPLED | wgpu::TextureUsageFlags::TRANSFER_DST
        });
        let texture_view = texture.create_default_texture_view();
        let temp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: texels.len() as u32,
            usage: wgpu::BufferUsageFlags::TRANSFER_SRC | wgpu::BufferUsageFlags::TRANSFER_DST
        });
        temp_buf.set_sub_data(0, &texels);
        //init_command_buf.copy_buffer_to_texture(); //TODO!

        // Create the bind group
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            r_address_mode: wgpu::AddressMode::ClampToEdge,
            s_address_mode: wgpu::AddressMode::ClampToEdge,
            t_address_mode: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            max_anisotropy: 0,
            compare_function: wgpu::CompareFunction::Always,
            border_color: wgpu::BorderColor::TransparentBlack,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create the render pipeline
        let (vs_bytes, fs_bytes) = fw::load_glsl_pair("cube");
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);

        let blend_state0 = device.create_blend_state(&wgpu::BlendStateDescriptor::REPLACE);
        let depth_stencil_state =
            device.create_depth_stencil_state(&wgpu::DepthStencilStateDescriptor::IGNORE);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: fw::SWAP_CHAIN_FORMAT,
                    samples: 1,
                }],
                depth_stencil_attachment: None,
            },
            blend_states: &[&blend_state0],
            depth_stencil_state: &depth_stencil_state,
        });

        // Done
        device.get_queue().submit(&[init_command_buf]);
        Cube {
            vertex_buf,
            index_buf,
            bind_group,
            pipeline,
        }
    }

    fn update(&mut self, _event: fw::winit::WindowEvent) {
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
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
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group);
            rpass.set_index_buffer(&self.index_buf, 0);
            rpass.set_vertex_buffers(&[(&self.vertex_buf, 0)]);
            rpass.draw(0..3, 0..1);
            rpass.end_pass();
        }

        device
            .get_queue()
            .submit(&[cmd_buf]);
    }
}

fn main() {
    fw::run::<Cube>();
}
