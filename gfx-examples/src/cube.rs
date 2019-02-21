mod framework;


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

fn create_texels(size: usize) -> Vec<u8> {
    use std::iter;

    (0 .. size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0*(id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0*(id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x*x + y*y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 5) as u8)
                .chain(iter::once(0xFF - (count * 15) as u8))
                .chain(iter::once(0xFF - (count * 50) as u8))
                .chain(iter::once(1))
        })
        .collect()
}

struct Cube {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
}

impl Cube {
    fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        mx_projection * mx_view
    }
}

impl framework::Example for Cube {
    fn init(device: &mut wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor) -> Self {
        use std::mem;

        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (vertex_data.len() * vertex_size) as u32,
            usage: wgpu::BufferUsageFlags::VERTEX | wgpu::BufferUsageFlags::TRANSFER_DST,
        });
        vertex_buf.set_sub_data(0, framework::cast_slice(&vertex_data));
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: (index_data.len() * 2) as u32,
            usage: wgpu::BufferUsageFlags::INDEX | wgpu::BufferUsageFlags::TRANSFER_DST,
        });
        index_buf.set_sub_data(0, framework::cast_slice(&index_data));

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStageFlags::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStageFlags::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStageFlags::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        // Create the texture
        let size = 256u32;
        let texels = create_texels(size as usize);
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            array_size: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8g8b8a8Unorm,
            usage: wgpu::TextureUsageFlags::SAMPLED | wgpu::TextureUsageFlags::TRANSFER_DST
        });
        let texture_view = texture.create_default_view();
        let temp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: texels.len() as u32,
            usage: wgpu::BufferUsageFlags::TRANSFER_SRC | wgpu::BufferUsageFlags::TRANSFER_DST
        });
        temp_buf.set_sub_data(0, &texels);
        init_encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &temp_buf,
                offset: 0,
                row_pitch: 4 * size,
                image_height: size,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                level: 0,
                slice: 0,
                origin: wgpu::Origin3d {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            texture_extent,
        );

        // Create other resources
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            r_address_mode: wgpu::AddressMode::ClampToEdge,
            s_address_mode: wgpu::AddressMode::ClampToEdge,
            t_address_mode: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            max_anisotropy: 0,
            compare_function: wgpu::CompareFunction::Always,
            border_color: wgpu::BorderColor::TransparentBlack,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            size: 64,
            usage: wgpu::BufferUsageFlags::UNIFORM | wgpu::BufferUsageFlags::TRANSFER_DST,
        });
        let mx_total = Self::generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        uniform_buf.set_sub_data(0, framework::cast_slice(&mx_ref[..]));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0 .. 64,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create the render pipeline
        let vs_bytes = framework::load_glsl("cube.vert", framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl("cube.frag", framework::ShaderStage::Fragment);
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::PipelineStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: wgpu::PipelineStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            },
            rasterization_state: wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: wgpu::MAX_DEPTH_BIAS_CLAMP,
            },
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color: wgpu::BlendDescriptor::REPLACE,
                    alpha: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWriteFlags::ALL,
                },
            ],
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[
                wgpu::VertexBufferDescriptor {
                    stride: vertex_size as u32,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            attribute_index: 0,
                            format: wgpu::VertexFormat::FloatR32G32B32A32,
                            offset: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            attribute_index: 1,
                            format: wgpu::VertexFormat::FloatR32G32,
                            offset: 4 * 4,
                        },
                    ],
                },
            ],
            sample_count: 1,
        });

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);
        Cube {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            bind_group,
            uniform_buf,
            pipeline,
        }
    }

    fn update(&mut self, event: wgpu::winit::WindowEvent) {
        if let wgpu::winit::WindowEvent::Resized(size) = event {
            let mx_total = Self::generate_matrix(size.width as f32 / size.height as f32);
            let mx_ref: &[f32; 16] = mx_total.as_ref();
            self.uniform_buf.set_sub_data(0, framework::cast_slice(&mx_ref[..]));
        }
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group);
            rpass.set_index_buffer(&self.index_buf, 0);
            rpass.set_vertex_buffers(&[(&self.vertex_buf, 0)]);
            rpass.draw_indexed(0 .. self.index_count as u32, 0, 0..1);
        }

        device
            .get_queue()
            .submit(&[encoder.finish()]);
    }
}

fn main() {
    framework::run::<Cube>("cube");
}
