#[path = "../framework.rs"]
mod framework;

const SKYBOX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

type Uniform = cgmath::Matrix4<f32>;
type Uniforms = [Uniform; 2];

fn raw_uniforms(uniforms: &Uniforms) -> [f32; 16 * 2] {
    let mut raw = [0f32; 16 * 2];
    raw[..16].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&uniforms[0])[..]);
    raw[16..].copy_from_slice(&AsRef::<[f32; 16]>::as_ref(&uniforms[1])[..]);
    raw
}

pub struct Skybox {
    aspect: f32,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    uniforms: Uniforms,
}

impl Skybox {
    fn generate_uniforms(aspect_ratio: f32) -> Uniforms {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(1.5f32, -5.0, 3.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_correction = framework::OPENGL_TO_WGPU_MATRIX;
        [mx_correction * mx_projection, mx_correction * mx_view]
    }
}

impl framework::Example for Skybox {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
    ) -> (Self, Option<wgpu::CommandBuffer>) {
        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        component_type: wgpu::TextureComponentType::Float,
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::Cube,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
            ],
            label: None,
        });

        // Create the render pipeline
        let vs_bytes = include_bytes!("shader.vert.spv");
        let fs_bytes = include_bytes!("shader.frag.spv");
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs_bytes[..])).unwrap());
        let fs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs_bytes[..])).unwrap());

        let aspect = sc_desc.width as f32 / sc_desc.height as f32;
        let uniforms = Self::generate_uniforms(aspect);
        let uniform_buf = device.create_buffer_with_data(
            bytemuck::cast_slice(&raw_uniforms(&uniforms)),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        // Create the render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Cw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            depth_stencil_state: None,
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Undefined,
        });

        let paths: [&'static [u8]; 6] = [
            &include_bytes!("images/posx.png")[..],
            &include_bytes!("images/negx.png")[..],
            &include_bytes!("images/posy.png")[..],
            &include_bytes!("images/negy.png")[..],
            &include_bytes!("images/posz.png")[..],
            &include_bytes!("images/negz.png")[..],
        ];

        // we set these multiple times, but whatever
        let (mut image_width, mut image_height) = (0, 0);
        let faces = paths
            .iter()
            .map(|png| {
                let png = std::io::Cursor::new(png);
                let decoder = png::Decoder::new(png);
                let (info, mut reader) = decoder.read_info().expect("can read info");
                image_width = info.width;
                image_height = info.height;
                let mut buf = vec![0; info.buffer_size()];
                reader.next_frame(&mut buf).expect("can read png frame");
                buf
            })
            .collect::<Vec<_>>();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SKYBOX_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        });

        for (i, image) in faces.iter().enumerate() {
            log::debug!(
                "Copying skybox image {} of size {},{} to gpu",
                i,
                image_width,
                image_height,
            );
            let image_buf = device.create_buffer_with_data(image, wgpu::BufferUsage::COPY_SRC);

            init_encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &image_buf,
                    offset: 0,
                    bytes_per_row: 4 * image_width,
                    rows_per_image: 0,
                },
                wgpu::TextureCopyView {
                    texture: &texture,
                    mip_level: 0,
                    array_layer: i as u32,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::Extent3d {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                },
            );
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: SKYBOX_FORMAT,
            dimension: wgpu::TextureViewDimension::Cube,
            aspect: wgpu::TextureAspect::default(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_layer_count: 6,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniform_buf.slice(..)),
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
            label: None,
        });
        (
            Self {
                pipeline,
                bind_group,
                uniform_buf,
                aspect,
                uniforms,
            },
            Some(init_encoder.finish()),
        )
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.aspect = sc_desc.width as f32 / sc_desc.height as f32;
        let uniforms = Skybox::generate_uniforms(self.aspect);
        let mx_total = uniforms[0] * uniforms[1];
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(bytemuck::cast_slice(mx_ref), &self.uniform_buf, 0);
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainOutput,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::CommandBuffer {
        // update rotation
        let rotation = cgmath::Matrix4::<f32>::from_angle_x(cgmath::Deg(0.25));
        self.uniforms[1] = self.uniforms[1] * rotation;
        queue.write_buffer(
            bytemuck::cast_slice(&raw_uniforms(&self.uniforms)),
            &self.uniform_buf,
            0,
        );

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3 as u32, 0..1);
        }
        encoder.finish()
    }
}

fn main() {
    framework::run::<Skybox>("skybox");
}
