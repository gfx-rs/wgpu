#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};

use wgpu::vertex_attr_array;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    #[allow(dead_code)]
    pos: [f32; 4],
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

fn create_vertices() -> Vec<Vertex> {
    vec![
        Vertex {
            pos: [100.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [100.0, 1000.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-100.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-100.0, 1000.0, 0.0, 1.0],
        },
    ]
}

fn create_texels(size: usize, cx: f32, cy: f32) -> Vec<u8> {
    use std::iter;

    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let mut x = 4.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let mut y = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let mut count = 0;
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 2) as u8)
                .chain(iter::once(0xFF - (count * 5) as u8))
                .chain(iter::once(0xFF - (count * 13) as u8))
                .chain(iter::once(1))
        })
        .collect()
}

struct Example {
    vertex_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    draw_pipeline: wgpu::RenderPipeline,
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 1000.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(0f32, 0.0, 10.0),
            cgmath::Point3::new(0f32, 50.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_correction = framework::OPENGL_TO_WGPU_MATRIX;
        mx_correction * mx_projection * mx_view
    }

    fn generate_mipmaps(
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        mip_count: u32,
    ) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                },
            ],
            label: None,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let vs_bytes = include_bytes!("blit.vert.spv");
        let fs_bytes = include_bytes!("blit.frag.spv");
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs_bytes[..])).unwrap());
        let fs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs_bytes[..])).unwrap());

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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: TEXTURE_FORMAT,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Undefined,
        });

        let views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    format: TEXTURE_FORMAT,
                    dimension: wgpu::TextureViewDimension::D2,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                })
            })
            .collect::<Vec<_>>();

        for target_mip in 1..mip_count as usize {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[target_mip - 1]),
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: None,
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &views[target_mip],
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::WHITE,
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }
    }
}

impl framework::Example for Example {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
    ) -> (Self, Option<wgpu::CommandBuffer>) {
        use std::mem;

        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let vertex_data = create_vertices();
        let vertex_buf = device.create_buffer_with_data(
            bytemuck::cast_slice(&vertex_data),
            wgpu::BufferUsage::VERTEX,
        );

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        component_type: wgpu::TextureComponentType::Float,
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        // Create the texture
        let mip_level_count = 9;
        let size = 1 << mip_level_count;
        let texels = create_texels(size as usize, -0.8, 0.156);
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::OUTPUT_ATTACHMENT
                | wgpu::TextureUsage::COPY_DST,
            label: None,
        });
        let texture_view = texture.create_default_view();
        let temp_buf =
            device.create_buffer_with_data(texels.as_slice(), wgpu::BufferUsage::COPY_SRC);
        init_encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &temp_buf,
                offset: 0,
                bytes_per_row: 4 * size,
                rows_per_image: 0,
            },
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            texture_extent,
        );

        // Create other resources
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: wgpu::CompareFunction::Undefined,
        });
        let mx_total = Self::generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_with_data(
            bytemuck::cast_slice(mx_ref),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0..64,
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
            label: None,
        });

        // Create the render pipeline
        let vs_bytes = include_bytes!("draw.vert.spv");
        let fs_bytes = include_bytes!("draw.frag.spv");
        let vs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs_bytes[..])).unwrap());
        let fs_module = device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs_bytes[..])).unwrap());

        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
            color_states: &[wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: vertex_size as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float4],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Done
        Self::generate_mipmaps(&mut init_encoder, &device, &texture, mip_level_count);

        let this = Example {
            vertex_buf,
            bind_group,
            uniform_buf,
            draw_pipeline,
        };
        (this, Some(init_encoder.finish()))
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
    ) -> Option<wgpu::CommandBuffer> {
        let mx_total = Self::generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();

        let temp_buf = device
            .create_buffer_with_data(bytemuck::cast_slice(mx_ref), wgpu::BufferUsage::COPY_SRC);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.uniform_buf, 0, 64);
        Some(encoder.finish())
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainOutput,
        device: &wgpu::Device,
    ) -> wgpu::CommandBuffer {
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
            rpass.set_pipeline(&self.draw_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, &self.vertex_buf, 0, 0);
            rpass.draw(0..4, 0..1);
        }

        encoder.finish()
    }
}

fn main() {
    framework::run::<Example>("mipmap");
}
