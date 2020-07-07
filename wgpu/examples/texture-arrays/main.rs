#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    _pos: [f32; 2],
    _tex_coord: [f32; 2],
    _index: u32,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Uniform {
    index: u32,
}

unsafe impl Pod for Uniform {}
unsafe impl Zeroable for Uniform {}

struct UniformWorkaroundData {
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
}

fn vertex(pos: [i8; 2], tc: [i8; 2], index: i8) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
        _index: index as u32,
    }
}

fn create_vertices() -> Vec<Vertex> {
    vec![
        // left rectangle
        vertex([-1, -1], [0, 1], 0),
        vertex([-1, 1], [0, 0], 0),
        vertex([0, 1], [1, 0], 0),
        vertex([0, -1], [1, 1], 0),
        // right rectangle
        vertex([0, -1], [0, 1], 1),
        vertex([0, 1], [0, 0], 1),
        vertex([1, 1], [1, 0], 1),
        vertex([1, -1], [1, 1], 1),
    ]
}

fn create_indices() -> Vec<u16> {
    vec![
        // Left rectangle
        0, 1, 2, // 1st
        2, 0, 3, // 2nd
        // Right rectangle
        4, 5, 6, // 1st
        6, 4, 7, // 2nd
    ]
}

#[derive(Copy, Clone)]
enum Color {
    RED,
    GREEN,
}

fn create_texture_data(color: Color) -> [u8; 4] {
    match color {
        Color::RED => [255, 0, 0, 255],
        Color::GREEN => [0, 255, 0, 255],
    }
}

struct Example {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_workaround_data: Option<UniformWorkaroundData>,
}

impl framework::Example for Example {
    fn needed_features() -> wgpu::Features {
        wgpu::Features::UNSIZED_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING
            | wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY
    }
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut uniform_workaround = false;
        let vs_module = device.create_shader_module(wgpu::include_spirv!("shader.vert.spv"));
        let fs_bytes: Vec<u8> = match device.features() {
            f if f.contains(wgpu::Features::UNSIZED_BINDING_ARRAY) => {
                include_bytes!("unsized-non-uniform.frag.spv").to_vec()
            }
            f if f.contains(wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING) => {
                include_bytes!("non-uniform.frag.spv").to_vec()
            }
            f if f.contains(wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING) => {
                uniform_workaround = true;
                include_bytes!("uniform.frag.spv").to_vec()
            }
            f if f.contains(wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY) => {
                include_bytes!("constant.frag.spv").to_vec()
            }
            _ => {
                panic!(
                    "Graphics adapter does not support any of the features needed for this example"
                );
            }
        };
        let fs_module = device.create_shader_module(wgpu::util::make_spirv(&fs_bytes));

        let vertex_size = std::mem::size_of::<Vertex>();
        let vertex_data = create_vertices();
        let vertex_buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&vertex_data),
            wgpu::BufferUsage::VERTEX,
        );

        let index_data = create_indices();
        let index_buffer = device
            .create_buffer_with_data(bytemuck::cast_slice(&index_data), wgpu::BufferUsage::INDEX);

        let uniform_workaround_data = if uniform_workaround {
            let buffer0 = device.create_buffer_with_data(
                &bytemuck::cast_slice(&[Uniform { index: 0 }]),
                wgpu::BufferUsage::UNIFORM,
            );
            let buffer1 = device.create_buffer_with_data(
                &bytemuck::cast_slice(&[Uniform { index: 1 }]),
                wgpu::BufferUsage::UNIFORM,
            );

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    bindings: &[wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::UniformBuffer {
                            dynamic: false,
                            min_binding_size: None,
                        },
                    )],
                    label: Some("uniform workaround bind group layout"),
                });

            let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buffer0.slice(..)),
                }],
                label: Some("uniform workaround bind group 0"),
            });

            let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buffer1.slice(..)),
                }],
                label: Some("uniform workaround bind group 1"),
            });

            Some(UniformWorkaroundData {
                bind_group_layout,
                bind_group0,
                bind_group1,
            })
        } else {
            None
        };

        let red_texture_data = create_texture_data(Color::RED);
        let green_texture_data = create_texture_data(Color::GREEN);

        let texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        };
        let red_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("red"),
            ..texture_descriptor
        });
        let green_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("green"),
            ..texture_descriptor
        });

        let red_texture_view = red_texture.create_default_view();
        let green_texture_view = green_texture.create_default_view();

        queue.write_texture(
            wgpu::TextureCopyView {
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                texture: &red_texture,
            },
            &red_texture_data,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 0,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        queue.write_texture(
            wgpu::TextureCopyView {
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                texture: &green_texture,
            },
            &green_texture_data,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 0,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    count: Some(2),
                    ..wgpu::BindGroupLayoutEntry::new(
                        0,
                        wgpu::ShaderStage::FRAGMENT,
                        wgpu::BindingType::SampledTexture {
                            component_type: wgpu::TextureComponentType::Float,
                            dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    )
                },
                wgpu::BindGroupLayoutEntry::new(
                    1,
                    wgpu::ShaderStage::FRAGMENT,
                    wgpu::BindingType::Sampler { comparison: false },
                ),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&[
                        red_texture_view,
                        green_texture_view,
                    ]),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let pipeline_layout = if let Some(ref workaround) = uniform_workaround_data {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout, &workaround.bind_group_layout],
            })
        } else {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            })
        };

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
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
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
                    attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2, 2 => Int],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            bind_group,
            pipeline,
            uniform_workaround_data,
        }
    }
    fn resize(
        &mut self,
        _sc_desc: &wgpu::SwapChainDescriptor,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        // noop
    }
    fn update(&mut self, _event: winit::event::WindowEvent) {
        // noop
    }
    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &impl futures::task::LocalSpawn,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..));
        if let Some(ref workaround) = self.uniform_workaround_data {
            rpass.set_bind_group(1, &workaround.bind_group0, &[]);
            rpass.draw_indexed(0..6, 0, 0..1);
            rpass.set_bind_group(1, &workaround.bind_group1, &[]);
            rpass.draw_indexed(6..12, 0, 0..1);
        } else {
            rpass.draw_indexed(0..12, 0, 0..1);
        }

        drop(rpass);

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("texture-arrays");
}
