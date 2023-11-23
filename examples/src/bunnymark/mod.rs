use bytemuck::{Pod, Zeroable};
use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{Key, NamedKey},
};

const MAX_BUNNIES: usize = 1 << 20;
const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4],
    size: [f32; 2],
    pad: [f32; 2],
}

#[repr(C, align(256))]
#[derive(Clone, Copy, Zeroable)]
struct Bunny {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: u32,
}

impl Bunny {
    fn update_data(&mut self, delta: f32, extent: &[u32; 2]) {
        self.position[0] += self.velocity[0] * delta;
        self.position[1] += self.velocity[1] * delta;
        self.velocity[1] += GRAVITY * delta;
        if (self.velocity[0] > 0.0 && self.position[0] + 0.5 * BUNNY_SIZE > extent[0] as f32)
            || (self.velocity[0] < 0.0 && self.position[0] - 0.5 * BUNNY_SIZE < 0.0)
        {
            self.velocity[0] *= -1.0;
        }
        if self.velocity[1] < 0.0 && self.position[1] < 0.5 * BUNNY_SIZE {
            self.velocity[1] *= -1.0;
        }
    }
}

/// Example struct holds references to wgpu resources and frame persistent data
struct Example {
    global_group: wgpu::BindGroup,
    local_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    bunnies: Vec<Bunny>,
    local_buffer: wgpu::Buffer,
    extent: [u32; 2],
    rng: WyRand,
}

impl Example {
    fn spawn_bunnies(&mut self) {
        let spawn_count = 64;
        let color = self.rng.generate::<u32>();
        println!(
            "Spawning {} bunnies, total at {}",
            spawn_count,
            self.bunnies.len() + spawn_count
        );
        for _ in 0..spawn_count {
            let speed = self.rng.generate::<f32>() * MAX_VELOCITY - (MAX_VELOCITY * 0.5);
            self.bunnies.push(Bunny {
                position: [0.0, 0.5 * (self.extent[1] as f32)],
                velocity: [speed, 0.0],
                color,
                _pad: 0,
            });
        }
    }

    fn render_inner(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let delta = 0.01;
        for bunny in self.bunnies.iter_mut() {
            bunny.update_data(delta, &self.extent);
        }

        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
        queue.write_buffer(&self.local_buffer, 0, unsafe {
            std::slice::from_raw_parts(
                self.bunnies.as_ptr() as *const u8,
                self.bunnies.len() * uniform_alignment as usize,
            )
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let clear_color = wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.global_group, &[]);
            for i in 0..self.bunnies.len() {
                let offset =
                    (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
                rpass.set_bind_group(1, &self.local_group, &[offset]);
                rpass.draw(0..4, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../../../wgpu-hal/examples/halmark/shader.wgsl"
            ))),
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Globals>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                label: None,
            });
        let local_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Bunny>() as _),
                    },
                    count: None,
                }],
                label: None,
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.view_formats[0],
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::default(),
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint16),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let texture = {
            let img_data = include_bytes!("../../../logo.png");
            let decoder = png::Decoder::new(std::io::Cursor::new(img_data));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).unwrap();

            let size = wgpu::Extent3d {
                width: info.width,
                height: info.height,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &buf,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(info.width * 4),
                    rows_per_image: None,
                },
                size,
            );
            texture
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let globals = Globals {
            mvp: glam::Mat4::orthographic_rh(
                0.0,
                config.width as f32,
                0.0,
                config.height as f32,
                -1.0,
                1.0,
            )
            .to_cols_array_2d(),
            size: [BUNNY_SIZE; 2],
            pad: [0.0; 2],
        };
        let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("global"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });
        let uniform_alignment =
            device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
        let local_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("local"),
            size: (MAX_BUNNIES as wgpu::BufferAddress) * uniform_alignment,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let global_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        let local_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &local_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(mem::size_of::<Bunny>() as _),
                }),
            }],
            label: None,
        });

        let rng = WyRand::new_seed(42);

        let mut ex = Example {
            pipeline,
            global_group,
            local_group,
            bunnies: Vec::new(),
            local_buffer,
            extent: [config.width, config.height],
            rng,
        };

        ex.spawn_bunnies();

        ex
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        if let winit::event::WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    logical_key: Key::Named(NamedKey::Space),
                    state: ElementState::Pressed,
                    ..
                },
            ..
        } = event
        {
            self.spawn_bunnies();
        }
    }

    fn resize(
        &mut self,
        _sc_desc: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        //empty
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.render_inner(view, device, queue);
    }
}

pub fn main() {
    crate::framework::run::<Example>("bunnymark");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "bunnymark",
    image_path: "/examples/src/bunnymark/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default(),
    // We're looking for very small differences, so look in the high percentiles.
    comparisons: &[
        wgpu_test::ComparisonType::Mean(0.05),
        wgpu_test::ComparisonType::Percentile {
            percentile: 0.99,
            threshold: 0.37,
        },
    ],
    _phantom: std::marker::PhantomData::<Example>,
};
