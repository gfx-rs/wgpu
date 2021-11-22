#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU32;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _tex_coord: [f32; 2],
    _index: u32,
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
    Red,
    Green,
}

fn create_texture_data(color: Color) -> [u8; 4] {
    match color {
        Color::Red => [255, 0, 0, 255],
        Color::Green => [0, 255, 0, 255],
    }
}

struct Example {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    uniform_workaround: bool,
}

impl framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::UNSIZED_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::PUSH_CONSTANTS
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 4,
            ..wgpu::Limits::default()
        }
    }
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut uniform_workaround = false;
        let vs_module = device.create_shader_module(&wgpu::include_spirv!("shader.vert.spv"));
        let fs_source = match device.features() {
            f if f.contains(wgpu::Features::UNSIZED_BINDING_ARRAY) => {
                wgpu::include_spirv_raw!("unsized-non-uniform.frag.spv")
            }
            f if f.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ) =>
            {
                wgpu::include_spirv_raw!("non-uniform.frag.spv")
            }
            f if f.contains(wgpu::Features::TEXTURE_BINDING_ARRAY) => {
                uniform_workaround = true;
                wgpu::include_spirv_raw!("uniform.frag.spv")
            }
            _ => unreachable!(),
        };
        let fs_module = unsafe { device.create_shader_module_spirv(&fs_source) };

        let vertex_size = std::mem::size_of::<Vertex>();
        let vertex_data = create_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_data = create_indices();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        let red_texture_data = create_texture_data(Color::Red);
        let green_texture_data = create_texture_data(Color::Green);

        let texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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

        let red_texture_view = red_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let green_texture_view = green_texture.create_view(&wgpu::TextureViewDescriptor::default());

        queue.write_texture(
            red_texture.as_image_copy(),
            &red_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(4).unwrap()),
                rows_per_image: None,
            },
            wgpu::Extent3d::default(),
        );
        queue.write_texture(
            green_texture.as_image_copy(),
            &green_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(4).unwrap()),
                rows_per_image: None,
            },
            wgpu::Extent3d::default(),
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(2),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: NonZeroU32::new(2),
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&[
                        &red_texture_view,
                        &green_texture_view,
                    ]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::SamplerArray(&[&sampler, &sampler]),
                },
            ],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: if uniform_workaround {
                &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    range: 0..4,
                }]
            } else {
                &[]
            },
        });

        let index_format = wgpu::IndexFormat::Uint16;

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: vertex_size as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Sint32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[config.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_format,
            uniform_workaround,
        }
    }
    fn resize(
        &mut self,
        _sc_desc: &wgpu::SurfaceConfiguration,
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
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view,
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
        rpass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        if self.uniform_workaround {
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::cast_slice(&[0]));
            rpass.draw_indexed(0..6, 0, 0..1);
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::cast_slice(&[1]));
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

// This fails due to an issue with naga https://github.com/gfx-rs/wgpu/issues/1532
#[test]
fn texture_arrays_constant() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/texture-arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters::default().failure(),
        tolerance: 0,
        max_outliers: 0,
    });
}

// This fails due to an issue with naga https://github.com/gfx-rs/wgpu/issues/1532
#[test]
fn texture_arrays_uniform() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/texture-arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::PUSH_CONSTANTS,
        base_test_parameters: framework::test_common::TestParameters::default().failure(),
        tolerance: 0,
        max_outliers: 0,
    });
}

// This fails due to an issue with naga https://github.com/gfx-rs/wgpu/issues/1532
#[test]
fn texture_arrays_non_uniform() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/texture-arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        base_test_parameters: framework::test_common::TestParameters::default().failure(),
        tolerance: 0,
        max_outliers: 0,
    });
}

// This fails due to an issue with naga https://github.com/gfx-rs/wgpu/issues/1532
#[test]
fn texture_arrays_unsized_non_uniform() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/texture-arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::UNSIZED_BINDING_ARRAY,
        base_test_parameters: framework::test_common::TestParameters::default().failure(),
        tolerance: 0,
        max_outliers: 0,
    });
}
