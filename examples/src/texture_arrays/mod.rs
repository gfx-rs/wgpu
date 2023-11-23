use bytemuck::{Pod, Zeroable};
use std::num::{NonZeroU32, NonZeroU64};
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
    Blue,
    White,
}

fn create_texture_data(color: Color) -> [u8; 4] {
    match color {
        Color::Red => [255, 0, 0, 255],
        Color::Green => [0, 255, 0, 255],
        Color::Blue => [0, 0, 255, 255],
        Color::White => [255, 255, 255, 255],
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

impl crate::framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_BINDING_ARRAY
    }
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut uniform_workaround = false;
        let base_shader_module = device.create_shader_module(wgpu::include_wgsl!("indexing.wgsl"));
        let env_override = match std::env::var("WGPU_TEXTURE_ARRAY_STYLE") {
            Ok(value) => match &*value.to_lowercase() {
                "nonuniform" | "non_uniform" => Some(true),
                "uniform" => Some(false),
                _ => None,
            },
            Err(_) => None,
        };
        let fragment_entry_point = match (device.features(), env_override) {
            (_, Some(false)) => {
                uniform_workaround = true;
                "uniform_main"
            }
            (_, Some(true)) => "non_uniform_main",
            (f, _)
                if f.contains(
                    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                ) =>
            {
                "non_uniform_main"
            }
            _ => {
                uniform_workaround = true;
                "uniform_main"
            }
        };
        let non_uniform_shader_module;
        // TODO: Because naga's capibilities are evaluated on validate, not on write, we cannot make a shader module with unsupported
        // capabilities even if we don't use it. So for now put it in a separate module.
        let fragment_shader_module = if !uniform_workaround {
            non_uniform_shader_module =
                device.create_shader_module(wgpu::include_wgsl!("non_uniform_indexing.wgsl"));
            &non_uniform_shader_module
        } else {
            &base_shader_module
        };

        println!("Using fragment entry point '{fragment_entry_point}'");

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

        let mut texture_index_buffer_contents = vec![0u32; 128];
        texture_index_buffer_contents[0] = 0;
        texture_index_buffer_contents[64] = 1;
        let texture_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&texture_index_buffer_contents),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let red_texture_data = create_texture_data(Color::Red);
        let green_texture_data = create_texture_data(Color::Green);
        let blue_texture_data = create_texture_data(Color::Blue);
        let white_texture_data = create_texture_data(Color::White);

        let texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        };
        let red_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("red"),
            view_formats: &[],
            ..texture_descriptor
        });
        let green_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("green"),
            view_formats: &[],
            ..texture_descriptor
        });
        let blue_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blue"),
            view_formats: &[],
            ..texture_descriptor
        });
        let white_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("white"),
            view_formats: &[],
            ..texture_descriptor
        });

        let red_texture_view = red_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let green_texture_view = green_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let blue_texture_view = blue_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let white_texture_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());

        queue.write_texture(
            red_texture.as_image_copy(),
            &red_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d::default(),
        );
        queue.write_texture(
            green_texture.as_image_copy(),
            &green_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d::default(),
        );
        queue.write_texture(
            blue_texture.as_image_copy(),
            &blue_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d::default(),
        );
        queue.write_texture(
            white_texture.as_image_copy(),
            &white_texture_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(2),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: NonZeroU32::new(2),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    },
                    count: None,
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
                    resource: wgpu::BindingResource::TextureViewArray(&[
                        &blue_texture_view,
                        &white_texture_view,
                    ]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::SamplerArray(&[&sampler, &sampler]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &texture_index_buffer,
                        offset: 0,
                        size: Some(NonZeroU64::new(4).unwrap()),
                    }),
                },
            ],
            layout: &bind_group_layout,
            label: Some("bind group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let index_format = wgpu::IndexFormat::Uint16;

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &base_shader_module,
                entry_point: "vert_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: vertex_size as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Sint32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: fragment_shader_module,
                entry_point: fragment_entry_point,
                targets: &[Some(config.view_formats[0].into())],
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
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        if self.uniform_workaround {
            rpass.set_bind_group(0, &self.bind_group, &[0]);
            rpass.draw_indexed(0..6, 0, 0..1);
            rpass.set_bind_group(0, &self.bind_group, &[256]);
            rpass.draw_indexed(6..12, 0, 0..1);
        } else {
            rpass.set_bind_group(0, &self.bind_group, &[0]);
            rpass.draw_indexed(0..12, 0, 0..1);
        }

        drop(rpass);

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("texture-arrays");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "texture-arrays",
    image_path: "/examples/src/texture_arrays/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::empty(),
    base_test_parameters: wgpu_test::TestParameters::default(),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.0)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST_UNIFORM: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "texture-arrays-uniform",
    image_path: "/examples/src/texture_arrays/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::empty(),
    base_test_parameters: wgpu_test::TestParameters::default(),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.0)],
    _phantom: std::marker::PhantomData::<Example>,
};

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST_NON_UNIFORM: crate::framework::ExampleTestParams =
    crate::framework::ExampleTestParams {
        name: "texture-arrays-non-uniform",
        image_path: "/examples/src/texture_arrays/screenshot.png",
        width: 1024,
        height: 768,
        optional_features:
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        base_test_parameters: wgpu_test::TestParameters::default(),
        comparisons: &[wgpu_test::ComparisonType::Mean(0.0)],
        _phantom: std::marker::PhantomData::<Example>,
    };
