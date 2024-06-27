use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::fmt::Display;
use wgpu::util::DeviceExt;

#[derive(PartialEq, Eq)]
enum RenderMode {
    SolidMesh,
    Points,
    Wireframe,
    WireframeThick,
}

impl RenderMode {
    fn get_shader(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        let shader_src = match self {
            RenderMode::SolidMesh => Cow::Borrowed(include_str!("shader/solid_mesh.wgsl")),
            RenderMode::Points => Cow::Borrowed(include_str!("shader/points.wgsl")),
            RenderMode::Wireframe => Cow::Borrowed(include_str!("shader/wireframe.wgsl")),
            RenderMode::WireframeThick => {
                Cow::Borrowed(include_str!("shader/wireframe_thick.wgsl"))
            }
        };

        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Solid {} Shader", self)),
            source: wgpu::ShaderSource::Wgsl(shader_src),
        })
    }

    fn get_topology(&self) -> wgpu::PrimitiveTopology {
        match self {
            RenderMode::SolidMesh => wgpu::PrimitiveTopology::TriangleList,
            RenderMode::Points => wgpu::PrimitiveTopology::PointList,
            RenderMode::Wireframe => wgpu::PrimitiveTopology::LineList,
            RenderMode::WireframeThick => wgpu::PrimitiveTopology::TriangleList,
        }
    }

    fn render_draw<'a, 'b>(
        &self,
        render_pass: &mut wgpu::RenderPass<'a>,
        index_count: u32,
        vertex_count: u32,
        index_buffer_slice: wgpu::BufferSlice<'b>,
    ) where
        'b: 'a,
    {
        match self {
            RenderMode::SolidMesh => {
                render_pass.set_index_buffer(index_buffer_slice, wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..index_count, 0, 0..1)
            }
            RenderMode::Points => render_pass.draw(0..vertex_count, 0..1),
            RenderMode::Wireframe => {
                let num_triangles = index_count / 3;
                render_pass.draw(0..6 * num_triangles, 0..1);
            }
            RenderMode::WireframeThick => {
                let num_triangles = index_count / 3;
                render_pass.draw(0..3 * 6 * num_triangles, 0..1);
            }
        }
    }
}

impl Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            RenderMode::SolidMesh => "SolidMesh".to_string(),
            RenderMode::Points => "Points".to_string(),
            RenderMode::Wireframe => "Wireframe".to_string(),
            RenderMode::WireframeThick => "WireframeThick".to_string(),
        };
        write!(f, "{}", str)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    world: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    screen_width: u32,
    screen_height: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    color: [f32; 4],
}

fn create_vertices() -> (Vec<Vertex>, [u32; 36]) {
    let vertex_data = vec![
        Vertex {
            position: [-1.0, -1.0, -1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [1.0, -1.0, -1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [1.0, -1.0, 1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [-1.0, -1.0, 1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [-1.0, 1.0, -1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [1.0, 1.0, -1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [1.0, 1.0, 1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
        Vertex {
            position: [-1.0, 1.0, 1.0],
            color: [0.0, 1.0, 0.0, 1.0], // Green
        },
    ];

    let index_data: [u32; 36] = [
        0, 1, 2, /* */ 0, 2, 3, // TOP
        4, 5, 6, /* */ 4, 6, 7, // FRONT
        3, 2, 6, /* */ 3, 6, 7, // BACK
        1, 0, 4, /* */ 1, 4, 5, // LEFT
        3, 0, 7, /* */ 0, 7, 4, // RIGHT
        2, 1, 6, /* */ 1, 6, 5, // BOTTOM
    ];

    (vertex_data, index_data)
}

struct Example<const RENDER_MODE: u8> {
    render_mode: RenderMode,
    uniform_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniforms: Uniforms,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl<const RENDER_MODE: u8> Example<RENDER_MODE> {
    fn update_matrix(&mut self, aspect_ratio: f32, screen_width: u32, screen_height: u32) {
        // Assuming the world matrix is identity
        let world_matrix = glam::Mat4::IDENTITY;

        // Projection matrix (perspective)
        let projection_matrix =
            glam::Mat4::perspective_rh_gl(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);

        // View matrix
        let eye = glam::Vec3::new(2.0, 2.0, 5.0);
        let center = glam::Vec3::new(0.0, 0.0, 0.0);
        let up = glam::Vec3::Y;
        let view_matrix = glam::Mat4::look_at_rh(eye, center, up);

        // Store the matrices
        self.uniforms.world = world_matrix.to_cols_array_2d();
        self.uniforms.view = view_matrix.to_cols_array_2d();
        self.uniforms.proj = projection_matrix.to_cols_array_2d();

        self.uniforms.screen_width = screen_width;
        self.uniforms.screen_height = screen_height;
    }
}

impl<const RENDER_MODE: u8> crate::framework::Example for Example<RENDER_MODE> {
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let (vertices, index) = create_vertices();

        let render_mode: RenderMode = match RENDER_MODE {
            0 => RenderMode::SolidMesh,
            1 => RenderMode::Points,
            2 => RenderMode::Wireframe,
            3 => RenderMode::WireframeThick,
            _ => RenderMode::Wireframe,
        };

        let positions: Vec<f32> = vertices.iter().flat_map(|v| v.position.to_vec()).collect();
        let colors: Vec<u32> = vertices
            .iter()
            .map(|v| {
                ((v.color[3] * 255.0) as u32) << 24 // Alpha
                    | ((v.color[0] * 255.0) as u32) << 16 // Red
                    | ((v.color[1] * 255.0) as u32) << 8 // Green
                    | ((v.color[2] * 255.0) as u32) // Blue
            })
            .collect();

        let positions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Position - Vertex buffer"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube color - Vertex buffer"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube - Index Buffer"),
            contents: bytemuck::cast_slice(&index),
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX,
        });

        let uniforms = Uniforms {
            world: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view: glam::Mat4::look_at_rh(
                glam::Vec3::new(0.0, 0.0, 5.0),
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::Y,
            )
            .to_cols_array_2d(),
            proj: glam::Mat4::perspective_rh_gl(
                45.0_f32.to_radians(),
                config.width as f32 / config.height as f32,
                0.1,
                100.0,
            )
            .to_cols_array_2d(),
            screen_width: config.width,
            screen_height: config.height,
            _padding: [0, 0],
        };

        let uniform_buffer_vs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube - Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    /*  Binding 0: uniform buffer */
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    /* Binding 1: positions */
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    /* Binding 2: colors */
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    /* Binding 3: indices */
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Layout"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer_vs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: color_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cube - Bind group layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = render_mode.get_shader(&device);
        let current_texture_format = config.format;

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "main_vertex",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "main_fragment",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: current_texture_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: render_mode.get_topology(),
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,

                unclipped_depth: false,
                polygon_mode: Default::default(),
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Done
        Example {
            render_mode,
            uniform_buffer: uniform_buffer_vs,
            vertex_buffer: positions_buffer,
            index_buffer,
            uniforms,
            pipeline,
            bind_group,
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.update_matrix(
            config.width as f32 / config.height as f32,
            config.width,
            config.height,
        );
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (vertices, index) = create_vertices();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            let index_buffer_slice = self.index_buffer.slice(..);
            self.render_mode.render_draw(
                &mut rpass,
                index.len() as u32,
                vertices.len() as u32,
                index_buffer_slice,
            );
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    let mut args = std::env::args();
    args.next();

    match args.nth(1).as_deref() {
        None => crate::framework::run::<Example<2>>("wireframe_vertex_pulling - Wireframe"),

        Some("solid") => {
            crate::framework::run::<Example<0>>("wireframe_vertex_pulling - Solid Mesh")
        }
        Some("points") => crate::framework::run::<Example<1>>("wireframe_vertex_pulling - Points"),
        Some("wireframe") => {
            crate::framework::run::<Example<2>>("wireframe_vertex_pulling - Wireframe")
        }
        Some("wireframe-thick") => {
            crate::framework::run::<Example<3>>("wireframe_vertex_pulling - Wireframe Thick")
        }

        Some(_) => crate::framework::run::<Example<2>>("wireframe_vertex_pulling - Wireframe"),
    }
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST_WIREFRAME_VERTEX_PULLING_SOLID: crate::framework::ExampleTestParams = {
    let test_parameter =
        wgpu_test::TestParameters::default().limits(wgpu::Limits::downlevel_defaults());

    crate::framework::ExampleTestParams {
        name: "wireframe_vertex_pulling solid",
        // Generated on AMD Phoenix 1 on Vk/Arch
        image_path: "/examples/src/wireframe_vertex_pulling/solid.png",
        width: 1920,
        height: 1200,
        optional_features: wgpu::Features::default(),
        base_test_parameters: test_parameter,
        comparisons: &[wgpu_test::ComparisonType::Mean(0.04)],
        _phantom: std::marker::PhantomData::<Example<0>>,
    }
};

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST_WIREFRAME_VERTEX_PULLING_POINTS: crate::framework::ExampleTestParams = {
    let test_parameter =
        wgpu_test::TestParameters::default().limits(wgpu::Limits::downlevel_defaults());

    crate::framework::ExampleTestParams {
        name: "wireframe_vertex_pulling points",
        // Generated on AMD Phoenix 1 on Vk/Arch
        image_path: "/examples/src/wireframe_vertex_pulling/points.png",
        width: 1920,
        height: 1200,
        optional_features: wgpu::Features::default(),
        base_test_parameters: test_parameter,
        comparisons: &[wgpu_test::ComparisonType::Mean(0.04)],
        _phantom: std::marker::PhantomData::<Example<1>>,
    }
};

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST_WIREFRAME_VERTEX_PULLING_WIREFRAME: crate::framework::ExampleTestParams = {
    let test_parameter =
        wgpu_test::TestParameters::default().limits(wgpu::Limits::downlevel_defaults());

    crate::framework::ExampleTestParams {
        name: "wireframe_vertex_pulling wireframe",
        // Generated on AMD Phoenix 1 on Vk/Arch
        image_path: "/examples/src/wireframe_vertex_pulling/wireframe.png",
        width: 1920,
        height: 1200,
        optional_features: wgpu::Features::default(),
        base_test_parameters: test_parameter,
        comparisons: &[wgpu_test::ComparisonType::Mean(0.04)],
        _phantom: std::marker::PhantomData::<Example<2>>,
    }
};

// TODO find why thick is not working
// #[cfg(test)]
// #[wgpu_test::gpu_test]
// static TEST_WIREFRAME_VERTEX_PULLING_WIREFRAME_THICK: crate::framework::ExampleTestParams = {
//     let test_parameter = wgpu_test::TestParameters::default()
//         .limits(wgpu::Limits::downlevel_defaults());
//
//     crate::framework::ExampleTestParams {
//         name: "wireframe_vertex_pulling wireframe thick",
//         // Generated on AMD Phoenix 1 on Vk/Arch
//         image_path: "/examples/src/wireframe_vertex_pulling/thick.png",
//         width: 1920,
//         height: 1200,
//         optional_features: wgpu::Features::default(),
//         base_test_parameters: test_parameter,
//         comparisons: &[
//             wgpu_test::ComparisonType::Mean(0.04),
//         ],
//         _phantom: std::marker::PhantomData::<Example<3>>,
//     }
// };
