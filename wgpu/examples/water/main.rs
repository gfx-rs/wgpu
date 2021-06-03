#[path = "../framework.rs"]
mod framework;

mod point_gen;

use bytemuck::{Pod, Zeroable};
use cgmath::Point3;
use std::{borrow::Cow, iter, mem};
use wgpu::util::DeviceExt;

///
/// Radius of the terrain.
///
/// Changing this value will change the size of the
/// water and terrain. Note however, that changes to
/// this value will require modification of the time
/// scale in the `render` method below.
///
const SIZE: f32 = 10.0;

///
/// Location of the camera.
/// Location of light is in terrain/water shaders.
///
const CAMERA: Point3<f32> = Point3 {
    x: -100.0,
    y: 50.0,
    z: 100.0,
};

struct Matrices {
    view: cgmath::Matrix4<f32>,
    flipped_view: cgmath::Matrix4<f32>,
    projection: cgmath::Matrix4<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
struct TerrainUniforms {
    view_projection: [f32; 16],
    clipping_plane: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
struct WaterUniforms {
    view: [f32; 16],
    projection: [f32; 16],
    time_size_width: [f32; 4],
    height: [f32; 4],
}

struct Uniforms {
    terrain_normal: TerrainUniforms,
    terrain_flipped: TerrainUniforms,
    water: WaterUniforms,
}

struct Example {
    water_vertex_buf: wgpu::Buffer,
    water_vertex_count: usize,
    water_bind_group_layout: wgpu::BindGroupLayout,
    water_bind_group: wgpu::BindGroup,
    water_uniform_buf: wgpu::Buffer,
    water_pipeline: wgpu::RenderPipeline,

    terrain_vertex_buf: wgpu::Buffer,
    terrain_vertex_count: usize,
    terrain_normal_bind_group: wgpu::BindGroup,
    ///
    /// Binds to the uniform buffer where the
    /// camera has been placed underwater.
    ///
    terrain_flipped_bind_group: wgpu::BindGroup,
    terrain_normal_uniform_buf: wgpu::Buffer,
    ///
    /// Contains uniform variables where the camera
    /// has been placed underwater.
    ///
    terrain_flipped_uniform_buf: wgpu::Buffer,
    terrain_pipeline: wgpu::RenderPipeline,

    reflect_view: wgpu::TextureView,

    depth_buffer: wgpu::TextureView,

    current_frame: usize,

    ///
    /// Used to prevent issues when rendering after
    /// minimizing the window.
    ///
    active: Option<usize>,
}

impl Example {
    ///
    /// Creates the view matrices, and the corrected projection matrix.
    ///
    fn generate_matrices(aspect_ratio: f32) -> Matrices {
        let projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 10.0, 400.0);
        let reg_view = cgmath::Matrix4::look_at_rh(
            CAMERA,
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_y(), //Note that y is up. Differs from other examples.
        );

        let scale = cgmath::Matrix4::from_nonuniform_scale(8.0, 1.5, 8.0);

        let reg_view = reg_view * scale;

        let flipped_view = cgmath::Matrix4::look_at_rh(
            cgmath::Point3::new(CAMERA.x, -CAMERA.y, CAMERA.z),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_y(),
        );

        let correction = framework::OPENGL_TO_WGPU_MATRIX;

        let flipped_view = flipped_view * scale;

        Matrices {
            view: reg_view,
            flipped_view,
            projection: correction * projection,
        }
    }

    fn generate_uniforms(width: u32, height: u32) -> Uniforms {
        let Matrices {
            view,
            flipped_view,
            projection,
        } = Self::generate_matrices(width as f32 / height as f32);

        Uniforms {
            terrain_normal: TerrainUniforms {
                view_projection: *(projection * view).as_ref(),
                clipping_plane: [0.0; 4],
            },
            terrain_flipped: TerrainUniforms {
                view_projection: *(projection * flipped_view).as_ref(),
                clipping_plane: [0., 1., 0., 0.],
            },
            water: WaterUniforms {
                view: *view.as_ref(),
                projection: *projection.as_ref(),
                time_size_width: [0.0, 1.0, SIZE * 2.0, width as f32],
                height: [height as f32, 0.0, 0.0, 0.0],
            },
        }
    }

    ///
    /// Initializes Uniforms and textures.
    ///
    fn initialize_resources(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        water_uniforms: &wgpu::Buffer,
        terrain_normal_uniforms: &wgpu::Buffer,
        terrain_flipped_uniforms: &wgpu::Buffer,
        water_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (wgpu::TextureView, wgpu::TextureView, wgpu::BindGroup) {
        // Matrices for our projection and view.
        // flipped_view is the view from under the water.
        let Uniforms {
            terrain_normal,
            terrain_flipped,
            water,
        } = Self::generate_uniforms(sc_desc.width, sc_desc.height);

        // Put the uniforms into buffers on the GPU
        queue.write_buffer(
            terrain_normal_uniforms,
            0,
            bytemuck::cast_slice(&[terrain_normal]),
        );
        queue.write_buffer(
            terrain_flipped_uniforms,
            0,
            bytemuck::cast_slice(&[terrain_flipped]),
        );
        queue.write_buffer(water_uniforms, 0, bytemuck::cast_slice(&[water]));

        let texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth_or_array_layers: 1,
        };

        let reflection_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Reflection Render Texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::RENDER_ATTACHMENT,
        });

        let draw_depth_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::RENDER_ATTACHMENT,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let depth_view = draw_depth_buffer.create_view(&wgpu::TextureViewDescriptor::default());

        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: water_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &reflection_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Water Bind Group"),
        });

        (
            reflection_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view,
            water_bind_group,
        )
    }
}

impl framework::Example for Example {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        // Size of one water vertex
        let water_vertex_size = mem::size_of::<point_gen::WaterVertexAttributes>();

        let water_vertices = point_gen::HexWaterMesh::generate(SIZE).generate_points();

        // Size of one terrain vertex
        let terrain_vertex_size = mem::size_of::<point_gen::TerrainVertexAttributes>();

        // Noise generation
        let terrain_noise = noise::OpenSimplex::new();

        // Random colouration
        let mut terrain_random = rand::thread_rng();

        // Generate terrain. The closure determines what each hexagon will look like.
        let terrain =
            point_gen::HexTerrainMesh::generate(SIZE, |point| -> point_gen::TerrainVertex {
                use noise::NoiseFn;
                use rand::Rng;
                let noise = terrain_noise.get([point[0] as f64 / 5.0, point[1] as f64 / 5.0]) + 0.1;

                let y = noise as f32 * 8.0;

                // Multiplies a colour by some random amount.
                fn mul_arr(mut arr: [u8; 4], by: f32) -> [u8; 4] {
                    arr[0] = (arr[0] as f32 * by).min(255.0) as u8;
                    arr[1] = (arr[1] as f32 * by).min(255.0) as u8;
                    arr[2] = (arr[2] as f32 * by).min(255.0) as u8;
                    arr
                }

                // Under water
                const DARK_SAND: [u8; 4] = [235, 175, 71, 255];
                // Coast
                const SAND: [u8; 4] = [217, 191, 76, 255];
                // Normal
                const GRASS: [u8; 4] = [122, 170, 19, 255];
                // Mountain
                const SNOW: [u8; 4] = [175, 224, 237, 255];

                // Random colouration.
                let random = terrain_random.gen::<f32>() * 0.2 + 0.9;

                // Choose colour.
                let colour = if y <= 0.0 {
                    DARK_SAND
                } else if y <= 0.8 {
                    SAND
                } else if y <= 3.0 {
                    GRASS
                } else {
                    SNOW
                };
                point_gen::TerrainVertex {
                    position: Point3 {
                        x: point[0],
                        y,
                        z: point[1],
                    },
                    colour: mul_arr(colour, random),
                }
            });

        // Generate the buffer data.
        let terrain_vertices = terrain.make_buffer_data();

        // Create the buffers on the GPU to hold the data.
        let water_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water vertices"),
            contents: bytemuck::cast_slice(&water_vertices),
            usage: wgpu::BufferUsage::VERTEX,
        });

        let terrain_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain vertices"),
            contents: bytemuck::cast_slice(&terrain_vertices),
            usage: wgpu::BufferUsage::VERTEX,
        });

        // Create the bind group layout. This is what our uniforms will look like.
        let water_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Water Bind Group Layout"),
                entries: &[
                    // Uniform variables such as projection/view.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<WaterUniforms>() as _,
                            ),
                        },
                        count: None,
                    },
                    // Reflection texture.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Depth texture for terrain.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Sampler to be able to sample the textures.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                ],
            });

        let terrain_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain Bind Group Layout"),
                entries: &[
                    // Regular uniform variables like view/projection.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<TerrainUniforms>() as _,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // Create our pipeline layouts.
        let water_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("water"),
                bind_group_layouts: &[&water_bind_group_layout],
                push_constant_ranges: &[],
            });

        let terrain_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain"),
                bind_group_layouts: &[&terrain_bind_group_layout],
                push_constant_ranges: &[],
            });

        let water_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Uniforms"),
            size: mem::size_of::<WaterUniforms>() as _,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_normal_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Normal Terrain Uniforms"),
            size: mem::size_of::<TerrainUniforms>() as _,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_flipped_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flipped Terrain Uniforms"),
            size: mem::size_of::<TerrainUniforms>() as _,
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group.
        // This puts values behind what was laid out in the bind group layout.

        let (reflect_view, depth_buffer, water_bind_group) = Self::initialize_resources(
            sc_desc,
            device,
            queue,
            &water_uniform_buf,
            &terrain_normal_uniform_buf,
            &terrain_flipped_uniform_buf,
            &water_bind_group_layout,
        );

        let terrain_normal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &terrain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_normal_uniform_buf.as_entire_binding(),
            }],
            label: Some("Terrain Normal Bind Group"),
        });
        let terrain_flipped_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &terrain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_flipped_uniform_buf.as_entire_binding(),
            }],
            label: Some("Terrain Flipped Bind Group"),
        });

        // Upload/compile them to GPU code.
        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Metal | wgpu::Backend::Vulkan => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION
            }
            _ => (), //TODO
        }
        let terrain_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("terrain.wgsl"))),
            flags,
        });
        let water_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("water"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("water.wgsl"))),
            flags,
        });

        // Create the render pipelines. These describe how the data will flow through the GPU, and what
        // constraints and modifiers it will have.
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water"),
            // The "layout" is what uniforms will be needed.
            layout: Some(&water_pipeline_layout),
            // Vertex shader and input buffers
            vertex: wgpu::VertexState {
                module: &water_module,
                entry_point: "vs_main",
                // Layout of our vertices. This should match the structs
                // which are uploaded to the GPU. This should also be
                // ensured by tagging on either a `#[repr(C)]` onto a
                // struct, or a `#[repr(transparent)]` if it only contains
                // one item, which is itself `repr(C)`.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: water_vertex_size as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Sint16x2, 1 => Sint8x4],
                }],
            },
            // Fragment shader and output targets
            fragment: Some(wgpu::FragmentState {
                module: &water_module,
                entry_point: "fs_main",
                // Describes how the colour will be interpolated
                // and assigned to the output attachment.
                targets: &[wgpu::ColorTargetState {
                    format: sc_desc.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Max,
                        },
                    }),
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            // How the triangles will be rasterized. This is more important
            // for the terrain because of the beneath-the water shot.
            // This is also dependent on how the triangles are being generated.
            primitive: wgpu::PrimitiveState {
                // What kind of data are we passing in?
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            // Describes how us writing to the depth/stencil buffer
            // will work. Since this is water, we need to read from the
            // depth buffer both as a texture in the shader, and as an
            // input attachment to do depth-testing. We don't write, so
            // depth_write_enabled is set to false. This is called
            // RODS or read-only depth stencil.
            depth_stencil: Some(wgpu::DepthStencilState {
                // We don't use stencil.
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            // No multisampling is used.
            multisample: wgpu::MultisampleState::default(),
        });

        // Same idea as the water pipeline.
        let terrain_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain"),
            layout: Some(&terrain_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &terrain_module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: terrain_vertex_size as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain_module,
                entry_point: "fs_main",
                targets: &[sc_desc.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
        });

        // Done
        Example {
            water_vertex_buf,
            water_vertex_count: water_vertices.len(),
            water_bind_group_layout,
            water_bind_group,
            water_uniform_buf,
            water_pipeline,

            terrain_vertex_buf,
            terrain_vertex_count: terrain_vertices.len(),
            terrain_normal_bind_group,
            terrain_flipped_bind_group,
            terrain_normal_uniform_buf,
            terrain_flipped_uniform_buf,
            terrain_pipeline,

            reflect_view,

            depth_buffer,

            current_frame: 0,

            active: Some(0),
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if sc_desc.width == 0 && sc_desc.height == 0 {
            // Stop rendering altogether.
            self.active = None;
            return;
        }
        self.active = Some(self.current_frame);

        // Regenerate all of the buffers and textures.

        let (reflect_view, depth_buffer, water_bind_group) = Self::initialize_resources(
            sc_desc,
            device,
            queue,
            &self.water_uniform_buf,
            &self.terrain_normal_uniform_buf,
            &self.terrain_flipped_uniform_buf,
            &self.water_bind_group_layout,
        );
        self.water_bind_group = water_bind_group;

        self.depth_buffer = depth_buffer;
        self.reflect_view = reflect_view;
    }

    #[allow(clippy::eq_op)]
    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        // Increment frame count regardless of if we draw.
        self.current_frame += 1;
        let back_color = wgpu::Color {
            r: 161.0 / 255.0,
            g: 246.0 / 255.0,
            b: 255.0 / 255.0,
            a: 1.0,
        };

        // Write the sin/cos values to the uniform buffer for the water.
        let (water_sin, water_cos) = ((self.current_frame as f32) / 600.0).sin_cos();
        queue.write_buffer(
            &self.water_uniform_buf,
            mem::size_of::<[f32; 16]>() as wgpu::BufferAddress * 2,
            bytemuck::cast_slice(&[water_sin, water_cos]),
        );

        // Only render valid frames. See resize method.
        if let Some(active) = self.active {
            if active >= self.current_frame {
                return;
            }
        } else {
            return;
        }

        // The encoder provides a way to turn our instructions here, into
        // a command buffer the GPU can understand.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Command Encoder"),
        });

        // First pass: render the reflection.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &self.reflect_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(back_color),
                        store: true,
                    },
                }],
                // We still need to use the depth buffer here
                // since the pipeline requires it.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            rpass.set_pipeline(&self.terrain_pipeline);
            rpass.set_bind_group(0, &self.terrain_flipped_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.terrain_vertex_buf.slice(..));
            rpass.draw(0..self.terrain_vertex_count as u32, 0..1);
        }
        // Terrain right side up. This time we need to use the
        // depth values, so we must use StoreOp::Store.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(back_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            rpass.set_pipeline(&self.terrain_pipeline);
            rpass.set_bind_group(0, &self.terrain_normal_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.terrain_vertex_buf.slice(..));
            rpass.draw(0..self.terrain_vertex_count as u32, 0..1);
        }
        // Render the water. This reads from the depth buffer, but does not write
        // to it, so it cannot be in the same render pass.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: None,
                    stencil_ops: None,
                }),
            });

            rpass.set_pipeline(&self.water_pipeline);
            rpass.set_bind_group(0, &self.water_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.water_vertex_buf.slice(..));
            rpass.draw(0..self.water_vertex_count as u32, 0..1);
        }

        queue.submit(iter::once(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("water");
}
