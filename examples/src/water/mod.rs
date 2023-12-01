mod point_gen;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use nanorand::{Rng, WyRand};
use std::{borrow::Cow, f32::consts, iter, mem};
use wgpu::util::DeviceExt;

///
/// Radius of the terrain.
///
/// Changing this value will change the size of the
/// water and terrain. Note however, that changes to
/// this value will require modification of the time
/// scale in the `render` method below.
///
const SIZE: f32 = 29.0;

///
/// Location of the camera.
/// Location of light is in terrain/water shaders.
///
const CAMERA: Vec3 = glam::Vec3::new(-200.0, 70.0, 200.0);

struct Matrices {
    view: glam::Mat4,
    flipped_view: glam::Mat4,
    projection: glam::Mat4,
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
    terrain_normal_uniform_buf: wgpu::Buffer,
    ///
    /// Contains uniform variables where the camera
    /// has been placed underwater.
    ///
    terrain_flipped_uniform_buf: wgpu::Buffer,
    terrain_pipeline: wgpu::RenderPipeline,

    /// A render bundle for drawing the terrain.
    ///
    /// This isn't really necessary, but it does make sure we have at
    /// least one use of `RenderBundleEncoder::set_bind_group` among
    /// the examples.
    terrain_bundle: wgpu::RenderBundle,

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
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 10.0, 400.0);
        let reg_view = glam::Mat4::look_at_rh(
            CAMERA,
            glam::Vec3::new(0f32, 0.0, 0.0),
            glam::Vec3::Y, //Note that y is up. Differs from other examples.
        );

        let scale = glam::Mat4::from_scale(glam::Vec3::new(8.0, 1.5, 8.0));

        let reg_view = reg_view * scale;

        let flipped_view = glam::Mat4::look_at_rh(
            glam::Vec3::new(CAMERA.x, -CAMERA.y, CAMERA.z),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );

        let flipped_view = flipped_view * scale;

        Matrices {
            view: reg_view,
            flipped_view,
            projection,
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
        config: &wgpu::SurfaceConfiguration,
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
        } = Self::generate_uniforms(config.width, config.height);

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
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let reflection_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Reflection Render Texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.view_formats[0],
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let draw_depth_buffer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Buffer"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Color Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Depth Sampler"),
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
                    resource: wgpu::BindingResource::Sampler(&color_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&depth_sampler),
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

impl crate::framework::Example for Example {
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        // Size of one water vertex
        let water_vertex_size = mem::size_of::<point_gen::WaterVertexAttributes>();

        let water_vertices = point_gen::HexWaterMesh::generate(SIZE).generate_points();

        // Size of one terrain vertex
        let terrain_vertex_size = mem::size_of::<point_gen::TerrainVertexAttributes>();

        // Noise generation
        let terrain_noise = noise::OpenSimplex::default();

        // Random colouration
        let mut terrain_random = WyRand::new_seed(42);

        // Generate terrain. The closure determines what each hexagon will look like.
        let terrain =
            point_gen::HexTerrainMesh::generate(SIZE, |point| -> point_gen::TerrainVertex {
                use noise::NoiseFn;
                let noise = terrain_noise.get([point[0] as f64 / 5.0, point[1] as f64 / 5.0]) + 0.1;

                let y = noise as f32 * 22.0;

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
                let random = terrain_random.generate::<f32>() * 0.2 + 0.9;

                // Choose colour.
                let colour = if y <= 0.0 {
                    DARK_SAND
                } else if y <= 0.8 {
                    SAND
                } else if y <= 10.0 {
                    GRASS
                } else {
                    SNOW
                };
                point_gen::TerrainVertex {
                    position: Vec3::new(point[0], y, point[1]),
                    colour: mul_arr(colour, random),
                }
            });

        // Generate the buffer data.
        let terrain_vertices = terrain.make_buffer_data();

        // Create the buffers on the GPU to hold the data.
        let water_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water vertices"),
            contents: bytemuck::cast_slice(&water_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let terrain_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain vertices"),
            contents: bytemuck::cast_slice(&terrain_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create the bind group layout. This is what our uniforms will look like.
        let water_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Water Bind Group Layout"),
                entries: &[
                    // Uniform variables such as projection/view.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Sampler to be able to sample the textures.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Sampler to be able to sample the textures.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
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
                        visibility: wgpu::ShaderStages::VERTEX,
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_normal_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Normal Terrain Uniforms"),
            size: mem::size_of::<TerrainUniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_flipped_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flipped Terrain Uniforms"),
            size: mem::size_of::<TerrainUniforms>() as _,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group.
        // This puts values behind what was laid out in the bind group layout.

        let (reflect_view, depth_buffer, water_bind_group) = Self::initialize_resources(
            config,
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

        // Binds to the uniform buffer where the
        // camera has been placed underwater.
        let terrain_flipped_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &terrain_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_flipped_uniform_buf.as_entire_binding(),
            }],
            label: Some("Terrain Flipped Bind Group"),
        });

        // Upload/compile them to GPU code.
        let terrain_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("terrain.wgsl"))),
        });
        let water_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("water.wgsl"))),
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
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Sint16x2, 1 => Sint8x4],
                }],
            },
            // Fragment shader and output targets
            fragment: Some(wgpu::FragmentState {
                module: &water_module,
                entry_point: "fs_main",
                // Describes how the colour will be interpolated
                // and assigned to the output attachment.
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.view_formats[0],
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
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
            multiview: None,
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
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain_module,
                entry_point: "fs_main",
                targets: &[Some(config.view_formats[0].into())],
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
            multiview: None,
        });

        // A render bundle to draw the terrain.
        let terrain_bundle = {
            let mut encoder =
                device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: None,
                    color_formats: &[Some(config.view_formats[0])],
                    depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: 1,
                    multiview: None,
                });
            encoder.set_pipeline(&terrain_pipeline);
            encoder.set_bind_group(0, &terrain_flipped_bind_group, &[]);
            encoder.set_vertex_buffer(0, terrain_vertex_buf.slice(..));
            encoder.draw(0..terrain_vertices.len() as u32, 0..1);
            encoder.finish(&wgpu::RenderBundleDescriptor::default())
        };

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
            terrain_normal_uniform_buf,
            terrain_flipped_uniform_buf,
            terrain_pipeline,
            terrain_bundle,

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
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if config.width == 0 && config.height == 0 {
            // Stop rendering altogether.
            self.active = None;
            return;
        }
        self.active = Some(self.current_frame);

        // Regenerate all of the buffers and textures.

        let (reflect_view, depth_buffer, water_bind_group) = Self::initialize_resources(
            config,
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
    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
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
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.reflect_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(back_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // We still need to use the depth buffer here
                // since the pipeline requires it.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.execute_bundles([&self.terrain_bundle]);
        }
        // Terrain right side up. This time we need to use the
        // depth values, so we must use StoreOp::Store.
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(back_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
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
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.water_pipeline);
            rpass.set_bind_group(0, &self.water_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.water_vertex_buf.slice(..));
            rpass.draw(0..self.water_vertex_count as u32, 0..1);
        }

        queue.submit(iter::once(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("water");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "water",
    image_path: "/examples/src/water/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        .downlevel_flags(wgpu::DownlevelFlags::READ_ONLY_DEPTH_STENCIL),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.01)],
    _phantom: std::marker::PhantomData::<Example>,
};
