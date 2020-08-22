#[path = "../framework.rs"]
mod framework;

use futures::task::{LocalSpawn, LocalSpawnExt};
use wgpu::util::DeviceExt;

const IMAGE_SIZE: u32 = 512;

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
    staging_belt: wgpu::util::StagingBelt,
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
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_COMPRESSION_BC
    }

    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        component_type: wgpu::TextureComponentType::Float,
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::Cube,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
            ],
        });

        // Create the render pipeline
        let vs_module = device.create_shader_module(wgpu::include_spirv!("shader.vert.spv"));
        let fs_module = device.create_shader_module(wgpu::include_spirv!("shader.frag.spv"));

        let aspect = sc_desc.width as f32 / sc_desc.height as f32;
        let uniforms = Self::generate_uniforms(aspect);
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&raw_uniforms(&uniforms)),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
                ..Default::default()
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[sc_desc.format.into()],
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
            ..Default::default()
        });

        let device_features = device.features();

        let (skybox_format, single_file) =
            if device_features.contains(wgt::Features::TEXTURE_COMPRESSION_BC) {
                (wgpu::TextureFormat::Bc1RgbaUnormSrgb, true)
            } else {
                (wgpu::TextureFormat::Rgba8UnormSrgb, false)
            };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: IMAGE_SIZE,
                height: IMAGE_SIZE,
                depth: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: skybox_format,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        });

        if single_file {
            log::debug!(
                "Copying BC1 skybox images of size {},{},6 to gpu",
                IMAGE_SIZE,
                IMAGE_SIZE,
            );

            let bc1_path: &[u8] = &include_bytes!("images/bc1.dds")[..];

            let mut dds_cursor = std::io::Cursor::new(bc1_path);
            let dds_file = ddsfile::Dds::read(&mut dds_cursor).unwrap();

            let block_width = 4;
            let block_size = 8;

            queue.write_texture(
                wgpu::TextureCopyView {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                },
                &dds_file.data,
                wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: block_size * ((IMAGE_SIZE + (block_width - 1)) / block_width),
                    rows_per_image: IMAGE_SIZE,
                },
                wgpu::Extent3d {
                    width: IMAGE_SIZE,
                    height: IMAGE_SIZE,
                    depth: 6,
                },
            );
        } else {
            let paths: [&'static [u8]; 6] = [
                &include_bytes!("images/posx.png")[..],
                &include_bytes!("images/negx.png")[..],
                &include_bytes!("images/posy.png")[..],
                &include_bytes!("images/negy.png")[..],
                &include_bytes!("images/posz.png")[..],
                &include_bytes!("images/negz.png")[..],
            ];

            let faces = paths
                .iter()
                .map(|png| {
                    let png = std::io::Cursor::new(png);
                    let decoder = png::Decoder::new(png);
                    let (info, mut reader) = decoder.read_info().expect("can read info");
                    let mut buf = vec![0; info.buffer_size()];
                    reader.next_frame(&mut buf).expect("can read png frame");
                    buf
                })
                .collect::<Vec<_>>();

            for (i, image) in faces.iter().enumerate() {
                log::debug!(
                    "Copying skybox image {} of size {},{} to gpu",
                    i,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                );
                queue.write_texture(
                    wgpu::TextureCopyView {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: i as u32,
                        },
                    },
                    &image,
                    wgpu::TextureDataLayout {
                        offset: 0,
                        bytes_per_row: 4 * IMAGE_SIZE,
                        rows_per_image: 0,
                    },
                    wgpu::Extent3d {
                        width: IMAGE_SIZE,
                        height: IMAGE_SIZE,
                        depth: 1,
                    },
                );
            }
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..wgpu::TextureViewDescriptor::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        Skybox {
            pipeline,
            bind_group,
            uniform_buf,
            aspect,
            uniforms,
            staging_belt: wgpu::util::StagingBelt::new(0x100),
        }
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
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &impl LocalSpawn,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // update rotation
        let rotation = cgmath::Matrix4::<f32>::from_angle_x(cgmath::Deg(0.25));
        self.uniforms[1] = self.uniforms[1] * rotation;
        let raw_uniforms = raw_uniforms(&self.uniforms);
        self.staging_belt
            .write_buffer(
                &mut encoder,
                &self.uniform_buf,
                0,
                wgpu::BufferSize::new((raw_uniforms.len() * 4) as wgpu::BufferAddress).unwrap(),
                device,
            )
            .copy_from_slice(bytemuck::cast_slice(&raw_uniforms));

        self.staging_belt.finish();

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3 as u32, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        let belt_future = self.staging_belt.recall();
        spawner.spawn_local(belt_future).unwrap();
    }
}

fn main() {
    framework::run::<Skybox>("skybox");
}
