use glam::{Affine3A, Mat4, Vec3};
use std::time::Instant;
use std::{env, mem};
use wgpu::ray_tracing::{
    AccelerationStructureFlags, AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries,
    BlasGeometrySizeDescriptors, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor,
    CommandEncoderRayTracing, CreateBlasDescriptor, CreateTlasDescriptor, DeviceRayTracing,
    TlasInstance, TlasPackage,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{include_wgsl, BufferUsages, SamplerDescriptor};

struct Example {
    tlas_package: TlasPackage,
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    blit_bind_group: wgpu::BindGroup,
    storage_texture: Texture,
    start: Instant,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE | wgpu::Features::RAY_QUERY
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut index_buffer = false;

        for arg in env::args() {
            if arg == "index_buffer" {
                index_buffer = true;
            } else {
                log::warn!("unknown arg {arg}")
            }
        }

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let blit_shader = device.create_shader_module(include_wgsl!("blit.wgsl"));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl for shader.wgsl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure,
                    count: None,
                },
            ],
        });

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl for blit.wgsl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let vertices: [f32; 9] = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0];

        let vertices_size_in_bytes = vertices.len() * 4;

        let indices: [u32; 3] = [0, 1, 2];

        let indices_size_in_bytes = indices.len() * 4;

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::BLAS_INPUT,
        });

        let index_buffer = if index_buffer {
            Some(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("vertex buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::BLAS_INPUT,
            }))
        } else {
            None
        };

        let blas_size_desc = BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            // 3 coordinates per vertex
            vertex_count: (vertices.len() / 3) as u32,
            index_format: index_buffer.as_ref().map(|_| wgpu::IndexFormat::Uint32),
            index_count: index_buffer.as_ref().map(|_| indices.len() as u32),
            flags: wgpu::ray_tracing::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &CreateBlasDescriptor {
                label: None,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                desc: vec![blas_size_desc],
            },
        );

        let tlas = device.create_tlas(&CreateTlasDescriptor {
            label: None,
            max_instances: 3,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });

        let mut tlas_package = TlasPackage::new(tlas, 3);

        *tlas_package.get_mut_single(0) = TlasInstance::new(
            &blas,
            Affine3A::from_translation(Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            })
            .to_cols_array(),
            0,
            0xff,
        );

        *tlas_package.get_mut_single(2) = TlasInstance::new(
            &blas,
            Affine3A::from_translation(Vec3 {
                x: -1.0,
                y: -1.0,
                z: -2.0,
            })
            .to_cols_array(),
            0,
            0xff,
        );

        *tlas_package.get_mut_single(2) = TlasInstance::new(
            &blas,
            Affine3A::from_translation(Vec3 {
                x: 1.0,
                y: -1.0,
                z: -2.0,
            })
            .to_cols_array(),
            0,
            0xff,
        );

        let uniforms = {
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(59.0_f32.to_radians(), 1.0, 0.001, 1000.0);

            Uniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }
        };

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_ref(&uniforms),
            usage: BufferUsages::UNIFORM,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder.build_acceleration_structures(
            Some(&BlasBuildEntry {
                blas: &blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &blas_size_desc,
                    vertex_buffer: &vertex_buffer,
                    first_vertex: 0,
                    vertex_stride: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    index_buffer: index_buffer.as_ref(),
                    index_buffer_offset: index_buffer.as_ref().map(|_| 0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            Some(&tlas_package),
        );

        queue.submit(Some(encoder.finish()));

        let storage_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: Default::default(),
            address_mode_v: Default::default(),
            address_mode_w: Default::default(),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 1.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline layout for shader.wgsl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline for shader.wgsl"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout for blit.wgsl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline for blit.wgsl"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: Default::default(),
                })],
            }),
            multiview: None,
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group for shader.wgsl"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &storage_tex.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas_package.tlas()),
                },
            ],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group for blit.wgsl"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &storage_tex.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            tlas_package,
            compute_pipeline,
            blit_pipeline,
            bind_group,
            blit_bind_group,
            storage_texture: storage_tex,
            start: Instant::now(),
        }
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.tlas_package
            .get_mut_single(0)
            .unwrap()
            .as_mut()
            .unwrap()
            .transform =
            Affine3A::from_rotation_y(self.start.elapsed().as_secs_f32() * 1000.0).to_cols_array();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(None, Some(&self.tlas_package));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.bind_group), &[]);
            cpass.dispatch_workgroups(
                self.storage_texture.width() / 8,
                self.storage_texture.height() / 8,
                1,
            );
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, Some(&self.blit_bind_group), &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-traced-triangle");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_traced_triangle",
    image_path: "/examples/src/ray_traced_triangle/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters {
        required_features: <Example as crate::framework::Example>::required_features(),
        required_limits: <Example as crate::framework::Example>::required_limits(),
        force_fxc: false,
        skips: vec![],
        failures: Vec::new(),
        required_downlevel_caps:
            <Example as crate::framework::Example>::required_downlevel_capabilities(),
    },
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
