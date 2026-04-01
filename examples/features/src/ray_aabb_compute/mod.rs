use std::{borrow::Cow, iter, mem};

use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use wgpu::StoreOp;

use crate::utils;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAabb {
    min: [f32; 3],
    max: [f32; 3],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

#[inline]
fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
    let row_0 = mat.matrix3.row(0);
    let row_1 = mat.matrix3.row(1);
    let row_2 = mat.matrix3.row(2);
    let translation = mat.translation;
    [
        row_0.x,
        row_0.y,
        row_0.z,
        translation.x,
        row_1.x,
        row_1.y,
        row_1.z,
        translation.y,
        row_2.x,
        row_2.y,
        row_2.z,
        translation.z,
    ]
}

struct Example {
    rt_target: wgpu::Texture,
    #[expect(dead_code)]
    rt_view: wgpu::TextureView,
    #[expect(dead_code)]
    sampler: wgpu::Sampler,
    #[expect(dead_code)]
    uniform_buf: wgpu::Buffer,
    #[expect(dead_code)]
    aabb_buf: wgpu::Buffer,
    tlas: wgpu::Tlas,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    animation_timer: utils::AnimationTimer,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::VERTEX_WRITABLE_STORAGE
            | wgpu::Features::EXPERIMENTAL_RAY_QUERY
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let aabb_data = [
            GpuAabb {
                min: [-3.5, -0.5, -0.5],
                max: [-1.5, 0.5, 0.5],
                _pad: [0.0; 2],
            },
            GpuAabb {
                min: [-0.5, -0.5, -0.5],
                max: [0.5, 0.5, 0.5],
                _pad: [0.0; 2],
            },
            GpuAabb {
                min: [1.5, -0.5, -0.5],
                max: [3.5, 0.5, 0.5],
                _pad: [0.0; 2],
            },
        ];

        let rt_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rt_target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });

        let rt_view = rt_target.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: None,
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rt_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let uniforms = {
            let view =
                Mat4::look_at_rh(Vec3::new(0.0, 0.5, 5.0), Vec3::new(0.0, 0.0, 0.0), Vec3::Y);
            let proj = Mat4::perspective_rh(
                59.0_f32.to_radians(),
                config.width as f32 / config.height as f32,
                0.001,
                1000.0,
            );

            Uniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AABB primitives"),
            contents: bytemuck::cast_slice(&aabb_data),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
        });

        let aabb_size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
            primitive_count: aabb_data.len() as u32,
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::AABBs {
                descriptors: vec![aabb_size_desc.clone()],
            },
        );

        let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 1,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_aabb_compute"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt_aabb"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_bind_group_layout = compute_pipeline.get_bind_group_layout(0);

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&rt_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: aabb_buf.as_entire_binding(),
                },
            ],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let blit_bind_group_layout = blit_pipeline.get_bind_group_layout(0);

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&rt_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        tlas[0] = Some(wgpu::TlasInstance::new(
            &blas,
            affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::IDENTITY,
                Vec3::new(0.0, 0.0, 0.0),
            )),
            0,
            0xff,
        ));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                    size: &aabb_size_desc,
                    stride: mem::size_of::<GpuAabb>() as wgpu::BufferAddress,
                    aabb_buffer: &aabb_buf,
                    primitive_offset: 0,
                }]),
            }),
            iter::once(&tlas),
        );

        queue.submit(Some(encoder.finish()));

        Example {
            rt_target,
            rt_view,
            sampler,
            uniform_buf,
            aabb_buf,
            tlas,
            compute_pipeline,
            compute_bind_group,
            blit_pipeline,
            blit_bind_group,
            animation_timer: utils::AnimationTimer::default(),
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        let anim_time = self.animation_timer.time();

        self.tlas[0].as_mut().unwrap().transform =
            affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::from_rotation_y(anim_time * 0.4),
                Vec3::new(0.0, 0.0, 0.0),
            ));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, Some(&self.compute_bind_group), &[]);
            cpass.dispatch_workgroups(self.rt_target.width() / 8, self.rt_target.height() / 8, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, Some(&self.blit_bind_group), &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-aabb");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
pub static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_aabb_compute",
    image_path: "/examples/features/src/ray_aabb_compute/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters::default()
        // Metal has no AABB intersection in ray queries yet; image compare fails.
        // https://github.com/gfx-rs/wgpu/pull/9304
        // https://github.com/gfx-rs/wgpu/issues/9100
        .expect_fail(wgpu_test::FailureCase::backend(wgpu::Backends::METAL)),
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
