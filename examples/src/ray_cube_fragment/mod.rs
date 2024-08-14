use std::{borrow::Cow, future::Future, iter, mem, pin::Pin, task, time::Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use rt::traits::*;
use wgpu::ray_tracing as rt;

// from cube
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: [[f32; 4]; 4],
    proj_inverse: [[f32; 4]; 4],
}

/// A wrapper for `pop_error_scope` futures that panics if an error occurs.
///
/// Given a future `inner` of an `Option<E>` for some error type `E`,
/// wait for the future to be ready, and panic if its value is `Some`.
///
/// This can be done simpler with `FutureExt`, but we don't want to add
/// a dependency just for this small case.
struct ErrorFuture<F> {
    inner: F,
}
impl<F: Future<Output = Option<wgpu::Error>>> Future for ErrorFuture<F> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<()> {
        let inner = unsafe { self.map_unchecked_mut(|me| &mut me.inner) };
        inner.poll(cx).map(|error| {
            if let Some(e) = error {
                panic!("Rendering {}", e);
            }
        })
    }
}

#[allow(dead_code)]
struct Example {
    uniforms: Uniforms,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    blas: rt::Blas,
    tlas_package: rt::TlasPackage,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    start_inst: Instant,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::RAY_QUERY | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let side_count = 8;

        let uniforms = {
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(
                59.0_f32.to_radians(),
                config.width as f32 / config.height as f32,
                0.001,
                1000.0,
            );

            Uniforms {
                view_inverse: view.inverse().to_cols_array_2d(),
                proj_inverse: proj.inverse().to_cols_array_2d(),
            }
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let blas_geo_size_desc = rt::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x4,
            vertex_count: vertex_data.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint16),
            index_count: Some(index_data.len() as u32),
            flags: rt::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &rt::CreateBlasDescriptor {
                label: None,
                flags: rt::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: rt::AccelerationStructureUpdateMode::Build,
            },
            rt::BlasGeometrySizeDescriptors::Triangles {
                desc: vec![blas_geo_size_desc.clone()],
            },
        );

        let tlas = device.create_tlas(&rt::CreateTlasDescriptor {
            label: None,
            flags: rt::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: rt::AccelerationStructureUpdateMode::Build,
            max_instances: side_count * side_count,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
            multiview: None,
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                },
            ],
        });

        let tlas_package = rt::TlasPackage::new(tlas, side_count * side_count);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(
            iter::once(&rt::BlasBuildEntry {
                blas: &blas,
                geometry: rt::BlasGeometries::TriangleGeometries(vec![rt::BlasTriangleGeometry {
                    size: &blas_geo_size_desc,
                    vertex_buffer: &vertex_buf,
                    first_vertex: 0,
                    vertex_stride: mem::size_of::<Vertex>() as u64,
                    index_buffer: Some(&index_buf),
                    index_buffer_offset: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
            // iter::empty(),
            iter::once(&tlas_package),
        );

        queue.submit(Some(encoder.finish()));

        let start_inst = Instant::now();

        Example {
            uniforms,
            uniform_buf,
            vertex_buf,
            index_buf,
            blas,
            tlas_package,
            pipeline,
            bind_group,
            start_inst,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let proj = Mat4::perspective_rh(
            59.0_f32.to_radians(),
            config.width as f32 / config.height as f32,
            0.001,
            1000.0,
        );

        self.uniforms.proj_inverse = proj.inverse().to_cols_array_2d();

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        // scene update
        {
            let dist = 12.0;

            let side_count = 8;

            let anim_time = self.start_inst.elapsed().as_secs_f64() as f32;

            for x in 0..side_count {
                for y in 0..side_count {
                    let instance = self
                        .tlas_package
                        .get_mut_single((x + y * side_count) as usize)
                        .unwrap();

                    let x = x as f32 / (side_count - 1) as f32;
                    let y = y as f32 / (side_count - 1) as f32;
                    let x = x * 2.0 - 1.0;
                    let y = y * 2.0 - 1.0;

                    let transform = Mat4::from_rotation_translation(
                        Quat::from_euler(
                            glam::EulerRot::XYZ,
                            anim_time * 0.5 * 0.342,
                            anim_time * 0.5 * 0.254,
                            anim_time * 0.5 * 0.832,
                        ),
                        Vec3 {
                            x: x * dist,
                            y: y * dist,
                            z: -24.0,
                        },
                    );
                    let transform = transform.transpose().to_cols_array()[..12]
                        .try_into()
                        .unwrap();

                    *instance = Some(rt::TlasInstance::new(&self.blas, transform, 0, 0xff));
                }
            }
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas_package));

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

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-cube");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_cube_fragment",
    image_path: "/examples/src/ray_cube_fragment/screenshot.png",
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
