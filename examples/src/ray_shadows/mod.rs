use std::{borrow::Cow, future::Future, iter, mem, pin::Pin, task, time::Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use wgpu::{vertex_attr_array, IndexFormat, VertexBufferLayout};

// from cube
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 3],
    _normal: [f32; 3],
}

fn vertex(pos: [f32; 3], normal: [f32; 3]) -> Vertex {
    Vertex {
        _pos: pos,
        _normal: normal,
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // base
        vertex([-1.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        vertex([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        vertex([1.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        vertex([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        //shadow caster
        vertex([-(1.0 / 3.0), 0.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([-(1.0 / 3.0), 2.0 / 3.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([1.0 / 3.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        vertex([1.0 / 3.0, 2.0 / 3.0, 1.0], [0.0, 0.0, 1.0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 1, //base
        4, 5, 6, 6, 7, 5,
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    vertex: Mat4,
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

struct Example {
    uniforms: Uniforms,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    start_inst: Instant,
}

const CAM_LOOK_AT: Vec3 = Vec3::new(0.0, 1.0, -1.5);

fn create_matrix(config: &wgpu::SurfaceConfiguration) -> Uniforms {
    let view = Mat4::look_at_rh(CAM_LOOK_AT, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(
        59.0_f32.to_radians(),
        config.width as f32 / config.height as f32,
        0.1,
        1000.0,
    );

    Uniforms {
        view_inverse: view.inverse(),
        proj_inverse: proj.inverse(),
        vertex: (proj * view),
    }
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
            | wgpu::Features::PUSH_CONSTANTS
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 12,
            ..wgpu::Limits::default()
        }
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let uniforms = create_matrix(config);

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

        let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: vertex_data.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint16),
            index_count: Some(index_data.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_geo_size_desc.clone()],
            },
        );

        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: 1,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure,
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::FRAGMENT,
                range: 0..12,
            }],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: Default::default(),
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
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

        let mut tlas_package = wgpu::TlasPackage::new(tlas);

        tlas_package[0] = Some(wgpu::TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &blas_geo_size_desc,
                        vertex_buffer: &vertex_buf,
                        first_vertex: 0,
                        vertex_stride: mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&index_buf),
                        index_buffer_offset: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            iter::once(&tlas_package),
        );

        queue.submit(Some(encoder.finish()));

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
                    resource: tlas_package.as_binding(),
                },
            ],
        });

        let start_inst = Instant::now();

        Example {
            uniforms,
            uniform_buf,
            vertex_buf,
            index_buf,
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
        self.uniforms = create_matrix(config);

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.uniforms]));
        queue.submit(None);
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        //device.push_error_scope(wgpu::ErrorFilter::Validation);
        const LIGHT_DISTANCE: f32 = 5.0;
        const TIME_SCALE: f32 = -0.2;
        const INITIAL_TIME: f32 = 1.0;
        let cos = (self.start_inst.elapsed().as_secs_f32() * TIME_SCALE + INITIAL_TIME).cos()
            * LIGHT_DISTANCE;
        let sin = (self.start_inst.elapsed().as_secs_f32() * TIME_SCALE + INITIAL_TIME).sin()
            * LIGHT_DISTANCE;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
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
            rpass.set_bind_group(0, Some(&self.bind_group), &[]);
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, &0.0_f32.to_ne_bytes());
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 4, &cos.to_ne_bytes());
            rpass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 8, &sin.to_ne_bytes());
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.set_index_buffer(self.index_buf.slice(..), IndexFormat::Uint16);
            rpass.draw_indexed(0..12, 0, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray-shadows");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_cube_shadows",
    image_path: "/examples/src/ray_shadows/screenshot.png",
    width: 1024,
    height: 768,
    optional_features: wgpu::Features::default(),
    base_test_parameters: wgpu_test::TestParameters {
        required_features: <Example as crate::framework::Example>::required_features(),
        required_limits: <Example as crate::framework::Example>::required_limits(),
        skips: vec![],
        failures: Vec::new(),
        required_downlevel_caps:
            <Example as crate::framework::Example>::required_downlevel_capabilities(),
        force_fxc: false,
    },
    comparisons: &[wgpu_test::ComparisonType::Mean(0.02)],
    _phantom: std::marker::PhantomData::<Example>,
};
