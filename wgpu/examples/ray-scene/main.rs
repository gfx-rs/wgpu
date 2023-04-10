use std::{borrow::Cow, future::Future, iter, mem, ops::Range, pin::Pin, task, time::Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use rt::traits::*;
use wgpu::ray_tracing as rt;

#[path = "../framework.rs"]
mod framework;

// from cube
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 3],
    _tex_coord: [f32; 2],
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
                panic!("Rendering {e}");
            }
        })
    }
}

#[derive(Debug, Clone, Default)]
struct RawSceneComponents {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    geometries: Vec<Range<usize>>,
    instances: Vec<(Range<usize>, Range<usize>)>, //vertex range, geometry range
}

#[allow(dead_code)]
struct SceneComponents {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    geometries: wgpu::Buffer,
    instances: wgpu::Buffer,
    bottom_level_acceleration_structures: Vec<rt::Blas>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceEntry {
    first_vertex: u32,
    first_geometry: u32,
}

fn load_model(scene: &mut RawSceneComponents, source: &[u8]) {
    let data = obj::ObjData::load_buf(source).unwrap();

    let start_vertex_index = scene.vertices.len();
    let start_geometry_index = scene.geometries.len();

    scene.vertices.extend(
        data.position
            .iter()
            // .zip(data.normal.iter())
            .map(|pos| Vertex {
                _pos: *pos,
                _tex_coord: [0.0, 0.0],
            }),
    );

    for object in data.objects {
        for group in object.groups {
            let start_index_index = scene.indices.len();
            for poly in group.polys {
                for end_index in 2..poly.0.len() {
                    for &index in &[0, end_index - 1, end_index] {
                        let obj::IndexTuple(position_id, _texture_id, _normal_id) = poly.0[index];
                        scene.indices.push(position_id as u16);
                    }
                }
            }
            scene
                .geometries
                .push(start_index_index..scene.indices.len());
        }
    }
    scene.instances.push((
        start_vertex_index..scene.vertices.len(),
        start_geometry_index..scene.geometries.len(),
    ));

    // dbg!(scene.vertices.len());
    // dbg!(scene.indices.len());
    // dbg!(&scene.geometries);
    // dbg!(&scene.instances);
}

fn upload_scene_components(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &RawSceneComponents,
) -> SceneComponents {
    let geometry_buffer_content = scene
        .geometries
        .iter()
        .map(|geometry| geometry.start as u32)
        .collect::<Vec<u32>>();

    let instance_buffer_content = scene
        .instances
        .iter()
        .map(|geometry| InstanceEntry {
            first_vertex: geometry.0.start as u32,
            first_geometry: geometry.1.start as u32,
        })
        .collect::<Vec<_>>();

    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertices"),
        contents: bytemuck::cast_slice(&scene.vertices),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT,
    });
    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Indices"),
        contents: bytemuck::cast_slice(&scene.indices),
        usage: wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::BLAS_INPUT,
    });
    let geometries = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Geometries"),
        contents: bytemuck::cast_slice(&geometry_buffer_content),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Instances"),
        contents: bytemuck::cast_slice(&instance_buffer_content),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let (size_descriptors, bottom_level_acceleration_structures): (Vec<_>, Vec<_>) = scene
        .instances
        .iter()
        .map(|(vertex_range, geometry_range)| {
            let size_desc: Vec<rt::BlasTriangleGeometrySizeDescriptor> = (*geometry_range)
                .clone()
                .into_iter()
                .map(|i| rt::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: vertex_range.end as u32 - vertex_range.start as u32,
                    index_format: Some(wgpu::IndexFormat::Uint16),
                    index_count: Some(
                        scene.geometries[i].end as u32 - scene.geometries[i].start as u32,
                    ),
                    flags: rt::AccelerationStructureGeometryFlags::OPAQUE,
                })
                .collect();

            let blas = device.create_blas(
                &rt::CreateBlasDescriptor {
                    label: None,
                    flags: rt::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: rt::AccelerationStructureUpdateMode::Build,
                },
                rt::BlasGeometrySizeDescriptors::Triangles {
                    desc: size_desc.clone(),
                },
            );
            (size_desc, blas)
        })
        .unzip();

    let build_entries: Vec<_> = scene
        .instances
        .iter()
        .zip(size_descriptors.iter())
        .zip(bottom_level_acceleration_structures.iter())
        .map(|(((vertex_range, geometry_range), size_desc), blas)| {
            let triangle_geometries: Vec<_> = size_desc
                .iter()
                .zip(geometry_range.clone().into_iter())
                .map(|(size, i)| rt::BlasTriangleGeometry {
                    size,
                    vertex_buffer: &vertices,
                    first_vertex: vertex_range.start as u32,
                    vertex_stride: mem::size_of::<Vertex>() as u64,
                    index_buffer: Some(&indices),
                    index_buffer_offset: Some(scene.geometries[i].start as u64 * 2),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect();

            rt::BlasBuildEntry {
                blas,
                geometry: rt::BlasGeometries::TriangleGeometries(triangle_geometries),
            }
        })
        .collect();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(build_entries.iter(), iter::empty());

    queue.submit(Some(encoder.finish()));

    SceneComponents {
        vertices,
        indices,
        geometries,
        instances,
        bottom_level_acceleration_structures,
    }
}

fn load_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> SceneComponents {
    let mut scene = RawSceneComponents::default();
    load_model(
        &mut scene,
        include_bytes!("../skybox/models/teslacyberv3.0.obj"),
    );

    load_model(&mut scene, include_bytes!("cube.obj"));

    upload_scene_components(device, queue, &scene)
}

#[allow(dead_code)]
struct Example {
    uniforms: Uniforms,
    uniform_buf: wgpu::Buffer,
    tlas_package: rt::TlasPackage,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    start_inst: Instant,
    scene_components: SceneComponents,
}

impl framework::Example for Example {
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
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
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

        let scene_components = load_scene(device, queue);

        let start_inst = Instant::now();

        Example {
            uniforms,
            uniform_buf,
            tlas_package,
            pipeline,
            bind_group,
            start_inst,
            scene_components,
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

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spawner: &framework::Spawner,
    ) {
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        // scene update
        {
            let dist = 3.0;

            let side_count = 2;

            let anim_time = self.start_inst.elapsed().as_secs_f64() as f32;

            for x in 0..side_count {
                for y in 0..side_count {
                    let instance = self
                        .tlas_package
                        .get_mut_single((x + y * side_count) as usize)
                        .unwrap();

                    let blas_index = (x + y) % 2;

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
                            z: -10.0,
                        },
                    );
                    let transform = transform.transpose().to_cols_array()[..12]
                        .try_into()
                        .unwrap();

                    *instance = Some(rt::TlasInstance::new(
                        &self.scene_components.bottom_level_acceleration_structures[blas_index],
                        transform,
                        0,
                        0xff,
                    ));
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
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));

        // If an error occurs, report it and panic.
        spawner.spawn_local(ErrorFuture {
            inner: device.pop_error_scope(),
        });
    }
}

fn main() {
    framework::run::<Example>("ray-cube");
}

#[test]
fn ray_cube_fragment() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/ray-cube-fragment/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters {
            required_features: <Example as framework::Example>::required_features(),
            required_downlevel_properties:
                <Example as framework::Example>::required_downlevel_capabilities(),
            required_limits: <Example as framework::Example>::required_limits(),
            failures: Vec::new(),
        },
        tolerance: 1,
        max_outliers: 1225, // Bounded by swiftshader
    });
}
