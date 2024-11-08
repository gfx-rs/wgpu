use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::f32::consts::PI;
use std::ops::IndexMut;
use std::{borrow::Cow, future::Future, iter, mem, ops::Range, pin::Pin, task, time::Instant};
use wgpu::util::DeviceExt;

// from cube
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Default)]
struct Vertex {
    pos: [f32; 3],
    _p0: [u32; 1],
    normal: [f32; 3],
    _p1: [u32; 1],
    uv: [f32; 2],
    _p2: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
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

#[derive(Debug, Clone, Default)]
struct RawSceneComponents {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    geometries: Vec<(Range<usize>, Material)>, // index range, material
    instances: Vec<(Range<usize>, Range<usize>)>, // vertex range, geometry range
}

struct SceneComponents {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    geometries: wgpu::Buffer,
    instances: wgpu::Buffer,
    bottom_level_acceleration_structures: Vec<wgpu::Blas>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceEntry {
    first_vertex: u32,
    first_geometry: u32,
    last_geometry: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct GeometryEntry {
    first_index: u32,
    _p0: [u32; 3],
    material: Material,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
struct Material {
    roughness_exponent: f32,
    metalness: f32,
    specularity: f32,
    _p0: [u32; 1],
    albedo: [f32; 3],
    _p1: [u32; 1],
}

fn load_model(scene: &mut RawSceneComponents, path: &str) {
    let path = env!("CARGO_MANIFEST_DIR").to_string() + "/src" + path;
    println!("{}", path);
    let mut object = obj::Obj::load(path).unwrap();
    object.load_mtls().unwrap();

    let data = object.data;

    let start_vertex_index = scene.vertices.len();
    let start_geometry_index = scene.geometries.len();

    let mut mapping = std::collections::HashMap::<(usize, usize, usize), usize>::new();

    let mut next_index = 0;

    for object in data.objects {
        for group in object.groups {
            let start_index_index = scene.indices.len();
            for poly in group.polys {
                for end_index in 2..poly.0.len() {
                    for &index in &[0, end_index - 1, end_index] {
                        let obj::IndexTuple(position_id, texture_id, normal_id) = poly.0[index];
                        let texture_id = texture_id.expect("uvs required");
                        let normal_id = normal_id.expect("normals required");

                        let index = *mapping
                            .entry((position_id, texture_id, normal_id))
                            .or_insert(next_index);
                        if index == next_index {
                            next_index += 1;

                            scene.vertices.push(Vertex {
                                pos: data.position[position_id],
                                uv: data.texture[texture_id],
                                normal: data.normal[normal_id],
                                ..Default::default()
                            })
                        }

                        scene.indices.push(index as u32);
                    }
                }
            }

            let mut material: Material = Default::default();

            if let Some(obj::ObjMaterial::Mtl(mat)) = group.material {
                if let Some(kd) = mat.kd {
                    material.albedo = kd;
                }
                if let Some(ns) = mat.ns {
                    material.roughness_exponent = ns;
                }
                if let Some(ka) = mat.ka {
                    material.metalness = ka[0];
                }
                if let Some(ks) = mat.ks {
                    material.specularity = ks[0];
                }
            }

            scene
                .geometries
                .push((start_index_index..scene.indices.len(), material));
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
        .map(|(index_range, material)| GeometryEntry {
            first_index: index_range.start as u32,
            material: *material,
            ..Default::default()
        })
        .collect::<Vec<_>>();

    let instance_buffer_content = scene
        .instances
        .iter()
        .map(|geometry| InstanceEntry {
            first_vertex: geometry.0.start as u32,
            first_geometry: geometry.1.start as u32,
            last_geometry: geometry.1.end as u32,
            _pad: 1,
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
            let size_desc: Vec<wgpu::BlasTriangleGeometrySizeDescriptor> = (*geometry_range)
                .clone()
                .map(|i| wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: vertex_range.end as u32 - vertex_range.start as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(
                        scene.geometries[i].0.end as u32 - scene.geometries[i].0.start as u32,
                    ),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                })
                .collect();

            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: size_desc.clone(),
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
                .zip(geometry_range.clone())
                .map(|(size, i)| wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: &vertices,
                    first_vertex: vertex_range.start as u32,
                    vertex_stride: mem::size_of::<Vertex>() as u64,
                    index_buffer: Some(&indices),
                    index_buffer_offset: Some(scene.geometries[i].0.start as u64 * 4),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect();

            wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(triangle_geometries),
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

    load_model(&mut scene, "/skybox/models/teslacyberv3.0.obj");
    load_model(&mut scene, "/ray_scene/cube.obj");

    upload_scene_components(device, queue, &scene)
}

struct Example {
    uniforms: Uniforms,
    uniform_buf: wgpu::Buffer,
    tlas_package: wgpu::TlasPackage,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    start_inst: Instant,
    scene_components: SceneComponents,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
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

        let scene_components = load_scene(device, queue);

        let uniforms = {
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
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
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: side_count * side_count,
        });

        let tlas_package = wgpu::TlasPackage::new(tlas);

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
                    binding: 5,
                    resource: tlas_package.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene_components.vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_components.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene_components.geometries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene_components.instances.as_entire_binding(),
                },
            ],
        });

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

        self.uniforms.proj_inverse = proj.inverse();

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        // scene update
        {
            let dist = 3.5;

            let side_count = 2;

            let anim_time = self.start_inst.elapsed().as_secs_f64() as f32;

            for x in 0..side_count {
                for y in 0..side_count {
                    let instance = self.tlas_package.index_mut(x + y * side_count);

                    let blas_index = (x + y)
                        % self
                            .scene_components
                            .bottom_level_acceleration_structures
                            .len();

                    let x = x as f32 / (side_count - 1) as f32;
                    let y = y as f32 / (side_count - 1) as f32;
                    let x = x * 2.0 - 1.0;
                    let y = y * 2.0 - 1.0;

                    let transform = Mat4::from_rotation_translation(
                        Quat::from_euler(
                            glam::EulerRot::XYZ,
                            anim_time * 0.5 * 0.342,
                            anim_time * 0.5 * 0.254,
                            anim_time * 0.5 * 0.832 + PI,
                        ),
                        Vec3 {
                            x: x * dist,
                            y: y * dist,
                            z: -14.0,
                        },
                    );
                    let transform = transform.transpose().to_cols_array()[..12]
                        .try_into()
                        .unwrap();
                    *instance = Some(wgpu::TlasInstance::new(
                        &self.scene_components.bottom_level_acceleration_structures[blas_index],
                        transform,
                        blas_index as u32,
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
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, Some(&self.bind_group), &[]);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

pub fn main() {
    crate::framework::run::<Example>("ray_scene");
}

#[cfg(test)]
#[wgpu_test::gpu_test]
static TEST: crate::framework::ExampleTestParams = crate::framework::ExampleTestParams {
    name: "ray_scene",
    image_path: "/examples/src/ray_scene/screenshot.png",
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
