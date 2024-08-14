use std::{borrow::Cow, future::Future, iter, mem, pin::Pin, task, time::Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use rt::traits::*;
use wgpu::{ray_tracing as rt, StoreOp};

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AccelerationStructureInstance {
    transform: [f32; 12],
    custom_index_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

impl std::fmt::Debug for AccelerationStructureInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("transform", &self.transform)
            .field("custom_index()", &self.custom_index())
            .field("mask()", &self.mask())
            .field(
                "shader_binding_table_record_offset()",
                &self.shader_binding_table_record_offset(),
            )
            .field("flags()", &self.flags())
            .field(
                "acceleration_structure_reference",
                &self.acceleration_structure_reference,
            )
            .finish()
    }
}

#[allow(dead_code)]
impl AccelerationStructureInstance {
    const LOW_24_MASK: u32 = 0x00ff_ffff;
    const MAX_U24: u32 = (1u32 << 24u32) - 1u32;

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

    #[inline]
    fn rows_to_affine(rows: &[f32; 12]) -> Affine3A {
        Affine3A::from_cols_array(&[
            rows[0], rows[3], rows[6], rows[9], rows[1], rows[4], rows[7], rows[10], rows[2],
            rows[5], rows[8], rows[11],
        ])
    }

    pub fn transform_as_affine(&self) -> Affine3A {
        Self::rows_to_affine(&self.transform)
    }
    pub fn set_transform(&mut self, transform: &Affine3A) {
        self.transform = Self::affine_to_rows(transform);
    }

    pub fn custom_index(&self) -> u32 {
        self.custom_index_and_mask & Self::LOW_24_MASK
    }

    pub fn mask(&self) -> u8 {
        (self.custom_index_and_mask >> 24) as u8
    }

    pub fn shader_binding_table_record_offset(&self) -> u32 {
        self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK
    }

    pub fn flags(&self) -> u8 {
        (self.shader_binding_table_record_offset_and_flags >> 24) as u8
    }

    pub fn set_custom_index(&mut self, custom_index: u32) {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        self.custom_index_and_mask =
            (custom_index & Self::LOW_24_MASK) | (self.custom_index_and_mask & !Self::LOW_24_MASK)
    }

    pub fn set_mask(&mut self, mask: u8) {
        self.custom_index_and_mask =
            (self.custom_index_and_mask & Self::LOW_24_MASK) | (u32::from(mask) << 24)
    }

    pub fn set_shader_binding_table_record_offset(
        &mut self,
        shader_binding_table_record_offset: u32,
    ) {
        debug_assert!(shader_binding_table_record_offset <= Self::MAX_U24, "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24);
        self.shader_binding_table_record_offset_and_flags = (shader_binding_table_record_offset
            & Self::LOW_24_MASK)
            | (self.shader_binding_table_record_offset_and_flags & !Self::LOW_24_MASK)
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.shader_binding_table_record_offset_and_flags =
            (self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK)
                | (u32::from(flags) << 24)
    }

    pub fn new(
        transform: &Affine3A,
        custom_index: u32,
        mask: u8,
        shader_binding_table_record_offset: u32,
        flags: u8,
        acceleration_structure_reference: u64,
    ) -> Self {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        debug_assert!(
            shader_binding_table_record_offset <= Self::MAX_U24,
            "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24
        );
        AccelerationStructureInstance {
            transform: Self::affine_to_rows(transform),
            custom_index_and_mask: (custom_index & Self::MAX_U24) | (u32::from(mask) << 24),
            shader_binding_table_record_offset_and_flags: (shader_binding_table_record_offset
                & Self::MAX_U24)
                | (u32::from(flags) << 24),
            acceleration_structure_reference,
        }
    }
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
    rt_target: wgpu::Texture,
    rt_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    blas: rt::Blas,
    tlas_package: rt::TlasPackage,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    start_inst: Instant,
}

impl crate::framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
            | wgpu::Features::VERTEX_WRITABLE_STORAGE
            | wgpu::Features::RAY_QUERY
            | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
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
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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
            usage: wgpu::BufferUsages::UNIFORM,
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
            label: Some("rt_computer"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt"),
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
            multiview: None,
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

        let mut tlas_package = rt::TlasPackage::new(tlas, side_count * side_count);

        let dist = 3.0;

        for x in 0..side_count {
            for y in 0..side_count {
                *tlas_package
                    .get_mut_single((x + y * side_count) as usize)
                    .unwrap() = Some(rt::TlasInstance::new(
                    &blas,
                    AccelerationStructureInstance::affine_to_rows(
                        &Affine3A::from_rotation_translation(
                            Quat::from_rotation_y(45.9_f32.to_radians()),
                            Vec3 {
                                x: x as f32 * dist,
                                y: y as f32 * dist,
                                z: -30.0,
                            },
                        ),
                    ),
                    0,
                    0xff,
                ));
            }
        }

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
            rt_target,
            rt_view,
            sampler,
            uniform_buf,
            vertex_buf,
            index_buf,
            blas,
            tlas_package,
            compute_pipeline,
            compute_bind_group,
            blit_pipeline,
            blit_bind_group,
            start_inst,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        let anim_time = self.start_inst.elapsed().as_secs_f64() as f32;

        self.tlas_package
            .get_mut_single(0)
            .unwrap()
            .as_mut()
            .unwrap()
            .transform =
            AccelerationStructureInstance::affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::from_euler(
                    glam::EulerRot::XYZ,
                    anim_time * 0.342,
                    anim_time * 0.254,
                    anim_time * 0.832,
                ),
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: -6.0,
                },
            ));

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas_package));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.dispatch_workgroups(self.rt_target.width() / 8, self.rt_target.height() / 8, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.blit_pipeline);
            rpass.set_bind_group(0, &self.blit_bind_group, &[]);
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
    name: "ray_cube_compute",
    image_path: "/examples/src/ray_cube_compute/screenshot.png",
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
