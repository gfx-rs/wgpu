use std::{iter, mem};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestingContext, TestParameters};

use rt::traits::*;
use wgpu::ray_tracing as rt;
use wgpu::util::DeviceExt;

use glam::{Affine3A, Quat, Vec3};

use mesh_gen::{AccelerationStructureInstance, Vertex};

mod mesh_gen;

fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        | wgpu::Features::VERTEX_WRITABLE_STORAGE
        | wgpu::Features::RAY_QUERY
        | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
}

fn execute(ctx: TestingContext) {
    let max_instances = 1000;
    let device = &ctx.device;

    let (vertex_data, index_data) = mesh_gen::create_vertices();

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
        max_instances,
    });

    let mut tlas_package = rt::TlasPackage::new(tlas, max_instances);

    for i in 0..10000 {
        for j in 0..max_instances {
            *tlas_package.get_mut_single(0).unwrap() = Some(rt::TlasInstance::new(
                &blas,
                AccelerationStructureInstance::affine_to_rows(
                    &Affine3A::from_rotation_translation(
                        Quat::from_rotation_y(45.9_f32.to_radians()),
                        Vec3 {
                            x: j as f32,
                            y: i as f32,
                            z: 0.0,
                        },
                    ),
                ),
                0,
                0xff,
            ));
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

        ctx.queue.submit(Some(encoder.finish()));
    }

    ctx.device.poll(wgpu::Maintain::Wait);
}

#[gpu_test]
static RAY_TRACING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(required_features()),
    )
    .run_sync(execute);
