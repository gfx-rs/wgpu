use std::{iter, mem};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

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

fn execute<const USE_INDEX_BUFFER: bool>(ctx: TestingContext) {
    let max_instances = 1000;
    let device = &ctx.device;

    let (vertex_data, index_data) = mesh_gen::create_vertices();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let (index_buf, index_offset, index_format, index_count) = if USE_INDEX_BUFFER {
        (
            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Index Buffer"),
                    contents: bytemuck::cast_slice(&index_data),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
                }),
            ),
            Some(0),
            Some(wgpu::IndexFormat::Uint16),
            Some(index_data.len() as u32)
        )
    } else {
        (None, None, None, None)
    };

    let blas_geo_size_desc = rt::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format,
        index_count,
        flags: rt::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = device.create_blas(
        &rt::CreateBlasDescriptor {
            label: None,
            flags: rt::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: rt::AccelerationStructureUpdateMode::Build,
        },
        rt::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let tlas = device.create_tlas(&rt::CreateTlasDescriptor {
        label: None,
        flags: rt::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: rt::AccelerationStructureUpdateMode::Build,
        max_instances,
    });

    let mut tlas_package = rt::TlasPackage::new(tlas);

    for i in 0..2500 {
        eprintln!("Setting TlasInstances in loop {}", i);
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
                    index_buffer: index_buf.as_ref(),
                    index_buffer_offset: index_offset,
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }),
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
    .run_sync(execute::<false>);
