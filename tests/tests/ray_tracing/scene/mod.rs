use std::{iter, mem};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

use wgpu::util::DeviceExt;

use glam::{Affine3A, Quat, Vec3};

mod mesh_gen;

fn acceleration_structure_build(ctx: &TestingContext, use_index_buffer: bool) {
    let max_instances = 1000;
    let device = &ctx.device;

    let (vertex_data, index_data) = mesh_gen::create_vertices();

    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertex_data),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&index_data),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
    });

    let index_format = wgpu::IndexFormat::Uint16;
    let index_count = index_data.len() as u32;

    let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format: use_index_buffer.then_some(index_format),
        index_count: use_index_buffer.then_some(index_count),
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
        max_instances,
    });

    let mut tlas_package = wgpu::TlasPackage::new(tlas);

    for j in 0..max_instances {
        tlas_package[j as usize] = Some(wgpu::TlasInstance::new(
            &blas,
            mesh_gen::affine_to_rows(&Affine3A::from_rotation_translation(
                Quat::from_rotation_y(45.9_f32.to_radians()),
                Vec3 {
                    x: j as f32,
                    y: 0.0,
                    z: 0.0,
                },
            )),
            0,
            0xff,
        ));
    }

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(
        iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: mem::size_of::<mesh_gen::Vertex>() as u64,
                index_buffer: use_index_buffer.then_some(&index_buffer),
                index_buffer_offset: use_index_buffer.then_some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas_package),
    );

    ctx.queue.submit(Some(encoder.finish()));

    ctx.device.poll(wgpu::Maintain::Wait);
}

#[gpu_test]
static ACCELERATION_STRUCTURE_BUILD_NO_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
    )
    .run_sync(|ctx| {
        acceleration_structure_build(&ctx, false);
    });

#[gpu_test]
static ACCELERATION_STRUCTURE_BUILD_WITH_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE),
    )
    .run_sync(|ctx| {
        acceleration_structure_build(&ctx, true);
    });
