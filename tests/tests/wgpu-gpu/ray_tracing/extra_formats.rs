use std::mem;
use std::time::Duration;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

use crate::ray_tracing::acceleration_structure_limits;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(EXTRA_FORMATS_QUERY);
}

const RAY_QUERY_INTERSECTION_TRIANGLE: u32 = 1;

/// Encodes a triangle vertex for a 4-component 16-bit float format.
fn vertex_f16x4(x: f32, y: f32, z: f32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    for (i, v) in [x, y, z, 0.0].into_iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    bytes
}

/// Encodes a triangle vertex for a 2-component 16-bit float format.
/// The missing z component defaults to 0.
fn vertex_f16x2(x: f32, y: f32) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    for (i, v) in [x, y].into_iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    bytes
}

/// Encodes a triangle vertex for a 2-component normalized 16-bit signed integer
/// format. The missing z component defaults to 0.
fn vertex_snorm16x2(x: f32, y: f32) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    for (i, v) in [x, y].into_iter().enumerate() {
        let snorm = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes[i * 2..i * 2 + 2].copy_from_slice(&snorm.to_le_bytes());
    }
    bytes
}

/// Encodes a triangle vertex for a 4-component normalized 16-bit signed integer
/// format. The w component is ignored by the acceleration structure build.
fn vertex_snorm16x4(x: f32, y: f32, z: f32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    for (i, v) in [x, y, z, 1.0].into_iter().enumerate() {
        let snorm = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes[i * 2..i * 2 + 2].copy_from_slice(&snorm.to_le_bytes());
    }
    bytes
}

fn vertex_f32x2(x: f32, y: f32) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    for (i, v) in [x, y].into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn vertex_f32x3(x: f32, y: f32, z: f32) -> [u8; 12] {
    let mut bytes = [0u8; 12];
    for (i, v) in [x, y, z].into_iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Traces a ray down the z axis from `z = -2` against triangles encoded in
/// every vertex format allowed for acceleration structure builds, and checks
/// that the committed intersection lands at the expected distance.
#[gpu_test]
static EXTRA_FORMATS_QUERY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS,
            ),
    )
    .run_async(extra_formats_query);

async fn extra_formats_query(ctx: TestingContext) {
    // Each triangle covers the xy origin. Signed normalized formats cannot
    // leave the [-1, 1] cube, so their triangles sit at z = 1 instead of z = 2,
    // and 2-component formats default the missing z to 0.
    let cases: &[(
        &str,
        VertexFormat,
        Box<dyn Fn() -> Vec<u8> + Send + Sync>,
        f32,
    )] = &[
        (
            "Float32x3",
            VertexFormat::Float32x3,
            Box::new(|| {
                [(-2.0, -2.0, 2.0), (6.0, -2.0, 2.0), (-2.0, 6.0, 2.0)]
                    .map(|(x, y, z)| vertex_f32x3(x, y, z))
                    .concat()
            }),
            4.0,
        ),
        (
            "Float16x4",
            VertexFormat::Float16x4,
            Box::new(|| {
                [(-2.0, -2.0, 2.0), (6.0, -2.0, 2.0), (-2.0, 6.0, 2.0)]
                    .map(|(x, y, z)| vertex_f16x4(x, y, z))
                    .concat()
            }),
            4.0,
        ),
        (
            "Snorm16x4",
            VertexFormat::Snorm16x4,
            Box::new(|| {
                [(-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)]
                    .map(|(x, y, z)| vertex_snorm16x4(x, y, z))
                    .concat()
            }),
            3.0,
        ),
        (
            "Float16x2",
            VertexFormat::Float16x2,
            Box::new(|| {
                [(-2.0, -2.0), (6.0, -2.0), (-2.0, 6.0)]
                    .map(|(x, y)| vertex_f16x2(x, y))
                    .concat()
            }),
            2.0,
        ),
        (
            "Snorm16x2",
            VertexFormat::Snorm16x2,
            Box::new(|| {
                [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]
                    .map(|(x, y)| vertex_snorm16x2(x, y))
                    .concat()
            }),
            2.0,
        ),
        (
            "Float32x2",
            VertexFormat::Float32x2,
            Box::new(|| {
                [(-2.0, -2.0), (6.0, -2.0), (-2.0, 6.0)]
                    .map(|(x, y)| vertex_f32x2(x, y))
                    .concat()
            }),
            2.0,
        ),
    ];

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("extra_formats.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let layout = pipeline.get_bind_group_layout(0);

    for (name, format, encode, expected_t) in cases {
        let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(*name),
            contents: &encode(),
            usage: BufferUsages::BLAS_INPUT,
        });
        let stride = format.size();

        let blas_size = BlasTriangleGeometrySizeDescriptor {
            vertex_format: *format,
            vertex_count: 3,
            index_format: Some(IndexFormat::Uint32),
            index_count: Some(3),
            flags: AccelerationStructureGeometryFlags::OPAQUE,
        };
        let blas = ctx.device.create_blas(
            &CreateBlasDescriptor {
                label: Some(*name),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_size.clone()],
            },
        );
        let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some(*name),
            max_instances: 1,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });
        tlas[0] = Some(TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));

        let indices = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(*name),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::BLAS_INPUT,
        });

        let output = ctx.device.create_buffer(&BufferDescriptor {
            label: Some(*name),
            size: mem::size_of::<[f32; 2]>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: Some(*name),
            size: mem::size_of::<[f32; 2]>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some(*name),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(*name) });
        encoder.build_acceleration_structures(
            &[BlasBuildEntry {
                blas: &blas,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size: &blas_size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: stride,
                    index_buffer: Some(&indices),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }],
            [&tlas],
        );
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(*name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        ctx.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(MapMode::Read, Result::unwrap);
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .await
        .unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let (t, kind) = (
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        );
        drop(bytes);
        readback.unmap();

        assert_eq!(
            kind as u32, RAY_QUERY_INTERSECTION_TRIANGLE,
            "{name}: expected a triangle hit"
        );
        assert!(
            (t - expected_t).abs() < 1e-4,
            "{name}: hit at t = {t}, expected {expected_t}"
        );
    }
}
