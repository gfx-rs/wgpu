use std::{mem, num::NonZeroU32, time::Duration};

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([BINDING_ARRAY_TLAS, BINDING_ARRAY_TLAS_NON_UNIFORM_INDEX]);
}

const RAY_QUERY_INTERSECTION_TRIANGLE: u32 = 1;

fn tlas_array_parameters() -> TestParameters {
    TestParameters::default()
        .instance_flags(InstanceFlags::GPU_BASED_VALIDATION)
        .features(Features::EXPERIMENTAL_RAY_QUERY | Features::ACCELERATION_STRUCTURE_BINDING_ARRAY)
        .limits(Limits {
            max_binding_array_elements_per_shader_stage: 8,
            max_acceleration_structures_per_shader_stage: 8,
            max_binding_array_acceleration_structure_elements_per_shader_stage: 8,
            ..Limits::default().using_minimum_supported_acceleration_structure_values()
        })
}

/// Traces a ray through `tlas_array[1]`, which must select the second
/// structure in the array.
#[apply(gpu_test!)]
static BINDING_ARRAY_TLAS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(tlas_array_parameters())
    .run_async(|ctx| async move {
        let hits = trace_tlas_array(&ctx, "constant_index", 1).await;
        assert_hit_distances(&hits, &[6.0]);
    });

/// Indexes the array with values loaded from a storage buffer by two
/// invocations of one workgroup, so each selects a different element through a
/// non-uniform index.
#[apply(gpu_test!)]
static BINDING_ARRAY_TLAS_NON_UNIFORM_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(tlas_array_parameters())
    .run_async(|ctx| async move {
        let hits = trace_tlas_array(&ctx, "non_uniform_index", 2).await;
        assert_hit_distances(&hits, &[4.0, 6.0]);
    });

/// Checks each invocation hit the triangle at the distance of the element it
/// selected.
fn assert_hit_distances(hits: &[[f32; 2]], expected: &[f32]) {
    assert_eq!(hits.len(), expected.len());
    for (invocation, (&[t, kind], &expected_t)) in hits.iter().zip(expected).enumerate() {
        assert_eq!(
            kind as u32, RAY_QUERY_INTERSECTION_TRIANGLE,
            "invocation {invocation}: expected a triangle hit"
        );
        assert!(
            (t - expected_t).abs() < 1e-4,
            "invocation {invocation}: hit at t = {t}, expected {expected_t}"
        );
    }
}

/// Builds two TLAS over one BLAS with different instance transforms, binds
/// them as a `binding_array`, and runs one workgroup of `entry_point` from
/// `tlas.wgsl`, whose workgroup size must equal `invocations`.
///
/// The BLAS is one triangle covering the xy origin at z = 2, so a ray from
/// (0, 0, -2) towards +z hits it at t = 4 through element 0 (identity
/// transform) and at t = 6 through element 1 (translated by 2 along z).
async fn trace_tlas_array(
    ctx: &TestingContext,
    entry_point: &str,
    invocations: u32,
) -> Vec<[f32; 2]> {
    let vertices: [[f32; 3]; 3] = [[-2.0, -2.0, 2.0], [6.0, -2.0, 2.0], [-2.0, 6.0, 2.0]];
    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("TLAS array vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::BLAS_INPUT,
    });
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("TLAS array indices"),
        contents: bytemuck::cast_slice(&[0u32, 1, 2]),
        usage: BufferUsages::BLAS_INPUT,
    });

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: vertices.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(3),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("TLAS array BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    let create_tlas = |label| {
        ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some(label),
            max_instances: 1,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        })
    };
    let mut tlas_a = create_tlas("TLAS A");
    let mut tlas_b = create_tlas("TLAS B");
    tlas_a[0] = Some(TlasInstance::from_blas_untransformed(&blas));
    tlas_b[0] = Some(TlasInstance::from_blas_transformed(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
    ));

    let select = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("TLAS array selector"),
        contents: bytemuck::cast_slice(&[0u32, 1]),
        usage: BufferUsages::STORAGE,
    });
    let output_size = u64::from(invocations) * mem::size_of::<[f32; 2]>() as u64;
    let output = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("TLAS array output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("TLAS array readback"),
        size: output_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_entry = |binding, read_only| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("TLAS array BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: Some(NonZeroU32::new(2).unwrap()),
                },
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("TLAS array pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let shader = ctx.device.create_shader_module(include_wgsl!("tlas.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

    let tlas_array: [&Tlas; 2] = [&tlas_a, &tlas_b];
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("TLAS array bind group"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructureArray(&tlas_array),
            },
            BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: select.as_entire_binding(),
            },
        ],
    });

    // Build and wait on the BLAS first so this test does not depend on Metal ordering
    // acceleration structure builds within one command buffer (fixed separately in #10292).
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS array BLAS build"),
        });
    encoder.build_acceleration_structures(
        &[BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertex_buffer,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as u64,
                index_buffer: Some(&index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }],
        [],
    );
    let blas_submission = ctx.queue.submit([encoder.finish()]);
    ctx.async_poll(PollType::Wait {
        submission_index: blas_submission,
        timeout: Some(Duration::from_secs(30)),
    })
    .await
    .unwrap();

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("TLAS array build + trace"),
        });
    encoder.build_acceleration_structures([], [&tlas_a, &tlas_b]);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("TLAS array trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    ctx.queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, Result::unwrap);
    ctx.async_poll(PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })
    .await
    .unwrap();
    let hits = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    hits
}
