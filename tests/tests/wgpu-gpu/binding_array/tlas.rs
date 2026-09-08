use std::{mem, num::NonZeroU32, time::Duration};

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(BINDING_ARRAY_TLAS);
    tests.push(BINDING_ARRAY_TLAS_DYNAMIC_INDEX);
}

const RAY_QUERY_INTERSECTION_TRIANGLE: u32 = 1;

/// Builds two TLAS over the same BLAS, binds them as a `binding_array`, traces
/// a ray through element 1 of the array, and checks that the committed
/// intersection lands at the expected distance.
#[gpu_test]
static BINDING_ARRAY_TLAS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::ACCELERATION_STRUCTURE_BINDING_ARRAY,
            )
            .limits(wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 8,
                max_acceleration_structures_per_shader_stage: 8,
                max_binding_array_acceleration_structure_elements_per_shader_stage: 8,
                ..wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
            }),
    )
    .run_async(binding_array_tlas);

async fn binding_array_tlas(ctx: TestingContext) {
    // One big triangle covering the xy origin at z = 2; the ray from
    // (0, 0, -2) towards +z hits it at t = 4.
    let vertices: [[f32; 3]; 3] = [[-2.0, -2.0, 2.0], [6.0, -2.0, 2.0], [-2.0, 6.0, 2.0]];

    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("binding_array TLAS vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::BLAS_INPUT,
    });
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("binding_array TLAS indices"),
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
            label: Some("binding_array BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    // Both TLAS reference the same BLAS with identity transforms.
    let mut tlas_a = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("binding_array TLAS A"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let mut tlas_b = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("binding_array TLAS B"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    for tlas in [&mut tlas_a, &mut tlas_b] {
        tlas[0] = Some(TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0,
            0xFF,
        ));
    }

    let output = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("binding_array output"),
        size: mem::size_of::<[f32; 2]>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("binding_array readback"),
        size: mem::size_of::<[f32; 2]>() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("tlas.wgsl"));
    // The reflected layout does not carry the array length for acceleration
    // structure arrays, so declare it explicitly.
    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("binding_array BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: Some(NonZeroU32::new(2).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("binding_array pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("binding_array pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let tlas_array: [&Tlas; 2] = [&tlas_a, &tlas_b];
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("binding_array bind group"),
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
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("binding_array build + trace"),
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
        [&tlas_a, &tlas_b],
    );
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("binding_array trace"),
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
        "expected a triangle hit through tlas_array[1]"
    );
    assert!((t - 4.0).abs() < 1e-4, "hit at t = {t}, expected 4");
}

/// Builds two TLAS over the same BLAS with different instance transforms,
/// binds them as a `binding_array`, and indexes the array with a value loaded
/// from a storage buffer. The non-uniform index requires the
/// `ACCELERATION_STRUCTURE_BINDING_ARRAY` capability and must select the
/// structure whose transform was applied.
#[gpu_test]
static BINDING_ARRAY_TLAS_DYNAMIC_INDEX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(
                wgpu::Features::EXPERIMENTAL_RAY_QUERY
                    | wgpu::Features::ACCELERATION_STRUCTURE_BINDING_ARRAY,
            )
            .limits(wgpu::Limits {
                max_binding_array_elements_per_shader_stage: 8,
                max_acceleration_structures_per_shader_stage: 8,
                max_binding_array_acceleration_structure_elements_per_shader_stage: 8,
                ..wgpu::Limits::default().using_minimum_supported_acceleration_structure_values()
            }),
    )
    .run_async(binding_array_tlas_dynamic_index);

async fn binding_array_tlas_dynamic_index(ctx: TestingContext) {
    // One big triangle covering the xy origin at z = 2; the ray from
    // (0, 0, -2) towards +z hits it at t = 4.
    let vertices: [[f32; 3]; 3] = [[-2.0, -2.0, 2.0], [6.0, -2.0, 2.0], [-2.0, 6.0, 2.0]];

    let vertex_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("dynamic index vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::BLAS_INPUT,
    });
    let index_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("dynamic index indices"),
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
            label: Some("dynamic index BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    // TLAS A keeps the identity transform (hit at t = 4); TLAS B's instance
    // translates the geometry two units further away (hit at t = 6).
    let mut tlas_a = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("dynamic index TLAS A"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let mut tlas_b = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("dynamic index TLAS B"),
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    tlas_a[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xFF,
    ));
    tlas_b[0] = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
        0,
        0xFF,
    ));

    // The shader indexes tlas_array with this value, which is non-uniform
    // because it is loaded from a storage buffer.
    let select = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("dynamic index selector"),
        contents: bytemuck::cast_slice(&[1u32]),
        usage: BufferUsages::STORAGE,
    });
    let output = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("dynamic index output"),
        size: mem::size_of::<[f32; 2]>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("dynamic index readback"),
        size: mem::size_of::<[f32; 2]>() as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("tlas.wgsl"));
    // The reflected layout does not carry the array length for acceleration
    // structure arrays, so declare it explicitly.
    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dynamic index BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: Some(NonZeroU32::new(2).unwrap()),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(4),
                    },
                    count: None,
                },
            ],
        });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("dynamic index pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dynamic index pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dynamic_index"),
            compilation_options: Default::default(),
            cache: None,
        });

    let tlas_array: [&Tlas; 2] = [&tlas_a, &tlas_b];
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dynamic index bind group"),
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

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dynamic index build + trace"),
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
        [&tlas_a, &tlas_b],
    );
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dynamic index trace"),
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
        "expected a triangle hit through tlas_array[select]"
    );
    // The selected element is TLAS B, whose instance moved the triangle two
    // units further from the ray origin.
    assert!((t - 6.0).abs() < 1e-4, "hit at t = {t}, expected 6");
}
