use std::{borrow::Cow, num::NonZeroU32};

use wgpu::util::DeviceExt;
use wgpu::*;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(BINDING_ARRAY_TLAS);
}

#[gpu_test]
static BINDING_ARRAY_TLAS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::GPU_BASED_VALIDATION)
            // Ray queries + acceleration structure bindings are gated behind this experimental feature.
            .features(
                Features::EXPERIMENTAL_RAY_QUERY | Features::ACCELERATION_STRUCTURE_BINDING_ARRAY,
            )
            .limits(Limits {
                max_binding_array_elements_per_shader_stage: 8,
                max_acceleration_structures_per_shader_stage: 8,
                max_binding_array_acceleration_structure_elements_per_shader_stage: 8,
                ..Limits::default().using_minimum_supported_acceleration_structure_values()
            }),
    )
    .run_async(|ctx| async move { binding_array_tlas(ctx).await });

async fn binding_array_tlas(ctx: TestingContext) {
    // Minimal shader that consumes a TLAS binding array.
    //
    // We don't need to actually "trace" anything for this test. We only need:
    // - Pipeline compilation to accept `binding_array<acceleration_structure>`
    // - Bind group creation to accept `BindingResource::AccelerationStructureArray`
    // - Encoder to successfully set the bind group and submit.
    //
    // Creating a `ray_query` and initializing it against element 0 forces the binding to be used.
    let shader = r#"
        enable wgpu_ray_query;
        enable wgpu_binding_array;

        @group(0) @binding(0)
        var tlas_array: binding_array<acceleration_structure>;

        @compute
        @workgroup_size(1, 1, 1)
        fn main() {
            var rq: ray_query;
            rayQueryInitialize(
                &rq,
                tlas_array[0],
                RayDesc(
                    0u,
                    0xffu,
                    0.001,
                    1000.0,
                    vec3f(0.0, 0.0, 0.0),
                    vec3f(0.0, 0.0, 1.0)
                )
            );
        }
    "#;

    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Binding Array TLAS"),
        source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
    });

    // Build a minimal BLAS + two TLAS so we can bind an array of TLAS.
    //
    // This follows the shapes used in the ray tracing examples.
    let vertex_data: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let index_data: [u16; 3] = [0, 1, 2];

    let vertex_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: BufferUsages::VERTEX | BufferUsages::BLAS_INPUT,
        });

    let index_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RT Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: BufferUsages::INDEX | BufferUsages::BLAS_INPUT,
        });

    let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgpu::VertexFormat::Float32x3,
        vertex_count: vertex_data.len() as u32,
        index_format: Some(wgpu::IndexFormat::Uint16),
        index_count: Some(index_data.len() as u32),
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = ctx.device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("BLAS"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_geo_size_desc.clone()],
        },
    );

    let mut tlas_a = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("TLAS A"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });

    let mut tlas_b = ctx.device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("TLAS B"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });

    // Put a single instance into each TLAS. Both reference the same BLAS.
    //
    // NOTE: This indexing API is how TLAS instances are populated in the examples.
    tlas_a[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xff,
    ));
    tlas_b[0] = Some(wgpu::TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        0,
        0xff,
    ));

    // Build BLAS and TLASes.
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RT Build"),
        });

    encoder.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &blas_geo_size_desc,
                vertex_buffer: &vertex_buf,
                first_vertex: 0,
                vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                index_buffer: Some(&index_buf),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        [&tlas_a, &tlas_b],
    );

    ctx.queue.submit(Some(encoder.finish()));

    // Bind group layout with a TLAS array binding.
    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("TLAS array BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: Some(NonZeroU32::new(2).unwrap()),
            }],
        });

    let tlas_refs: [&Tlas; 2] = [&tlas_a, &tlas_b];

    let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("TLAS array BG"),
        layout: &bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::AccelerationStructureArray(&tlas_refs),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("TLAS array pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("TLAS array pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Dispatch"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}
