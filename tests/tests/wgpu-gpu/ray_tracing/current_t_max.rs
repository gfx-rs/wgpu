use std::mem::size_of;

use crate::ray_tracing::acceleration_structure_limits;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry, BindingResource,
    BlasAABBGeometrySizeDescriptor, BlasAabbGeometry, BlasBuildEntry, BlasGeometries,
    BlasGeometrySizeDescriptors, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    CreateBlasDescriptor, CreateTlasDescriptor, TlasInstance,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

/// Bounds of a procedural primitive, laid out as an AABB BLAS expects.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AabbPrimitive {
    min: [f32; 3],
    max: [f32; 3],
}

/// Number of `f32` values that the shaders write.
const OUTPUT_COUNT: usize = 5;

/// Parameters shared by the tests in this module.
fn parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .limits(acceleration_structure_limits())
        .features(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// Registers the tests in this module.
pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(CURRENT_T_MAX_TRACKS_GENERATED_HIT);
    tests.push(CURRENT_T_MAX_KEEPS_NEAREST_OF_TWO_HITS);
}

/// Checks that `rayQueryGetCurrentTMax` follows a generated hit through a ray query.
#[apply(gpu_test!)]
static CURRENT_T_MAX_TRACKS_GENERATED_HIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(current_t_max_tracks_generated_hit);

/// Traces one ray through an AABB and checks the values that the shader reads.
async fn current_t_max_tracks_generated_hit(ctx: TestingContext) {
    let values = trace(
        &ctx,
        &[AabbPrimitive {
            min: [-1.0, -1.0, 2.0],
            max: [1.0, 1.0, 4.0],
        }],
        "single_hit",
    )
    .await;

    assert_eq!(
        values[..5],
        [100.0, 3.0, 3.0, 3.0, 1.0],
        "t max before the hit, after a hit, after a farther hit, after the query, and the iteration count"
    );
}

/// Checks that the nearest generated hit sets the `t` max when the ray reaches two AABBs.
#[apply(gpu_test!)]
static CURRENT_T_MAX_KEEPS_NEAREST_OF_TWO_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(parameters())
    .run_async(current_t_max_keeps_nearest_of_two_hits);

/// Traces one ray through two AABBs, generating the near `t` of each, and checks the result.
async fn current_t_max_keeps_nearest_of_two_hits(ctx: TestingContext) {
    let values = trace(
        &ctx,
        &[
            AabbPrimitive {
                min: [-1.0, -1.0, 2.0],
                max: [1.0, 1.0, 4.0],
            },
            AabbPrimitive {
                min: [-1.0, -1.0, 6.0],
                max: [1.0, 1.0, 8.0],
            },
        ],
        "nearest_of_two",
    )
    .await;

    assert_eq!(
        values[..3],
        [100.0, 2.0, 2.0],
        "t max before the first hit, t max after the query, and the AABB count"
    );
}

/// Builds a scene of AABBs, runs the compute entry point on it, and returns the shader output.
async fn trace(ctx: &TestingContext, aabbs: &[AabbPrimitive], entry_point: &str) -> Vec<f32> {
    let aabb_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(aabbs),
        usage: BufferUsages::BLAS_INPUT,
    });

    let blas_size = BlasAABBGeometrySizeDescriptor {
        primitive_count: aabbs.len() as u32,
        flags: AccelerationStructureGeometryFlags::empty(),
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: Some("AABB BLAS"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![blas_size.clone()],
        },
    );

    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: Some("TLAS"),
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

    let mut build_encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("build"),
        });
    build_encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                size: &blas_size,
                stride: size_of::<AabbPrimitive>() as BufferAddress,
                aabb_buffer: &aabb_buffer,
                primitive_offset: 0,
            }]),
        }],
        [&tlas],
    );
    ctx.queue.submit([build_encoder.finish()]);

    let output_size = (OUTPUT_COUNT * size_of::<f32>()) as BufferAddress;
    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let cpu_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("cpu output"),
        size: output_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("current_t_max.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &cpu_buffer, 0, output_size);
    ctx.queue.submit([encoder.finish()]);

    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = cpu_buffer.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&data).to_vec()
}
