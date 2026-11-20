use std::sync::mpsc;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    include_wgsl, AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry, BlasBuildEntry,
    BlasGeometries, BlasGeometrySizeDescriptors, BlasTriangleGeometry,
    BlasTriangleGeometrySizeDescriptor, BufferDescriptor, BufferUsages, ComputePassDescriptor,
    ComputePipelineDescriptor, CreateBlasDescriptor, CreateTlasDescriptor, Features, TlasInstance,
    VertexFormat,
};
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

use super::acceleration_structure_limits;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(BLAS_TRANSFORM_AND_CULLING);
}

// Ray flag bits, as the SPIR-V ray flags that `RayDesc::flags` uses.
const CULL_BACK_FACING: u32 = 0x10;
const CULL_FRONT_FACING: u32 = 0x20;
const SKIP_TRIANGLES: u32 = 0x100;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    origin: [f32; 3],
    flags: u32,
    direction: [f32; 3],
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Hit {
    kind: u32,
    t: f32,
    front_face: u32,
}

const NO_HIT: u32 = 0;
const TRIANGLE_HIT: u32 = 1;

/// Rays fired along the z axis at a triangle that is moved by the BLAS geometry
/// transform, which wgpu specifies as a row-major 3x4 matrix.
#[apply(gpu_test!)]
static BLAS_TRANSFORM_AND_CULLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(acceleration_structure_limits())
            .features(Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_sync(blas_transform_and_culling);

fn blas_transform_and_culling(ctx: TestingContext) {
    let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("vertices"),
        contents: bytemuck::cast_slice(&[[0_f32, 0., 0.], [2., 0., 0.], [0., 2., 0.]]),
        usage: BufferUsages::BLAS_INPUT,
    });
    // Scale x by 2 and translate by (0.5, 0.5, 3). The matrix starts 16 bytes into
    // the buffer, so the transform buffer offset is exercised as well.
    let mut matrix = vec![0_f32; 4];
    matrix.extend([2., 0., 0., 0.5, 0., 1., 0., 0.5, 0., 0., 1., 3.]);
    let transform = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("row-major transform"),
        contents: bytemuck::cast_slice(&matrix),
        usage: BufferUsages::BLAS_INPUT,
    });

    let size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: None,
        index_count: None,
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = ctx.device.create_blas(
        &CreateBlasDescriptor {
            label: None,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE
                | AccelerationStructureFlags::USE_TRANSFORM,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![size.clone()],
        },
    );
    let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(TlasInstance::from_blas_untransformed(&blas));

    let probe = |origin, direction, flags| Probe {
        origin,
        flags,
        direction,
        padding: 0,
    };
    let from_below = |x, flags| probe([x, 1., -1.], [0., 0., 1.], flags);
    let from_above = |flags| probe([1., 1., 8.], [0., 0., -1.], flags);
    let probes = [
        // The triangle now sits at z = 3, so the rays hit it 4 units away. The second
        // ray is outside the untransformed triangle and misses it unless x is scaled.
        from_below(1., 0),
        from_below(3., 0),
        from_below(1., CULL_BACK_FACING),
        from_below(1., CULL_FRONT_FACING),
        from_above(0),
        from_above(CULL_BACK_FACING),
        from_above(CULL_FRONT_FACING),
        from_below(1., SKIP_TRIANGLES),
    ];

    let probe_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("probes"),
        contents: bytemuck::cast_slice(&probes),
        usage: BufferUsages::STORAGE,
    });
    let hit_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("hits"),
        size: (probes.len() * size_of::<Hit>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: hit_buffer.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = ctx
        .device
        .create_shader_module(include_wgsl!("transform.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            BindGroupEntry {
                binding: 1,
                resource: probe_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: hit_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(
        [&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &size,
                vertex_buffer: &vertices,
                first_vertex: 0,
                vertex_stride: size_of::<[f32; 3]>() as u64,
                index_buffer: None,
                first_index: None,
                transform_buffer: Some(&transform),
                transform_buffer_offset: Some(16),
            }]),
        }],
        [],
    );
    // Wait for the BLAS so the TLAS build cannot race with it on Metal.
    let blas_build = ctx.queue.submit([encoder.finish()]);
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: blas_build,
            timeout: None,
        })
        .unwrap();

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures([], [&tlas]);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(probes.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&hit_buffer, 0, &readback, 0, hit_buffer.size());
    ctx.queue.submit([encoder.finish()]);

    let (send, recv) = mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |res| {
        res.unwrap();
        send.send(()).unwrap()
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    recv.recv().unwrap();
    let hits: Vec<Hit> = bytemuck::cast_slice(&readback.get_mapped_range(..).unwrap()).to_vec();

    assert_eq!(hits[0].kind, TRIANGLE_HIT, "transformed triangle");
    assert_hit_distance(&hits[0], 4.0);
    assert_eq!(hits[1].kind, TRIANGLE_HIT, "x scale of the transform");
    assert_hit_distance(&hits[1], 4.0);

    // The two sides of the triangle face opposite ways. Culling back faces must keep
    // exactly the front-facing hit and culling front faces exactly the back-facing one.
    assert_ne!(hits[0].front_face, hits[4].front_face, "facing flips");
    for (unculled, cull_back, cull_front) in [(0, 2, 3), (4, 5, 6)] {
        assert_eq!(hits[unculled].kind, TRIANGLE_HIT, "unculled ray {unculled}");
        let front_face = hits[unculled].front_face != 0;
        assert_eq!(
            hits[cull_back].kind == TRIANGLE_HIT,
            front_face,
            "CULL_BACK_FACING for ray {unculled}"
        );
        assert_eq!(
            hits[cull_front].kind == TRIANGLE_HIT,
            !front_face,
            "CULL_FRONT_FACING for ray {unculled}"
        );
    }
    assert_eq!(hits[7].kind, NO_HIT, "skip triangles");
}

fn assert_hit_distance(hit: &Hit, expected: f32) {
    assert!(
        (hit.t - expected).abs() < 1e-4,
        "hit distance {} != {expected}",
        hit.t
    );
}
