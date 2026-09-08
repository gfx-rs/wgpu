use std::time::Duration;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(TRANSFORMED_QUERY_SEMANTICS);
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Probe {
    origin: [f32; 3],
    flags: u32,
    direction: [f32; 3],
    mask: u32,
    near: f32,
    far: f32,
    accept: u32,
    padding: u32,
}

fn with_flags(probe: Probe, flags: u32) -> Probe {
    Probe { flags, ..probe }
}

#[gpu_test]
static TRANSFORMED_QUERY_SEMANTICS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(super::acceleration_structure_limits())
            .features(Features::EXPERIMENTAL_RAY_QUERY),
    )
    .run_async(transformed_query_semantics);

async fn transformed_query_semantics(ctx: TestingContext) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let mut upload = device.create_command_encoder(&Default::default());
    let mut input = |label, bytes: &[u8]| {
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Pending writes and an earlier non-AS submission both precede the first build.
        queue.write_buffer(&staging, 0, bytes);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        upload.copy_buffer_to_buffer(&staging, 0, &buffer, 0, bytes.len() as u64);
        buffer
    };
    let vertices = input(
        "vertices",
        bytemuck::cast_slice(&[[0_f32, 0., 0.], [2., 0., 0.], [0., 2., 0.]]),
    );
    let indices = input("indices", bytemuck::cast_slice(&[0_u32, 1, 2]));
    let mut matrices = vec![0_f32; 4];
    matrices.extend([2., 0.5, 0., 1., 0., 3., 0., -2., 0., 0., 1., 4.]);
    let transform = input(
        "row-major transform at byte 16",
        bytemuck::cast_slice(&matrices),
    );
    let aabbs = input("aabb", bytemuck::cast_slice(&[-1_f32, -1., 2., 1., 1., 4.]));
    queue.submit([upload.finish()]);

    let triangle_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: 3,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(3),
        flags: AccelerationStructureGeometryFlags::empty(),
    };
    let aabb_size = BlasAABBGeometrySizeDescriptor {
        primitive_count: 1,
        flags: AccelerationStructureGeometryFlags::empty(),
    };
    let descriptor = CreateBlasDescriptor {
        label: None,
        flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: AccelerationStructureUpdateMode::Build,
    };
    let triangle = device.create_blas(
        &CreateBlasDescriptor {
            flags: descriptor.flags | AccelerationStructureFlags::USE_TRANSFORM,
            ..descriptor
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![triangle_size.clone()],
        },
    );
    let sphere = device.create_blas(
        &descriptor,
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![aabb_size.clone()],
        },
    );
    let opaque_triangle_size = BlasTriangleGeometrySizeDescriptor {
        flags: AccelerationStructureGeometryFlags::OPAQUE,
        ..triangle_size.clone()
    };
    let opaque_aabb_size = BlasAABBGeometrySizeDescriptor {
        flags: AccelerationStructureGeometryFlags::OPAQUE,
        ..aabb_size.clone()
    };
    let opaque_triangle = device.create_blas(
        &CreateBlasDescriptor {
            flags: descriptor.flags | AccelerationStructureFlags::USE_TRANSFORM,
            ..descriptor
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![opaque_triangle_size.clone()],
        },
    );
    let opaque_sphere = device.create_blas(
        &descriptor,
        BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![opaque_aabb_size.clone()],
        },
    );
    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 5,
        flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    tlas[0] = Some(TlasInstance::new(
        &triangle,
        [0., -1., 0., 3., 2., 0., 0., 1., 0., 0., 0.5, 2.],
        37,
        1,
    ));
    tlas[1] = Some(TlasInstance::new(
        &sphere,
        [1., 0., 0., 10., 0., 1., 0., 0., 0., 0., 1., 0.],
        91,
        2,
    ));
    tlas[2] = Some(TlasInstance::new(
        &triangle,
        [-1., 0., 0., -4., 0., 1., 0., 0., 0., 0., 1., 0.],
        53,
        4,
    ));

    tlas[3] = Some(TlasInstance::new(
        &opaque_triangle,
        [0., -1., 0., 3., 2., 0., 0., 1., 0., 0., 0.5, 2.],
        137,
        8,
    ));
    tlas[4] = Some(TlasInstance::new(
        &opaque_sphere,
        [1., 0., 0., 10., 0., 1., 0., 0., 0., 0., 1., 0.],
        191,
        16,
    ));

    let triangle_entry = BlasBuildEntry {
        blas: &triangle,
        geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
            size: &triangle_size,
            vertex_buffer: &vertices,
            first_vertex: 0,
            vertex_stride: 12,
            index_buffer: Some(&indices),
            first_index: Some(0),
            transform_buffer: Some(&transform),
            transform_buffer_offset: Some(16),
        }]),
    };
    let sphere_entry = BlasBuildEntry {
        blas: &sphere,
        geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
            size: &aabb_size,
            stride: 24,
            aabb_buffer: &aabbs,
            primitive_offset: 0,
        }]),
    };
    let opaque_triangle_entry = BlasBuildEntry {
        blas: &opaque_triangle,
        geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
            size: &opaque_triangle_size,
            vertex_buffer: &vertices,
            first_vertex: 0,
            vertex_stride: 12,
            index_buffer: Some(&indices),
            first_index: Some(0),
            transform_buffer: Some(&transform),
            transform_buffer_offset: Some(16),
        }]),
    };
    let opaque_sphere_entry = BlasBuildEntry {
        blas: &opaque_sphere,
        geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
            size: &opaque_aabb_size,
            stride: 24,
            aabb_buffer: &aabbs,
            primitive_offset: 0,
        }]),
    };
    let mut build = device.create_command_encoder(&Default::default());
    build.build_acceleration_structures(
        [
            &triangle_entry,
            &sphere_entry,
            &opaque_triangle_entry,
            &opaque_sphere_entry,
        ],
        [&tlas],
    );
    queue.submit([build.finish()]);

    let base = Probe {
        origin: [3.5, 5.5, 0.],
        flags: 0,
        direction: [0., 0., 1.],
        mask: 1,
        near: 0.,
        far: 10.,
        accept: 1,
        padding: 0,
    };
    let sphere_probe = Probe {
        origin: [10., 0., 0.],
        mask: 2,
        ..base
    };
    let opaque_probe = Probe { mask: 8, ..base };
    let opaque_sphere_probe = Probe {
        mask: 16,
        ..sphere_probe
    };
    // Flags follow SPIR-V Ray Flags, as required by Naga's RayDesc contract.
    let cases = [
        ("triangle", base, 1, 4., 37, 0),
        ("mask disjoint", Probe { mask: 2, ..base }, 0, 0., 0, 0),
        ("mask zero", Probe { mask: 0, ..base }, 0, 0., 0, 0),
        ("too near", Probe { far: 3.5, ..base }, 0, 0., 0, 0),
        ("too far", Probe { near: 4.5, ..base }, 0, 0., 0, 0),
        ("reject triangle", Probe { accept: 0, ..base }, 0, 0., 0, 0),
        (
            "force opaque",
            Probe {
                flags: 1,
                accept: 0,
                ..base
            },
            1,
            4.,
            37,
            0,
        ),
        ("force nonopaque", with_flags(base, 2), 1, 4., 37, 0),
        ("cull back", with_flags(base, 0x10), 0, 0., 0, 0),
        ("cull front", with_flags(base, 0x20), 1, 4., 37, 0),
        (
            "front ray culled",
            Probe {
                flags: 0x20,
                origin: [3.5, 5.5, 8.],
                direction: [0., 0., -1.],
                ..base
            },
            0,
            0.,
            0,
            0,
        ),
        (
            "front ray accepted",
            Probe {
                flags: 0x10,
                origin: [3.5, 5.5, 8.],
                direction: [0., 0., -1.],
                ..base
            },
            1,
            4.,
            37,
            0,
        ),
        ("opaque triangle control", opaque_probe, 1, 4., 137, 3),
        (
            "opaque triangle forced nonopaque rejects",
            Probe {
                flags: 2,
                accept: 0,
                ..opaque_probe
            },
            0,
            0.,
            0,
            0,
        ),
        ("cull opaque", with_flags(opaque_probe, 0x40), 0, 0., 0, 0),
        ("cull nonopaque", with_flags(base, 0x80), 0, 0., 0, 0),
        ("skip triangles", with_flags(base, 0x100), 0, 0., 0, 0),
        (
            "skip aabbs keeps triangle",
            with_flags(base, 0x200),
            1,
            4.,
            37,
            0,
        ),
        (
            "mirrored instance",
            Probe {
                origin: [-6.25, -0.5, 0.],
                mask: 4,
                ..base
            },
            1,
            4.,
            53,
            2,
        ),
        ("generated sphere", sphere_probe, 2, 2., 91, 1),
        ("opaque sphere control", opaque_sphere_probe, 2, 2., 191, 4),
        (
            "opaque sphere forced nonopaque",
            with_flags(opaque_sphere_probe, 2),
            2,
            2.,
            191,
            4,
        ),
        (
            "forced opaque sphere",
            with_flags(sphere_probe, 1),
            2,
            2.,
            91,
            1,
        ),
        (
            "opaque sphere culled",
            with_flags(opaque_sphere_probe, 0x40),
            0,
            0.,
            0,
            0,
        ),
        (
            "nonopaque sphere culled",
            with_flags(sphere_probe, 0x80),
            0,
            0.,
            0,
            0,
        ),
        (
            "back cull keeps sphere",
            with_flags(sphere_probe, 0x10),
            2,
            2.,
            91,
            1,
        ),
        (
            "front cull keeps sphere",
            with_flags(sphere_probe, 0x20),
            2,
            2.,
            91,
            1,
        ),
        (
            "opaque cull keeps nonopaque",
            with_flags(base, 0x40),
            1,
            4.,
            37,
            0,
        ),
        (
            "reject sphere",
            Probe {
                accept: 0,
                ..sphere_probe
            },
            0,
            0.,
            0,
            0,
        ),
        ("skip aabbs", with_flags(sphere_probe, 0x200), 0, 0., 0, 0),
        (
            "skip triangles keeps aabb",
            with_flags(sphere_probe, 0x100),
            2,
            2.,
            91,
            1,
        ),
        ("terminate first", with_flags(sphere_probe, 4), 2, 2., 91, 1),
    ];
    for (label, probe, ..) in &cases {
        assert!(
            (probe.flags & 0xc3).count_ones() <= 1,
            "{label}: opacity flags"
        );
        assert!(
            (probe.flags & 0x30).count_ones() <= 1,
            "{label}: face flags"
        );
        assert!(
            (probe.flags & 0x300).count_ones() <= 1,
            "{label}: geometry flags"
        );
        assert!(
            probe.flags & 0x100 == 0 || probe.flags & 0x30 == 0,
            "{label}: skip and face flags"
        );
    }
    let probes: Vec<_> = cases.iter().map(|c| c.1).collect();
    let probes = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&probes),
        usage: BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&BufferDescriptor {
        label: None,
        size: cases.len() as u64 * 32,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: None,
        size: output.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(include_wgsl!("semantics.wgsl"));
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("compute"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: probes.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    for rebuild in [false, true] {
        let mut encoder = device.create_command_encoder(&Default::default());
        if rebuild {
            encoder.build_acceleration_structures(
                [
                    &triangle_entry,
                    &sphere_entry,
                    &opaque_triangle_entry,
                    &opaque_sphere_entry,
                ],
                [&tlas],
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(MapMode::Read, Result::unwrap);
        ctx.async_poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .await
        .unwrap();
        {
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            let (hits, remainder) = words.as_chunks::<8>();
            assert!(remainder.is_empty());
            assert_eq!(hits.len(), cases.len());
            for (case, hit) in cases.iter().zip(hits) {
                let (label, _, kind, distance, custom, instance) = case;
                assert_eq!(hit[0], *kind, "{label}, rebuild={rebuild}");
                if *kind == 0 {
                    assert_eq!(*hit, [0; 8], "{label}: miss output");
                } else {
                    assert!(
                        (f32::from_bits(hit[1]) - distance).abs() < 1e-4,
                        "{label}: distance"
                    );
                    assert_eq!(&hit[2..6], &[*custom, *instance, 0, 0], "{label}: IDs");
                    if *kind == 1 {
                        for &b in &hit[6..8] {
                            assert!(
                                (f32::from_bits(b) - 0.25).abs() < 1e-4,
                                "{label}: barycentrics"
                            );
                        }
                    }
                }
            }
        }
        readback.unmap();
        for vertex_query in [false, true] {
            let count = if vertex_query { 1 } else { cases.len() as u32 };
            let actual = render_queries(&ctx, &shader, &tlas, &probes, count, vertex_query).await;
            let (hits, remainder) = actual.as_chunks::<4>();
            assert!(remainder.is_empty());
            assert_eq!(hits.len(), count as usize);
            for (case, hit) in cases.iter().zip(hits) {
                assert_eq!(hit[0], case.2, "{}: render kind", case.0);
                if case.2 == 0 {
                    assert_eq!(*hit, [0; 4], "{}: render miss output", case.0);
                } else {
                    assert!(
                        (f32::from_bits(hit[1]) - case.3).abs() < 1e-4,
                        "{}: render distance",
                        case.0
                    );
                    assert_eq!(&hit[2..4], &[case.4, case.5], "{}: render IDs", case.0);
                }
            }
        }
    }
}

async fn render_queries(
    ctx: &TestingContext,
    shader: &ShaderModule,
    tlas: &Tlas,
    probes: &Buffer,
    count: u32,
    vertex_query: bool,
) -> Vec<u32> {
    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: shader,
                entry_point: Some(if vertex_query {
                    "vertex_query"
                } else {
                    "vertex"
                }),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: shader,
                entry_point: Some(if vertex_query {
                    "fragment_vertex"
                } else {
                    "fragment"
                }),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba32Uint,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
    let group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: probes.as_entire_binding(),
            },
        ],
    });
    let extent = Extent3d {
        width: count,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = ctx.device.create_texture(&TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let readback = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 512,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let attachments = [Some(RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512),
                rows_per_image: Some(1),
            },
        },
        extent,
    );
    ctx.queue.submit([encoder.finish()]);
    readback.slice(..).map_async(MapMode::Read, Result::unwrap);
    ctx.async_poll(PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    })
    .await
    .unwrap();
    let words = {
        let view = readback.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, u32>(&view[..count as usize * 16]).to_vec()
    };
    readback.unmap();
    words
}
