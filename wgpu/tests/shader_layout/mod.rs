use std::{borrow::Cow, fmt::Write};

use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DownlevelFlags, Limits, Maintain, MapMode, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::common::{initialize_test, TestParameters, TestingContext};

#[derive(Clone, Copy)]
enum StorageType {
    Uniform,
    Storage,
}

impl StorageType {
    fn as_str(&self) -> &'static str {
        match self {
            StorageType::Uniform => "uniform",
            StorageType::Storage => "storage",
        }
    }
}

struct ShaderLayoutTest {
    member_types: &'static [&'static str],
    accessors: &'static [&'static [&'static str]],
    output_values: &'static [u32],
    uniform_failures: Backends,
    storage_failures: Backends,
}

const MAX_BUFFER_SIZE: u64 = 128;
const TESTS: &[ShaderLayoutTest] = &[
    ShaderLayoutTest {
        member_types: &["f32"],
        accessors: &[&[""]],
        output_values: &[1],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec2<f32>"],
        accessors: &[&[".x", ".y"]],
        output_values: &[1, 2],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec3<f32>"],
        accessors: &[&[".x", ".y", ".z"]],
        output_values: &[1, 2, 3],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec4<f32>"],
        accessors: &[&[".x", ".y", ".z", ".w"]],
        output_values: &[1, 2, 3, 4],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["u32"],
        accessors: &[&[""]],
        output_values: &[1],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec2<u32>"],
        accessors: &[&[".x", ".y"]],
        output_values: &[1, 2],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec3<u32>"],
        accessors: &[&[".x", ".y", ".z"]],
        output_values: &[1, 2, 3],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["vec4<u32>"],
        accessors: &[&[".x", ".y", ".z", ".w"]],
        output_values: &[1, 2, 3, 4],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat2x2<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", //
            "[1].x", "[1].y", //
        ]],
        output_values: &[1, 2, 3, 4],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat2x3<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", //
            "[1].x", "[1].y", "[1].z", //
        ]],
        output_values: &[1, 2, 3, 5, 6, 7],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat2x4<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", "[0].w", //
            "[1].x", "[1].y", "[1].z", "[1].w", //
        ]],
        output_values: &[1, 2, 3, 4, 5, 6, 7, 8],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::DX12,
    },
    ShaderLayoutTest {
        member_types: &["mat3x2<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", //
            "[1].x", "[1].y", //
            "[2].x", "[2].y", //
        ]],
        output_values: &[1, 2, 3, 4, 5, 6],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::DX12,
    },
    ShaderLayoutTest {
        member_types: &["mat3x3<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", //
            "[1].x", "[1].y", "[1].z", //
            "[2].x", "[2].y", "[2].z", //
        ]],
        output_values: &[1, 2, 3, 5, 6, 7, 9, 10, 11],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat3x4<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", "[0].w", //
            "[1].x", "[1].y", "[1].z", "[1].w", //
            "[2].x", "[2].y", "[2].z", "[2].w", //
        ]],
        output_values: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat4x2<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", //
            "[1].x", "[1].y", //
            "[2].x", "[2].y", //
            "[3].x", "[3].y", //
        ]],
        output_values: &[1, 2, 3, 4, 5, 6, 7, 8],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::DX12,
    },
    ShaderLayoutTest {
        member_types: &["mat4x3<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", //
            "[1].x", "[1].y", "[1].z", //
            "[2].x", "[2].y", "[2].z", //
            "[3].x", "[3].y", "[3].z", //
        ]],
        output_values: &[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
    ShaderLayoutTest {
        member_types: &["mat4x4<f32>"],
        accessors: &[&[
            "[0].x", "[0].y", "[0].z", "[0].w", //
            "[1].x", "[1].y", "[1].z", "[1].w", //
            "[2].x", "[2].y", "[2].z", "[2].w", //
            "[3].x", "[3].y", "[3].z", "[3].w", //
        ]],
        output_values: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        uniform_failures: Backends::empty(),
        storage_failures: Backends::empty(),
    },
];

fn input_layout_test(ctx: TestingContext, storage_type: StorageType) {
    let source = String::from(include_str!("input_layout.wgsl"));

    // We go from [1, MAX] for indices so that unwritten values (cleared zeros)
    // won't be confused for a valid written offset of 0.
    let values: Vec<_> = (1..=(MAX_BUFFER_SIZE as u32 / 4)).collect();

    let bgl = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: match storage_type {
                            StorageType::Uniform => wgpu::BufferBindingType::Uniform,
                            StorageType::Storage => {
                                wgpu::BufferBindingType::Storage { read_only: true }
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let input_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("input buffer"),
        size: MAX_BUFFER_SIZE,
        usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("output buffer"),
        size: MAX_BUFFER_SIZE,
        usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mapping_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("mapping buffer"),
        size: MAX_BUFFER_SIZE,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let pll = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let mut fail = false;
    for test in TESTS {
        let test_name = format!("{:?}", test.member_types);

        let mut input_members = String::new();
        for (idx, &ty) in test.member_types.iter().enumerate() {
            writeln!(&mut input_members, "member_{idx}: {ty},").unwrap();
        }

        let mut body = String::new();
        for (member_idx, &member_accessors) in test.accessors.iter().enumerate() {
            for &member_accessor in member_accessors {
                // We bitcast as the values are really u32/i32 values.
                writeln!(
                    &mut body,
                    "output[i] = bitcast<u32>(input.member_{member_idx}{member_accessor});"
                )
                .unwrap();
                writeln!(&mut body, "i += 1u;").unwrap();
            }
        }

        let processed = source
            .replace("{{storage_type}}", storage_type.as_str())
            .replace("{{input_members}}", &input_members)
            .replace("{{body}}", &body);

        let sm = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("shader {test_name}")),
            source: ShaderSource::Wgsl(Cow::Owned(processed)),
        });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("pipeline {test_name}")),
                layout: Some(&pll),
                module: &sm,
                entry_point: "cs_main",
            });

        ctx.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&values));

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        encoder.clear_buffer(&output_buffer, 0, None);
        encoder.clear_buffer(&mapping_buffer, 0, None);

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(&format!("cpass {test_name}")),
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
        drop(cpass);

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &mapping_buffer, 0, MAX_BUFFER_SIZE);

        ctx.queue.submit(Some(encoder.finish()));

        let _ = mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
        ctx.device.poll(Maintain::Wait);

        let mapped = mapping_buffer.slice(..).get_mapped_range();

        let typed: &[u32] = bytemuck::cast_slice(&*mapped);

        let left = &typed[..test.output_values.len()];
        let right = &*test.output_values;
        let failure = left != right;
        if failure {
            eprintln!("Inner test failure. Actual {left:?}. Expected {right:?}. Test {test_name}");
        }
        let backend_failures = match storage_type {
            StorageType::Uniform => test.uniform_failures,
            StorageType::Storage => test.storage_failures,
        };
        if failure != backend_failures.contains(ctx.adapter.get_info().backend.into()) {
            fail |= true;
            if !failure {
                eprintln!("Unexpected test success. Test {test_name}");
            }
        }
        drop(mapped);

        mapping_buffer.unmap();
    }
    assert!(!fail);
}

#[test]
fn input_layout_uniform() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            input_layout_test(ctx, StorageType::Uniform);
        },
    );
}

#[test]
fn input_layout_storage() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            input_layout_test(ctx, StorageType::Storage);
        },
    );
}
