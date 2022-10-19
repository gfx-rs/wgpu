use std::{borrow::Cow, fmt::Write};

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
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

#[derive(Clone, Copy)]
enum OutputType {
    Float,
    Uint,
    Sint,
}

impl OutputType {
    fn as_str(&self) -> &'static str {
        match self {
            OutputType::Float => "f32",
            OutputType::Uint => "u32",
            OutputType::Sint => "i32",
        }
    }
}

const MAX_BUFFER_SIZE: u64 = 128;
#[rustfmt::skip]
const TESTS: &[(&[&str], &[&[&str]], &[u32], OutputType)] = &[
    // member types    accessors                   output values        output type
    (&["f32"],       &[&[""]],                     &[1],                OutputType::Float),
    (&["vec2<f32>"], &[&[".x", ".y"]],             &[1, 2],             OutputType::Float),
    (&["vec3<f32>"], &[&[".x", ".y", ".z"]],       &[1, 2, 3],          OutputType::Float),
    (&["vec4<f32>"], &[&[".x", ".y", ".z", ".w"]], &[1, 2, 3, 4],       OutputType::Float),

    (&["mat2x2<f32>"], &[&["[0].x", "[0].y", "[1].x", "[1].y"]],                   &[1, 2, 3, 4],  OutputType::Float),
    (&["mat2x3<f32>"], &[&["[0].x", "[0].y", "[0].z", "[1].x", "[1].y", "[1].z"]], &[1, 2, 3, 5, 6, 7],  OutputType::Float),
    (&["mat2x4<f32>"], &[&["[0].x", "[0].y", "[0].z", "[0].w", "[1].x", "[1].y", "[1].z", "[1].w"]], &[1, 2, 3, 4, 5, 6, 7, 8],  OutputType::Float),
    
    (&["mat3x2<f32>"], &[&["[0].x", "[0].y", "[1].x", "[1].y", "[2].x", "[2].y"]], &[1, 2, 3, 4, 5, 6],  OutputType::Float),
    (&["mat3x3<f32>"], &[&["[0].x", "[0].y", "[0].z", "[1].x", "[1].y", "[1].z", "[2].x", "[2].y", "[2].z"]], &[1, 2, 3, 5, 6, 7, 9, 10, 11],  OutputType::Float),
    (&["mat3x4<f32>"], &[&["[0].x", "[0].y", "[0].z", "[0].w", "[1].x", "[1].y", "[1].z", "[1].w", "[2].x", "[2].y", "[2].z", "[2].w"]], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  OutputType::Float),
    
    (&["mat4x2<f32>"], &[&["[0].x", "[0].y", "[1].x", "[1].y", "[2].x", "[2].y", "[3].x", "[3].y"]], &[1, 2, 3, 4, 5, 6, 7, 8],  OutputType::Float),
    (&["mat4x3<f32>"], &[&["[0].x", "[0].y", "[0].z", "[1].x", "[1].y", "[1].z", "[2].x", "[2].y", "[2].z", "[3].x", "[3].y", "[3].z"]], &[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15],  OutputType::Float),
    (&["mat4x4<f32>"], &[&["[0].x", "[0].y", "[0].z", "[0].w", "[1].x", "[1].y", "[1].z", "[1].w", "[2].x", "[2].y", "[2].z", "[2].w", "[3].x", "[3].y", "[3].z", "[3].w"]], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  OutputType::Float),
];

fn input_layout_test(ctx: TestingContext, storage_type: StorageType) {
    let source = String::from(include_str!("input_layout.wgsl"));

    // We go from [1, MAX] for indices so that unwritten values (cleared zeros)
    // won't be confused for a valid written offset of 0.
    let uint_values: Vec<_> = (1..=(MAX_BUFFER_SIZE / 4) as u32).collect();
    let sint_values: Vec<_> = (1..=(MAX_BUFFER_SIZE / 4) as i32).collect();
    let float_values: Vec<_> = (1..=(MAX_BUFFER_SIZE / 4)).map(|v| v as f32).collect();

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

    for &(member_types, accessors, output_values, output_type) in TESTS {
        let test_name = format!("{member_types:?}");

        let mut input_members = String::new();
        for (idx, &ty) in member_types.iter().enumerate() {
            writeln!(&mut input_members, "member_{idx}: {ty},").unwrap();
        }

        let mut body = String::new();
        for (member_idx, &member_accessors) in accessors.iter().enumerate() {
            for &member_accessor in member_accessors {
                writeln!(
                    &mut body,
                    "output[i] = input.member_{member_idx}{member_accessor};"
                )
                .unwrap();
                writeln!(&mut body, "i += 1u;").unwrap();
            }
        }

        let processed = source
            .replace("{{storage_type}}", storage_type.as_str())
            .replace("{{input_members}}", &input_members)
            .replace("{{body}}", &body)
            .replace("{{output_type}}", output_type.as_str());

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

        let bytes: &[u8] = match output_type {
            OutputType::Float => bytemuck::cast_slice(&float_values),
            OutputType::Uint => bytemuck::cast_slice(&uint_values),
            OutputType::Sint => bytemuck::cast_slice(&sint_values),
        };

        ctx.queue.write_buffer(&input_buffer, 0, bytes);

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
        match output_type {
            OutputType::Float => {
                let typed: &[f32] = bytemuck::cast_slice(&*mapped);
                let expected: Vec<f32> = output_values.iter().map(|&v| v as f32).collect();
                assert_eq!(
                    &typed[..expected.len()],
                    expected,
                    "Left is actual. Right is expected. Inner test {test_name}"
                );
            }
            OutputType::Uint => {
                let typed: &[u32] = bytemuck::cast_slice(&*mapped);
                assert_eq!(
                    &typed[..output_values.len()],
                    output_values,
                    "Left is actual. Right is expected."
                );
            }
            OutputType::Sint => {
                let typed: &[i32] = bytemuck::cast_slice(&*mapped);
                let expected: Vec<i32> = output_values.iter().map(|&v| v as i32).collect();
                assert_eq!(
                    &typed[..expected.len()],
                    expected,
                    "Left is actual. Right is expected."
                );
            }
        }
        drop(mapped);

        mapping_buffer.unmap();
    }
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
