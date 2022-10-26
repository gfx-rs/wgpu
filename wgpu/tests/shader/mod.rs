use std::{borrow::Cow};

use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Maintain, MapMode, PipelineLayoutDescriptor, PushConstantRange,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::common::TestingContext;

mod struct_layout;

#[derive(Clone, Copy)]
enum StorageType {
    Uniform,
    Storage,
    PushConstant,
}

impl StorageType {
    fn as_str(&self) -> &'static str {
        match self {
            StorageType::Uniform => "uniform",
            StorageType::Storage => "storage",
            StorageType::PushConstant => "push_constant",
        }
    }
}

struct ShaderTest {
    name: String,
    members: String,
    body: String,
    input_values: Vec<u32>,
    output_values: Vec<u32>,
    output_initialization: u32,
}

const MAX_BUFFER_SIZE: u64 = 128;

fn shader_input_output_test(
    ctx: TestingContext,
    storage_type: StorageType,
    tests: Vec<ShaderTest>,
) {
    let source = String::from(include_str!("shader_test.wgsl"));


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
                            StorageType::Storage | StorageType::PushConstant => {
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
            push_constant_ranges: match storage_type {
                StorageType::PushConstant => &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..MAX_BUFFER_SIZE as u32,
                }],
                _ => &[],
            },
        });

    let mut fail = false;
    for test in tests {
        let test_name = test.name;

        let mut processed = source
            .replace("{{storage_type}}", storage_type.as_str())
            .replace("{{input_members}}", &test.members)
            .replace("{{body}}", &test.body);

        if let StorageType::PushConstant = storage_type {
            processed = processed.replace("@group(0) @binding(0)", "");
        }

        let sm = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(&format!("shader {test_name}")),
            source: ShaderSource::Wgsl(Cow::Borrowed(&processed)),
        });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("pipeline {test_name}")),
                layout: Some(&pll),
                module: &sm,
                entry_point: "cs_main",
            });

        let output_pre_init_data = vec![test.output_initialization; MAX_BUFFER_SIZE as usize / 4];
        ctx.queue.write_buffer(&output_buffer, 0, bytemuck::cast_slice(&output_pre_init_data));

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(&format!("cpass {test_name}")),
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);

        match storage_type {
            StorageType::Uniform | StorageType::Storage => {
                ctx.queue
                    .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&test.input_values));
            }
            StorageType::PushConstant => cpass.set_push_constants(0, bytemuck::cast_slice(&test.input_values)),
        }

        cpass.dispatch_workgroups(1, 1, 1);
        drop(cpass);

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &mapping_buffer, 0, MAX_BUFFER_SIZE);

        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
        ctx.device.poll(Maintain::Wait);

        let mapped = mapping_buffer.slice(..).get_mapped_range();

        let typed: &[u32] = bytemuck::cast_slice(&*mapped);

        let left = &typed[..test.output_values.len()];
        let right = test.output_values;
        let failure = left != right;
        if failure {
            eprintln!(
                "Inner test failure. Actual {:?}. Expected {:?}. Test {test_name}",
                left.iter().copied().collect::<Vec<_>>(),
                right.iter().copied().collect::<Vec<_>>(),
            );
        }
        let backend_failures = match storage_type {
            StorageType::Uniform => Backends::empty(),
            StorageType::Storage => Backends::empty(),
            StorageType::PushConstant => Backends::empty(),
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
