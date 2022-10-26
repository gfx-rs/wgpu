//! Infrastructure for testing particular behavior of shaders across platforms.
//!
//! The tests take the form of a input buffer filled with u32 data. A compute
//! shader is run on the input buffer which generates an output buffer. This
//! buffer is then read and compared to a given output.

use std::borrow::Cow;

use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Maintain, MapMode, PipelineLayoutDescriptor, PushConstantRange,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::common::TestingContext;

mod struct_layout;

#[derive(Clone, Copy, PartialEq)]
enum InputStorageType {
    Uniform,
    Storage,
    PushConstant,
}

impl InputStorageType {
    fn as_str(&self) -> &'static str {
        match self {
            InputStorageType::Uniform => "uniform",
            InputStorageType::Storage => "storage",
            InputStorageType::PushConstant => "push_constant",
        }
    }
}

/// Describes a single test of a shader.
struct ShaderTest {
    /// Human readable name
    name: String,
    /// This text will be the body of the `Input` struct. Replaces "{{input_members}}"
    /// in the shader_test shader.
    input_members: String,
    /// This text will be the body of the compute shader. Replaces "{{body}}"
    /// in the shader_test shader.
    body: String,
    /// List of values will be written to the input buffer.
    input_values: Vec<u32>,
    /// List of expected outputs from the shader.
    output_values: Vec<u32>,
    /// Value to pre-initialize the output buffer to. Often u32::MAX so
    /// that writing a 0 looks different than not writing a value at all.
    output_initialization: u32,
    /// Which backends this test will fail on. If the test passes on this
    /// backend when it shouldn't, an assert will be raised.
    failures: Backends,
}

const MAX_BUFFER_SIZE: u64 = 128;

/// Runs the given shader tests with the given storage_type for the input_buffer.
fn shader_input_output_test(
    ctx: TestingContext,
    storage_type: InputStorageType,
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
                        // We don't use this buffer for push constants, but for simplicity
                        // we just use the storage buffer binding.
                        ty: match storage_type {
                            InputStorageType::Uniform => wgpu::BufferBindingType::Uniform,
                            InputStorageType::Storage | InputStorageType::PushConstant => {
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
                InputStorageType::PushConstant => &[PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..MAX_BUFFER_SIZE as u32,
                }],
                _ => &[],
            },
        });

    let mut fail = false;
    for test in tests {
        assert!(test.input_values.len() <= MAX_BUFFER_SIZE as usize / 4);
        assert!(test.output_values.len() <= MAX_BUFFER_SIZE as usize / 4);

        let test_name = test.name;

        // -- Building shader + pipeline --

        let mut processed = source
            .replace("{{storage_type}}", storage_type.as_str())
            .replace("{{input_members}}", &test.input_members)
            .replace("{{body}}", &test.body);

        // Add the bindings for all inputs besides push constants.
        processed = if matches!(storage_type, InputStorageType::PushConstant) {
            processed.replace("{{input_bindings}}", "")
        } else {
            processed.replace("{{input_bindings}}", "@group(0) @binding(0)")
        };

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

        // -- Initializing data --

        let output_pre_init_data = vec![test.output_initialization; MAX_BUFFER_SIZE as usize / 4];
        ctx.queue.write_buffer(
            &output_buffer,
            0,
            bytemuck::cast_slice(&output_pre_init_data),
        );

        match storage_type {
            InputStorageType::Uniform | InputStorageType::Storage => {
                ctx.queue
                    .write_buffer(&input_buffer, 0, bytemuck::cast_slice(&test.input_values));
            }
            _ => {
                // Init happens in the compute pass
            }
        }

        // -- Run test --

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(&format!("cpass {test_name}")),
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bg, &[]);

        if let InputStorageType::PushConstant = storage_type {
            cpass.set_push_constants(0, bytemuck::cast_slice(&test.input_values))
        }

        cpass.dispatch_workgroups(1, 1, 1);
        drop(cpass);

        // -- Pulldown data --

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &mapping_buffer, 0, MAX_BUFFER_SIZE);

        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
        ctx.device.poll(Maintain::Wait);

        let mapped = mapping_buffer.slice(..).get_mapped_range();

        let typed: &[u32] = bytemuck::cast_slice(&*mapped);

        // -- Check results --

        let left = &typed[..test.output_values.len()];
        let right = test.output_values;
        let failure = left != right;
        // We don't immediately panic to let all tests execute
        if failure {
            eprintln!(
                "Inner test failure. Actual {:?}. Expected {:?}. Test {test_name}",
                left.to_vec(),
                right.to_vec(),
            );
        }
        if failure
            != test
                .failures
                .contains(ctx.adapter.get_info().backend.into())
        {
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
