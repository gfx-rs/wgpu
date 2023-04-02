//! Infrastructure for testing particular behavior of shaders across platforms.
//!
//! The tests take the form of a input buffer filled with u32 data. A compute
//! shader is run on the input buffer which generates an output buffer. This
//! buffer is then read and compared to a given output.

use std::{borrow::Cow, fmt::Debug};

use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Maintain, MapMode, PipelineLayoutDescriptor, PushConstantRange,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::common::TestingContext;

mod numeric_builtins;
mod struct_layout;
mod zero_init_workgroup_mem;

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
    custom_struct_members: String,
    /// This text will be the body of the compute shader. Replaces "{{body}}"
    /// in the shader_test shader.
    body: String,
    /// This text will be the input type of the compute shader. Replaces "{{input_type}}".
    ///
    /// Defaults to "CustomStruct"
    input_type: String,
    /// This text will be the output type of the compute shader. Replaces "{{output_type}}".
    ///
    /// Defaults to "array<u32>".
    output_type: String,
    /// List of values will be written to the input buffer.
    input_values: Vec<u32>,
    /// List of lists of valid expected outputs from the shader.
    output_values: Vec<Vec<u32>>,
    /// Function which compares the output values to the resulting values and
    /// prints a message on failure.
    ///
    /// Defaults [`Self::default_comparison_function`].
    output_comparison_fn: fn(&str, &[u32], &[Vec<u32>]) -> bool,
    /// Value to pre-initialize the output buffer to. Often u32::MAX so
    /// that writing a 0 looks different than not writing a value at all.
    ///
    /// Defaults to u32::MAX.
    output_initialization: u32,
    /// Which backends this test will fail on. If the test passes on this
    /// backend when it shouldn't, an assert will be raised.
    ///
    /// Defaults to Backends::empty().
    failures: Backends,
}
impl ShaderTest {
    fn default_comparison_function<O: bytemuck::Pod + Debug + PartialEq>(
        test_name: &str,
        actual_values: &[u32],
        expected_values: &[Vec<u32>],
    ) -> bool {
        let cast_actual = bytemuck::cast_slice::<u32, O>(actual_values);

        // When printing the error message, we want to trim `cast_actual` to the length
        // of the longest set of expected values. This tracks that value.
        let mut max_relevant_value_count = 0;

        for expected in expected_values {
            let cast_expected = bytemuck::cast_slice::<u32, O>(expected);

            // We shorten the actual to the length of the expected.
            if &cast_actual[0..cast_expected.len()] == cast_expected {
                return true;
            }

            max_relevant_value_count = max_relevant_value_count.max(cast_expected.len());
        }

        // We haven't found a match, lets print an error.

        eprint!(
            "Inner test failure. Actual {:?}. Expected",
            &cast_actual[0..max_relevant_value_count]
        );

        if expected_values.len() != 1 {
            eprint!(" one of: ");
        } else {
            eprint!(": ");
        }

        for (idx, expected) in expected_values.iter().enumerate() {
            let cast_expected = bytemuck::cast_slice::<u32, O>(expected);
            eprint!("{cast_expected:?}");
            if idx + 1 != expected_values.len() {
                eprint!(" ");
            }
        }

        eprintln!(". Test {test_name}");

        false
    }

    fn new<I: bytemuck::Pod, O: bytemuck::Pod + Debug + PartialEq>(
        name: String,
        custom_struct_members: String,
        body: String,
        input_values: &[I],
        output_values: &[O],
    ) -> Self {
        Self {
            name,
            custom_struct_members,
            body,
            input_type: String::from("CustomStruct"),
            output_type: String::from("array<u32>"),
            input_values: bytemuck::cast_slice(input_values).to_vec(),
            output_values: vec![bytemuck::cast_slice(output_values).to_vec()],
            output_comparison_fn: Self::default_comparison_function::<O>,
            output_initialization: u32::MAX,
            failures: Backends::empty(),
        }
    }

    /// Add another set of possible outputs. If any of the given
    /// output values are seen it's considered a success (i.e. this is OR, not AND).
    ///
    /// Assumes that this type O is the same as the O provided to new.
    fn extra_output_values<O: bytemuck::Pod + Debug + PartialEq>(
        mut self,
        output_values: &[O],
    ) -> Self {
        self.output_values
            .push(bytemuck::cast_slice(output_values).to_vec());

        self
    }

    fn failures(mut self, failures: Backends) -> Self {
        self.failures = failures;

        self
    }
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

        // This isn't terribly efficient but the string is short and it's a test.
        // The body and input members are the longest part, so do them last.
        let mut processed = source
            .replace("{{storage_type}}", storage_type.as_str())
            .replace("{{input_type}}", &test.input_type)
            .replace("{{output_type}}", &test.output_type)
            .replace("{{input_members}}", &test.custom_struct_members)
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
            timestamp_writes: &[],
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

        let typed: &[u32] = bytemuck::cast_slice(&mapped);

        // -- Check results --

        let failure = !(test.output_comparison_fn)(&test_name, typed, &test.output_values);
        // We don't immediately panic to let all tests execute
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
