//! Tests for the timestamp normalization algorithm's utility functions.
//!
//! Because they involve multiple kinds of hand-rolled math operations,
//! we do testing to ensure the overall operation (which is very simple)
//! works correctly.

use nanorand::Rng;
use wgpu::{util::DeviceExt, Limits};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uint96(u32, u32, u32);

impl Uint96 {
    fn from_u128(value: u128) -> Self {
        let a = (value & 0xFFFF_FFFF) as u32;
        let b = ((value >> 32) & 0xFFFF_FFFF) as u32;
        let c = ((value >> 64) & 0xFFFF_FFFF) as u32;

        Self(a, b, c)
    }

    fn as_u128(&self) -> u128 {
        ((self.2 as u128) << 64) | ((self.1 as u128) << 32) | (self.0 as u128)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct U64MulU32Input {
    left: u64,
    right: u32,
    _pad: u32,
}

impl U64MulU32Input {
    fn new(left: u64, right: u32) -> Self {
        Self {
            left,
            right,
            _pad: 0,
        }
    }
}

fn assert_u64_mul_u32(left: u64, right: u32, computed: Uint96) {
    let real = left as u128 * right as u128;

    let computed = computed.as_u128();

    assert_eq!(
        computed, real,
        "{left} * {right} should be {real} but is {computed}"
    );
}

#[gpu_test]
static U64_MUL_U32: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(Limits {
                max_storage_buffer_binding_size: 256 * 1024 * 1024,
                ..Limits::downlevel_defaults()
            }),
    )
    .run_sync(test_u64_mul_u32);

fn test_u64_mul_u32(ctx: TestingContext) {
    const TOTAL_RANDOM_INPUTS: usize = 1_000_000;
    const MANUAL_INPUTS: usize = 2;

    const TOTAL_INPUTS: usize = TOTAL_RANDOM_INPUTS + MANUAL_INPUTS;

    let mut inputs = Vec::with_capacity(TOTAL_INPUTS);

    inputs.push(U64MulU32Input::new(2, 2));
    inputs.push(U64MulU32Input::new(u64::MAX, u32::MAX));

    // Smoke test the algorithm by generating 1M random inputs, and checking the results.
    let mut generator = nanorand::WyRand::new_seed(0xDEAD_BEEF);

    for _ in 0..TOTAL_RANDOM_INPUTS {
        let left = generator.generate::<u64>();
        let right = generator.generate::<u32>();

        inputs.push(U64MulU32Input::new(left, right));
    }

    assert_eq!(TOTAL_INPUTS, inputs.len());

    let output_bytes = process_shader(
        ctx,
        bytemuck::cast_slice(&inputs),
        include_str!("u64_mul_u32.wgsl"),
    );
    let output_values = bytemuck::pod_collect_to_vec(&output_bytes);

    for (&input, &output) in inputs.iter().zip(output_values.iter()) {
        assert_u64_mul_u32(input.left, input.right, output);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShiftRightU96Input {
    value: Uint96,
    shift: u32,
}

impl ShiftRightU96Input {
    fn new(value: u128, shift: u32) -> Self {
        assert!(shift <= 32);
        assert!(value >> 96 == 0);

        Self {
            value: Uint96::from_u128(value),
            shift,
        }
    }
}

fn assert_shift_right_u96(value: Uint96, shift: u32, computed: Uint96) {
    let value = value.as_u128();

    let real = value >> shift;

    let computed = computed.as_u128();

    assert_eq!(
        computed, real,
        "{value:X} >> {shift} should be {real:X} but is {computed:X}",
    );
}

#[gpu_test]
static SHIFT_RIGHT_U96: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .limits(Limits {
                max_storage_buffer_binding_size: 256 * 1024 * 1024,
                ..Limits::downlevel_defaults()
            }),
    )
    .run_sync(test_shift_right_u96);

fn test_shift_right_u96(ctx: TestingContext) {
    const TOTAL_RANDOM_INPUTS: usize = 1_000_000;
    const TOTAL_SHIFT_INPUTS: usize = 33;
    const MANUAL_INPUTS: usize = 1;

    const TOTAL_INPUTS: usize = TOTAL_RANDOM_INPUTS + TOTAL_SHIFT_INPUTS + MANUAL_INPUTS;

    let mut inputs = Vec::with_capacity(TOTAL_INPUTS);

    inputs.push(ShiftRightU96Input::new(1, 1));

    for shift in 0..TOTAL_SHIFT_INPUTS {
        // 96 bit number with a visually recognizable pattern.
        const INTERESTING_NUMBER: u128 = 0x1234_5678_9ABC_DEF0_1234_5678;

        inputs.push(ShiftRightU96Input::new(INTERESTING_NUMBER, shift as u32));
    }

    // Smoke test the algorithm by generating 1M random inputs, and checking the results.
    let mut generator = nanorand::WyRand::new_seed(0xDEAD_BEEF);

    for _ in 0..TOTAL_RANDOM_INPUTS {
        // nanorand doesn't have generate_range for u128, so just chop the top bits off.
        let value = generator.generate::<u128>() >> 32;
        let shift = generator.generate_range(0..=32);

        inputs.push(ShiftRightU96Input::new(value, shift));
    }

    assert_eq!(TOTAL_INPUTS, inputs.len());

    let output_bytes = process_shader(
        ctx,
        bytemuck::cast_slice(&inputs),
        include_str!("shift_right_u96.wgsl"),
    );

    let output_values = bytemuck::pod_collect_to_vec(&output_bytes);

    for (&input, &output) in inputs.iter().zip(output_values.iter()) {
        assert_shift_right_u96(input.value, input.shift, output);
    }
}

fn process_shader(ctx: TestingContext, inputs: &[u8], entry_point_src: &str) -> Vec<u8> {
    let common_src = include_str!("../../../../wgpu-core/src/timestamp_normalization/common.wgsl");

    let full_source = format!("{common_src}\n{entry_point_src}");

    let shader_module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("u64_mul_u32"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("u64_mul_u32"),
            layout: None,
            module: &shader_module,
            entry_point: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let input_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: inputs,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let output_size = (size_of::<Uint96>() * inputs.len()) as u64;

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let pulldown_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pulldown Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = pipeline.get_bind_group_layout(0);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Compute Pass"),
        timestamp_writes: None,
    });

    cpass.set_pipeline(&pipeline);
    cpass.set_bind_group(0, &bg, &[]);
    cpass.dispatch_workgroups(inputs.len().div_ceil(256) as u32, 1, 1);

    drop(cpass);

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &pulldown_buffer, 0, output_size);

    ctx.queue.submit([encoder.finish()]);
    pulldown_buffer.map_async(wgpu::MapMode::Read, .., |_| {});

    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    let value = pulldown_buffer.get_mapped_range(..).to_vec();

    value
}
