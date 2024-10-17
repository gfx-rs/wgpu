use wgpu::{DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

fn create_numeric_builtin_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[rustfmt::skip]
    let clamp_values: &[(f32, f32, f32, &[f32])] = &[
        // value - low - high - valid outputs

        // normal clamps
        (   20.0,  0.0,  10.0,  &[10.0]),
        (  -10.0,  0.0,  10.0,  &[0.0]),
        (    5.0,  0.0,  10.0,  &[5.0]),

        // med-of-three or min/max
        (    3.0,  2.0,  1.0,   &[1.0, 2.0]),
    ];

    for &(input, low, high, output) in clamp_values {
        let mut test = ShaderTest::new(
            format!("clamp({input}, {low}, {high}) == {output:?}"),
            String::from("value: f32, low: f32, high: f32"),
            String::from("output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));"),
            &[input, low, high],
            &[output[0]],
        );
        for &extra in &output[1..] {
            test = test.extra_output_values(&[extra]);
        }

        tests.push(test);
    }

    tests
}

#[gpu_test]
static NUMERIC_BUILTINS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_numeric_builtin_test(),
        )
    });

fn create_int64_atomic_min_max_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let test = ShaderTest::new(
        "atomicMax".into(),
        "value: u64".into(),
        "atomicMin(&output, 0lu); atomicMax(&output, 2lu);".into(),
        &[0],
        &[2],
    )
    .output_type("atomic<u64>".into());

    tests.push(test);

    let test = ShaderTest::new(
        "atomicMin".into(),
        "value: u64".into(),
        "atomicMax(&output, 100lu); atomicMin(&output, 4lu);".into(),
        &[0],
        &[4],
    )
    .output_type("atomic<u64>".into());

    tests.push(test);

    tests
}

#[gpu_test]
static INT64_ATOMIC_MIN_MAX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgt::Features::SHADER_INT64 | wgt::Features::SHADER_INT64_ATOMIC_MIN_MAX)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_int64_atomic_min_max_test(),
        )
    });

fn create_int64_atomic_all_ops_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let test = ShaderTest::new(
        "atomicAdd".into(),
        "value: u64".into(),
        "atomicStore(&output, 0lu); atomicAdd(&output, 1lu); atomicAdd(&output, 1lu);".into(),
        &[0],
        &[2],
    )
    .output_type("atomic<u64>".into());

    tests.push(test);

    let test = ShaderTest::new(
        "atomicAnd".into(),
        "value: u64".into(),
        "atomicStore(&output, 31lu); atomicAnd(&output, 30lu); atomicAnd(&output, 3lu);".into(),
        &[0],
        &[2],
    )
    .output_type("atomic<u64>".into());

    tests.push(test);

    let test = ShaderTest::new(
        "atomicOr".into(),
        "value: u64".into(),
        "atomicStore(&output, 0lu); atomicOr(&output, 3lu); atomicOr(&output, 6lu);".into(),
        &[0],
        &[7],
    )
    .output_type("atomic<u64>".into());

    tests.push(test);

    tests
}

#[gpu_test]
static INT64_ATOMIC_ALL_OPS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgt::Features::SHADER_INT64 | wgt::Features::SHADER_INT64_ATOMIC_ALL_OPS)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_int64_atomic_all_ops_test(),
        )
    });

fn create_float32_atomic_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let test = ShaderTest::new(
        "atomicAdd".into(),
        "value: f32".into(),
        "atomicStore(&output, 0.0); atomicAdd(&output, -0.50); atomicAdd(&output, 1.75);".into(),
        &[0_f32],
        &[1.25_f32],
    )
    .output_type("atomic<f32>".into());

    tests.push(test);

    let test = ShaderTest::new(
        "atomicAdd".into(),
        "value: f32".into(),
        "atomicStore(&output, 0.0); atomicSub(&output, -2.5); atomicSub(&output, 3.0);".into(),
        &[0_f32],
        &[-0.5_f32],
    )
    .output_type("atomic<f32>".into());

    tests.push(test);

    tests
}

#[gpu_test]
static FLOAT32_ATOMIC: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgt::Features::SHADER_FLOAT32_ATOMIC)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_float32_atomic_test())
    });

// See https://github.com/gfx-rs/wgpu/issues/5276
/*
fn create_int64_polyfill_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let u64_clz_values: &[(u64, u32)] = &[
        (u64::MAX, 0),
        (1, 63),
        (2, 62),
        (3, 62),
        (1 << 63, 0),
        (1 << 62, 1),
        (0, 64),
    ];

    for &(input, output) in u64_clz_values {
        let test = ShaderTest::new(
            format!("countLeadingZeros({input}lu) == {output:?}"),
            String::from("value: u64"),
            String::from("output[0] = u32(countLeadingZeros(input.value));"),
            &[input],
            &[output],
        );

        tests.push(test);
    }

    let i64_clz_values: &[(i64, u32)] = &[
        (i64::MAX, 1),
        (i64::MIN, 0),
        (1, 63),
        (1 << 62, 1),
        (-1 << 62, 0),
        (0, 64),
        (-1, 0),
    ];

    for &(input, output) in i64_clz_values {
        let test = ShaderTest::new(
            format!("countLeadingZeros({input}li) == {output:?}"),
            String::from("value: i64"),
            String::from("output[0] = u32(countLeadingZeros(input.value));"),
            &[input],
            &[output],
        );

        tests.push(test);
    }

    let u64_flb_values: &[(u64, u32)] = &[
        (u64::MAX, 63),
        (1, 0),
        (2, 1),
        (3, 1),
        (1 << 63, 63),
        (1 << 62, 62),
        (0, u32::MAX),
    ];

    for &(input, output) in u64_flb_values {
        let test = ShaderTest::new(
            format!("firstLeadingBit({input}lu) == {output:?}"),
            String::from("value: u64"),
            String::from("output[0] = u32(firstLeadingBit(input.value));"),
            &[input],
            &[output],
        );

        tests.push(test);
    }

    let i64_flb_values: &[(i64, u32)] = &[
        (i64::MAX, 62),
        (i64::MIN, 62),
        (1, 0),
        (1 << 62, 62),
        (-1 << 62, 61),
        (0, u32::MAX),
        (-1, u32::MAX),
    ];

    for &(input, output) in i64_flb_values {
        let test = ShaderTest::new(
            format!("firstLeadingBit({input}li) == {output:?}"),
            String::from("value: i64"),
            String::from("output[0] = u32(firstLeadingBit(input.value));"),
            &[input],
            &[output],
        );

        tests.push(test);
    }

    tests
}

#[gpu_test]
static INT64_POLYFILL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_INT64)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_int64_polyfill_test())
    });
*/
