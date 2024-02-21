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
