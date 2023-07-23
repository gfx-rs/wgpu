use wasm_bindgen_test::*;
use wgpu::{DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{initialize_test, TestParameters};

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
            format!("clamp({input}, 0.0, 10.0) == {output:?})"),
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

#[test]
#[wasm_bindgen_test]
fn numeric_builtins() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            shader_input_output_test(
                ctx,
                InputStorageType::Storage,
                create_numeric_builtin_test(),
            );
        },
    );
}
