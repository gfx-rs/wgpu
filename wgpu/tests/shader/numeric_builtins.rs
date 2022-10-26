use wgpu::{Backends, DownlevelFlags, Limits};

use crate::{
    common::{initialize_test, TestParameters},
    shader::{shader_input_output_test, InputStorageType, ShaderTest},
};

fn create_numeric_builtin_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[rustfmt::skip]
    let clamp_values: &[(f32, f32, f32, f32)] = &[
        // value - low - high - valid outputs

        // normal clamps
        (   20.0,  0.0,  10.0,  10.0),
        (  -10.0,  0.0,  10.0,  0.0),
        (    5.0,  0.0,  10.0,  5.0),

        // med-of-three or min/max
        (    3.0,  2.0,  1.0,   1.0),
    ];

    for &(input, low, high, output) in clamp_values {
        tests.push(ShaderTest {
            name: format!("clamp({input}, 0.0, 10.0) == {output})"),
            input_members: String::from("value: f32, low: f32, high: f32"),
            body: String::from(
                "output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));",
            ),
            input_values: bytemuck::cast_slice(&[input, low, high]).to_vec(),
            output_values: bytemuck::cast_slice(&[output]).to_vec(),
            output_initialization: u32::MAX,
            failures: Backends::empty(),
        });
    }

    tests
}

#[test]
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
