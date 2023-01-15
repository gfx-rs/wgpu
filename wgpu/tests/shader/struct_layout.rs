use std::fmt::Write;

use wasm_bindgen_test::*;
use wgpu::{Backends, DownlevelFlags, Features, Limits};

use crate::{
    common::{initialize_test, TestParameters},
    shader::{shader_input_output_test, InputStorageType, ShaderTest, MAX_BUFFER_SIZE},
};

fn create_struct_layout_tests(storage_type: InputStorageType) -> Vec<ShaderTest> {
    let input_values: Vec<_> = (0..(MAX_BUFFER_SIZE as u32 / 4)).collect();

    let mut tests = Vec::new();

    // Vector tests
    for components in [2, 3, 4] {
        for ty in ["f32", "u32", "i32"] {
            let input_members = format!("member: vec{components}<{ty}>,");
            // There's 2 possible ways to load a component of a vector:
            // - Do `input.member.x` (direct)
            // - Store `input.member` in a variable; do `var.x` (loaded)
            let mut direct = String::new();
            let mut loaded = String::from("let loaded = input.member;");
            let component_accessors = ["x", "y", "z", "w"]
                .into_iter()
                .take(components)
                .enumerate();
            for (idx, component) in component_accessors {
                writeln!(
                    direct,
                    "output[{idx}] = bitcast<u32>(input.member.{component});"
                )
                .unwrap();
                writeln!(loaded, "output[{idx}] = bitcast<u32>(loaded.{component});").unwrap();
            }

            tests.push(ShaderTest::new(
                format!("vec{components}<{ty}> - direct"),
                input_members.clone(),
                direct,
                &input_values,
                &(0..components as u32).collect::<Vec<_>>(),
            ));

            tests.push(ShaderTest::new(
                format!("vec{components}<{ty}> - loaded"),
                input_members.clone(),
                loaded,
                &input_values,
                &(0..components as u32).collect::<Vec<_>>(),
            ));
        }
    }

    // Matrix tests
    for columns in [2, 3, 4] {
        for rows in [2, 3, 4] {
            let ty = format!("mat{columns}x{rows}<f32>");
            let input_members = format!("member: {ty},");
            // There's 3 possible ways to load a component of a matrix:
            // - Do `input.member[0].x` (direct)
            // - Store `input.member[0]` in a variable; do `var.x` (vector_loaded)
            // - Store `input.member` in a variable; do `var[0].x` (fully_loaded)
            let mut direct = String::new();
            let mut vector_loaded = String::new();
            let mut fully_loaded = String::from("let loaded = input.member;");
            for column in 0..columns {
                writeln!(vector_loaded, "let vec_{column} = input.member[{column}];").unwrap();
            }

            let mut output_values = Vec::new();

            let mut current_output_idx = 0;
            let mut current_input_idx = 0;
            for column in 0..columns {
                let component_accessors = ["x", "y", "z", "w"].into_iter().take(rows);
                for component in component_accessors {
                    writeln!(
                        direct,
                        "output[{current_output_idx}] = bitcast<u32>(input.member[{column}].{component});"
                    )
                    .unwrap();
                    writeln!(
                        vector_loaded,
                        "output[{current_output_idx}] = bitcast<u32>(vec_{column}.{component});"
                    )
                    .unwrap();
                    writeln!(
                        fully_loaded,
                        "output[{current_output_idx}] = bitcast<u32>(loaded[{column}].{component});"
                    )
                    .unwrap();

                    output_values.push(current_input_idx);
                    current_input_idx += 1;
                    current_output_idx += 1;
                }
                // Round to next vec4 if we're matrices with vec3 columns
                if rows == 3 {
                    current_input_idx += 1;
                }
            }

            // https://github.com/gfx-rs/naga/issues/1785
            let failures = if storage_type == InputStorageType::Uniform && rows == 2 {
                Backends::GL
            } else {
                Backends::empty()
            };

            tests.push(
                ShaderTest::new(
                    format!("{ty} - direct"),
                    input_members.clone(),
                    direct,
                    &input_values,
                    &output_values,
                )
                .failures(failures),
            );

            tests.push(
                ShaderTest::new(
                    format!("{ty} - vector loaded"),
                    input_members.clone(),
                    vector_loaded,
                    &input_values,
                    &output_values,
                )
                .failures(failures),
            );

            tests.push(
                ShaderTest::new(
                    format!("{ty} - fully loaded"),
                    input_members.clone(),
                    fully_loaded,
                    &input_values,
                    &output_values,
                )
                .failures(failures),
            );
        }
    }

    // Vec3 alignment tests
    for ty in ["f32", "u32", "i32"] {
        let members = format!("_vec: vec3<{ty}>,\nscalar: {ty},");
        let direct = String::from("output[0] = bitcast<u32>(input.scalar);");

        tests.push(ShaderTest::new(
            format!("vec3<{ty}>, {ty} alignment"),
            members,
            direct,
            &input_values,
            &[3],
        ));
    }

    // Mat3 alignment tests
    for ty in ["f32", "u32", "i32"] {
        for columns in [2, 3, 4] {
            let members = format!("_mat: mat{columns}x3<f32>,\nscalar: {ty},");
            let direct = String::from("output[0] = bitcast<u32>(input.scalar);");

            tests.push(ShaderTest::new(
                format!("mat{columns}x3<f32>, {ty} alignment"),
                members,
                direct,
                &input_values,
                &[columns * 4],
            ));
        }
    }

    tests
}

#[test]
#[wasm_bindgen_test]
fn uniform_input() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            shader_input_output_test(
                ctx,
                InputStorageType::Uniform,
                create_struct_layout_tests(InputStorageType::Uniform),
            );
        },
    );
}

#[test]
#[wasm_bindgen_test]
fn storage_input() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            shader_input_output_test(
                ctx,
                InputStorageType::Storage,
                create_struct_layout_tests(InputStorageType::Storage),
            );
        },
    );
}

#[test]
#[wasm_bindgen_test]
fn push_constant_input() {
    initialize_test(
        TestParameters::default()
            .features(Features::PUSH_CONSTANTS)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits {
                max_push_constant_size: MAX_BUFFER_SIZE as u32,
                ..Limits::downlevel_defaults()
            })
            .backend_failure(Backends::GL),
        |ctx| {
            shader_input_output_test(
                ctx,
                InputStorageType::PushConstant,
                create_struct_layout_tests(InputStorageType::PushConstant),
            );
        },
    );
}
