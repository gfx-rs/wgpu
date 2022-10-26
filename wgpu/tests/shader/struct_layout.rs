use std::fmt::Write;

use wgpu::{DownlevelFlags, Limits, Features};

use crate::{
    common::{initialize_test, TestParameters},
    shader::{ShaderTest, StorageType, shader_input_output_test, MAX_BUFFER_SIZE},
};

fn create_struct_layout_tests() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    // Vector tests
    for components in [2, 3, 4] {
        for ty in ["f32", "u32", "i32"] {
            let members = format!("member: vec{components}<{ty}>,");
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

            tests.push(ShaderTest {
                name: format!("vec{components}<{ty}> - direct"),
                members: members.clone(),
                body: direct,
                output_values: (1..=components as u32).collect(),
            });

            tests.push(ShaderTest {
                name: format!("vec{components}<{ty}> - loaded"),
                members,
                body: loaded,
                output_values: (1..=components as u32).collect(),
            });
        }
    }

    // Matrix tests
    for columns in [2, 3, 4] {
        for rows in [2, 3, 4] {
            let ty = format!("mat{columns}x{rows}<f32>");
            let members = format!("member: {ty},");
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

                    output_values.push(current_input_idx + 1);
                    current_input_idx += 1;
                    current_output_idx += 1;
                }
                // Round to next vec4 if we're matrices with vec3 columns
                if rows == 3 {
                    current_input_idx += 1;
                }
            }

            tests.push(ShaderTest {
                name: format!("{ty} - direct"),
                members: members.clone(),
                body: direct,
                output_values: output_values.clone(),
            });

            tests.push(ShaderTest {
                name: format!("{ty} - vector loaded"),
                members: members.clone(),
                body: vector_loaded,
                output_values: output_values.clone(),
            });

            tests.push(ShaderTest {
                name: format!("{ty} - fully loaded"),
                members,
                body: fully_loaded,
                output_values,
            });
        }
    }

    // Vec3 alignment tests
    for ty in ["f32", "u32", "i32"] {
        let members = format!("_vec: vec3<{ty}>,\nscalar: {ty},");
        let direct = String::from("output[0] = bitcast<u32>(input.scalar);");

        tests.push(ShaderTest {
            name: format!("vec3<{ty}>, {ty} alignment"),
            members,
            body: direct,
            output_values: vec![4],
        });
    }

    tests
}

#[test]
fn input_layout_uniform() {
    initialize_test(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
        |ctx| {
            shader_input_output_test(ctx, StorageType::Uniform, create_struct_layout_tests());
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
            shader_input_output_test(ctx, StorageType::Storage, create_struct_layout_tests());
        },
    );
}

#[test]
fn input_layout_push_constant() {
    initialize_test(
        TestParameters::default()
            .features(Features::PUSH_CONSTANTS)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits {
                max_push_constant_size: MAX_BUFFER_SIZE as u32,
                ..Limits::downlevel_defaults()
            }),
        |ctx| {
            shader_input_output_test(ctx, StorageType::PushConstant, create_struct_layout_tests());
        },
    );
}
