use std::fmt::Write;

use wgpu::{Backends, DownlevelFlags, Features, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest, MAX_BUFFER_SIZE};
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

#[gpu_test]
static UNIFORM_INPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            // Validation errors thrown by the SPIR-V validator https://github.com/gfx-rs/wgpu/issues/4371
            .expect_fail(
                FailureCase::backend(wgpu::Backends::VULKAN)
                    .validation_error("a matrix with stride 8 not satisfying alignment to 16"),
            )
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Uniform,
            create_struct_layout_tests(InputStorageType::Uniform),
        )
    });

#[gpu_test]
static STORAGE_INPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_struct_layout_tests(InputStorageType::Storage),
        )
    });

#[gpu_test]
static PUSH_CONSTANT_INPUT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PUSH_CONSTANTS)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits {
                max_push_constant_size: MAX_BUFFER_SIZE as u32,
                ..Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::PushConstant,
            create_struct_layout_tests(InputStorageType::PushConstant),
        )
    });

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

            // https://github.com/gfx-rs/wgpu/issues/4371
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

    // Test for https://github.com/gfx-rs/wgpu/issues/5262.
    //
    // The struct is supposed to have a size of 32 and alignment of 16.
    for ty in ["f32", "u32", "i32"] {
        let header = format!("struct Inner {{ vec: vec3<{ty}>, scalar1: u32, scalar2: u32 }}");
        let members = String::from("arr: array<Inner, 2>");
        let direct = String::from(
            "\
            output[0] = bitcast<u32>(input.arr[0].vec.x);
            output[1] = bitcast<u32>(input.arr[0].vec.y);
            output[2] = bitcast<u32>(input.arr[0].vec.z);
            output[3] = bitcast<u32>(input.arr[0].scalar1);
            output[4] = bitcast<u32>(input.arr[0].scalar2);
            output[5] = bitcast<u32>(input.arr[1].vec.x);
            output[6] = bitcast<u32>(input.arr[1].vec.y);
            output[7] = bitcast<u32>(input.arr[1].vec.z);
            output[8] = bitcast<u32>(input.arr[1].scalar1);
            output[9] = bitcast<u32>(input.arr[1].scalar2);
        ",
        );

        tests.push(
            ShaderTest::new(
                format!("Alignment of 24 byte struct with a vec3<{ty}>"),
                members,
                direct,
                &input_values,
                &[0, 1, 2, 3, 4, 8, 9, 10, 11, 12],
            )
            .header(header),
        );
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

    // Nested struct and array test.
    //
    // This tries to exploit all the weird edge cases of the struct layout algorithm.
    {
        let header =
            String::from("struct Inner { scalar: f32, member: array<vec3<f32>, 2>, scalar2: f32 }");
        let members = String::from("inner: Inner, scalar3: f32, vector: vec3<f32>, scalar4: f32");
        let direct = String::from(
            "\
            output[0] = bitcast<u32>(input.inner.scalar);
            output[1] = bitcast<u32>(input.inner.member[0].x);
            output[2] = bitcast<u32>(input.inner.member[0].y);
            output[3] = bitcast<u32>(input.inner.member[0].z);
            output[4] = bitcast<u32>(input.inner.member[1].x);
            output[5] = bitcast<u32>(input.inner.member[1].y);
            output[6] = bitcast<u32>(input.inner.member[1].z);
            output[7] = bitcast<u32>(input.inner.scalar2);
            output[8] = bitcast<u32>(input.scalar3);
            output[9] = bitcast<u32>(input.vector.x);
            output[10] = bitcast<u32>(input.vector.y);
            output[11] = bitcast<u32>(input.vector.z);
            output[12] = bitcast<u32>(input.scalar4);
        ",
        );

        tests.push(
            ShaderTest::new(
                String::from("nested struct and array"),
                members,
                direct,
                &input_values,
                &[
                    0, // inner.scalar
                    4, 5, 6, // inner.member[0]
                    8, 9, 10, // inner.member[1]
                    12, // scalar2
                    16, // scalar3
                    20, 21, 22, // vector
                    23, // scalar4
                ],
            )
            .header(header),
        );
    }

    tests
}

#[gpu_test]
static UNIFORM_INPUT_INT64: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_INT64)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_64bit_struct_layout_tests(),
        )
    });

#[gpu_test]
static STORAGE_INPUT_INT64: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_INT64)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_64bit_struct_layout_tests(),
        )
    });

#[gpu_test]
static PUSH_CONSTANT_INPUT_INT64: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_INT64 | Features::PUSH_CONSTANTS)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits {
                max_push_constant_size: MAX_BUFFER_SIZE as u32,
                ..Limits::downlevel_defaults()
            }),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::PushConstant,
            create_64bit_struct_layout_tests(),
        )
    });

fn create_64bit_struct_layout_tests() -> Vec<ShaderTest> {
    let input_values: Vec<_> = (0..(MAX_BUFFER_SIZE as u32 / 4)).collect();

    let mut tests = Vec::new();

    // 64 bit alignment tests
    for ty in ["u64", "i64"] {
        let members = format!("scalar: {ty},");
        let direct = String::from(
            "\
            output[0] = u32(bitcast<u64>(input.scalar) & 0xFFFFFFFF);
            output[1] = u32((bitcast<u64>(input.scalar) >> 32) & 0xFFFFFFFF);
        ",
        );

        tests.push(ShaderTest::new(
            format!("{ty} alignment"),
            members,
            direct,
            &input_values,
            &[0, 1],
        ));
    }

    // Nested struct and array test.
    //
    // This tries to exploit all the weird edge cases of the struct layout algorithm.
    // We dont go as all-out as the other nested struct test because
    // all our primitives are twice as wide and we have only so much buffer to spare.
    {
        let header = String::from(
            "struct Inner { scalar: u64, scalar32: u32, member: array<vec3<u64>, 2> }",
        );
        let members = String::from("inner: Inner");
        let direct = String::from(
            "\
            output[0] = u32(bitcast<u64>(input.inner.scalar) & 0xFFFFFFFF);
            output[1] = u32((bitcast<u64>(input.inner.scalar) >> 32) & 0xFFFFFFFF);
            output[2] = bitcast<u32>(input.inner.scalar32);
            for (var index = 0u; index < 2u; index += 1u) {
                for (var component = 0u; component < 3u; component += 1u) {
                    output[3 + index * 6 + component * 2] = u32(bitcast<u64>(input.inner.member[index][component]) & 0xFFFFFFFF);
                    output[4 + index * 6 + component * 2] = u32((bitcast<u64>(input.inner.member[index][component]) >> 32) & 0xFFFFFFFF);
                }
            }
        ",
        );

        tests.push(
            ShaderTest::new(
                String::from("nested struct and array"),
                members,
                direct,
                &input_values,
                &[
                    0, 1, // inner.scalar
                    2, // inner.scalar32
                    8, 9, 10, 11, 12, 13, // inner.member[0]
                    16, 17, 18, 19, 20, 21, // inner.member[1]
                ],
            )
            .header(header),
        );
    }
    {
        let header = String::from("struct Inner { scalar32: u32, scalar: u64, scalar32_2: u32 }");
        let members = String::from("inner: Inner, vector: vec3<i64>");
        let direct = String::from(
            "\
            output[0] = bitcast<u32>(input.inner.scalar32);
            output[1] = u32(bitcast<u64>(input.inner.scalar) & 0xFFFFFFFF);
            output[2] = u32((bitcast<u64>(input.inner.scalar) >> 32) & 0xFFFFFFFF);
            output[3] = bitcast<u32>(input.inner.scalar32_2);
            output[4] = u32(bitcast<u64>(input.vector.x) & 0xFFFFFFFF);
            output[5] = u32((bitcast<u64>(input.vector.x) >> 32) & 0xFFFFFFFF);
            output[6] = u32(bitcast<u64>(input.vector.y) & 0xFFFFFFFF);
            output[7] = u32((bitcast<u64>(input.vector.y) >> 32) & 0xFFFFFFFF);
            output[8] = u32(bitcast<u64>(input.vector.z) & 0xFFFFFFFF);
            output[9] = u32((bitcast<u64>(input.vector.z) >> 32) & 0xFFFFFFFF);
        ",
        );

        tests.push(
            ShaderTest::new(
                String::from("nested struct and array"),
                members,
                direct,
                &input_values,
                &[
                    0, // inner.scalar32
                    2, 3, // inner.scalar
                    4, // inner.scalar32_2
                    8, 9, 10, 11, 12, 13, // vector
                ],
            )
            .header(header),
        );
    }

    tests
}

#[gpu_test]
static UNIFORM_INPUT_F16: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_F16)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_16bit_struct_layout_test(),
        )
    });

#[gpu_test]
static STORAGE_INPUT_F16: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_F16)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_16bit_struct_layout_test(),
        )
    });

fn create_16bit_struct_layout_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    fn f16asu16(f32: f32) -> u16 {
        half::f16::from_f32(f32).to_bits()
    }

    // 16 bit alignment tests
    {
        let members =
            "scalar1: f16, scalar2: f16, v3: vec3<f16>, tuck_in: f16, scalar4: f16, larger: u32";
        let direct = String::from(
            "\
            output[0] = u32(input.scalar1);
            output[1] = u32(input.scalar2);
            output[2] = u32(input.v3.x);
            output[3] = u32(input.v3.y);
            output[4] = u32(input.v3.z);
            output[5] = u32(input.tuck_in);
            output[6] = u32(input.scalar4);
            output[7] = u32(extractBits(input.larger, 0u, 16u));
            output[8] = u32(extractBits(input.larger, 16u, 16u));
        ",
        );

        tests.push(ShaderTest::new(
            "f16 alignment".into(),
            members.into(),
            direct,
            &[
                f16asu16(0.0),
                f16asu16(1.0),
                f16asu16(2.0),
                f16asu16(3.0),
                f16asu16(4.0),
                f16asu16(5.0),
                f16asu16(6.0),
                f16asu16(7.0),
                f16asu16(8.0),
                f16asu16(9.0),
                10_u16,
                11_u16,
                // Some extra values to help debug if the test fails.
                12_u16,
                13_u16,
                14_u16,
                15_u16,
                16_u16,
                17_u16,
                18_u16,
                19_u16,
                20_u16,
            ],
            &[
                0, // scalar1
                1, // scalar2
                4, 5, 6,  // v3
                7,  // tuck_in
                8,  // scalar4
                10, // larger[0..16]
                11, // larger[16..32]
            ],
        ));
    }

    // Matrix tests
    {
        let members = "m2: mat2x2h, m3: mat3x3h, m4: mat4x4h";
        let direct = String::from(
            "\
            output[0] = u32(input.m2[0].x);
            output[1] = u32(input.m2[0].y);
            output[2] = u32(input.m2[1].x);
            output[3] = u32(input.m2[1].y);

            output[4] = u32(input.m3[0].x);
            output[5] = u32(input.m3[0].y);
            output[6] = u32(input.m3[0].z);
            output[7] = u32(input.m3[1].x);
            output[8] = u32(input.m3[1].y);
            output[9] = u32(input.m3[1].z);
            output[10] = u32(input.m3[2].x);
            output[11] = u32(input.m3[2].y);
            output[12] = u32(input.m3[2].z);

            output[13] = u32(input.m4[0].x);
            output[14] = u32(input.m4[0].y);
            output[15] = u32(input.m4[0].z);
            output[16] = u32(input.m4[0].w);
            output[17] = u32(input.m4[1].x);
            output[18] = u32(input.m4[1].y);
            output[19] = u32(input.m4[1].z);
            output[20] = u32(input.m4[1].w);
            output[21] = u32(input.m4[2].x);
            output[22] = u32(input.m4[2].y);
            output[23] = u32(input.m4[2].z);
            output[24] = u32(input.m4[2].w);
            output[25] = u32(input.m4[3].x);
            output[26] = u32(input.m4[3].y);
            output[27] = u32(input.m4[3].z);
            output[28] = u32(input.m4[3].w);
        ",
        );

        tests.push(ShaderTest::new(
            "f16 matrix alignment".into(),
            members.into(),
            direct,
            &(0..32).map(|x| f16asu16(x as f32)).collect::<Vec<_>>(),
            &[
                0, 1, // m2[0]
                2, 3, // m2[1]
                //
                4, 5, 6, // m3[0]
                8, 9, 10, // m3[1]
                12, 13, 14, // m3[2]
                //
                16, 17, 18, 19, // m4[0]
                20, 21, 22, 23, // m4[1]
                24, 25, 26, 27, // m4[2]
                28, 29, 30, 31, // m4[3]
            ],
        ));
    }

    // Insert `enable f16;` header
    tests
        .into_iter()
        .map(|test| test.header("enable f16;".into()))
        .collect()
}
