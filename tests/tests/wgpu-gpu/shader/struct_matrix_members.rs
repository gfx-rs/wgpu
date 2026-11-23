//! `matCx2` members of structs outside the uniform address space.

use wgpu::{Backends, DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([STRUCT_MATRIX_MEMBER_STORE, STRUCT_MATRIX_MEMBER_POINTER]);
}

const INPUT_MEMBERS: &str = "a: f32,";
const INPUT_VALUES: &[f32] = &[0.5];

fn output_matrix(first: usize, expr: &str, columns: usize, rows: usize) -> String {
    let mut out = String::new();
    for c in 0..columns {
        for r in 0..rows {
            let index = first + c * rows + r;
            out += &format!("output[{index}] = bitcast<u32>({expr}[{c}][{r}]);\n");
        }
    }
    out
}

fn store_tests() -> Vec<ShaderTest> {
    let r_outputs = format!(
        "{}{}",
        output_matrix(0, "r.M", 2, 2),
        output_matrix(4, "r.Minv", 2, 2)
    );
    let r_expected = [1.0f32, 2.0, 3.0, 0.5, 5.0, 6.0, 7.0, 8.0];

    let mut tests = Vec::new();

    // https://github.com/gfx-rs/wgpu/issues/10658
    tests.push(
        ShaderTest::new(
            String::from("issue 10658"),
            String::from(INPUT_MEMBERS),
            format!("let r = make(input.a);\n{r_outputs}"),
            INPUT_VALUES,
            &r_expected,
        )
        .header(String::from(
            "struct R { M: mat2x2<f32>, Minv: mat2x2<f32> }
            fn make(a: f32) -> R {
                var r: R;
                r.M = mat2x2<f32>(1.0, 2.0, 3.0, a);
                r.Minv = mat2x2<f32>(5.0, 6.0, 7.0, 8.0);
                return r;
            }",
        )),
    );

    tests.push(
        ShaderTest::new(
            String::from("stores inside if/else"),
            String::from(INPUT_MEMBERS),
            format!("let r = make(input.a);\n{r_outputs}"),
            INPUT_VALUES,
            &r_expected,
        )
        .header(String::from(
            "struct R { M: mat2x2<f32>, Minv: mat2x2<f32> }
            fn make(a: f32) -> R {
                var r: R;
                if a > 0.0 {
                    r.M = mat2x2<f32>(1.0, 2.0, 3.0, a);
                    r.Minv = mat2x2<f32>(5.0, 6.0, 7.0, 8.0);
                } else {
                    r.M = mat2x2<f32>();
                    r.Minv = mat2x2<f32>();
                }
                return r;
            }",
        )),
    );

    tests.push(
        ShaderTest::new(
            String::from("mat3x3 and mat2x2 members"),
            String::from(INPUT_MEMBERS),
            format!(
                "let r = make(input.a);\n{}{}",
                output_matrix(0, "r.M", 3, 3),
                output_matrix(9, "r.Minv", 2, 2)
            ),
            INPUT_VALUES,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 5.0, 6.0, 7.0, 8.0,
            ],
        )
        .header(String::from(
            "struct R { M: mat3x3<f32>, Minv: mat2x2<f32> }
            fn make(a: f32) -> R {
                var r: R;
                r.M = mat3x3<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, a);
                r.Minv = mat2x2<f32>(5.0, 6.0, 7.0, 8.0);
                return r;
            }",
        )),
    );

    let column_stores = "
        var i = 2u;
        var j = 1u;
        {var}.M[0] = vec2(1.0, 2.0);
        {var}.M[1][0] = 3.0;
        {var}.M[1][1] = input.a;
        {var}.M[i] = vec2(5.0, 6.0);
        {var}.M[i][j] = 7.0;
        {var}.f = 8.0;
    ";
    let column_expected = [1.0f32, 2.0, 3.0, 0.5, 5.0, 7.0, 8.0];
    for (name, header, var) in [
        ("function", "", "r"),
        ("private", "var<private> p: R;", "p"),
        ("workgroup", "var<workgroup> w: R;", "w"),
        ("array element", "", "arr[1]"),
    ] {
        let declaration = match name {
            "function" => "var r: R;",
            "array element" => "var arr: array<R, 2>;",
            _ => "",
        };
        tests.push(
            ShaderTest::new(
                format!("column and component stores, {name}"),
                String::from(INPUT_MEMBERS),
                format!(
                    "{declaration}\n{}\n{}output[6] = bitcast<u32>({var}.f);",
                    column_stores.replace("{var}", var),
                    output_matrix(0, &format!("{var}.M"), 3, 2),
                ),
                INPUT_VALUES,
                &column_expected,
            )
            .header(format!("struct R {{ M: mat3x2<f32>, f: f32 }}\n{header}")),
        );
    }

    tests
}

#[apply(gpu_test!)]
static STRUCT_MATRIX_MEMBER_STORE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| shader_input_output_test(ctx, InputStorageType::Storage, store_tests()));

fn pointer_tests() -> Vec<ShaderTest> {
    vec![
        // https://github.com/gfx-rs/wgpu/issues/7415
        ShaderTest::new(
            String::from("issue 7415"),
            String::from(INPUT_MEMBERS),
            format!(
                "var s: ContainsMat2x2;\nfoo(&s.m, input.a);\n{}",
                output_matrix(0, "s.m", 2, 2)
            ),
            INPUT_VALUES,
            &[1.0f32, 2.0, 3.0, 0.5],
        )
        .header(String::from(
            "struct ContainsMat2x2 { m: mat2x2<f32> }
            fn foo(p: ptr<function, mat2x2<f32>>, a: f32) {
                *p = mat2x2<f32>(1.0, 2.0, 3.0, a);
            }",
        )),
        ShaderTest::new(
            String::from("column store and load through pointer"),
            String::from(INPUT_MEMBERS),
            format!(
                "var s: S;
                s.m = mat3x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
                store_column(&s.m, input.a);
                output[6] = bitcast<u32>(read_matrix(&s.m));
                {}",
                output_matrix(0, "s.m", 3, 2)
            ),
            INPUT_VALUES,
            &[1.0f32, 2.0, 0.5, 7.0, 5.0, 6.0, 6.0],
        )
        .header(String::from(
            "struct S { a: f32, m: mat3x2<f32> }
            fn store_column(p: ptr<function, mat3x2<f32>>, a: f32) {
                (*p)[1] = vec2(a, 7.0);
            }
            fn read_matrix(p: ptr<function, mat3x2<f32>>) -> f32 {
                return (*p)[2].y;
            }",
        )),
        ShaderTest::new(
            String::from("pointer to private struct member"),
            String::from(INPUT_MEMBERS),
            format!("foo(&p.m, input.a);\n{}", output_matrix(0, "p.m", 2, 2)),
            INPUT_VALUES,
            &[1.0f32, 2.0, 3.0, 0.5],
        )
        .header(String::from(
            "struct S { a: f32, m: mat2x2<f32> }
            var<private> p: S;
            fn foo(m: ptr<private, mat2x2<f32>>, a: f32) {
                *m = mat2x2<f32>(1.0, 2.0, 3.0, a);
            }",
        )),
    ]
}

#[apply(gpu_test!)]
static STRUCT_MATRIX_MEMBER_POINTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults())
            // Naga's SPIR-V backend panics on pointers to struct members passed to
            // functions: https://github.com/gfx-rs/wgpu/issues/6875
            .skip(FailureCase::backend(Backends::VULKAN)),
    )
    .run_async(|ctx| shader_input_output_test(ctx, InputStorageType::Storage, pointer_tests()));
