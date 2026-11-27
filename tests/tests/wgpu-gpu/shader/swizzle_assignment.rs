//! Tests for the WGSL `swizzle_assignment` language extension.
//!
//! <https://www.w3.org/TR/WGSL/#language_extension-swizzle_assignment>

use wgpu::{DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([SWIZZLE_ASSIGNMENT, SWIZZLE_ASSIGNMENT_EVALUATION_ORDER]);
}

/// A swizzle assignment to a vector, and the vector it should leave behind.
struct Case {
    /// The type of the vector.
    ty: &'static str,
    /// The initial value of the vector.
    initial: &'static str,
    /// The statement to test. `{v}` stands for the memory view of the vector,
    /// which may be a pointer, and `{value}` for a reference to it.
    statement: &'static str,
    /// The components of the vector after the statement, as `u32` bits.
    expected: Vec<u32>,
}

fn cases() -> Vec<Case> {
    let u = |ty, statement, expected: &[u32]| Case {
        ty,
        initial: if ty == "vec4u" {
            "vec4u(1, 2, 3, 4)"
        } else {
            "vec3u(1, 2, 3)"
        },
        statement,
        expected: expected.to_vec(),
    };
    vec![
        u("vec4u", "{v}.zx = vec2u(8, 9);", &[9, 2, 8, 4]),
        u(
            "vec4u",
            "{v}.xyzw = vec4u(10, 11, 12, 13);",
            &[10, 11, 12, 13],
        ),
        u(
            "vec4u",
            "{v}.wzyx = vec4u(10, 11, 12, 13);",
            &[13, 12, 11, 10],
        ),
        u("vec4u", "{v}.ab = vec2(5, 6);", &[1, 2, 6, 5]),
        // The right-hand side is evaluated before the vector is written.
        u("vec4u", "{v}.xy = vec2({v}.y, {v}.x);", &[2, 1, 3, 4]),
        // Chained swizzles.
        u("vec4u", "{v}.yz.yx = vec2u(99, 100);", &[1, 100, 99, 4]),
        u("vec4u", "{v}.xxyy.zxy.xy = vec2u(5, 6);", &[6, 5, 3, 4]),
        u("vec4u", "{v}.yz.y = 99u;", &[1, 2, 99, 4]),
        u("vec4u", "{v}.zy.x++;", &[1, 2, 4, 4]),
        // Indexing a swizzle view chooses one of its components.
        u("vec4u", "{v}.zy[1] = 50u;", &[1, 50, 3, 4]),
        u("vec4u", "{v}.wx[input.one] = 7u;", &[7, 2, 3, 4]),
        u("vec4u", "{v}.xy[input.one] = 7u;", &[1, 7, 3, 4]),
        u("vec4u", "{v}.wzy[i32(input.one) + 1] = 7u;", &[1, 7, 3, 4]),
        u("vec4u", "{v}.wzyx[input.one] += 7u;", &[1, 2, 10, 4]),
        // Compound assignments.
        u("vec4u", "{v}.xz += vec2u(10, 20);", &[11, 2, 23, 4]),
        u("vec4u", "{v}.xz *= 3u;", &[3, 2, 9, 4]),
        u("vec4u", "{v}.yw <<= vec2u(1, 2);", &[1, 4, 3, 16]),
        u("vec4u", "{v}.wzyx -= vec4u(1);", &[0, 1, 2, 3]),
        u("vec3u", "{v}.zx = vec2u(8, 9);", &[9, 2, 8]),
        Case {
            ty: "vec3i",
            initial: "vec3i(10, 20, 30)",
            statement: "{v}.zx = vec2i(40, -60);",
            expected: [-60i32, 20, 40].map(|x| x as u32).to_vec(),
        },
        Case {
            ty: "vec2f",
            initial: "vec2f(1.0, 2.0)",
            statement: "{v}.yx = {value} + {value};",
            expected: [4.0f32, 2.0].map(f32::to_bits).to_vec(),
        },
    ]
}

/// Where the vector lives, and how the statement refers to it.
#[derive(Clone, Copy, Debug)]
enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Storage,
}

fn create_swizzle_assignment_tests() -> Vec<ShaderTest> {
    let mut tests = Vec::new();
    for case in cases() {
        for space in [
            AddressSpace::Function,
            AddressSpace::Private,
            AddressSpace::Workgroup,
            AddressSpace::Storage,
        ] {
            for use_pointer in [false, true] {
                let Case {
                    ty,
                    initial,
                    statement,
                    ref expected,
                } = case;

                let (header, declaration, var) = match space {
                    AddressSpace::Function => (String::new(), format!("var v: {ty};"), "v"),
                    AddressSpace::Private => (format!("var<private> v: {ty};"), String::new(), "v"),
                    AddressSpace::Workgroup => {
                        (format!("var<workgroup> v: {ty};"), String::new(), "v")
                    }
                    AddressSpace::Storage => (String::new(), String::new(), "output[0]"),
                };
                let (pointer, view) = if use_pointer {
                    (format!("let p = &{var};"), "p")
                } else {
                    (String::new(), var)
                };
                let value = if use_pointer { "(*p)" } else { var };
                let statement = statement.replace("{v}", view).replace("{value}", value);
                let store_output = match space {
                    AddressSpace::Storage => String::new(),
                    _ => (0..expected.len())
                        .map(|i| format!("output[{i}] = bitcast<u32>(v[{i}]);\n"))
                        .collect(),
                };

                let body = format!(
                    "
                    {declaration}
                    {var} = {initial};
                    {pointer}
                    {statement}
                    {store_output}
                    "
                );
                let mut test = ShaderTest::new(
                    format!("{space:?}, pointer: {use_pointer}: {statement}"),
                    String::from("one: u32"),
                    body,
                    &[1u32],
                    expected,
                )
                .header(header);
                if let AddressSpace::Storage = space {
                    test = test.output_type(format!("array<{ty}>"));
                }
                tests.push(test);
            }
        }
    }
    tests
}

#[apply(gpu_test!)]
static SWIZZLE_ASSIGNMENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_swizzle_assignment_tests(),
        )
    });

/// The examples of evaluation order from the specification.
///
/// <https://www.w3.org/TR/WGSL/#example-swizzle-assign-order-of-evaluation>
/// <https://www.w3.org/TR/WGSL/#example-compound-swizzle-assign-order-of-evaluation>
fn create_evaluation_order_tests() -> Vec<ShaderTest> {
    let header = String::from(
        "
        var<private> v: array<vec4u, 1>;

        fn update_and_index() -> u32 {
            v[0] += vec4u(2, 4, 8, 10);
            return 0;
        }

        fn update_and_rhs() -> vec2u {
            v[0].x = 900;
            v[0].z = 100;
            return vec2u(55, 66);
        }
        ",
    );
    [
        ("=", [66u32, 55, 100, 10]),
        // The selected components are read before the right-hand side is
        // evaluated, and the others after.
        ("+=", [68, 59, 100, 10]),
    ]
    .into_iter()
    .map(|(op, expected)| {
        ShaderTest::new(
            format!("v[update_and_index()].yx {op} update_and_rhs()"),
            String::from("one: u32"),
            format!(
                "
                v[0] = vec4u(0, 0, 0, 0);
                v[update_and_index()].yx {op} update_and_rhs();
                output[0] = v[0].x;
                output[1] = v[0].y;
                output[2] = v[0].z;
                output[3] = v[0].w;
                "
            ),
            &[1u32],
            &expected,
        )
        .header(header.clone())
    })
    .collect()
}

#[apply(gpu_test!)]
static SWIZZLE_ASSIGNMENT_EVALUATION_ORDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_evaluation_order_tests(),
        )
    });
