//! Tests for the WGSL front end.
#![cfg(feature = "wgsl-in")]

fn check(input: &str, snapshot: &str) {
    let output = naga::front::wgsl::parse_str(input)
        .expect_err("expected parser error")
        .emit_to_string(input);
    if output != snapshot {
        for diff in diff::lines(&output, snapshot) {
            match diff {
                diff::Result::Left(l) => println!("-{}", l),
                diff::Result::Both(l, _) => println!(" {}", l),
                diff::Result::Right(r) => println!("+{}", r),
            }
        }
        panic!("Error snapshot failed");
    }
}

#[test]
fn function_without_identifier() {
    check(
        "fn () {}",
        r###"error: expected identifier, found '('
  ┌─ wgsl:1:4
  │
1 │ fn () {}
  │    ^ expected identifier

"###,
    );
}

#[test]
fn invalid_integer() {
    check(
        "fn foo([location(1.)] x: i32) {}",
        r###"error: expected identifier, found '['
  ┌─ wgsl:1:8
  │
1 │ fn foo([location(1.)] x: i32) {}
  │        ^ expected identifier

"###,
    );
}

#[test]
fn invalid_float() {
    check(
        "let scale: f32 = 1.1.;",
        r###"error: expected ';', found '.'
  ┌─ wgsl:1:21
  │
1 │ let scale: f32 = 1.1.;
  │                     ^ expected ';'

"###,
    );
}

#[test]
fn invalid_texture_sample_type() {
    check(
        "let x: texture_2d<f16>;",
        r###"error: texture sample type must be one of f32, i32 or u32, but found f16
  ┌─ wgsl:1:19
  │
1 │ let x: texture_2d<f16>;
  │                   ^^^ must be one of f32, i32 or u32

"###,
    );
}

#[test]
fn unknown_identifier() {
    check(
        r###"
              fn f(x: f32) -> f32 {
                  return x * schmoo;
              }
          "###,
        r###"error: no definition in scope for identifier: 'schmoo'
  ┌─ wgsl:3:30
  │
3 │                   return x * schmoo;
  │                              ^^^^^^ unknown identifier

"###,
    );
}

#[test]
fn negative_index() {
    check(
        r#"
            fn main() -> f32 {
                let a = array<f32, 3>(0., 1., 2.);
                return a[-1];
            }
        "#,
        r#"error: expected unsigned integer constant expression, found `-1`
  ┌─ wgsl:4:26
  │
4 │                 return a[-1];
  │                          ^^ expected unsigned integer

"#,
    );
}

#[test]
fn bad_texture() {
    check(
        r#"
            [[group(0), binding(0)]] var sampler : sampler;

            [[stage(fragment)]]
            fn main() -> [[location(0)]] vec4<f32> {
                let a = 3;
                return textureSample(a, sampler, vec2<f32>(0.0));
            }
        "#,
        r#"error: expected an image, but found 'a' which is not an image
  ┌─ wgsl:7:38
  │
7 │                 return textureSample(a, sampler, vec2<f32>(0.0));
  │                                      ^ not an image

"#,
    );
}

#[test]
fn bad_type_cast() {
    check(
        r#"
            fn x() -> i32 {
                return i32(vec2<f32>(0.0));
            }
        "#,
        r#"error: cannot cast a vec2<f32> to a i32
  ┌─ wgsl:3:27
  │
3 │                 return i32(vec2<f32>(0.0));
  │                           ^^^^^^^^^^^^^^^^ cannot cast a vec2<f32> to a i32

"#,
    );
}

#[test]
fn bad_texture_sample_type() {
    check(
        r#"
            [[group(0), binding(0)]] var sampler : sampler;
            [[group(0), binding(1)]] var texture : texture_2d<bool>;

            [[stage(fragment)]]
            fn main() -> [[location(0)]] vec4<f32> {
                return textureSample(texture, sampler, vec2<f32>(0.0));
            }
        "#,
        r#"error: texture sample type must be one of f32, i32 or u32, but found bool
  ┌─ wgsl:3:63
  │
3 │             [[group(0), binding(1)]] var texture : texture_2d<bool>;
  │                                                               ^^^^ must be one of f32, i32 or u32

"#,
    );
}

#[test]
fn bad_for_initializer() {
    check(
        r#"
            fn x() {
                for ({};;) {}
            }
        "#,
        r#"error: for(;;) initializer is not an assignment or a function call: '{}'
  ┌─ wgsl:3:22
  │
3 │                 for ({};;) {}
  │                      ^^ not an assignment or function call

"#,
    );
}

#[test]
fn unknown_storage_class() {
    check(
        r#"
            [[group(0), binding(0)]] var<bad> texture: texture_2d<f32>;
        "#,
        r#"error: unknown storage class: 'bad'
  ┌─ wgsl:2:42
  │
2 │             [[group(0), binding(0)]] var<bad> texture: texture_2d<f32>;
  │                                          ^^^ unknown storage class

"#,
    );
}

#[test]
fn unknown_attribute() {
    check(
        r#"
            [[a]]
            fn x() {}
        "#,
        r#"error: unknown attribute: 'a'
  ┌─ wgsl:2:15
  │
2 │             [[a]]
  │               ^ unknown attribute

"#,
    );
}

#[test]
fn unknown_built_in() {
    check(
        r#"
            fn x([[builtin(unknown_built_in)]] y: u32) {}
        "#,
        r#"error: unknown builtin: 'unknown_built_in'
  ┌─ wgsl:2:28
  │
2 │             fn x([[builtin(unknown_built_in)]] y: u32) {}
  │                            ^^^^^^^^^^^^^^^^ unknown builtin

"#,
    );
}

#[test]
fn unknown_access() {
    check(
        r#"
            var<storage,unknown_access> x: array<u32>;
        "#,
        r#"error: unknown access: 'unknown_access'
  ┌─ wgsl:2:25
  │
2 │             var<storage,unknown_access> x: array<u32>;
  │                         ^^^^^^^^^^^^^^ unknown access

"#,
    );
}

#[test]
fn unknown_shader_stage() {
    check(
        r#"
            [[stage(geometry)]] fn main() {}
        "#,
        r#"error: unknown shader stage: 'geometry'
  ┌─ wgsl:2:21
  │
2 │             [[stage(geometry)]] fn main() {}
  │                     ^^^^^^^^ unknown shader stage

"#,
    );
}

#[test]
fn unknown_ident() {
    check(
        r#"
            fn main() {
                let a = b;
            }
        "#,
        r#"error: no definition in scope for identifier: 'b'
  ┌─ wgsl:3:25
  │
3 │                 let a = b;
  │                         ^ unknown identifier

"#,
    );
}

#[test]
fn unknown_scalar_type() {
    check(
        r#"
            let a: vec2<something>;
        "#,
        r#"error: unknown scalar type: 'something'
  ┌─ wgsl:2:25
  │
2 │             let a: vec2<something>;
  │                         ^^^^^^^^^ unknown scalar type
  │
  = note: Valid scalar types are f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool

"#,
    );
}

#[test]
fn unknown_type() {
    check(
        r#"
            let a: Vec<f32>;
        "#,
        r#"error: unknown type: 'Vec'
  ┌─ wgsl:2:20
  │
2 │             let a: Vec<f32>;
  │                    ^^^ unknown type

"#,
    );
}

#[test]
fn unknown_storage_format() {
    check(
        r#"
            let storage: texture_storage_1d<rgba>;
        "#,
        r#"error: unknown storage format: 'rgba'
  ┌─ wgsl:2:45
  │
2 │             let storage: texture_storage_1d<rgba>;
  │                                             ^^^^ unknown storage format

"#,
    );
}

#[test]
fn unknown_conservative_depth() {
    check(
        r#"
            [[early_depth_test(abc)]] fn main() {}
        "#,
        r#"error: unknown conservative depth: 'abc'
  ┌─ wgsl:2:32
  │
2 │             [[early_depth_test(abc)]] fn main() {}
  │                                ^^^ unknown conservative depth

"#,
    );
}

#[test]
fn zero_array_stride() {
    check(
        r#"
            type zero = [[stride(0)]] array<f32>;
        "#,
        r#"error: array stride must not be zero
  ┌─ wgsl:2:34
  │
2 │             type zero = [[stride(0)]] array<f32>;
  │                                  ^ array stride must not be zero

"#,
    );
}

#[test]
fn struct_member_zero_size() {
    check(
        r#"
            struct Bar {
                [[size(0)]] data: array<f32>;
            };
        "#,
        r#"error: struct member size or alignment must not be 0
  ┌─ wgsl:3:24
  │
3 │                 [[size(0)]] data: array<f32>;
  │                        ^ struct member size or alignment must not be 0

"#,
    );
}

#[test]
fn struct_member_zero_align() {
    check(
        r#"
            struct Bar {
                [[align(0)]] data: array<f32>;
            };
        "#,
        r#"error: struct member size or alignment must not be 0
  ┌─ wgsl:3:25
  │
3 │                 [[align(0)]] data: array<f32>;
  │                         ^ struct member size or alignment must not be 0

"#,
    );
}

#[test]
fn inconsistent_binding() {
    check(
        r#"
        fn foo([[builtin(vertex_index), location(0)]] x: u32) {}
        "#,
        r#"error: input/output binding is not consistent
  ┌─ wgsl:2:16
  │
2 │         fn foo([[builtin(vertex_index), location(0)]] x: u32) {}
  │                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ input/output binding is not consistent

"#,
    );
}

#[test]
fn unknown_local_function() {
    check(
        r#"
            fn x() {
                for (a();;) {}
            }
        "#,
        r#"error: unknown local function `a`
  ┌─ wgsl:3:22
  │
3 │                 for (a();;) {}
  │                      ^ unknown local function

"#,
    );
}

#[test]
fn let_type_mismatch() {
    check(
        r#"
            let x: i32 = 1.0;
        "#,
        r#"error: the type of `x` is expected to be [1]
  ┌─ wgsl:2:17
  │
2 │             let x: i32 = 1.0;
  │                 ^ definition of `x`

"#,
    );
}

#[test]
fn local_var_type_mismatch() {
    check(
        r#"
            fn foo() {
                var x: f32 = 1;
            }
        "#,
        r#"error: the type of `x` is expected to be [1]
  ┌─ wgsl:3:21
  │
3 │                 var x: f32 = 1;
  │                     ^ definition of `x`

"#,
    );
}

#[test]
fn local_var_missing_type() {
    check(
        r#"
            fn foo() {
                var x;
            }
        "#,
        r#"error: variable `x` needs a type
  ┌─ wgsl:3:21
  │
3 │                 var x;
  │                     ^ definition of `x`

"#,
    );
}

#[test]
fn postfix_pointers() {
    check(
        r#"
            fn main() {
                var v: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                let pv = &v;
                let a = *pv[3]; // Problematic line
            }
        "#,
        r#"error: the value indexed by a `[]` subscripting expression must not be a pointer
  ┌─ wgsl:5:26
  │
5 │                 let a = *pv[3]; // Problematic line
  │                          ^^ expression is a pointer

"#,
    );

    check(
        r#"
            struct S { m: i32; };
            fn main() {
                var s: S = S(42);
                let ps = &s;
                let a = *ps.m; // Problematic line
            }
        "#,
        r#"error: the value accessed by a `.member` expression must not be a pointer
  ┌─ wgsl:6:26
  │
6 │                 let a = *ps.m; // Problematic line
  │                          ^^ expression is a pointer

"#,
    );
}

macro_rules! check_validation_error {
    // We want to support an optional guard expression after the pattern, so
    // that we can check values we can't match against, like strings.
    // Unfortunately, we can't simply include `$( if $guard:expr )?` in the
    // pattern, because Rust treats `?` as a repetition operator, and its count
    // (0 or 1) will not necessarily match `$source`.
    ( $( $source:literal ),* : $pattern:pat ) => {
        check_validation_error!( @full $( $source ),* : $pattern if true ; "");
    };

    ( $( $source:literal ),* : $pattern:pat if $guard:expr ) => {
        check_validation_error!( @full $( $source ),* : $pattern if $guard ; stringify!( $guard ) );
    };

    ( @full $( $source:literal ),* : $pattern:pat if $guard:expr ; $guard_string:expr ) => {
        $(
            let error = validation_error($source);
            if ! matches!(&error, $pattern if $guard) {
                eprintln!("validation error does not match pattern:\n\
                           source code: {}\n\
                           \n\
                           actual result:\n\
                           {:#?}\n\
                           \n\
                           expected match for pattern:\n\
                           {}{}",
                          stringify!($source),
                          error,
                          stringify!($pattern),
                          $guard_string);
                panic!("validation error does not match pattern");
            }
        )*
    };
}

fn validation_error(source: &str) -> Result<naga::valid::ModuleInfo, naga::valid::ValidationError> {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(err) => {
            eprintln!("WGSL parse failed:");
            panic!("{}", err.emit_to_string(source));
        }
    };
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
}

#[test]
fn invalid_arrays() {
    check_validation_error! {
        "type Bad = array<array<f32>, 4>;",
        "type Bad = array<sampler, 4>;",
        "type Bad = array<texture_2d<f32>, 4>;":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidArrayBaseType(_),
            ..
        })
    }

    check_validation_error! {
        r#"
            [[block]] struct Block { value: f32; };
            type Bad = array<Block, 4>;
        "#:
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::NestedTopLevel,
            ..
        })
    }

    check_validation_error! {
        r#"
            type Bad = [[stride(2)]] array<f32, 4>;
        "#:
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InsufficientArrayStride { stride: 2, base_size: 4 },
            ..
        })
    }

    check_validation_error! {
        "type Bad = array<f32, true>;",
        r#"
            let length: f32 = 2.718;
            type Bad = array<f32, length>;
        "#:
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidArraySizeConstant(_),
            ..
        })
    }

    check_validation_error! {
        "type Bad = array<f32, 0>;",
        "type Bad = array<f32, -1>;":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::NonPositiveArrayLength(_),
            ..
        })
    }
}

#[test]
fn invalid_structs() {
    check_validation_error! {
        "struct Bad { data: sampler; };",
        "struct Bad { data: texture_2d<f32>; };":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidData(_),
            ..
        })
    }

    check_validation_error! {
        "struct Bad { data: array<f32>; other: f32; };":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidDynamicArray(_, _),
            ..
        })
    }
}

#[test]
fn invalid_functions() {
    check_validation_error! {
        "fn unacceptable_unsized(arg: array<f32>) { }",
        "fn unacceptable_unsized(arg: ptr<storage, array<f32>>) { }",
        "
        struct Unsized { data: array<f32>; };
        fn unacceptable_unsized(arg: Unsized) { }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            error: naga::valid::FunctionError::InvalidArgumentType {
                index: 0,
                name: argument_name,
            },
            ..
        })
        if function_name == "unacceptable_unsized" && argument_name == "arg"
    }

    check_validation_error! {
        "
        struct Unsized { data: array<f32>; };
        fn acceptable_pointer_to_unsized(arg: ptr<workgroup, Unsized>) { }
        ":
        Ok(_)
    }

    check_validation_error! {
        "
        struct Unsized { data: array<f32>; };
        fn unacceptable_uniform_class(arg: ptr<uniform, f32>) { }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            error: naga::valid::FunctionError::InvalidArgumentPointerClass {
                index: 0,
                name: argument_name,
                class: naga::StorageClass::Uniform,
            },
            ..
        })
        if function_name == "unacceptable_uniform_class" && argument_name == "arg"
    }
}

#[test]
fn pointer_type_equivalence() {
    check_validation_error! {
        r#"
            fn f(pv: ptr<function, vec2<f32>>, pf: ptr<function, f32>) { }

            fn g() {
               var m: mat2x2<f32>;
               let pv: ptr<function, vec2<f32>> = &m.x;
               let pf: ptr<function, f32> = &m.x.x;

               f(pv, pf);
            }
        "#:
        Ok(_)
    }
}

#[test]
fn missing_bindings() {
    check_validation_error! {
        "
        [[stage(vertex)]]
        fn vertex(input: vec4<f32>) -> [[location(0)]] vec4<f32> {
           return input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            error: naga::valid::EntryPointError::Argument(
                0,
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation_error! {
        "
        [[stage(vertex)]]
        fn vertex([[location(0)]] input: vec4<f32>, more_input: f32) -> [[location(0)]] vec4<f32> {
           return input + more_input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            error: naga::valid::EntryPointError::Argument(
                1,
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation_error! {
        "
        [[stage(vertex)]]
        fn vertex([[location(0)]] input: vec4<f32>) -> vec4<f32> {
           return input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            error: naga::valid::EntryPointError::Result(
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation_error! {
        "
        struct VertexIn {
          [[location(0)]] pos: vec4<f32>;
          uv: vec2<f32>;
        };

        [[stage(vertex)]]
        fn vertex(input: VertexIn) -> [[location(0)]] vec4<f32> {
           return input.pos;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            error: naga::valid::EntryPointError::Argument(
                0,
                naga::valid::VaryingError::MemberMissingBinding(1),
            ),
            ..
        })
    }
}

#[test]
fn invalid_access() {
    check_validation_error! {
        "
        fn array_by_value(a: array<i32, 5>, i: i32) -> i32 {
            return a[i];
        }
        ",
        "
        fn matrix_by_value(m: mat4x4<f32>, i: i32) -> vec4<f32> {
            return m[i];
        }
        ":
        Err(naga::valid::ValidationError::Function {
            error: naga::valid::FunctionError::Expression {
                error: naga::valid::ExpressionError::IndexMustBeConstant(_),
                ..
            },
            ..
        })
    }

    check_validation_error! {
        r#"
            fn main() -> f32 {
                let a = array<f32, 3>(0., 1., 2.);
                return a[3];
            }
        "#:
        Err(naga::valid::ValidationError::Function {
            error: naga::valid::FunctionError::Expression {
                error: naga::valid::ExpressionError::IndexOutOfBounds(_, _),
                ..
            },
            ..
        })
    }
}

#[test]
fn valid_access() {
    check_validation_error! {
        "
        fn vector_by_value(v: vec4<i32>, i: i32) -> i32 {
            return v[i];
        }
        ",
        "
        fn matrix_dynamic(m: mat4x4<f32>, i: i32, j: i32) -> f32 {
            var temp: mat4x4<f32> = m;
            // Dynamically indexing the column vector applies
            // `Access` to a `ValuePointer`.
            return temp[i][j];
        }
        ",
        "
        fn main() {
            var v: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
            let pv = &v;
            let a = (*pv)[3];
        }
        ":
        Ok(_)
    }
}

#[test]
fn invalid_local_vars() {
    check_validation_error! {
        "
        struct Unsized { data: array<f32>; };
        fn local_ptr_dynamic_array(okay: ptr<storage, Unsized>) {
            var not_okay: ptr<storage, array<f32>> = &(*okay).data;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            error: naga::valid::FunctionError::LocalVariable {
                name: local_var_name,
                error: naga::valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        })
        if local_var_name == "not_okay"
    }
}

#[test]
fn dead_code() {
    check_validation_error! {
        "
        fn dead_code_after_if(condition: bool) -> i32 {
            if (condition) {
                return 1;
            } else {
                return 2;
            }
            return 3;
        }
        ":
        Ok(_)
    }
    check_validation_error! {
        "
        fn dead_code_after_block() -> i32 {
            {
                return 1;
            }
            return 2;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            error: naga::valid::FunctionError::InstructionsAfterReturn,
            ..
        })
    }
}
