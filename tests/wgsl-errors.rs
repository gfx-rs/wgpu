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
        r###"error: expected floating-point literal, found `1.1.`
  ┌─ wgsl:1:18
  │
1 │ let scale: f32 = 1.1.;
  │                  ^^^^ expected floating-point literal
  │
  = note: invalid float literal

"###,
    );
}

#[test]
fn invalid_scalar_width() {
    check(
        "let scale: f32 = 1.1f1000;",
        r###"error: invalid width of `1000` for literal
  ┌─ wgsl:1:18
  │
1 │ let scale: f32 = 1.1f1000;
  │                  ^^^^^^^^ invalid width
  │
  = note: valid widths are 8, 16, 32, 64

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
        r#"error: expected non-negative integer constant expression, found `-1`
  ┌─ wgsl:4:26
  │
4 │                 return a[-1];
  │                          ^^ expected non-negative integer

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
        "[[block]] struct Bad { data: ptr<storage, f32>; };":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidBlockType(_),
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

    // A *valid* way to pass an unsized value.
    check_validation_error! {
        "
        struct Unsized { data: array<f32>; };
        fn acceptable_ptr_to_unsized(okay: ptr<storage, Unsized>) { }
        ":
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
            var not_okay: ptr<storage, array<f32>> = okay.data;
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
