/*!
Tests for the WGSL front end.
*/
#![cfg(feature = "wgsl-in")]
#![allow(
    // We need to investiagate these.
    clippy::result_large_err
)]

use naga::{
    compact::KeepUnused,
    valid::{self, Capabilities},
};

#[track_caller]
fn check(input: &str, snapshot: &str) {
    let output = match naga::front::wgsl::parse_str(input) {
        Ok(_) => panic!("expected parser error, but parsing succeeded!"),
        Err(err) => err.emit_to_string(input),
    };
    if output != snapshot {
        for diff in diff::lines(snapshot, &output) {
            match diff {
                diff::Result::Left(l) => println!("-{l}"),
                diff::Result::Both(l, _) => println!(" {l}"),
                diff::Result::Right(r) => println!("+{r}"),
            }
        }
        panic!("Error snapshot failed");
    }
}

#[track_caller]
fn check_success(input: &str) {
    match naga::front::wgsl::parse_str(input) {
        Ok(_) => {}
        Err(err) => {
            panic!(
                "expected success, but parsing failed with:\n{}",
                err.emit_to_string(input)
            );
        }
    }
}

#[test]
fn very_negative_integers() {
    // wgpu#4492
    check(
        "const i32min = -0x80000000i;",
        r###"error: numeric literal not representable by target type: `0x80000000i`
  ┌─ wgsl:1:17
  │
1 │ const i32min = -0x80000000i;
  │                 ^^^^^^^^^^^ numeric literal not representable by target type

"###,
    );
}

#[test]
fn reserved_identifier_prefix() {
    check(
        "var __bad;",
        r###"error: Identifier starts with a reserved prefix: `__bad`
  ┌─ wgsl:1:5
  │
1 │ var __bad;
  │     ^^^^^ invalid identifier

"###,
    );
}

#[test]
fn function_without_identifier() {
    check(
        "fn () {}",
        r###"error: expected identifier, found "("
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
        r###"error: expected identifier, found "["
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
        "const scale: f32 = 1.1.;",
        r###"error: expected identifier, found ";"
  ┌─ wgsl:1:24
  │
1 │ const scale: f32 = 1.1.;
  │                        ^ expected identifier

"###,
    );
}

#[test]
fn invalid_texture_sample_type() {
    check(
        "const x: texture_2d<bool>;",
        r###"error: texture sample type must be one of f32, i32 or u32, but found bool
  ┌─ wgsl:1:21
  │
1 │ const x: texture_2d<bool>;
  │                     ^^^^ must be one of f32, i32 or u32

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
        r###"error: no definition in scope for identifier: `schmoo`
  ┌─ wgsl:3:30
  │
3 │                   return x * schmoo;
  │                              ^^^^^^ unknown identifier

"###,
    );
}

#[test]
fn bad_texture() {
    check(
        r#"
            @group(0) @binding(0) var sampler1 : sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                let a = 3;
                return textureSample(a, sampler1, vec2<f32>(0.0));
            }
        "#,
        r#"error: expected an image, but found `a` which is not an image
  ┌─ wgsl:7:38
  │
7 │                 return textureSample(a, sampler1, vec2<f32>(0.0));
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
  ┌─ wgsl:3:28
  │
3 │                 return i32(vec2<f32>(0.0));
  │                            ^^^^^^^^^^^^^^ cannot cast a vec2<f32> to a i32

"#,
    );
}

#[test]
fn cross_vec2() {
    check(
        r#"
            fn x() -> f32 {
                return cross(vec2(0., 1.), vec2(0., 1.));
            }
        "#,
        "\
error: wrong type passed as argument #1 to `cross`
  ┌─ wgsl:3:24
  │
3 │                 return cross(vec2(0., 1.), vec2(0., 1.));
  │                        ^^^^^ ^^^^^^^^^^^^ argument #1 has type `vec2<{AbstractFloat}>`
  │
  = note: `cross` accepts the following types for argument #1:
  = note: allowed type: vec3<{AbstractFloat}>
  = note: allowed type: vec3<f32>
  = note: allowed type: vec3<f16>
  = note: allowed type: vec3<f64>

",
    );
}

#[test]
fn cross_vec4() {
    check(
        r#"
            fn x() -> f32 {
                return cross(vec4(0., 1., 2., 3.), vec4(0., 1., 2., 3.));
            }
        "#,
        "\
error: wrong type passed as argument #1 to `cross`
  ┌─ wgsl:3:24
  │
3 │                 return cross(vec4(0., 1., 2., 3.), vec4(0., 1., 2., 3.));
  │                        ^^^^^ ^^^^^^^^^^^^^^^^^^^^ argument #1 has type `vec4<{AbstractFloat}>`
  │
  = note: `cross` accepts the following types for argument #1:
  = note: allowed type: vec3<{AbstractFloat}>
  = note: allowed type: vec3<f32>
  = note: allowed type: vec3<f16>
  = note: allowed type: vec3<f64>

",
    );
}

#[test]
fn type_not_constructible() {
    check(
        r#"
            fn x() {
                _ = atomic<i32>(0);
            }
        "#,
        r#"error: type `atomic` is not constructible
  ┌─ wgsl:3:21
  │
3 │                 _ = atomic<i32>(0);
  │                     ^^^^^^ type is not constructible

"#,
    );
}

#[test]
fn type_not_inferable() {
    check(
        r#"
            fn x() {
                _ = mat2x2();
            }
        "#,
        r#"error: type can't be inferred
  ┌─ wgsl:3:21
  │
3 │                 _ = mat2x2();
  │                     ^^^^^^ type can't be inferred

"#,
    );
}

#[test]
fn unexpected_constructor_parameters() {
    check(
        r#"
            fn x() {
                _ = i32(0, 1);
            }
        "#,
        r#"error: unexpected components
  ┌─ wgsl:3:28
  │
3 │                 _ = i32(0, 1);
  │                            ^ unexpected components

"#,
    );
}

#[test]
fn constructor_parameter_type_mismatch() {
    check(
        r#"
            fn x() {
                _ = mat2x2<f32>(array(0, 1), vec2(2, 3));
            }
        "#,
        "error: automatic conversions cannot convert `array<{AbstractInt}, 2>` to `vec2<f32>`
  ┌─ wgsl:3:21
  │
3 │                 _ = mat2x2<f32>(array(0, 1), vec2(2, 3));
  │                     ^^^^^^^^^^^ ^^^^^^^^^^^ this expression has type array<{AbstractInt}, 2>
  │                     │\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20
  │                     a value of type vec2<f32> is required here

",
    );
}

#[test]
fn vector_constructor_incorrect_component_count() {
    // Too few components
    check(
        r#"
            fn x() {
                _ = vec4(1, 2, 3);
            }
        "#,
        r#"error: Constructor expects 4 components, found 3
  ┌─ wgsl:3:21
  │
3 │                 _ = vec4(1, 2, 3);
  │                     ^^^^^^^^^^^^^ see msg

"#,
    );

    // Too many components
    check(
        r#"
            fn x() {
                _ = vec4(1, 2, 3, 4, 5);
            }
        "#,
        r#"error: Constructor expects 4 components, found 5
  ┌─ wgsl:3:21
  │
3 │                 _ = vec4(1, 2, 3, 4, 5);
  │                     ^^^^^^^^^^^^^^^^^^^ see msg

"#,
    );

    // The outer constructor has the correct number of components, but only
    // because the inner constructor has too many.
    check(
        r#"
            fn x() {
                _ = vec4(1, vec2(2, 3, 4));
            }
        "#,
        r#"error: Constructor expects 2 components, found 3
  ┌─ wgsl:3:29
  │
3 │                 _ = vec4(1, vec2(2, 3, 4));
  │                             ^^^^^^^^^^^^^ see msg

"#,
    );
}

#[test]
fn bad_texture_sample_type() {
    check(
        r#"
            @group(0) @binding(0) var sampler1 : sampler;
            @group(0) @binding(1) var texture : texture_2d<bool>;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                return textureSample(texture, sampler1, vec2<f32>(0.0));
            }
        "#,
        r#"error: texture sample type must be one of f32, i32 or u32, but found bool
  ┌─ wgsl:3:60
  │
3 │             @group(0) @binding(1) var texture : texture_2d<bool>;
  │                                                            ^^^^ must be one of f32, i32 or u32

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
        r#"error: for(;;) initializer is not an assignment or a function call: `{}`
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
            @group(0) @binding(0) var<bad> texture: texture_2d<f32>;
        "#,
        r#"error: unknown address space: `bad`
  ┌─ wgsl:2:39
  │
2 │             @group(0) @binding(0) var<bad> texture: texture_2d<f32>;
  │                                       ^^^ unknown address space

"#,
    );
}

#[test]
fn unknown_attribute() {
    check(
        r#"
            @a
            fn x() {}
        "#,
        r#"error: unknown attribute: `a`
  ┌─ wgsl:2:14
  │
2 │             @a
  │              ^ unknown attribute

"#,
    );
}

#[test]
fn unknown_built_in() {
    check(
        r#"
            fn x(@builtin(unknown_built_in) y: u32) {}
        "#,
        r#"error: unknown builtin: `unknown_built_in`
  ┌─ wgsl:2:27
  │
2 │             fn x(@builtin(unknown_built_in) y: u32) {}
  │                           ^^^^^^^^^^^^^^^^ unknown builtin

"#,
    );
}

#[test]
fn unknown_access() {
    check(
        r#"
            var<storage,unknown_access> x: array<u32>;
        "#,
        r#"error: unknown access: `unknown_access`
  ┌─ wgsl:2:25
  │
2 │             var<storage,unknown_access> x: array<u32>;
  │                         ^^^^^^^^^^^^^^ unknown access

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
        r#"error: no definition in scope for identifier: `b`
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
            const a = vec2<vec2f>();
        "#,
        r#"error: unknown scalar type: `vec2f`
  ┌─ wgsl:2:28
  │
2 │             const a = vec2<vec2f>();
  │                            ^^^^^ unknown scalar type
  │
  = note: Valid scalar types are f32, f64, i32, u32, bool

"#,
    );
}

#[test]
fn unknown_type() {
    check(
        r#"
            const a: Vec = 10;
        "#,
        r#"error: unknown type: `Vec`
  ┌─ wgsl:2:22
  │
2 │             const a: Vec = 10;
  │                      ^^^ unknown type

"#,
    );
}

#[test]
fn unknown_storage_format() {
    check(
        r#"
            const storage1: texture_storage_1d<rgba>;
        "#,
        r#"error: unknown storage format: `rgba`
  ┌─ wgsl:2:48
  │
2 │             const storage1: texture_storage_1d<rgba>;
  │                                                ^^^^ unknown storage format

"#,
    );
}

#[test]
fn unknown_conservative_depth() {
    check(
        r#"
            @early_depth_test(abc) fn main() {}
        "#,
        r#"error: unknown conservative depth: `abc`
  ┌─ wgsl:2:31
  │
2 │             @early_depth_test(abc) fn main() {}
  │                               ^^^ unknown conservative depth

"#,
    );
}

#[test]
fn struct_member_size_too_low() {
    check(
        r#"
            struct Bar {
                @size(0) data: array<f32>
            }
        "#,
        r#"error: struct member size must be at least 4
  ┌─ wgsl:3:23
  │
3 │                 @size(0) data: array<f32>
  │                       ^ must be at least 4

"#,
    );
}

#[test]
fn struct_member_align_too_low() {
    check(
        r#"
            struct Bar {
                @align(8) data: vec3<f32>
            }
        "#,
        r#"error: struct member alignment must be at least 16
  ┌─ wgsl:3:24
  │
3 │                 @align(8) data: vec3<f32>
  │                        ^ must be at least 16

"#,
    );
}

#[test]
fn struct_member_non_po2_align() {
    check(
        r#"
            struct Bar {
                @align(7) data: array<f32>
            }
        "#,
        r#"error: struct member alignment must be a power of 2
  ┌─ wgsl:3:24
  │
3 │                 @align(7) data: array<f32>
  │                        ^ must be a power of 2

"#,
    );
}

#[test]
fn inconsistent_binding() {
    check(
        r#"
        fn foo(@builtin(vertex_index) @location(0) x: u32) {}
        "#,
        r#"error: input/output binding is not consistent
  ┌─ wgsl:2:16
  │
2 │         fn foo(@builtin(vertex_index) @location(0) x: u32) {}
  │                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ input/output binding is not consistent

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
        r#"error: no definition in scope for identifier: `a`
  ┌─ wgsl:3:22
  │
3 │                 for (a();;) {}
  │                      ^ unknown identifier

"#,
    );
}

#[test]
fn let_type_mismatch() {
    check(
        r#"
            const x: i32 = 1.0;
        "#,
        r#"error: the type of `x` is expected to be `i32`, but got `{AbstractFloat}`
  ┌─ wgsl:2:19
  │
2 │             const x: i32 = 1.0;
  │                   ^ definition of `x`

"#,
    );

    check(
        r#"
            fn foo() {
                let x: f32 = true;
            }
        "#,
        r#"error: the type of `x` is expected to be `f32`, but got `bool`
  ┌─ wgsl:3:21
  │
3 │                 let x: f32 = true;
  │                     ^ definition of `x`

"#,
    );
}

#[test]
fn var_type_mismatch() {
    check(
        r#"
            fn foo() {
                var x: f32 = 1u;
            }
        "#,
        r#"error: the type of `x` is expected to be `f32`, but got `u32`
  ┌─ wgsl:3:21
  │
3 │                 var x: f32 = 1u;
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
        r#"error: declaration of `x` needs a type specifier or initializer
  ┌─ wgsl:3:21
  │
3 │                 var x;
  │                     ^ needs a type specifier or initializer

"#,
    );
}

#[test]
fn reserved_keyword() {
    // global var
    check(
        r#"
            var bool: bool = true;
        "#,
        r###"error: name `bool` is a reserved keyword
  ┌─ wgsl:2:17
  │
2 │             var bool: bool = true;
  │                 ^^^^ definition of `bool`

"###,
    );

    // global constant
    check(
        r#"
            const break: bool = true;
            fn foo() {
                var foo = break;
            }
        "#,
        r###"error: name `break` is a reserved keyword
  ┌─ wgsl:2:19
  │
2 │             const break: bool = true;
  │                   ^^^^^ definition of `break`

"###,
    );

    // local let
    check(
        r#"
            fn foo() {
                let atomic: f32 = 1.0;
            }
        "#,
        r###"error: name `atomic` is a reserved keyword
  ┌─ wgsl:3:21
  │
3 │                 let atomic: f32 = 1.0;
  │                     ^^^^^^ definition of `atomic`

"###,
    );

    // local var
    check(
        r#"
            fn foo() {
                var sampler: f32 = 1.0;
            }
        "#,
        r###"error: name `sampler` is a reserved keyword
  ┌─ wgsl:3:21
  │
3 │                 var sampler: f32 = 1.0;
  │                     ^^^^^^^ definition of `sampler`

"###,
    );

    // fn name
    check(
        r#"
            fn break() {}
        "#,
        r###"error: name `break` is a reserved keyword
  ┌─ wgsl:2:16
  │
2 │             fn break() {}
  │                ^^^^^ definition of `break`

"###,
    );

    // struct
    check(
        r#"
            struct array {}
        "#,
        r###"error: name `array` is a reserved keyword
  ┌─ wgsl:2:20
  │
2 │             struct array {}
  │                    ^^^^^ definition of `array`

"###,
    );

    // struct member
    check(
        r#"
            struct Foo { sampler: f32 }
        "#,
        r###"error: name `sampler` is a reserved keyword
  ┌─ wgsl:2:26
  │
2 │             struct Foo { sampler: f32 }
  │                          ^^^^^^^ definition of `sampler`

"###,
    );
}

#[test]
fn module_scope_identifier_redefinition() {
    // const
    check(
        r#"
            const foo: bool = true;
            const foo: bool = true;
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:19
  │
2 │             const foo: bool = true;
  │                   ^^^ previous definition of `foo`
3 │             const foo: bool = true;
  │                   ^^^ redefinition of `foo`

"###,
    );
    // var
    check(
        r#"
            var foo: bool = true;
            var foo: bool = true;
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:17
  │
2 │             var foo: bool = true;
  │                 ^^^ previous definition of `foo`
3 │             var foo: bool = true;
  │                 ^^^ redefinition of `foo`

"###,
    );

    // let and var
    check(
        r#"
            var foo: bool = true;
            const foo: bool = true;
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:17
  │
2 │             var foo: bool = true;
  │                 ^^^ previous definition of `foo`
3 │             const foo: bool = true;
  │                   ^^^ redefinition of `foo`

"###,
    );

    // function
    check(
        r#"fn foo() {}
                fn bar() {}
                fn foo() {}"#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:1:4
  │
1 │ fn foo() {}
  │    ^^^ previous definition of `foo`
2 │                 fn bar() {}
3 │                 fn foo() {}
  │                    ^^^ redefinition of `foo`

"###,
    );

    // let and function
    check(
        r#"
            const foo: bool = true;
            fn foo() {}
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:19
  │
2 │             const foo: bool = true;
  │                   ^^^ previous definition of `foo`
3 │             fn foo() {}
  │                ^^^ redefinition of `foo`

"###,
    );
}

#[test]
fn matrix_with_bad_type() {
    check(
        r#"
            fn main() {
                let m = mat2x2<i32>();
            }
        "#,
        r#"error: matrix scalar type must be floating-point, but found `i32`
  ┌─ wgsl:3:32
  │
3 │                 let m = mat2x2<i32>();
  │                                ^^^ must be floating-point (e.g. `f32`)

"#,
    );

    check(
        r#"
            fn main() {
                var m: mat3x3<i32>;
            }
        "#,
        r#"error: matrix scalar type must be floating-point, but found `i32`
  ┌─ wgsl:3:31
  │
3 │                 var m: mat3x3<i32>;
  │                               ^^^ must be floating-point (e.g. `f32`)

"#,
    );
}

#[test]
fn matrix_constructor_inferred() {
    check(
        r#"
            const m: mat2x2<f64> = mat2x2<f32>(vec2(0), vec2(1));
        "#,
        r#"error: the type of `m` is expected to be `mat2x2<f64>`, but got `mat2x2<f32>`
  ┌─ wgsl:2:19
  │
2 │             const m: mat2x2<f64> = mat2x2<f32>(vec2(0), vec2(1));
  │                   ^ definition of `m`

"#,
    );
}

#[test]
fn float16_requires_enable() {
    check(
        r#"
            const a: f16 = 1.0;
        "#,
        r#"error: the `f16` enable extension is not enabled
  ┌─ wgsl:2:22
  │
2 │             const a: f16 = 1.0;
  │                      ^^^ the `f16` "Enable Extension" is needed for this functionality, but it is not currently enabled.
  │
  = note: You can enable this extension by adding `enable f16;` at the top of the shader, before any other items.

"#,
    );

    check(
        r#"
            const a = 1.0h;
        "#,
        r#"error: the `f16` enable extension is not enabled
  ┌─ wgsl:2:23
  │
2 │             const a = 1.0h;
  │                       ^^^^ the `f16` "Enable Extension" is needed for this functionality, but it is not currently enabled.
  │
  = note: You can enable this extension by adding `enable f16;` at the top of the shader, before any other items.

"#,
    );
}

#[test]
fn multiple_enables_valid() {
    check_success(
        r#"
            enable f16;
            enable f16;
            const a: f16 = 1.0h;
        "#,
    );
}

/// Check the result of validating a WGSL program against a pattern.
///
/// Unless you are generating code programmatically, the
/// `check_validation` macro will probably be more convenient to
/// use.
macro_rules! check_one_validation {
    ( $source:expr, $pattern:pat $( if $guard:expr )? ) => {
        let source = $source;
        let error = validation_error($source, naga::valid::Capabilities::default());
        #[allow(clippy::redundant_pattern_matching)]
        if ! matches!(&error, $pattern $( if $guard )? ) {
            eprintln!("validation error does not match pattern:\n\
                       source code: {}\n\
                       \n\
                       actual result:\n\
                       {:#?}\n\
                       \n\
                       expected match for pattern:\n\
                       {}",
                      &source,
                      error,
                      stringify!($pattern));
            $( eprintln!("if {}", stringify!($guard)); )?
            panic!("validation error does not match pattern");
        }
    };
    ( $source:expr, $pattern:pat $( if $guard:expr )?, $capabilities:expr ) => {
        let source = $source;
        let error = validation_error($source, $capabilities);
        #[allow(clippy::redundant_pattern_matching)]
        if ! matches!(&error, $pattern $( if $guard )? ) {
            eprintln!("validation error does not match pattern:\n\
                       source code: {}\n\
                       \n\
                       actual result:\n\
                       {:#?}\n\
                       \n\
                       expected match for pattern:\n\
                       {}",
                      &source,
                      error,
                      stringify!($pattern));
            $( eprintln!("if {}", stringify!($guard)); )?
            panic!("validation error does not match pattern");
        }
    }
}

macro_rules! check_validation {
    // We want to support an optional guard expression after the pattern, so
    // that we can check values we can't match against, like strings.
    // Unfortunately, we can't simply include `$( if $guard:expr )?` in the
    // pattern, because Rust treats `?` as a repetition operator, and its count
    // (0 or 1) will not necessarily match `$source`.
    ( $( $source:literal ),* : $pattern:pat ) => {
        $(
            check_one_validation!($source, $pattern);
        )*
    };
    ( $( $source:literal ),* : $pattern:pat, $capabilities:expr ) => {
        $(
            check_one_validation!($source, $pattern, $capabilities);
        )*
    };
    ( $( $source:literal ),* : $pattern:pat if $guard:expr ) => {
        $(
            check_one_validation!($source, $pattern if $guard);
        )*
    };
    ( $( $source:literal ),* : $pattern:pat if $guard:expr, $capabilities:expr ) => {
        $(
            check_one_validation!($source, $pattern if $guard, $capabilities);
        )*
    }
}

#[track_caller]
fn validation_error(
    source: &str,
    caps: naga::valid::Capabilities,
) -> Result<naga::valid::ModuleInfo, naga::valid::ValidationError> {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(err) => {
            eprintln!("WGSL parse failed:");
            panic!("{}", err.emit_to_string(source));
        }
    };
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
        .validate(&module)
        .map_err(|e| e.into_inner()) // TODO: Add tests for spans, too?
}

#[test]
fn int64_capability() {
    check_validation! {
        "var input: u64;",
        "var input: i64;":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::WidthError(naga::valid::WidthError::MissingCapability {flag: "SHADER_INT64",..}),
            ..
        })
    }
}

#[test]
fn float16_capability() {
    check_validation! {
        "enable f16; var input: f16;",
        "enable f16; var input: vec2<f16>;":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::WidthError(naga::valid::WidthError::MissingCapability {flag: "FLOAT16",..}),
            ..
        })
    }
}

#[test]
fn float16_in_push_constant() {
    check_validation! {
        "enable f16; var<push_constant> input: f16;",
        "enable f16; var<push_constant> input: vec2<f16>;",
        "enable f16; var<push_constant> input: mat4x4<f16>;",
        "enable f16; struct S { a: f16 }; var<push_constant> input: S;",
        "enable f16; struct S1 { a: f16 }; struct S2 { a : S1 } var<push_constant> input: S2;":
        Err(naga::valid::ValidationError::GlobalVariable {
            source: naga::valid::GlobalVariableError::InvalidPushConstantType(
                naga::valid::PushConstantError::InvalidScalar(
                    naga::Scalar::F16
                )
            ),
            ..
        }),
        naga::valid::Capabilities::SHADER_FLOAT16 | naga::valid::Capabilities::PUSH_CONSTANT
    }
}

#[test]
fn float16_in_atomic() {
    check_validation! {
        "enable f16; var<storage> a: atomic<f16>;":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::InvalidAtomicWidth(
                naga::ScalarKind::Float,
                2
            ),
            ..
        }),
        naga::valid::Capabilities::SHADER_FLOAT16
    }
}

#[test]
fn invalid_arrays() {
    check_validation! {
        "alias Bad = array<array<f32>, 4>;",
        "alias Bad = array<sampler, 4>;",
        "alias Bad = array<texture_2d<f32>, 4>;":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::InvalidArrayBaseType(_),
            ..
        })
    }

    check_validation! {
        "var<uniform> input: array<u64, 2>;",
        "var<uniform> input: array<vec2<u32>, 2>;":
        Err(naga::valid::ValidationError::GlobalVariable {
            source: naga::valid::GlobalVariableError::Alignment(naga::AddressSpace::Uniform,_,_),
            ..
        }),
        naga::valid::Capabilities::SHADER_INT64
    }

    check_validation! {
        r#"
            fn main() -> f32 {
                let a = array<f32, 3>(0., 1., 2.);
                return a[-1];
            }
        "#:
        Err(
            naga::valid::ValidationError::Function {
                name,
                source: naga::valid::FunctionError::Expression {
                    source: naga::valid::ExpressionError::NegativeIndex(_),
                    ..
                },
                ..
            }
        )
            if name == "main"
    }

    check(
        "alias Bad = array<f32, true>;",
        r###"error: must be a const-expression that resolves to a concrete integer scalar (`u32` or `i32`)
  ┌─ wgsl:1:24
  │
1 │ alias Bad = array<f32, true>;
  │                        ^^^^ must resolve to `u32` or `i32`

"###,
    );

    check(
        r#"
            const length: f32 = 2.718;
            alias Bad = array<f32, length>;
        "#,
        r###"error: must be a const-expression that resolves to a concrete integer scalar (`u32` or `i32`)
  ┌─ wgsl:3:36
  │
3 │             alias Bad = array<f32, length>;
  │                                    ^^^^^^ must resolve to `u32` or `i32`

"###,
    );

    check(
        "alias Bad = array<f32, 0>;",
        r###"error: array element count must be positive (> 0)
  ┌─ wgsl:1:24
  │
1 │ alias Bad = array<f32, 0>;
  │                        ^ must be positive

"###,
    );

    check(
        "alias Bad = array<f32, -1>;",
        r###"error: array element count must be positive (> 0)
  ┌─ wgsl:1:24
  │
1 │ alias Bad = array<f32, -1>;
  │                        ^^ must be positive

"###,
    );
}

#[test]
fn discard_in_wrong_stage() {
    check_validation! {
        "@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x == 3u {
        discard;
    }
}":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Compute,
            source: naga::valid::EntryPointError::ForbiddenStageOperations,
            ..
        })
    }

    check_validation! {
        "@vertex
fn main() -> @builtin(position) vec4<f32> {
    if true {
        discard;
    }
    return vec4<f32>();
}":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            source: naga::valid::EntryPointError::ForbiddenStageOperations,
            ..
        })
    }
}

#[test]
fn invalid_structs() {
    check_validation! {
        "struct Bad { data: sampler }",
        "struct Bad { data: texture_2d<f32> }":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::InvalidData(_),
            ..
        })
    }

    check_validation! {
        "struct Bad { data: array<f32>, other: f32, }":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::InvalidDynamicArray(_, _),
            ..
        })
    }

    check_validation! {
        "struct Empty {}":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::EmptyStruct,
            ..
        })
    }
}

#[test]
fn struct_type_mismatch_in_assignment() {
    check_validation!(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };
        fn main() {
            var x: Bar = Bar(1);
            x = Foo(1);
        }
        ":
        Err(naga::valid::ValidationError::Function {
            handle: _,
            name: function_name,
            source: naga::valid::FunctionError::InvalidStoreTypes { .. },
        })
        // The validation error is reported at the call, i.e., in `main`
        if function_name == "main"
    );
}

#[test]
fn struct_type_mismatch_in_let_decl() {
    check(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };
        fn main() {
            let x: Bar = Foo(1);
        }
        ",
        "error: the type of `x` is expected to be `Bar`, but got `Foo`
  ┌─ wgsl:5:17
  │
5 │             let x: Bar = Foo(1);
  │                 ^ definition of `x`

",
    );
}

#[test]
fn struct_type_mismatch_in_return_value() {
    check_validation!(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };
        fn bar() -> Bar {
            return Foo(1);
        }
        ":
        Err(naga::valid::ValidationError::Function {
            handle: _,
            name: function_name,
            source: naga::valid::FunctionError::InvalidReturnType { .. }
        }) if function_name == "bar"
    );
}

#[test]
fn struct_type_mismatch_in_argument() {
    check_validation!(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };
        fn bar(a: Bar) {}
        fn main() {
            bar(Foo(1));
        }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::InvalidCall {
                function: _,
                error: naga::valid::CallError::ArgumentType { index, .. },
            },
            ..
        })
        // The validation error is reported at the call, i.e., in `main`
        if function_name == "main" && *index == 0
    );
}

#[test]
fn struct_type_mismatch_in_global_var() {
    check(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };

        var<uniform> foo: Foo = Bar(1);
        ",
        "error: the type of `foo` is expected to be `Foo`, but got `Bar`
  ┌─ wgsl:5:22
  │
5 │         var<uniform> foo: Foo = Bar(1);
  │                      ^^^ definition of `foo`

",
    );
}

#[test]
fn struct_type_mismatch_in_global_const() {
    check(
        "
        struct Foo { a: u32 };
        struct Bar { a: u32 };

        const foo: Foo = Bar(1);
        ",
        "error: the type of `foo` is expected to be `Foo`, but got `Bar`
  ┌─ wgsl:5:15
  │
5 │         const foo: Foo = Bar(1);
  │               ^^^ definition of `foo`

",
    );
}

#[test]
fn invalid_functions() {
    check_validation! {
        "fn unacceptable_unsized(arg: array<f32>) { }",
        "
        struct Unsized { data: array<f32> }
        fn unacceptable_unsized(arg: Unsized) { }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::InvalidArgumentType {
                index: 0,
                name: argument_name,
            },
            ..
        })
        if function_name == "unacceptable_unsized" && argument_name == "arg"
    }

    // Pointer's address space cannot hold unsized data.
    check_validation! {
        "fn unacceptable_unsized(arg: ptr<workgroup, array<f32>>) { }",
        "
        struct Unsized { data: array<f32> }
        fn unacceptable_unsized(arg: ptr<workgroup, Unsized>) { }
        ":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::InvalidPointerToUnsized {
                base: _,
                space: naga::AddressSpace::WorkGroup,
            },
            ..
        })
    }

    // Pointers of these address spaces cannot be passed as arguments.
    check_validation! {
        "fn unacceptable_ptr_space(arg: ptr<storage, array<f32>>) { }":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::InvalidArgumentPointerSpace {
                index: 0,
                name: argument_name,
                space: naga::AddressSpace::Storage { .. },
            },
            ..
        })
        if function_name == "unacceptable_ptr_space" && argument_name == "arg"
    }
    check_validation! {
        "fn unacceptable_ptr_space(arg: ptr<uniform, f32>) { }":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::InvalidArgumentPointerSpace {
                index: 0,
                name: argument_name,
                space: naga::AddressSpace::Uniform,
            },
            ..
        })
        if function_name == "unacceptable_ptr_space" && argument_name == "arg"
    }
    check_validation! {
        "fn unacceptable_ptr_space(arg: ptr<workgroup, f32>) { }":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::InvalidArgumentPointerSpace {
                index: 0,
                name: argument_name,
                space: naga::AddressSpace::WorkGroup,
            },
            ..
        })
        if function_name == "unacceptable_ptr_space" && argument_name == "arg"
    }

    check_validation! {
        "
        struct AFloat {
          said_float: f32
        };
        @group(0) @binding(0)
        var<storage> float: AFloat;

        fn return_pointer() -> ptr<storage, f32> {
           return &float.said_float;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::NonConstructibleReturnType,
            ..
        })
        if function_name == "return_pointer"
    }

    check_validation! {
        "
        @group(0) @binding(0)
        var<storage> atom: atomic<u32>;

        fn return_atomic() -> atomic<u32> {
           return atom;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            source: naga::valid::FunctionError::NonConstructibleReturnType,
            ..
        })
        if function_name == "return_atomic"
    }
}

#[test]
fn invalid_return_type() {
    check_validation! {
        "fn invalid_return_type() -> i32 { return 0u; }":
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::InvalidReturnType { .. },
            ..
        })
    };
}

#[test]
fn pointer_type_equivalence() {
    check_validation! {
        r#"
            fn f(pv: ptr<function, vec2<f32>>) { }

            fn g() {
               var m: mat2x2<f32>;
               let pv: ptr<function, vec2<f32>> = &m[0];

               f(pv);
            }
        "#:
        Ok(_)
    }
}

#[test]
fn missing_bindings() {
    check_validation! {
        "
        @fragment
        fn fragment(_input: vec4<f32>) -> @location(0) vec4<f32> {
           return _input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Fragment,
            source: naga::valid::EntryPointError::Argument(
                0,
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation! {
        "
        @fragment
        fn fragment(@location(0) _input: vec4<f32>, more_input: f32) -> @location(0) vec4<f32> {
           return _input + more_input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Fragment,
            source: naga::valid::EntryPointError::Argument(
                1,
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation! {
        "
        @fragment
        fn fragment(@location(0) _input: vec4<f32>) -> vec4<f32> {
           return _input;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Fragment,
            source: naga::valid::EntryPointError::Result(
                naga::valid::VaryingError::MissingBinding,
            ),
            ..
        })
    }

    check_validation! {
        "
        struct FragmentIn {
          @location(0) pos: vec4<f32>,
          uv: vec2<f32>
        }

        @fragment
        fn fragment(_input: FragmentIn) -> @location(0) vec4<f32> {
           return _input.pos;
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Fragment,
            source: naga::valid::EntryPointError::Argument(
                0,
                naga::valid::VaryingError::MemberMissingBinding(1),
            ),
            ..
        })
    }
}

#[test]
fn missing_bindings2() {
    check_validation! {
        "
        @vertex
        fn vertex() {}
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            source: naga::valid::EntryPointError::MissingVertexOutputPosition,
            ..
        })
    }

    check_validation! {
        "
        struct VertexOut {
            @location(0) a: vec4<f32>,
        }

        @vertex
        fn vertex() -> VertexOut {
            return VertexOut(vec4<f32>());
        }
        ":
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            source: naga::valid::EntryPointError::MissingVertexOutputPosition,
            ..
        })
    }
}

#[test]
fn invalid_blend_src() {
    // Missing capability.
    check_validation! {
        "
        enable dual_source_blending;
        struct FragmentOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(0) @blend_src(1) output1: vec4<f32>,
        }
        @fragment
        fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::UnsupportedCapability(Capabilities::DUAL_SOURCE_BLENDING),
                ),
                ..
            },
        )
    }

    // Missing enable directive.
    // Note that this is a parsing error, not a validation error.
    check("
        struct FragmentOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(0) @blend_src(1) output1: vec4<f32>,
        }
        @fragment
        fn main(@builtin(position) position: vec4<f32>) -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(0.0)); }
        ",
        r###"error: the `dual_source_blending` enable extension is not enabled
  ┌─ wgsl:3:27
  │
3 │             @location(0) @blend_src(0) output0: vec4<f32>,
  │                           ^^^^^^^^^ the `dual_source_blending` "Enable Extension" is needed for this functionality, but it is not currently enabled.
  │
  = note: You can enable this extension by adding `enable dual_source_blending;` at the top of the shader, before any other items.

"###,
    );

    // Using blend_src on an input.
    check_validation! {
        "
        enable dual_source_blending;
        @fragment
        fn main(@location(0) @blend_src(0) input: f32) -> vec4f { return vec4f(0.0); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Argument(
                    0,
                    naga::valid::VaryingError::InvalidInputAttributeInStage("blend_src", naga::ShaderStage::Fragment),
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Using blend_src as output on something that isn't a fragment shader.
    check_validation! {
        "
        enable dual_source_blending;
        struct VertexOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(0) @blend_src(1) output1: vec4<f32>,
        }
        @vertex
        fn main() -> VertexOutput { return VertexOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Vertex,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::InvalidAttributeInStage("blend_src", naga::ShaderStage::Vertex),
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Invalid blend_src index.
    check_validation! {
        "
        enable dual_source_blending;
        struct FragmentOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(0) @blend_src(2) output1: vec4<f32>,
        }
        @fragment
        fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::InvalidBlendSrcIndex { location: 0, blend_src: 2 },
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Using a location other than 1 on blend_src
    check_validation! {
        "
        enable dual_source_blending;
        struct FragmentOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(1) @blend_src(1) output1: vec4<f32>,
        }
        @fragment
        fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::InvalidBlendSrcIndex { location: 1, blend_src: 1 },
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Using same blend_src several times.
    check_validation! {
        "
        enable dual_source_blending;
        struct FragmentOutput {
            @location(0) @blend_src(1) output0: vec4<f32>,
            @location(0) @blend_src(1) output1: vec4<f32>,
        }
        @fragment
        fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::BindingCollisionBlendSrc { blend_src: 1 },
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Two attributes, only one has blend_src
    check_validation! {
        "
        enable dual_source_blending;
        struct FragmentOutput {
            @location(0) @blend_src(0) output0: vec4<f32>,
            @location(1) output1: vec4<f32>,
        }
        @fragment
        fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), vec4(1.0)); }
        ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::IncompleteBlendSrcUsage,
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Single attribute using blend_src.
    check_validation! {
        "
            enable dual_source_blending;
            struct FragmentOutput {
                @location(0) @blend_src(0) output0: vec4<f32>,
            }
            @fragment
            fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0)); }
            ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::IncompleteBlendSrcUsage,
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }

    // Mixed output types.
    check_validation! {
        "
            enable dual_source_blending;
            struct FragmentOutput {
                @location(0) @blend_src(0) output0: vec4<f32>,
                @location(0) @blend_src(1) output1: f32,
            }
            @fragment
            fn main() -> FragmentOutput { return FragmentOutput(vec4(0.0), 1.0); }
            ":
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Fragment,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::BlendSrcOutputTypeMismatch { blend_src_0_type: _, blend_src_1_type: _ },
                ),
                ..
            },
        ),
        Capabilities::DUAL_SOURCE_BLENDING
    }
}

#[test]
fn invalid_access() {
    check_validation! {
        r#"
            fn main() -> f32 {
                let a = array<f32, 3>(0., 1., 2.);
                return a[3];
            }
        "#:
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::Expression {
                source: naga::valid::ExpressionError::IndexOutOfBounds(_, _),
                ..
            },
            ..
        })
    }
}

#[test]
fn valid_access() {
    check_validation! {
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

    check_validation! {
        "
        fn matrix_by_value(m: mat4x4<f32>, i: i32) -> vec4<f32> {
            return m[i];
        }
        ":
        Ok(_)
    }
}

#[test]
fn invalid_local_vars() {
    check_validation! {
        "
        struct Unsized { data: array<f32> }
        fn local_ptr_dynamic_array(okay: ptr<storage, Unsized>) {
            var not_okay: ptr<storage, array<f32>> = &(*okay).data;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: naga::valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        })
        if local_var_name == "not_okay"
    }

    check_validation! {
        "
        fn f() {
            var x: atomic<u32>;
        }
        ":
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::LocalVariable {
                name: local_var_name,
                source: naga::valid::LocalVariableError::InvalidType(_),
                ..
            },
            ..
        })
        if local_var_name == "x"
    }
}

#[test]
fn invalid_runtime_sized_arrays() {
    // You can't have structs whose last member is an unsized struct. An unsized
    // array may only appear as the last member of a struct used directly as a
    // variable's store type.
    check_validation! {
        "
        struct Unsized {
            arr: array<f32>
        }

        struct Outer {
            legit: i32,
            _unsized: Unsized
        }

        @group(0) @binding(0) var<storage> outer: Outer;

        fn fetch(i: i32) -> f32 {
           return outer._unsized.arr[i];
        }
        ":
        Err(naga::valid::ValidationError::Type {
            name: struct_name,
            source: naga::valid::TypeError::InvalidDynamicArray(member_name, _),
            ..
        })
        if struct_name == "Outer" && member_name == "_unsized"
    }
}

#[test]
fn select() {
    let snapshots = [
        (
            "
        fn select_pointers(which: bool) -> i32 {
            var x: i32 = 1;
            var y: i32 = 2;
            let p = select(&x, &y, which);
            return *p;
        }
        ",
            "\
error: unexpected argument type for `select` call
  ┌─ wgsl:5:28
  │
5 │             let p = select(&x, &y, which);
  │                            ^^ this value of type `ptr<function, i32>`
  │
  = note: expected a scalar or a `vecN` of scalars

",
        ),
        (
            "
        fn select_arrays(which: bool) -> i32 {
            var x: array<i32, 4>;
            var y: array<i32, 4>;
            let s = select(x, y, which);
            return s[0];
        }
        ",
            "\
error: unexpected argument type for `select` call
  ┌─ wgsl:5:28
  │
5 │             let s = select(x, y, which);
  │                            ^ this value of type `array<i32, 4>`
  │
  = note: expected a scalar or a `vecN` of scalars

",
        ),
        (
            "
        struct S { member: i32 }
        fn select_structs(which: bool) -> S {
            var x: S = S(1);
            var y: S = S(2);
            let s = select(x, y, which);
            return s;
        }
        ",
            "\
error: unexpected argument type for `select` call
  ┌─ wgsl:6:28
  │
6 │             let s = select(x, y, which);
  │                            ^ this value of type `S`
  │
  = note: expected a scalar or a `vecN` of scalars

",
        ),
        (
            "
        @compute @workgroup_size(1, 1)
        fn main() {
            // Bad: `9001` isn't a `bool`.
            _ = select(1, 2, 9001);
        }
        ",
            "\
error: Expected boolean expression for condition argument of `select`, got something else
  ┌─ wgsl:5:17
  │
5 │             _ = select(1, 2, 9001);
  │                 ^^^^^^ see msg

",
        ),
        (
            "
        @compute @workgroup_size(1, 1)
        fn main() {
            // Bad: `bool` and abstract int args. don't match.
            _ = select(true, 1, false);
        }
        ",
            "\
error: type mismatch for reject and accept values in `select` call
  ┌─ wgsl:5:24
  │
5 │             _ = select(true, 1, false);
  │                        ^^^^  ^ accept value of type `{AbstractInt}`
  │                        │      
  │                        reject value of type `bool`

",
        ),
    ];

    for (input, snapshot) in snapshots {
        check(input, snapshot);
    }
}

#[test]
fn missing_default_case() {
    check_validation! {
        "
        fn test_missing_default_case() {
          switch(0) {
            case 0: {}
          }
        }
        ":
        Err(
            naga::valid::ValidationError::Function {
                source: naga::valid::FunctionError::MissingDefaultCase,
                ..
            },
        )
    }
}

#[test]
fn wrong_access_mode() {
    // The assignments to `global.i` should be forbidden, because they are in
    // variables whose access mode is `read`, not `read_write`.
    check_validation! {
        "
            struct Globals {
                i: i32
            }

            @group(0) @binding(0)
            var<storage> globals: Globals;

            fn store(v: i32) {
                globals.i = v;
            }
        ",
        "
            struct Globals {
                i: i32
            }

            @group(0) @binding(0)
            var<uniform> globals: Globals;

            fn store(v: i32) {
                globals.i = v;
            }
        ":
        Err(
            naga::valid::ValidationError::Function {
                name,
                source: naga::valid::FunctionError::InvalidStorePointer(_),
                ..
            },
        )
            if name == "store"
    }
}

#[test]
fn io_shareable_types() {
    for numeric in "i32 u32 f32".split_whitespace() {
        let types = format!("{numeric} vec2<{numeric}> vec3<{numeric}> vec4<{numeric}>");
        for ty in types.split_whitespace() {
            check_one_validation! {
                &format!("@vertex
                          fn f(@location(0) arg: {ty}) -> @builtin(position) vec4<f32>
                          {{ return vec4<f32>(0.0); }}"),
                Ok(_module)
            }
        }
    }

    for ty in "bool
               vec2<bool> vec3<bool> vec4<bool>
               array<f32,4>
               mat2x2<f32>
               ptr<function,f32>"
        .split_whitespace()
    {
        check_one_validation! {
            &format!("@vertex
                          fn f(@location(0) arg: {ty}) -> @builtin(position) vec4<f32>
                          {{ return vec4<f32>(0.0); }}"),
            Err(
                naga::valid::ValidationError::EntryPoint {
                    stage: naga::ShaderStage::Vertex,
                    name,
                    source: naga::valid::EntryPointError::Argument(
                        0,
                        naga::valid::VaryingError::NotIOShareableType(
                            _,
                        ),
                    ),
                },
            )
            if name == "f"
        }
    }
}

#[test]
fn host_shareable_types() {
    // Host-shareable, constructible types.
    let types = "i32 u32 f32
                 vec2<i32> vec3<u32> vec4<f32>
                 mat4x4<f32>
                 array<mat4x4<f32>,4>
                 AStruct";
    for ty in types.split_whitespace() {
        check_one_validation! {
            &format!("struct AStruct {{ member: array<mat4x4<f32>, 8> }};
                      @group(0) @binding(0) var<uniform> ubuf: {ty};
                      @group(0) @binding(1) var<storage> sbuf: {ty};"),
            Ok(_module)
        }
    }

    // Host-shareable but not constructible types.
    let types = "atomic<i32> atomic<u32>
                 array<atomic<u32>,4>
                 array<u32>
                 AStruct";
    for ty in types.split_whitespace() {
        check_one_validation! {
            &format!("struct AStruct {{ member: array<atomic<u32>, 8> }};
                      @group(0) @binding(1) var<storage> sbuf: {ty};"),
            Ok(_module)
        }
    }

    // Types that are neither host-shareable nor constructible.
    for ty in "bool ptr<storage,i32>".split_whitespace() {
        check_one_validation! {
            &format!("@group(0) @binding(0) var<storage> sbuf: {ty};"),
            Err(
                naga::valid::ValidationError::GlobalVariable {
                    name,
                    handle: _,
                    source: naga::valid::GlobalVariableError::MissingTypeFlags { .. },
                },
            )
            if name == "sbuf"
        }

        check_one_validation! {
            &format!("@group(0) @binding(0) var<uniform> ubuf: {ty};"),
            Err(naga::valid::ValidationError::GlobalVariable {
                    name,
                    handle: _,
                    source: naga::valid::GlobalVariableError::MissingTypeFlags { .. },
                },
            )
            if name == "ubuf"
        }
    }
}

#[test]
fn var_init() {
    check_validation! {
        "
        var<workgroup> initialized: u32 = 0u;
        ":
        Err(
            naga::valid::ValidationError::GlobalVariable {
                source: naga::valid::GlobalVariableError::InitializerNotAllowed(naga::AddressSpace::WorkGroup),
                ..
            },
        )
    }
}

#[test]
fn misplaced_break_if() {
    check(
        "
        fn test_misplaced_break_if() {
            loop {
                break if true;
            }
        }
        ",
        r###"error: A break if is only allowed in a continuing block
  ┌─ wgsl:4:17
  │
4 │                 break if true;
  │                 ^^^^^^^^ not in a continuing block

"###,
    );
}

#[test]
fn break_if_bad_condition() {
    check_validation! {
        "
        fn test_break_if_bad_condition() {
            loop {
                continuing {
                    break if 1;
                }
            }
        }
        ":
        Err(
            naga::valid::ValidationError::Function {
                source: naga::valid::FunctionError::InvalidIfType(_),
                ..
            },
        )
    }
}

#[test]
fn swizzle_assignment() {
    check(
        "
        fn f() {
            var v = vec2(0);
            v.xy = vec2(1);
        }
    ",
        r###"error: invalid left-hand side of assignment
  ┌─ wgsl:4:13
  │
4 │             v.xy = vec2(1);
  │             ^^^^ cannot assign to this expression
  │
  = note: WGSL does not support assignments to swizzles
  = note: consider assigning each component individually

"###,
    );
}

#[test]
fn binary_statement() {
    check(
        "
        fn f() {
            3 + 5;
        }
    ",
        r###"error: expected assignment or increment/decrement, found "+"
  ┌─ wgsl:3:15
  │
3 │             3 + 5;
  │               ^ expected assignment or increment/decrement

"###,
    );
}

#[test]
fn assign_to_expr() {
    check(
        "
        fn f() {
            3 + 5 = 10;
        }
        ",
        r###"error: expected assignment or increment/decrement, found "+"
  ┌─ wgsl:3:15
  │
3 │             3 + 5 = 10;
  │               ^ expected assignment or increment/decrement

"###,
    );
}

#[test]
fn assign_to_let() {
    check(
        "
        fn f() {
            let a = 10;
	        a = 20;
        }
        ",
        r###"error: invalid left-hand side of assignment
  ┌─ wgsl:3:17
  │
3 │             let a = 10;
  │                 ^ this is an immutable binding
4 │             a = 20;
  │             ^ cannot assign to this expression
  │
  = note: consider declaring `a` with `var` instead of `let`

"###,
    );

    check(
        "
        fn f() {
            let a = array(1, 2);
			a[0] = 1;
        }
        ",
        r###"error: invalid left-hand side of assignment
  ┌─ wgsl:3:17
  │
3 │             let a = array(1, 2);
  │                 ^ this is an immutable binding
4 │             a[0] = 1;
  │             ^^^^ cannot assign to this expression
  │
  = note: consider declaring `a` with `var` instead of `let`

"###,
    );

    check(
        "
        struct S { a: i32 }

        fn f() {
            let a = S(10);
	        a.a = 20;
        }
        ",
        r###"error: invalid left-hand side of assignment
  ┌─ wgsl:5:17
  │
5 │             let a = S(10);
  │                 ^ this is an immutable binding
6 │             a.a = 20;
  │             ^^^ cannot assign to this expression
  │
  = note: consider declaring `a` with `var` instead of `let`

"###,
    );
}

#[test]
fn recursive_function() {
    check(
        "
        fn f() {
            f();
        }
        ",
        r###"error: declaration of `f` is recursive
  ┌─ wgsl:2:12
  │
2 │         fn f() {
  │            ^
3 │             f();
  │             ^ uses itself here

"###,
    );
}

#[test]
fn cyclic_function() {
    check(
        "
        fn f() {
            g();
        }
        fn g() {
            f();
        }
        ",
        r###"error: declaration of `f` is cyclic
  ┌─ wgsl:2:12
  │
2 │         fn f() {
  │            ^
3 │             g();
  │             ^ uses `g`
4 │         }
5 │         fn g() {
  │            ^
6 │             f();
  │             ^ ending the cycle

"###,
    );
}

#[test]
fn switch_signed_unsigned_mismatch() {
    check(
        "
        fn x(y: u32) {
            switch y {
                case 1i: {}
            }
        }
        ",
        r###"error: invalid `switch` case selector value
  ┌─ wgsl:4:22
  │
4 │                 case 1i: {}
  │                      ^^ `switch` case selector must have the same type as the `switch` selector expression

"###,
    );

    check(
        "
        fn x(y: i32) {
            switch y {
                case 1u: {}
            }
        }
        ",
        r###"error: invalid `switch` case selector value
  ┌─ wgsl:4:22
  │
4 │                 case 1u: {}
  │                      ^^ `switch` case selector must have the same type as the `switch` selector expression

"###,
    );
}

#[test]
fn switch_invalid_type() {
    check(
        "
        fn x(y: f32) {
            switch y {
                case 1: {}
            }
        }
        ",
        r###"error: invalid `switch` selector
  ┌─ wgsl:3:20
  │
3 │             switch y {
  │                    ^ `switch` selector must be a scalar integer

"###,
    );

    check(
        "
        fn x(y: vec2<i32>) {
            switch y {
                case 1: {}
            }
        }
        ",
        r###"error: invalid `switch` selector
  ┌─ wgsl:3:20
  │
3 │             switch y {
  │                    ^ `switch` selector must be a scalar integer

"###,
    );

    check(
        "
        fn x() {
            switch 0 {
                case 1.0: {}
            }
        }
    ",
        r###"error: invalid `switch` case selector value
  ┌─ wgsl:4:22
  │
4 │                 case 1.0: {}
  │                      ^^^ `switch` case selector must be a scalar integer const expression

"###,
    );
}

#[test]
fn switch_non_const_case() {
    check(
        "
        fn x(y: i32) {
            switch 0 {
                case y: {}
            }
        }
    ",
        r###"error: invalid `switch` case selector value
  ┌─ wgsl:4:22
  │
4 │                 case y: {}
  │                      ^ `switch` case selector must be a scalar integer const expression

"###,
    );
}

#[test]
fn function_returns_void() {
    check(
        "
        fn x() {
	        let a = vec2<f32>(1.0, 2.0);
        }

        fn b() {
	        let a = x();
        }
    ",
        r###"error: function does not return any value
  ┌─ wgsl:7:18
  │
7 │             let a = x();
  │                     ^
  │
  = note: perhaps you meant to call the function in a separate statement?

"###,
    )
}

#[test]
fn function_must_use_unused() {
    check(
        r#"
@must_use
fn use_me(a: i32) -> i32 {
  return 10;
}

fn useless() -> i32 {
  use_me(1);
  return 0;
}
"#,
        r#"error: unused return value from function annotated with @must_use
  ┌─ wgsl:8:3
  │
8 │   use_me(1);
  │   ^^^^^^
  │
  = note: function 'use_me' is declared with `@must_use` attribute
  = note: use a phony assignment or declare a value using the function call as the initializer

"#,
    );
}

#[test]
fn function_must_use_returns_void() {
    check(
        r#"
@must_use
fn use_me(a: i32) {
  let x = a;
}
"#,
        r#"error: function annotated with @must_use but does not return any value
  ┌─ wgsl:2:2
  │
2 │ @must_use
  │  ^^^^^^^^
3 │ fn use_me(a: i32) {
  │    ^^^^^^^^^^^^^
  │
  = note: declare a return type or remove the attribute

"#,
    );
}

#[test]
fn function_must_use_repeated() {
    check(
        r#"
@must_use
@must_use
fn use_me(a: i32) -> i32 {
  return 10;
}
"#,
        r#"error: repeated attribute: `must_use`
  ┌─ wgsl:3:2
  │
3 │ @must_use
  │  ^^^^^^^^ repeated attribute

"#,
    );
}

#[test]
fn struct_member_must_use() {
    check(
        r#"
struct S {
  @must_use a: i32,
}
"#,
        r#"error: unknown attribute: `must_use`
  ┌─ wgsl:3:4
  │
3 │   @must_use a: i32,
  │    ^^^^^^^^ unknown attribute

"#,
    )
}

#[test]
fn function_param_redefinition_as_param() {
    check(
        "
        fn x(a: f32, a: vec2<f32>) {}
    ",
        "error: redefinition of `a`
  ┌─ wgsl:2:14
  │
2 │         fn x(a: f32, a: vec2<f32>) {}
  │              ^       ^ redefinition of `a`
  │              │\x20\x20\x20\x20\x20\x20\x20\x20
  │              previous definition of `a`

",
    )
}

#[test]
fn function_param_redefinition_as_local() {
    check(
        "
        fn x(a: f32) {
			let a = 0.0;
		}
    ",
        r###"error: redefinition of `a`
  ┌─ wgsl:2:14
  │
2 │         fn x(a: f32) {
  │              ^ previous definition of `a`
3 │             let a = 0.0;
  │                 ^ redefinition of `a`

"###,
    )
}

#[test]
fn struct_redefinition() {
    check(
        "
        struct Foo { a: u32 };
        struct Foo { a: u32 };
    ",
        "error: redefinition of `Foo`
  ┌─ wgsl:2:16
  │
2 │         struct Foo { a: u32 };
  │                ^^^ previous definition of `Foo`
3 │         struct Foo { a: u32 };
  │                ^^^ redefinition of `Foo`

",
    );
}

#[test]
fn struct_member_redefinition() {
    check(
        "
        struct A {
            a: f32,
            a: f32,
        }
    ",
        r###"error: redefinition of `a`
  ┌─ wgsl:3:13
  │
3 │             a: f32,
  │             ^ previous definition of `a`
4 │             a: f32,
  │             ^ redefinition of `a`

"###,
    )
}

#[test]
fn function_must_return_value() {
    check_validation!(
        "fn func() -> i32 {
        }":
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::InvalidReturnType { .. },
            ..
        })
    );
    check_validation!(
        "fn func(x: i32) -> i32 {
            let y = x + 10;
        }":
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::InvalidReturnType { .. },
            ..
        })
    );
}

#[test]
fn constructor_type_error_span() {
    check(
        "
        fn unfortunate() {
            var a: array<i32, 1> = array<i32, 1>(1.0);
        }
    ",
        "error: automatic conversions cannot convert `{AbstractFloat}` to `i32`
  ┌─ wgsl:3:36
  │
3 │             var a: array<i32, 1> = array<i32, 1>(1.0);
  │                                    ^^^^^^^^^^^^^ ^^^ this expression has type {AbstractFloat}
  │                                    │\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20
  │                                    a value of type i32 is required here

",
    )
}

#[test]
fn global_initialization_type_mismatch() {
    check(
        "
        var<private> a: vec2<f32> = vec2<i32>(1i, 2i);
    ",
        r###"error: the type of `a` is expected to be `vec2<f32>`, but got `vec2<i32>`
  ┌─ wgsl:2:22
  │
2 │         var<private> a: vec2<f32> = vec2<i32>(1i, 2i);
  │                      ^ definition of `a`

"###,
    )
}

#[test]
fn binding_array_local() {
    check_validation! {
        "fn f() { var x: binding_array<sampler, 4>; }":
        Err(_)
    }
}

#[test]
fn binding_array_private() {
    check_validation! {
        "var<private> x: binding_array<sampler, 4>;":
        Err(_)
    }
}

#[test]
fn binding_array_non_struct() {
    check_validation! {
        "var<storage> x: binding_array<i32, 4>;":
        Err(naga::valid::ValidationError::Type {
            source: naga::valid::TypeError::BindingArrayBaseTypeNotStruct(_),
            ..
        })
    }
}

#[test]
fn compaction_preserves_spans() {
    let source = r#"
        fn f() {
           var a: i32 = -(-(-(-42i)));
           var x: array<i32,1>;
           var y = x[1.0];
        }
        @compute @workgroup_size(1)
        fn main() {
            f();
        }
    "#;
    // The error span should be on `x[1.0]`, which is at characters 108..114.
    let mut module = naga::front::wgsl::parse_str(source).expect("source ought to parse");
    naga::compact::compact(&mut module, KeepUnused::No);
    let err = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .expect_err("source ought to fail validation");

    // Ideally this would all just be a `matches!` with a big pattern,
    // but the `Span` API is full of opaque structs.
    let mut spans = err.spans();

    // The first span is the whole function.
    let _ = spans.next().expect("error should have at least one span");

    // The second span is the invalid indexing expression.
    let dest_span = spans
        .next()
        .expect("error should have at least two spans")
        .0;
    if !matches!(
        dest_span.to_range(),
        Some(core::ops::Range {
            start: 108,
            end: 114
        })
    ) {
        panic!("Error message has wrong span:\n\n{err:#?}");
    }
}

#[test]
fn limit_braced_statement_nesting() {
    let too_many_braces = "fn f() {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{";

    let expected_diagnostic = r###"error: brace nesting limit reached
  ┌─ wgsl:1:135
  │
1 │ fn f() {{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{
  │                                                                                                                                       ^ limit reached at this brace
  │
  = note: nesting limit is currently set to 127

"###;

    // In debug builds, we might actually overflow the stack before exercising this error case,
    // depending on the platform and the `RUST_MIN_STACK` env. var. Use a thread with a custom
    // stack size that works on all platforms.
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 2 /* MB */)
        .spawn(|| check(too_many_braces, expected_diagnostic))
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn too_many_unclosed_loops() {
    let too_many_braces = "fn f() {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
       loop {
           ";

    let expected_diagnostic = r###"error: brace nesting limit reached
    ┌─ wgsl:128:13
    │
128 │        loop {
    │             ^ limit reached at this brace
    │
    = note: nesting limit is currently set to 127

"###;

    // In debug builds, we might actually overflow the stack before exercising this error case,
    // depending on the platform and the `RUST_MIN_STACK` env. var. Use a thread with a custom
    // stack size that works on all platforms.
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 2 /* MB */)
        .spawn(|| check(too_many_braces, expected_diagnostic))
        .unwrap()
        .join()
        .unwrap()
}

#[test]
fn local_const_wrong_type() {
    check(
        "
        fn f() {
            const c: i32 = 5u;
        }
        ",
        r###"error: the type of `c` is expected to be `i32`, but got `u32`
  ┌─ wgsl:3:19
  │
3 │             const c: i32 = 5u;
  │                   ^ definition of `c`

"###,
    );
}

#[test]
fn local_const_from_let() {
    check(
        "
        fn f() {
            let a = 5;
            const c = a;
        }
        ",
        r###"error: this operation is not supported in a const context
  ┌─ wgsl:4:23
  │
4 │             const c = a;
  │                       ^ operation not supported here

"###,
    );
}

#[test]
fn local_const_from_var() {
    check(
        "
        fn f() {
            var a = 5;
            const c = a;
        }
        ",
        r###"error: this operation is not supported in a const context
  ┌─ wgsl:4:23
  │
4 │             const c = a;
  │                       ^ operation not supported here

"###,
    );
}

#[test]
fn local_const_from_override() {
    check(
        "
        override o: i32;
        fn f() {
            const c = o;
        }
        ",
        r###"error: Unexpected override-expression
  ┌─ wgsl:4:23
  │
4 │             const c = o;
  │                       ^ see msg

"###,
    );
}

#[test]
fn local_const_from_global_var() {
    check(
        "
        var v: i32;
        fn f() {
            const c = v;
        }
        ",
        r###"error: Unexpected runtime-expression
  ┌─ wgsl:4:23
  │
4 │             const c = v;
  │                       ^ see msg

"###,
    );
}

#[test]
fn only_one_swizzle_type() {
    check(
        "
        const ok1 = vec2(0.0, 0.0).xy;
        const ok2 = vec2(0.0, 0.0).rg;
        const err = vec2(0.0, 0.0).xg;
        ",
        r###"error: invalid field accessor `xg`
  ┌─ wgsl:4:36
  │
4 │         const err = vec2(0.0, 0.0).xg;
  │                                    ^^ invalid accessor

"###,
    );
}

#[test]
fn const_assert_must_be_const() {
    check(
        "
        fn foo() {
            let a = 5;
            const_assert a != 0;
        }
        ",
        r###"error: this operation is not supported in a const context
  ┌─ wgsl:4:26
  │
4 │             const_assert a != 0;
  │                          ^ operation not supported here

"###,
    );
}

#[test]
fn const_assert_must_be_bool() {
    check(
        "
            const_assert(5); // 5 is not bool
        ",
        r###"error: must be a const-expression that resolves to a `bool`
  ┌─ wgsl:2:26
  │
2 │             const_assert(5); // 5 is not bool
  │                          ^ must resolve to `bool`

"###,
    );
}

#[test]
fn const_assert_failed() {
    check(
        "
            const_assert(false);
        ",
        r###"error: `const_assert` failure
  ┌─ wgsl:2:26
  │
2 │             const_assert(false);
  │                          ^^^^^ evaluates to `false`

"###,
    );
}

#[test]
fn reject_utf8_bom() {
    check(
        "\u{FEFF}fn main() {}",
        r#"error: expected global item (`struct`, `const`, `var`, `alias`, `fn`, `diagnostic`, `enable`, `requires`, `;`) or the end of the file, found "\u{feff}"
  ┌─ wgsl:1:1
  │
1 │ ﻿fn main() {}
  │  expected global item (`struct`, `const`, `var`, `alias`, `fn`, `diagnostic`, `enable`, `requires`, `;`) or the end of the file

"#,
    );
}

#[test]
fn matrix_vector_pointers() {
    check(
        "fn foo() {
            var v: vec2<f32>;
            let p = &v[0];
        }",
        r#"error: cannot take the address of a vector component
  ┌─ wgsl:3:22
  │
3 │             let p = &v[0];
  │                      ^^^^ invalid operand for address-of

"#,
    );

    check(
        "fn foo() {
            var v: vec2<f32>;
            let p = &v.x;
        }",
        r#"error: cannot take the address of a vector component
  ┌─ wgsl:3:22
  │
3 │             let p = &v.x;
  │                      ^^^ invalid operand for address-of

"#,
    );

    check(
        "fn foo() {
            var m: mat2x2<f32>;
            let p = &m[0][0];
        }",
        r#"error: cannot take the address of a vector component
  ┌─ wgsl:3:22
  │
3 │             let p = &m[0][0];
  │                      ^^^^^^^ invalid operand for address-of

"#,
    );
}

#[test]
fn vector_logical_ops() {
    // Const context
    check(
        "const and = vec2(true, false) && vec2(false, false);",
        r###"error: Cannot apply the binary op to the arguments
  ┌─ wgsl:1:13
  │
1 │ const and = vec2(true, false) && vec2(false, false);
  │             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ see msg

"###,
    );

    check(
        "const or = vec2(true, false) || vec2(false, false);",
        r###"error: Cannot apply the binary op to the arguments
  ┌─ wgsl:1:12
  │
1 │ const or = vec2(true, false) || vec2(false, false);
  │            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ see msg

"###,
    );

    // Runtime context
    check(
        "fn foo(a: vec2<bool>, b: vec2<bool>) {
            let y = a && b;
        }",
        r#"error: Incompatible operands: LogicalAnd(vec2<bool>, _)

"#,
    );

    check(
        "fn foo(a: vec2<bool>, b: vec2<bool>) {
            let y = a || b;
        }",
        r#"error: Incompatible operands: LogicalOr(vec2<bool>, _)

"#,
    );
}

#[test]
fn issue7165() {
    // Regression test for https://github.com/gfx-rs/wgpu/issues/7165
    let shader = "
        struct Struct { a: u32 }
        fn invalid_return_type(a: Struct) -> i32 { return a; }
    ";

    let module = naga::front::wgsl::parse_str(shader).unwrap();
    let err = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_err();

    // This is a proxy for doing the following (with an error
    // handler installed so it doesn't immediately panic):
    //
    // ```
    // device.create_shader_module(wgpu::ShaderModuleDescriptor {
    //     label,
    //     source: wgpu::ShaderSource::Naga(module),
    // });
    // ```
    //
    // `ShaderSource::Naga` causes the implementation to proceed with an empty
    // module source, which (prior to the fix for #7165) could panic when
    // rendering an error if the module contained spans.
    let _location = err.location("");
}

#[test]
fn wrong_argument_count() {
    check(
        "fn foo() -> f32 {
            return sin();
        }",
        r#"error: wrong number of arguments: expected 1, found 0
  ┌─ wgsl:2:20
  │
2 │             return sin();
  │                    ^^^ wrong number of arguments

"#,
    );
}

#[test]
fn too_many_arguments() {
    check(
        "fn foo() -> f32 {
            return sin(1.0, 2.0);
        }",
        r#"error: too many arguments passed to `sin`
  ┌─ wgsl:2:20
  │
2 │             return sin(1.0, 2.0);
  │                    ^^^      ^^^ unexpected argument #2
  │
  = note: The `sin` function accepts at most 1 argument(s)

"#,
    );
}

#[test]
fn too_many_arguments_2() {
    check(
        "fn foo() -> f32 {
            return distance(vec2<f32>(), 0i);
        }",
        r#"error: wrong type passed as argument #2 to `distance`
  ┌─ wgsl:2:20
  │
2 │             return distance(vec2<f32>(), 0i);
  │                    ^^^^^^^^              ^^ argument #2 has type `i32`
  │
  = note: `distance` accepts the following types for argument #2:
  = note: allowed type: f32
  = note: allowed type: f16
  = note: allowed type: f64
  = note: allowed type: vec2<f32>
  = note: allowed type: vec2<f16>
  = note: allowed type: vec2<f64>
  = note: allowed type: vec3<f32>
  = note: allowed type: vec3<f16>
  = note: allowed type: vec3<f64>
  = note: allowed type: vec4<f32>
  = note: allowed type: vec4<f16>
  = note: allowed type: vec4<f64>

"#,
    );
}

#[test]
fn inconsistent_type() {
    check(
        "fn foo() -> f32 {
            return dot(vec4<f32>(), vec3<f32>());
        }",
        r#"error: inconsistent type passed as argument #2 to `dot`
  ┌─ wgsl:2:20
  │
2 │             return dot(vec4<f32>(), vec3<f32>());
  │                    ^^^ ^^^^^^^^^^   ^^^^^^^^^^ argument #2 has type vec3<f32>
  │                        │             
  │                        this argument has type vec4<f32>, which constrains subsequent arguments
  │
  = note: Because argument #1 has type vec4<f32>, only the following types
  = note: (or types that automatically convert to them) are accepted for argument #2:
  = note: allowed type: vec4<f32>

"#,
    );
}

#[test]
fn more_inconsistent_type() {
    #[track_caller]
    fn variant(call: &str) {
        let input = format!(
            r#"
            fn f() {{ var x = {call}; }}
        "#
        );
        let result = naga::front::wgsl::parse_str(&input);
        let Err(ref err) = result else {
            panic!("expected ParseError, got {result:#?}");
        };
        if !err.message().contains("inconsistent type") {
            panic!("expected 'inconsistent type' error, got {result:#?}");
        }
    }

    variant("min(1.0, 1i)");
    variant("min(1i, 1.0)");
    variant("min(1i, 1f)");
    variant("min(1f, 1i)");

    variant("clamp(1, 1.0, 1i)");
    variant("clamp(1, 1i, 1.0)");
    variant("clamp(1, 1i, 1f)");
    variant("clamp(1, 1f, 1i)");
    variant("clamp(1.0, 1, 1i)");
    variant("clamp(1.0, 1.0, 1i)");
    variant("clamp(1.0, 1i, 1)");
    variant("clamp(1.0, 1i, 1.0)");
    variant("clamp(1.0, 1i, 1i)");
    variant("clamp(1.0, 1i, 1f)");
    variant("clamp(1.0, 1f, 1i)");
    variant("clamp(1i, 1, 1.0)");
    variant("clamp(1i, 1, 1f)");
    variant("clamp(1i, 1.0, 1)");
    variant("clamp(1i, 1.0, 1.0)");
    variant("clamp(1i, 1.0, 1i)");
    variant("clamp(1i, 1.0, 1f)");
    variant("clamp(1i, 1i, 1.0)");
    variant("clamp(1i, 1i, 1f)");
    variant("clamp(1i, 1f, 1)");
    variant("clamp(1i, 1f, 1.0)");
    variant("clamp(1i, 1f, 1i)");
    variant("clamp(1i, 1f, 1f)");
    variant("clamp(1f, 1, 1i)");
    variant("clamp(1f, 1.0, 1i)");
    variant("clamp(1f, 1i, 1)");
    variant("clamp(1f, 1i, 1.0)");
    variant("clamp(1f, 1i, 1i)");
    variant("clamp(1f, 1i, 1f)");
    variant("clamp(1f, 1f, 1i)");
}

/// Naga should not crash just because the type of a
/// bad argument is a struct.
#[test]
fn struct_names_in_argument_errors() {
    #[track_caller]
    fn variant(argument: &str) -> Result<naga::Module, naga::front::wgsl::ParseError> {
        let input = format!(
            r#"
                struct A {{ x: i32, }};
                fn f() {{ _ = sin({argument}); }}
            "#
        );
        naga::front::wgsl::parse_str(&input)
    }

    assert!(variant("1.0").is_ok());
    assert!(variant("1").is_ok());
    assert!(variant("1i").is_err());
    assert!(variant("A()").is_err());
}

/// Naga should not crash just because the type of a
/// bad conversion operand is a struct.
#[test]
fn struct_names_in_conversion_errors() {
    #[track_caller]
    fn variant(argument: &str) -> Result<naga::Module, naga::front::wgsl::ParseError> {
        let input = format!(
            r#"
                struct A {{ x: i32, }};
                fn f() {{ _ = i32({argument}); }}
            "#
        );
        naga::front::wgsl::parse_str(&input)
    }

    assert!(variant("1.0").is_ok());
    assert!(variant("1").is_ok());
    assert!(variant("1i").is_ok());
    assert!(variant("A()").is_err());
}

/// Naga should not crash just because the type of a
/// bad initializer is a struct.
#[test]
fn struct_names_in_init_errors() {
    #[track_caller]
    fn variant(init: &str) -> Result<naga::Module, naga::front::wgsl::ParseError> {
        let input = format!(
            r#"
                struct A {{ x: i32, }};
                fn f() {{ var y: i32 = {init}; }}
            "#
        );
        naga::front::wgsl::parse_str(&input)
    }

    assert!(variant("1").is_ok());
    assert!(variant("1i").is_ok());
    assert!(variant("1.0").is_err());
    assert!(variant("A()").is_err());
}

/// Constant evaluation with interesting values.
#[test]
fn const_eval_value_errors() {
    #[track_caller]
    fn variant(expr: &str) -> Result<naga::Module, naga::front::wgsl::ParseError> {
        let input = format!(
            r#"
                fn f() {{ _ = {expr}; }}
            "#
        );
        naga::front::wgsl::parse_str(&input)
    }

    assert!(variant("1/1").is_ok());
    assert!(variant("1/0").is_err());

    assert!(variant("f32(abs(1))").is_ok());
    assert!(variant("f32(abs(-9223372036854775807))").is_ok());
    assert!(variant("f32(abs(-9223372036854775807 - 1))").is_ok());
}

#[test]
fn subgroup_invalid_broadcast() {
    check_validation! {
        r#"
            fn main(id: u32) {
                subgroupBroadcast(123, id);
            }
        "#:
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::InvalidSubgroup(
                naga::valid::SubgroupError::InvalidInvocationIdExprType(_),
            ),
            ..
        }),
        naga::valid::Capabilities::SUBGROUP
    }
    check_validation! {
        r#"
            fn main(id: u32) {
                quadBroadcast(123, id);
            }
        "#:
        Err(naga::valid::ValidationError::Function {
            source: naga::valid::FunctionError::InvalidSubgroup(
                naga::valid::SubgroupError::InvalidInvocationIdExprType(_),
            ),
            ..
        }),
        naga::valid::Capabilities::SUBGROUP
    }
}

#[test]
fn invalid_clip_distances() {
    // Missing capability.
    check_validation! {
        r#"
            enable clip_distances;
            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @builtin(clip_distances) clip_distances: array<f32, 8>,
            }

            @vertex
            fn vs_main() -> VertexOutput {
                var out: VertexOutput;
                return out;
            }
        "#:
        Err(
            naga::valid::ValidationError::EntryPoint {
                stage: naga::ShaderStage::Vertex,
                source: naga::valid::EntryPointError::Result(
                    naga::valid::VaryingError::UnsupportedCapability(Capabilities::CLIP_DISTANCE),
                ),
                ..
            },
        )
    }

    // Missing enable directive.
    // Note that this is a parsing error, not a validation error.
    check(
        r#"
            @vertex
            fn vs_main() -> @builtin(clip_distances) array<f32, 8> {
                var out: array<f32, 8>;
                return out;
            }
        "#,
        r###"error: the `clip_distances` enable extension is not enabled
  ┌─ wgsl:3:38
  │
3 │             fn vs_main() -> @builtin(clip_distances) array<f32, 8> {
  │                                      ^^^^^^^^^^^^^^ the `clip_distances` "Enable Extension" is needed for this functionality, but it is not currently enabled.
  │
  = note: You can enable this extension by adding `enable clip_distances;` at the top of the shader, before any other items.

"###,
    );

    // Maximum clip distances exceeded
    check_validation! {
        r#"
            enable clip_distances;
            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @builtin(clip_distances) clip_distances: array<f32, 9>,
            }

            @vertex
            fn vs_main() -> VertexOutput {
                var out: VertexOutput;
                return out;
            }
        "#:
        Err(naga::valid::ValidationError::EntryPoint {
            stage: naga::ShaderStage::Vertex,
            source: naga::valid::EntryPointError::Result(
                naga::valid::VaryingError::InvalidBuiltInType(naga::ir::BuiltIn::ClipDistance)
            ),
            ..
        }),
        naga::valid::Capabilities::CLIP_DISTANCE
    }
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_large_array() {
    // The total size of an array is not resolved until validation. Type aliases
    // don't get spans so the error isn't very helpful.
    check_validation! {
        "alias LargeArray = array<u32, (1 << 28) + 1>;":
        Err(naga::valid::ValidationError::Layouter(
                naga::proc::LayoutError {
                    inner: naga::proc::LayoutErrorInner::TooLarge,
                    ..
                }
        ))
    }
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_array_of_arrays() {
    // If the size of the base type of an array is oversize, the error is raised
    // during lowering. Anonymous types don't get spans so this error isn't very
    // helpful.
    check(
        "alias ArrayOfArrays = array<array<u32, (1 << 28) + 1>, 22>;",
        r#"error: type is too large
 = note: the maximum size is 1073741824 bytes

"#,
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_override_array() {
    // The validation that occurs after override processing should reject any
    // arrays that were overridden to be larger than the maximum size. Type
    // aliases don't get spans so the error isn't very helpful.
    let source = r#"
            override SIZE: u32 = 1;
            alias ArrayOfOverrideArrays = array<u32, SIZE>;

            var<workgroup> global: ArrayOfOverrideArrays;

            @compute @workgroup_size(64)
            fn main() {
                let used = &global;
            }
        "#;
    let module = naga::front::wgsl::parse_str(source).expect("module should parse");
    let info = valid::Validator::new(Default::default(), valid::Capabilities::all())
        .validate(&module)
        .expect("module should validate");

    let overrides = hashbrown::HashMap::from([(String::from("SIZE"), f64::from((1 << 28) + 1))]);
    let err = naga::back::pipeline_constants::process_overrides(&module, &info, None, &overrides)
        .unwrap_err();
    let naga::back::pipeline_constants::PipelineConstantError::ValidationError(err) = err else {
        panic!("expected a validation error, got {err:?}");
    };
    assert!(matches!(
        err.into_inner(),
        naga::valid::ValidationError::Layouter(naga::proc::LayoutError {
            inner: naga::proc::LayoutErrorInner::TooLarge,
            ..
        }),
    ));
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_array_in_struct() {
    // If a struct member is oversize, the error is raised during lowering.
    // For struct members we can associate the error with the member.
    check(
        r#"
            struct ContainsLargeArray {
                arr: array<u32, (1 << 28) + 1>,
            }
        "#,
        r#"error: struct member is too large
  ┌─ wgsl:3:17
  │
3 │                 arr: array<u32, (1 << 28) + 1>,
  │                 ^^^ this member exceeds the maximum size
  │
  = note: the maximum size is 1073741824 bytes

"#,
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_two_arrays_in_struct() {
    // The total size of a struct is checked during lowering. For a struct,
    // we can associate the error with the struct itself.
    check(
        r#"
            struct TwoArrays {
                arr1: array<u32, 1 << 27>,
                arr2: array<u32, (1 << 27) + 1>,
            }
        "#,
        "error: type is too large
  ┌─ wgsl:2:13
  │\x20\x20
2 │ ╭             struct TwoArrays {
3 │ │                 arr1: array<u32, 1 << 27>,
4 │ │                 arr2: array<u32, (1 << 27) + 1>,
  │ ╰───────────────────────────────────────────────^ this type exceeds the maximum size
  │\x20\x20
  = note: the maximum size is 1073741824 bytes

",
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn max_type_size_array_of_structs() {
    // The total size of an array is not resolved until validation. Type aliases
    // don't get spans so the error isn't very helpful.
    check_validation! {
        r#"
            struct NotVeryBigStruct {
                data: u32,
            }
            alias BigArrayOfStructs = array<NotVeryBigStruct, (1 << 28) + 1>;
        "#:
        Err(naga::valid::ValidationError::Layouter(
                naga::proc::LayoutError {
                    inner: naga::proc::LayoutErrorInner::TooLarge,
                    ..
                }
        ))
    }
}
