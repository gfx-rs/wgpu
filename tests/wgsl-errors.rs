/*!
Tests for the WGSL front end.
*/
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
fn reserved_identifier_prefix() {
    check(
        "var __bad;",
        r###"error: Identifier starts with a reserved prefix: '__bad'
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
            @group(0) @binding(0) var sampler1 : sampler;

            @fragment
            fn main() -> @location(0) vec4<f32> {
                let a = 3;
                return textureSample(a, sampler1, vec2<f32>(0.0));
            }
        "#,
        r#"error: expected an image, but found 'a' which is not an image
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
fn type_not_inferrable() {
    check(
        r#"
            fn x() {
                _ = vec2();
            }
        "#,
        r#"error: type can't be inferred
  ┌─ wgsl:3:21
  │
3 │                 _ = vec2();
  │                     ^^^^ type can't be inferred

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
  ┌─ wgsl:3:27
  │
3 │                 _ = i32(0, 1);
  │                           ^^ unexpected components

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
        r#"error: invalid type for constructor component at index [0]
  ┌─ wgsl:3:33
  │
3 │                 _ = mat2x2<f32>(array(0, 1), vec2(2, 3));
  │                                 ^^^^^^^^^^^ invalid component type

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
            @group(0) @binding(0) var<bad> texture: texture_2d<f32>;
        "#,
        r#"error: unknown address space: 'bad'
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
        r#"error: unknown attribute: 'a'
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
        r#"error: unknown builtin: 'unknown_built_in'
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
        r#"error: unknown access: 'unknown_access'
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
            let storage1: texture_storage_1d<rgba>;
        "#,
        r#"error: unknown storage format: 'rgba'
  ┌─ wgsl:2:46
  │
2 │             let storage1: texture_storage_1d<rgba>;
  │                                              ^^^^ unknown storage format

"#,
    );
}

#[test]
fn unknown_conservative_depth() {
    check(
        r#"
            @early_depth_test(abc) fn main() {}
        "#,
        r#"error: unknown conservative depth: 'abc'
  ┌─ wgsl:2:31
  │
2 │             @early_depth_test(abc) fn main() {}
  │                               ^^^ unknown conservative depth

"#,
    );
}

#[test]
fn struct_member_zero_size() {
    check(
        r#"
            struct Bar {
                @size(0) data: array<f32>
            }
        "#,
        r#"error: struct member size or alignment must not be 0
  ┌─ wgsl:3:23
  │
3 │                 @size(0) data: array<f32>
  │                       ^ struct member size or alignment must not be 0

"#,
    );
}

#[test]
fn struct_member_zero_align() {
    check(
        r#"
            struct Bar {
                @align(0) data: array<f32>
            }
        "#,
        r#"error: struct member size or alignment must not be 0
  ┌─ wgsl:3:24
  │
3 │                 @align(0) data: array<f32>
  │                        ^ struct member size or alignment must not be 0

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
        r#"error: the type of `x` is expected to be `f32`
  ┌─ wgsl:2:17
  │
2 │             let x: i32 = 1.0;
  │                 ^ definition of `x`

"#,
    );

    check(
        r#"
            fn foo() {
                let x: f32 = true;
            }
        "#,
        r#"error: the type of `x` is expected to be `bool`
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
            let x: f32 = 1;
        "#,
        r#"error: the type of `x` is expected to be `i32`
  ┌─ wgsl:2:17
  │
2 │             let x: f32 = 1;
  │                 ^ definition of `x`

"#,
    );

    check(
        r#"
            fn foo() {
                var x: f32 = 1u32;
            }
        "#,
        r#"error: the type of `x` is expected to be `u32`
  ┌─ wgsl:3:21
  │
3 │                 var x: f32 = 1u32;
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
            struct S { m: i32 };
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

#[test]
fn reserved_keyword() {
    // global var
    check(
        r#"
            var bool: bool = true;
        "#,
        r###"error: name ` bool: bool = true;` is a reserved keyword
  ┌─ wgsl:2:16
  │
2 │             var bool: bool = true;
  │                ^^^^^^^^^^^^^^^^^^^ definition of ` bool: bool = true;`

"###,
    );

    // global constant
    check(
        r#"
            let break: bool = true;
            fn foo() {
                var foo = break;
            }
        "#,
        r###"error: name `break` is a reserved keyword
  ┌─ wgsl:2:17
  │
2 │             let break: bool = true;
  │                 ^^^^^ definition of `break`

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
    // let
    check(
        r#"
            let foo: bool = true;
            let foo: bool = true;
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:17
  │
2 │             let foo: bool = true;
  │                 ^^^ previous definition of `foo`
3 │             let foo: bool = true;
  │                 ^^^ redefinition of `foo`

"###,
    );
    // var
    check(
        r#"
            var foo: bool = true;
            var foo: bool = true;
        "#,
        r###"error: redefinition of ` foo: bool = true;`
  ┌─ wgsl:2:16
  │
2 │             var foo: bool = true;
  │                ^^^^^^^^^^^^^^^^^^ previous definition of ` foo: bool = true;`
3 │             var foo: bool = true;
  │                ^^^^^^^^^^^^^^^^^^ redefinition of ` foo: bool = true;`

"###,
    );

    // let and var
    check(
        r#"
            var foo: bool = true;
            let foo: bool = true;
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:16
  │
2 │             var foo: bool = true;
  │                ^^^^^^^^^^^^^^^^^^ previous definition of ` foo: bool = true;`
3 │             let foo: bool = true;
  │                 ^^^ redefinition of `foo`

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
            let foo: bool = true;
            fn foo() {}
        "#,
        r###"error: redefinition of `foo`
  ┌─ wgsl:2:17
  │
2 │             let foo: bool = true;
  │                 ^^^ previous definition of `foo`
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
                let m: mat3x3<i32>;
            }
        "#,
        r#"error: matrix scalar type must be floating-point, but found `i32`
  ┌─ wgsl:3:31
  │
3 │                 let m: mat3x3<i32>;
  │                               ^^^ must be floating-point (e.g. `f32`)

"#,
    );
}

/// Check the result of validating a WGSL program against a pattern.
///
/// Unless you are generating code programmatically, the
/// `check_validation_error` macro will probably be more convenient to
/// use.
macro_rules! check_one_validation {
    ( $source:expr, $pattern:pat $( if $guard:expr )? ) => {
        let source = $source;
        let error = validation_error($source);
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
    ( $( $source:literal ),* : $pattern:pat if $guard:expr ) => {
        $(
            check_one_validation!($source, $pattern if $guard);
        )*
    }
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
    .map_err(|e| e.into_inner()) // TODO: Add tests for spans, too?
}

#[test]
fn invalid_arrays() {
    check_validation! {
        "type Bad = array<array<f32>, 4>;",
        "type Bad = array<sampler, 4>;",
        "type Bad = array<texture_2d<f32>, 4>;":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidArrayBaseType(_),
            ..
        })
    }

    check_validation! {
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

    check_validation! {
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
    check_validation! {
        "struct Bad { data: sampler }",
        "struct Bad { data: texture_2d<f32> }":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidData(_),
            ..
        })
    }

    check_validation! {
        "struct Bad { data: array<f32>, other: f32, }":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::InvalidDynamicArray(_, _),
            ..
        })
    }

    check_validation! {
        "struct Empty {}":
        Err(naga::valid::ValidationError::Type {
            error: naga::valid::TypeError::EmptyStruct,
            ..
        })
    }
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
            error: naga::valid::FunctionError::InvalidArgumentType {
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
            error: naga::valid::TypeError::InvalidPointerToUnsized {
                base: _,
                space: naga::AddressSpace::WorkGroup { .. },
            },
            ..
        })
    }

    // Pointers of these storage classes cannot be passed as arguments.
    check_validation! {
        "fn unacceptable_ptr_space(arg: ptr<storage, array<f32>>) { }":
        Err(naga::valid::ValidationError::Function {
            name: function_name,
            error: naga::valid::FunctionError::InvalidArgumentPointerSpace {
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
            error: naga::valid::FunctionError::InvalidArgumentPointerSpace {
                index: 0,
                name: argument_name,
                space: naga::AddressSpace::Uniform,
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
            error: naga::valid::FunctionError::NonConstructibleReturnType,
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
            error: naga::valid::FunctionError::NonConstructibleReturnType,
            ..
        })
        if function_name == "return_atomic"
    }
}

#[test]
fn pointer_type_equivalence() {
    check_validation! {
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
    check_validation! {
        "
        @vertex
        fn vertex(_input: vec4<f32>) -> @location(0) vec4<f32> {
           return _input;
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

    check_validation! {
        "
        @vertex
        fn vertex(@location(0) _input: vec4<f32>, more_input: f32) -> @location(0) vec4<f32> {
           return _input + more_input;
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

    check_validation! {
        "
        @vertex
        fn vertex(@location(0) _input: vec4<f32>) -> vec4<f32> {
           return _input;
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

    check_validation! {
        "
        struct VertexIn {
          @location(0) pos: vec4<f32>,
          uv: vec2<f32>
        }

        @vertex
        fn vertex(_input: VertexIn) -> @location(0) vec4<f32> {
           return _input.pos;
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
    check_validation! {
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

    check_validation! {
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
    check_validation! {
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
    check_validation! {
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
            error: naga::valid::TypeError::InvalidDynamicArray(member_name, _),
            ..
        })
        if struct_name == "Outer" && member_name == "_unsized"
    }
}

#[test]
fn select() {
    check_validation! {
        "
        fn select_pointers(which: bool) -> i32 {
            var x: i32 = 1;
            var y: i32 = 2;
            let p = select(&x, &y, which);
            return *p;
        }
        ",
        "
        fn select_arrays(which: bool) -> i32 {
            var x: array<i32, 4>;
            var y: array<i32, 4>;
            let s = select(x, y, which);
            return s[0];
        }
        ",
        "
        struct S { member: i32 }
        fn select_structs(which: bool) -> S {
            var x: S = S(1);
            var y: S = S(2);
            let s = select(x, y, which);
            return s;
        }
        ":
        Err(
            naga::valid::ValidationError::Function {
                name,
                error: naga::valid::FunctionError::Expression {
                    error: naga::valid::ExpressionError::InvalidSelectTypes,
                    ..
                },
                ..
            },
        )
        if name.starts_with("select_")
    }
}

#[test]
fn last_case_falltrough() {
    check_validation! {
        "
        fn test_falltrough() {
          switch(0) {
            default: {}
            case 0: {
              fallthrough;
            }
          }
        }
        ":
        Err(
            naga::valid::ValidationError::Function {
                error: naga::valid::FunctionError::LastCaseFallTrough,
                ..
            },
        )
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
                error: naga::valid::FunctionError::MissingDefaultCase,
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
                error: naga::valid::FunctionError::InvalidStorePointer(_),
                ..
            },
        )
            if name == "store"
    }
}

#[test]
fn io_shareable_types() {
    for numeric in "i32 u32 f32".split_whitespace() {
        let types = format!(
            "{} vec2<{}> vec3<{}> vec4<{}>",
            numeric, numeric, numeric, numeric
        );
        for ty in types.split_whitespace() {
            check_one_validation! {
                &format!("@vertex
                          fn f(@location(0) arg: {}) -> @builtin(position) vec4<f32>
                          {{ return vec4<f32>(0.0); }}",
                         ty),
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
                          fn f(@location(0) arg: {}) -> @builtin(position) vec4<f32>
                          {{ return vec4<f32>(0.0); }}",
                     ty),
            Err(
                naga::valid::ValidationError::EntryPoint {
                    stage: naga::ShaderStage::Vertex,
                    name,
                    error: naga::valid::EntryPointError::Argument(
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
                      @group(0) @binding(0) var<uniform> ubuf: {};
                      @group(0) @binding(1) var<storage> sbuf: {};",
                     ty, ty),
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
                      @group(0) @binding(1) var<storage> sbuf: {};",
                     ty),
            Ok(_module)
        }
    }

    // Types that are neither host-shareable nor constructible.
    for ty in "bool ptr<storage,i32>".split_whitespace() {
        check_one_validation! {
            &format!("@group(0) @binding(0) var<storage> sbuf: {};", ty),
            Err(
                naga::valid::ValidationError::GlobalVariable {
                    name,
                    handle: _,
                    error: naga::valid::GlobalVariableError::MissingTypeFlags { .. },
                },
            )
            if name == "sbuf"
        }

        check_one_validation! {
            &format!("@group(0) @binding(0) var<uniform> ubuf: {};", ty),
            Err(naga::valid::ValidationError::GlobalVariable {
                    name,
                    handle: _,
                    error: naga::valid::GlobalVariableError::MissingTypeFlags { .. },
                },
            )
            if name == "ubuf"
        }
    }
}
