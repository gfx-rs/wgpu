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
  = note: valid width is 32

"###,
    );
}

macro_rules! check_validation_error {
    ( $( $source:literal ),* : $pattern:pat ) => {
        $(
            let error = validation_error($source);
            if ! matches!(error, $pattern) {
                eprintln!("validation error does not match pattern:\n\
                        {:?}\n\
                        \n\
                        expected match for pattern:\n\
                        {}",
                       error,
                          stringify!($pattern));
                panic!("validation error does not match pattern");
            }
        )*
    }
}

fn validation_error(source: &str) -> Result<naga::valid::ModuleInfo, naga::valid::ValidationError> {
    let module = naga::front::wgsl::parse_str(source).expect("expected WGSL parse to succeed");
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
            error: naga::valid::TypeError::NestedBlock,
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
