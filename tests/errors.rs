#[cfg(feature = "wgsl-in")]
macro_rules! err {
    ($value:expr, @$snapshot:literal) => {
        ::insta::assert_snapshot!(
            naga::front::wgsl::parse_str($value)
                .expect_err("expected parser error")
                .emit_to_string(),
            @$snapshot
        );
    };
}

#[cfg(feature = "wgsl-in")]
#[test]
fn function_without_identifier() {
    err!(
        "fn () {}",
        @r###"
    error: expected identifier, found '('
      ┌─ wgsl:1:4
      │
    1 │ fn () {}
      │    ^ expected identifier

    "###
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn invalid_integer() {
    err!(
        "fn foo([location(1.)] x: i32) {}",
        @r###"
    error: expected identifier, found '['
      ┌─ wgsl:1:8
      │
    1 │ fn foo([location(1.)] x: i32) {}
      │        ^ expected identifier

    "###
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn invalid_float() {
    err!(
        "const scale: f32 = 1.1.;",
        @r###"
    error: expected floating-point literal, found `1.1.`
      ┌─ wgsl:1:20
      │
    1 │ const scale: f32 = 1.1.;
      │                    ^^^^ expected floating-point literal

    "###
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn invalid_scalar_width() {
    err!(
        "const scale: f32 = 1.1f1000;",
        @r###"
    error: invalid width of `1000` for literal
      ┌─ wgsl:1:20
      │
    1 │ const scale: f32 = 1.1f1000;
      │                    ^^^^^^^^ invalid width
      │
      = note: valid width is 32

    "###
    );
}

#[cfg(feature = "wgsl-in")]
#[test]
fn invalid_accessor() {
    err!(
        r###"
        [[stage(vertex)]]
        fn vs_main() {
            var color: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
            var i: f32 = color.a;
        }
    "###,
        @r###"
    error: invalid field accessor `a`
      ┌─ wgsl:5:32
      │
    5 │             var i: f32 = color.a;
      │                                ^ invalid accessor

    "###
    );
}
