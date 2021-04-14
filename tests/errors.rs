#[cfg(feature = "wgsl-in")]
fn check(input: &str, snapshot: &str) {
    let output = naga::front::wgsl::parse_str(input)
        .expect_err("expected parser error")
        .emit_to_string();
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

#[cfg(feature = "wgsl-in")]
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

#[cfg(feature = "wgsl-in")]
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

#[cfg(feature = "wgsl-in")]
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

#[cfg(feature = "wgsl-in")]
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

#[cfg(feature = "wgsl-in")]
#[test]
fn invalid_accessor() {
    check(
        r###"
[[stage(vertex)]]
fn vs_main() {
    var color: vec3<f32> = vec3<f32>(1.0, 2.0, 3.0);
    var i: f32 = color.a;
}
"###,
        r###"error: invalid field accessor `a`
  ┌─ wgsl:5:24
  │
5 │     var i: f32 = color.a;
  │                        ^ invalid accessor

"###,
    );
}
