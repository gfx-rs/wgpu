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
