#[cfg(feature = "wgsl-in")]
macro_rules! err {
    ($value:expr, @$snapshot:literal) => {
        ::insta::assert_snapshot!(naga::front::wgsl::parse_str($value).expect_err("expected parser error").to_string(), @$snapshot);
    };
}

#[cfg(feature = "wgsl-in")]
#[test]
fn function_without_identifier() {
    err!(
        "fn () {}",
        @"error while parsing WGSL in scopes [FunctionDecl] at line 1 pos 4: unexpected token Paren('('), expected ident"
    );
}
