// Tests that ensure that various constructs that should not compile do not compile.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("compile_tests/fail/*.rs");
}
