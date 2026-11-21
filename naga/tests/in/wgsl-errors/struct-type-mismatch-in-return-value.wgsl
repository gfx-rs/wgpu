struct Foo { a: u32 };
struct Bar { a: u32 };
fn bar() -> Bar {
    return Foo(1);
}
