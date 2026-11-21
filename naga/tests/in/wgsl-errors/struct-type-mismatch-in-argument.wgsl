struct Foo { a: u32 };
struct Bar { a: u32 };
fn bar(a: Bar) {}
fn main() {
    bar(Foo(1));
}
