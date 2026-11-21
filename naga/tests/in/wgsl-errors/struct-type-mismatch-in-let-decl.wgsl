struct Foo { a: u32 };
struct Bar { a: u32 };
fn main() {
    let x: Bar = Foo(1);
}
