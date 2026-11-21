struct Foo { a: u32 };
struct Bar { a: u32 };

var<uniform> foo: Foo = Bar(1);
