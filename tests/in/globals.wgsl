// Global variable & constant declarations

let Foo: bool = true;

var<workgroup> wg : array<f32, 10u>;
var<workgroup> at: atomic<u32>;

struct Foo {
    v3: vec3<f32>;
    // test packed vec3
    v1: f32;
};
[[group(0), binding(1)]]
var<storage> alignment: Foo;

[[stage(compute), workgroup_size(1)]]
fn main() {
    wg[3] = alignment.v1;
    atomicStore(&at, 2u);

    // Valid, Foo and at is in function scope
    var Foo: f32 = 1.0;
    var at: bool = true;
}
