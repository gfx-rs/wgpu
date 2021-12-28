struct Foo {
    v3_: vec3<f32>;
    v1_: f32;
};

let Foo_2: bool = true;

var<workgroup> wg: array<f32,10u>;
var<workgroup> at_1: atomic<u32>;
[[group(0), binding(1)]]
var<storage> alignment: Foo;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    var Foo_1: f32 = 1.0;
    var at: bool = true;

    let _e7 = alignment.v1_;
    wg[3] = _e7;
    let _e12 = alignment.v3_.x;
    wg[2] = _e12;
    atomicStore((&at_1), 2u);
    return;
}
