struct Foo {
    v3_: vec3<f32>;
    v1_: f32;
};

struct Dummy {
    arr: [[stride(8)]] array<vec2<f32>>;
};

let Foo_2: bool = true;

var<workgroup> wg: array<f32,10u>;
var<workgroup> at_1: atomic<u32>;
[[group(0), binding(1)]]
var<storage> alignment: Foo;
[[group(0), binding(2)]]
var<storage> dummy: Dummy;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    var Foo_1: f32 = 1.0;
    var at: bool = true;

    let _e8 = alignment.v1_;
    wg[3] = _e8;
    let _e13 = alignment.v3_.x;
    wg[2] = _e13;
    atomicStore((&at_1), 2u);
    return;
}
