struct Foo {
    v3_: vec3<f32>,
    v1_: f32,
}

let Foo_2: bool = true;

var<workgroup> wg: array<f32,10u>;
var<workgroup> at_1: atomic<u32>;
@group(0) @binding(1) 
var<storage, read_write> alignment: Foo;
@group(0) @binding(2) 
var<storage> dummy: array<vec2<f32>>;
@group(0) @binding(3) 
var<uniform> float_vecs: array<vec4<f32>,20>;

@stage(compute) @workgroup_size(1, 1, 1) 
fn main() {
    var Foo_1: f32 = 1.0;
    var at: bool = true;

    let _e9 = alignment.v1_;
    wg[3] = _e9;
    let _e14 = alignment.v3_.x;
    wg[2] = _e14;
    alignment.v1_ = 4.0;
    wg[1] = f32(arrayLength((&dummy)));
    atomicStore((&at_1), 2u);
    return;
}
