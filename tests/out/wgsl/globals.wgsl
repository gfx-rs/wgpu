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
    var unnamed: vec3<f32>;
    var unnamed_1: vec2<f32>;
    var idx: i32 = 1;
    var Foo_1: f32 = 1.0;
    var at: bool = true;

    let _e9 = alignment.v1_;
    wg[3] = _e9;
    let _e14 = alignment.v3_.x;
    wg[2] = _e14;
    let _e16 = alignment.v3_;
    unnamed = _e16;
    let _e19 = alignment.v3_;
    unnamed_1 = _e19.zx;
    alignment.v1_ = 4.0;
    wg[1] = f32(arrayLength((&dummy)));
    atomicStore((&at_1), 2u);
    alignment.v3_ = vec3<f32>(1.0);
    alignment.v3_.x = 1.0;
    alignment.v3_.x = 2.0;
    let _e42 = idx;
    alignment.v3_[_e42] = 3.0;
    return;
}
