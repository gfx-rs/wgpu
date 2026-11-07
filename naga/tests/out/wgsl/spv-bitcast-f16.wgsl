enable f16;

struct B {
    a: u32,
    h2_: vec2<f16>,
    h4_: vec4<f16>,
    u2_: vec2<u32>,
    i: i32,
    f: f32,
}

@group(0) @binding(0)
var<storage, read_write> buf: B;

fn main_1() {
    let _e7 = buf.a;
    let _e8 = buf.h2_;
    let _e9 = buf.h4_;
    buf.h2_ = bitcast<vec2<f16>>(_e7);
    buf.a = bitcast<u32>(_e8);
    buf.u2_ = bitcast<vec2<u32>>(_e9);
    buf.i = bitcast<i32>(_e7);
    buf.f = bitcast<f32>(_e7);
    let _e15 = buf.u2_;
    buf.h4_ = bitcast<vec4<f16>>(_e15);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    main_1();
}
