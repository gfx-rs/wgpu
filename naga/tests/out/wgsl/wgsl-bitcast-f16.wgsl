enable f16;

const bitcast_vec2_to_scalar: u32 = 1006648320u;
const bitcast_vec2_to_vec4_: vec4<f16> = vec4<f16>(0.000000059604645h, 0h, 0.000000059604645h, 0h);
const bitcast_vec4_to_vec2_: vec2<u32> = vec2<u32>(1006648320u, 1006648320u);

@compute @workgroup_size(1, 1, 1)
fn main() {
    var h2_: vec2<f16> = vec2<f16>(1h, 2h);
    var h4_: vec4<f16> = vec4<f16>(1h, 2h, 3h, 4h);
    var i1_: i32 = 0i;
    var u1_: u32 = 0u;
    var f1_: f32 = 0f;
    var i2_: vec2<i32> = vec2(0i);
    var u2_: vec2<u32> = vec2(0u);
    var f2_: vec2<f32> = vec2(0f);

    let _e29 = h2_;
    i1_ = bitcast<i32>(_e29);
    let _e31 = h2_;
    u1_ = bitcast<u32>(_e31);
    let _e33 = h2_;
    f1_ = bitcast<f32>(_e33);
    let _e35 = h4_;
    i2_ = bitcast<vec2<i32>>(_e35);
    let _e37 = h4_;
    u2_ = bitcast<vec2<u32>>(_e37);
    let _e39 = h4_;
    f2_ = bitcast<vec2<f32>>(_e39);
    let _e41 = i1_;
    h2_ = bitcast<vec2<f16>>(_e41);
    let _e43 = u1_;
    h2_ = bitcast<vec2<f16>>(_e43);
    let _e45 = f1_;
    h2_ = bitcast<vec2<f16>>(_e45);
    let _e47 = i2_;
    h4_ = bitcast<vec4<f16>>(_e47);
    let _e49 = u2_;
    h4_ = bitcast<vec4<f16>>(_e49);
    let _e51 = f2_;
    h4_ = bitcast<vec4<f16>>(_e51);
    let _e53 = h2_;
    h2_ = bitcast<vec2<f16>>(_e53);
    let _e55 = u1_;
    u1_ = bitcast<u32>(_e55);
    u1_ = bitcast<u32>(4294967295u);
    f1_ = bitcast<f32>(1i);
    i1_ = bitcast<i32>(1f);
    h2_ = bitcast<vec2<f16>>(1i);
    h2_ = bitcast<vec2<f16>>(vec2<f16>(1h, 2h));
    h4_ = bitcast<vec4<f16>>(vec2<i32>(1i, 2i));
    u2_ = bitcast<vec2<u32>>(vec2<u32>(4294967295u, 1u));
    return;
}
