enable wgpu_int16;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var i2_: vec2<i16> = vec2<i16>(i16(1), i16(2));
    var u2_: vec2<u16> = vec2<u16>(u16(1), u16(2));
    var i4_: vec4<i16> = vec4<i16>(i16(1), i16(2), i16(3), i16(4));
    var u4_: vec4<u16> = vec4<u16>(u16(1), u16(2), u16(3), u16(4));
    var i1_: i32 = 0i;
    var u1_: u32 = 0u;
    var f1_: f32 = 0f;
    var iv2_: vec2<i32> = vec2(0i);
    var uv2_: vec2<u32> = vec2(0u);
    var fv2_: vec2<f32> = vec2(0f);

    let _e35 = i2_;
    i1_ = bitcast<i32>(_e35);
    let _e37 = u2_;
    u1_ = bitcast<u32>(_e37);
    let _e39 = u2_;
    f1_ = bitcast<f32>(_e39);
    let _e41 = i4_;
    uv2_ = bitcast<vec2<u32>>(_e41);
    let _e43 = u4_;
    fv2_ = bitcast<vec2<f32>>(_e43);
    let _e45 = u1_;
    i2_ = bitcast<vec2<i16>>(_e45);
    let _e47 = f1_;
    u2_ = bitcast<vec2<u16>>(_e47);
    let _e49 = fv2_;
    i4_ = bitcast<vec4<i16>>(_e49);
    let _e51 = iv2_;
    u4_ = bitcast<vec4<u16>>(_e51);
    return;
}
