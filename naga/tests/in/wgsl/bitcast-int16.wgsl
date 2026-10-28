enable wgpu_int16;

@compute @workgroup_size(1)
fn main() {
    var i2 = vec2<i16>(i16(1), i16(2));
    var u2 = vec2<u16>(u16(1), u16(2));
    var i4 = vec4<i16>(i16(1), i16(2), i16(3), i16(4));
    var u4 = vec4<u16>(u16(1), u16(2), u16(3), u16(4));

    var i1 = 0i;
    var u1 = 0u;
    var f1 = 0.0f;
    var iv2 = vec2<i32>(0);
    var uv2 = vec2<u32>(0u);
    var fv2 = vec2<f32>(0.0);

    i1 = bitcast<i32>(i2);
    u1 = bitcast<u32>(u2);
    f1 = bitcast<f32>(u2);
    uv2 = bitcast<vec2<u32>>(i4);
    fv2 = bitcast<vec2<f32>>(u4);

    i2 = bitcast<vec2<i16>>(u1);
    u2 = bitcast<vec2<u16>>(f1);
    i4 = bitcast<vec4<i16>>(fv2);
    u4 = bitcast<vec4<u16>>(iv2);
}
