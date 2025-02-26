@compute @workgroup_size(1)
fn main() {
    var i2 = vec2<i32>(0);
    var i3 = vec3<i32>(0);
    var i4 = vec4<i32>(0);

    var u2 = vec2<u32>(0u);
    var u3 = vec3<u32>(0u);
    var u4 = vec4<u32>(0u);

    var f2 = vec2<f32>(0.0);
    var f3 = vec3<f32>(0.0);
    var f4 = vec4<f32>(0.0);

    u2 = bitcast<vec2<u32>>(i2);
    u3 = bitcast<vec3<u32>>(i3);
    u4 = bitcast<vec4<u32>>(i4);

    i2 = bitcast<vec2<i32>>(u2);
    i3 = bitcast<vec3<i32>>(u3);
    i4 = bitcast<vec4<i32>>(u4);

    f2 = bitcast<vec2<f32>>(i2);
    f3 = bitcast<vec3<f32>>(i3);
    f4 = bitcast<vec4<f32>>(i4);
}
