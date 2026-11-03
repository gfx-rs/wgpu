enable f16;

@compute @workgroup_size(1)
fn main() {
    var h2 = vec2<f16>(1.0h, 2.0h);
    var h4 = vec4<f16>(1.0h, 2.0h, 3.0h, 4.0h);

    var i1 = 0i;
    var u1 = 0u;
    var f1 = 0.0f;
    var i2 = vec2<i32>(0);
    var u2 = vec2<u32>(0u);
    var f2 = vec2<f32>(0.0);

    i1 = bitcast<i32>(h2);
    u1 = bitcast<u32>(h2);
    f1 = bitcast<f32>(h2);

    i2 = bitcast<vec2<i32>>(h4);
    u2 = bitcast<vec2<u32>>(h4);
    f2 = bitcast<vec2<f32>>(h4);

    h2 = bitcast<vec2<f16>>(i1);
    h2 = bitcast<vec2<f16>>(u1);
    h2 = bitcast<vec2<f16>>(f1);

    h4 = bitcast<vec4<f16>>(i2);
    h4 = bitcast<vec4<f16>>(u2);
    h4 = bitcast<vec4<f16>>(f2);

    h2 = bitcast<vec2<f16>>(h2);
    u1 = bitcast<u32>(u1);

    u1 = bitcast<u32>(4294967295);
    f1 = bitcast<f32>(1);
    i1 = bitcast<i32>(1.0);
    h2 = bitcast<vec2<f16>>(1);
    h2 = bitcast<vec2<f16>>(vec2(1, 2));
    h4 = bitcast<vec4<f16>>(vec2(1, 2));
    u2 = bitcast<vec2<u32>>(vec2(4294967295, 1));
}
