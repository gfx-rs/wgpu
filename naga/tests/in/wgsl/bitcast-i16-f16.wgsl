enable f16;
enable wgpu_int16;

@compute @workgroup_size(1)
fn main() {
    var h = 1.0h;
    var i = i16(1);
    var u = u16(2);
    var h4 = vec4<f16>(1.0h, 2.0h, 3.0h, 4.0h);
    var u4 = vec4<u16>(u16(1), u16(2), u16(3), u16(4));

    i = bitcast<i16>(h);
    u = bitcast<u16>(h);
    h = bitcast<f16>(i);
    h = bitcast<f16>(u);
    u4 = bitcast<vec4<u16>>(h4);
    h4 = bitcast<vec4<f16>>(u4);
}
