enable f16;
enable wgpu_int16;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var h: f16 = 1h;
    var i: i16 = i16(1);
    var u: u16 = u16(2);
    var h4_: vec4<f16> = vec4<f16>(1h, 2h, 3h, 4h);
    var u4_: vec4<u16> = vec4<u16>(u16(1), u16(2), u16(3), u16(4));

    let _e18 = h;
    i = bitcast<i16>(_e18);
    let _e20 = h;
    u = bitcast<u16>(_e20);
    let _e22 = i;
    h = bitcast<f16>(_e22);
    let _e24 = u;
    h = bitcast<f16>(_e24);
    let _e26 = h4_;
    u4_ = bitcast<vec4<u16>>(_e26);
    let _e28 = u4_;
    h4_ = bitcast<vec4<f16>>(_e28);
    return;
}
