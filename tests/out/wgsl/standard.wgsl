@fragment 
fn derivatives(@builtin(position) foo: vec4<f32>) -> @location(0) vec4<f32> {
    var x: vec4<f32>;
    var y: vec4<f32>;
    var z: vec4<f32>;

    let _e1 = dpdxCoarse(foo);
    x = _e1;
    let _e3 = dpdyCoarse(foo);
    y = _e3;
    let _e5 = fwidthCoarse(foo);
    z = _e5;
    let _e7 = dpdxFine(foo);
    x = _e7;
    let _e8 = dpdyFine(foo);
    y = _e8;
    let _e9 = fwidthFine(foo);
    z = _e9;
    let _e10 = dpdx(foo);
    x = _e10;
    let _e11 = dpdy(foo);
    y = _e11;
    let _e12 = fwidth(foo);
    z = _e12;
    let _e13 = x;
    let _e14 = y;
    let _e16 = z;
    return ((_e13 + _e14) * _e16);
}
