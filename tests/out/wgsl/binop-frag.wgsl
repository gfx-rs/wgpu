var<private> o_color: vec4<f32>;

fn testBinOpVecFloat(a: vec4<f32>, b: f32) {
    var a1: vec4<f32>;
    var b1: f32;
    var v: vec4<f32>;

    a1 = a;
    b1 = b;
    let _e5: vec4<f32> = a1;
    v = (_e5 * 2.0);
    let _e8: vec4<f32> = a1;
    v = (_e8 / vec4<f32>(2.0));
    let _e12: vec4<f32> = a1;
    v = (_e12 + vec4<f32>(2.0));
    let _e16: vec4<f32> = a1;
    v = (_e16 - vec4<f32>(2.0));
    return;
}

fn testBinOpFloatVec(a2: vec4<f32>, b2: f32) {
    var a3: vec4<f32>;
    var b3: f32;
    var v1: vec4<f32>;

    a3 = a2;
    b3 = b2;
    let _e5: vec4<f32> = a3;
    let _e6: f32 = b3;
    v1 = (_e5 * _e6);
    let _e8: vec4<f32> = a3;
    let _e9: f32 = b3;
    v1 = (_e8 / vec4<f32>(_e9));
    let _e12: vec4<f32> = a3;
    let _e13: f32 = b3;
    v1 = (_e12 + vec4<f32>(_e13));
    let _e16: vec4<f32> = a3;
    let _e17: f32 = b3;
    v1 = (_e16 - vec4<f32>(_e17));
    return;
}

fn testBinOpIVecInt(a4: vec4<i32>, b4: i32) {
    var a5: vec4<i32>;
    var b5: i32;
    var v2: vec4<i32>;

    a5 = a4;
    b5 = b4;
    let _e5: vec4<i32> = a5;
    let _e6: i32 = b5;
    v2 = (_e5 * _e6);
    let _e8: vec4<i32> = a5;
    let _e9: i32 = b5;
    v2 = (_e8 / vec4<i32>(_e9));
    let _e12: vec4<i32> = a5;
    let _e13: i32 = b5;
    v2 = (_e12 + vec4<i32>(_e13));
    let _e16: vec4<i32> = a5;
    let _e17: i32 = b5;
    v2 = (_e16 - vec4<i32>(_e17));
    let _e20: vec4<i32> = a5;
    let _e21: i32 = b5;
    v2 = (_e20 & vec4<i32>(_e21));
    let _e24: vec4<i32> = a5;
    let _e25: i32 = b5;
    v2 = (_e24 | vec4<i32>(_e25));
    let _e28: vec4<i32> = a5;
    let _e29: i32 = b5;
    v2 = (_e28 ^ vec4<i32>(_e29));
    let _e32: vec4<i32> = a5;
    let _e33: i32 = b5;
    v2 = (_e32 >> vec4<u32>(u32(_e33)));
    let _e37: vec4<i32> = a5;
    let _e38: i32 = b5;
    v2 = (_e37 << vec4<u32>(u32(_e38)));
    return;
}

fn testBinOpIntIVec(a6: i32, b6: vec4<i32>) {
    var a7: i32;
    var b7: vec4<i32>;
    var v3: vec4<i32>;

    a7 = a6;
    b7 = b6;
    let _e5: i32 = a7;
    let _e6: vec4<i32> = b7;
    v3 = (_e5 * _e6);
    let _e8: i32 = a7;
    let _e9: vec4<i32> = b7;
    v3 = (vec4<i32>(_e8) + _e9);
    let _e12: i32 = a7;
    let _e13: vec4<i32> = b7;
    v3 = (vec4<i32>(_e12) - _e13);
    let _e16: i32 = a7;
    let _e17: vec4<i32> = b7;
    v3 = (vec4<i32>(_e16) & _e17);
    let _e20: i32 = a7;
    let _e21: vec4<i32> = b7;
    v3 = (vec4<i32>(_e20) | _e21);
    let _e24: i32 = a7;
    let _e25: vec4<i32> = b7;
    v3 = (vec4<i32>(_e24) ^ _e25);
    return;
}

fn testBinOpUVecUint(a8: vec4<u32>, b8: u32) {
    var a9: vec4<u32>;
    var b9: u32;
    var v4: vec4<u32>;

    a9 = a8;
    b9 = b8;
    let _e5: vec4<u32> = a9;
    let _e6: u32 = b9;
    v4 = (_e5 * _e6);
    let _e8: vec4<u32> = a9;
    let _e9: u32 = b9;
    v4 = (_e8 / vec4<u32>(_e9));
    let _e12: vec4<u32> = a9;
    let _e13: u32 = b9;
    v4 = (_e12 + vec4<u32>(_e13));
    let _e16: vec4<u32> = a9;
    let _e17: u32 = b9;
    v4 = (_e16 - vec4<u32>(_e17));
    let _e20: vec4<u32> = a9;
    let _e21: u32 = b9;
    v4 = (_e20 & vec4<u32>(_e21));
    let _e24: vec4<u32> = a9;
    let _e25: u32 = b9;
    v4 = (_e24 | vec4<u32>(_e25));
    let _e28: vec4<u32> = a9;
    let _e29: u32 = b9;
    v4 = (_e28 ^ vec4<u32>(_e29));
    let _e32: vec4<u32> = a9;
    let _e33: u32 = b9;
    v4 = (_e32 >> vec4<u32>(_e33));
    let _e36: vec4<u32> = a9;
    let _e37: u32 = b9;
    v4 = (_e36 << vec4<u32>(_e37));
    return;
}

fn testBinOpUintUVec(a10: u32, b10: vec4<u32>) {
    var a11: u32;
    var b11: vec4<u32>;
    var v5: vec4<u32>;

    a11 = a10;
    b11 = b10;
    let _e5: u32 = a11;
    let _e6: vec4<u32> = b11;
    v5 = (_e5 * _e6);
    let _e8: u32 = a11;
    let _e9: vec4<u32> = b11;
    v5 = (vec4<u32>(_e8) + _e9);
    let _e12: u32 = a11;
    let _e13: vec4<u32> = b11;
    v5 = (vec4<u32>(_e12) - _e13);
    let _e16: u32 = a11;
    let _e17: vec4<u32> = b11;
    v5 = (vec4<u32>(_e16) & _e17);
    let _e20: u32 = a11;
    let _e21: vec4<u32> = b11;
    v5 = (vec4<u32>(_e20) | _e21);
    let _e24: u32 = a11;
    let _e25: vec4<u32> = b11;
    v5 = (vec4<u32>(_e24) ^ _e25);
    return;
}

fn main1() {
    let _e1: vec4<f32> = o_color;
    let _e4: vec4<f32> = vec4<f32>(1.0);
    o_color.x = _e4.x;
    o_color.y = _e4.y;
    o_color.z = _e4.z;
    o_color.w = _e4.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}
