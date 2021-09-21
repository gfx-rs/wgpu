struct BST {
    data: i32;
};

var<private> o_color: vec4<f32>;

fn testBinOpVecFloat(a: vec4<f32>, b: f32) {
    var a1: vec4<f32>;
    var b1: f32;
    var v: vec4<f32>;

    a1 = a;
    b1 = b;
    let e5: vec4<f32> = a1;
    v = (e5 * 2.0);
    let e8: vec4<f32> = a1;
    v = (e8 / vec4<f32>(2.0));
    let e12: vec4<f32> = a1;
    v = (e12 + vec4<f32>(2.0));
    let e16: vec4<f32> = a1;
    v = (e16 - vec4<f32>(2.0));
    return;
}

fn testBinOpFloatVec(a2: vec4<f32>, b2: f32) {
    var a3: vec4<f32>;
    var b3: f32;
    var v1: vec4<f32>;

    a3 = a2;
    b3 = b2;
    let e5: vec4<f32> = a3;
    let e6: f32 = b3;
    v1 = (e5 * e6);
    let e8: vec4<f32> = a3;
    let e9: f32 = b3;
    v1 = (e8 / vec4<f32>(e9));
    let e12: vec4<f32> = a3;
    let e13: f32 = b3;
    v1 = (e12 + vec4<f32>(e13));
    let e16: vec4<f32> = a3;
    let e17: f32 = b3;
    v1 = (e16 - vec4<f32>(e17));
    return;
}

fn testBinOpIVecInt(a4: vec4<i32>, b4: i32) {
    var a5: vec4<i32>;
    var b5: i32;
    var v2: vec4<i32>;

    a5 = a4;
    b5 = b4;
    let e5: vec4<i32> = a5;
    let e6: i32 = b5;
    v2 = (e5 * e6);
    let e8: vec4<i32> = a5;
    let e9: i32 = b5;
    v2 = (e8 / vec4<i32>(e9));
    let e12: vec4<i32> = a5;
    let e13: i32 = b5;
    v2 = (e12 + vec4<i32>(e13));
    let e16: vec4<i32> = a5;
    let e17: i32 = b5;
    v2 = (e16 - vec4<i32>(e17));
    let e20: vec4<i32> = a5;
    let e21: i32 = b5;
    v2 = (e20 & vec4<i32>(e21));
    let e24: vec4<i32> = a5;
    let e25: i32 = b5;
    v2 = (e24 | vec4<i32>(e25));
    let e28: vec4<i32> = a5;
    let e29: i32 = b5;
    v2 = (e28 ^ vec4<i32>(e29));
    let e32: vec4<i32> = a5;
    let e33: i32 = b5;
    v2 = (e32 >> vec4<u32>(u32(e33)));
    let e37: vec4<i32> = a5;
    let e38: i32 = b5;
    v2 = (e37 << vec4<u32>(u32(e38)));
    return;
}

fn testBinOpIntIVec(a6: i32, b6: vec4<i32>) {
    var a7: i32;
    var b7: vec4<i32>;
    var v3: vec4<i32>;

    a7 = a6;
    b7 = b6;
    let e5: i32 = a7;
    let e6: vec4<i32> = b7;
    v3 = (e5 * e6);
    let e8: i32 = a7;
    let e9: vec4<i32> = b7;
    v3 = (vec4<i32>(e8) + e9);
    let e12: i32 = a7;
    let e13: vec4<i32> = b7;
    v3 = (vec4<i32>(e12) - e13);
    let e16: i32 = a7;
    let e17: vec4<i32> = b7;
    v3 = (vec4<i32>(e16) & e17);
    let e20: i32 = a7;
    let e21: vec4<i32> = b7;
    v3 = (vec4<i32>(e20) | e21);
    let e24: i32 = a7;
    let e25: vec4<i32> = b7;
    v3 = (vec4<i32>(e24) ^ e25);
    return;
}

fn testBinOpUVecUint(a8: vec4<u32>, b8: u32) {
    var a9: vec4<u32>;
    var b9: u32;
    var v4: vec4<u32>;

    a9 = a8;
    b9 = b8;
    let e5: vec4<u32> = a9;
    let e6: u32 = b9;
    v4 = (e5 * e6);
    let e8: vec4<u32> = a9;
    let e9: u32 = b9;
    v4 = (e8 / vec4<u32>(e9));
    let e12: vec4<u32> = a9;
    let e13: u32 = b9;
    v4 = (e12 + vec4<u32>(e13));
    let e16: vec4<u32> = a9;
    let e17: u32 = b9;
    v4 = (e16 - vec4<u32>(e17));
    let e20: vec4<u32> = a9;
    let e21: u32 = b9;
    v4 = (e20 & vec4<u32>(e21));
    let e24: vec4<u32> = a9;
    let e25: u32 = b9;
    v4 = (e24 | vec4<u32>(e25));
    let e28: vec4<u32> = a9;
    let e29: u32 = b9;
    v4 = (e28 ^ vec4<u32>(e29));
    let e32: vec4<u32> = a9;
    let e33: u32 = b9;
    v4 = (e32 >> vec4<u32>(e33));
    let e36: vec4<u32> = a9;
    let e37: u32 = b9;
    v4 = (e36 << vec4<u32>(e37));
    return;
}

fn testBinOpUintUVec(a10: u32, b10: vec4<u32>) {
    var a11: u32;
    var b11: vec4<u32>;
    var v5: vec4<u32>;

    a11 = a10;
    b11 = b10;
    let e5: u32 = a11;
    let e6: vec4<u32> = b11;
    v5 = (e5 * e6);
    let e8: u32 = a11;
    let e9: vec4<u32> = b11;
    v5 = (vec4<u32>(e8) + e9);
    let e12: u32 = a11;
    let e13: vec4<u32> = b11;
    v5 = (vec4<u32>(e12) - e13);
    let e16: u32 = a11;
    let e17: vec4<u32> = b11;
    v5 = (vec4<u32>(e16) & e17);
    let e20: u32 = a11;
    let e21: vec4<u32> = b11;
    v5 = (vec4<u32>(e20) | e21);
    let e24: u32 = a11;
    let e25: vec4<u32> = b11;
    v5 = (vec4<u32>(e24) ^ e25);
    return;
}

fn testStructConstructor() {
    var tree: BST = BST(1);

}

fn main1() {
    let e1: vec4<f32> = o_color;
    let e4: vec4<f32> = vec4<f32>(1.0);
    o_color.x = e4.x;
    o_color.y = e4.y;
    o_color.z = e4.z;
    o_color.w = e4.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}
