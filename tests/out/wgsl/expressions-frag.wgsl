struct BST {
    data: i32;
};

var<private> global: f32;
var<private> o_color: vec4<f32>;

fn testBinOpVecFloat(a: vec4<f32>, b: f32) {
    var a_1: vec4<f32>;
    var b_1: f32;
    var v: vec4<f32>;

    a_1 = a;
    b_1 = b;
    let e5: vec4<f32> = a_1;
    v = (e5 * 2.0);
    let e8: vec4<f32> = a_1;
    v = (e8 / vec4<f32>(2.0));
    let e12: vec4<f32> = a_1;
    v = (e12 + vec4<f32>(2.0));
    let e16: vec4<f32> = a_1;
    v = (e16 - vec4<f32>(2.0));
    return;
}

fn testBinOpFloatVec(a_2: vec4<f32>, b_2: f32) {
    var a_3: vec4<f32>;
    var b_3: f32;
    var v_1: vec4<f32>;

    a_3 = a_2;
    b_3 = b_2;
    let e5: vec4<f32> = a_3;
    let e6: f32 = b_3;
    v_1 = (e5 * e6);
    let e8: vec4<f32> = a_3;
    let e9: f32 = b_3;
    v_1 = (e8 / vec4<f32>(e9));
    let e12: vec4<f32> = a_3;
    let e13: f32 = b_3;
    v_1 = (e12 + vec4<f32>(e13));
    let e16: vec4<f32> = a_3;
    let e17: f32 = b_3;
    v_1 = (e16 - vec4<f32>(e17));
    return;
}

fn testBinOpIVecInt(a_4: vec4<i32>, b_4: i32) {
    var a_5: vec4<i32>;
    var b_5: i32;
    var v_2: vec4<i32>;

    a_5 = a_4;
    b_5 = b_4;
    let e5: vec4<i32> = a_5;
    let e6: i32 = b_5;
    v_2 = (e5 * e6);
    let e8: vec4<i32> = a_5;
    let e9: i32 = b_5;
    v_2 = (e8 / vec4<i32>(e9));
    let e12: vec4<i32> = a_5;
    let e13: i32 = b_5;
    v_2 = (e12 + vec4<i32>(e13));
    let e16: vec4<i32> = a_5;
    let e17: i32 = b_5;
    v_2 = (e16 - vec4<i32>(e17));
    let e20: vec4<i32> = a_5;
    let e21: i32 = b_5;
    v_2 = (e20 & vec4<i32>(e21));
    let e24: vec4<i32> = a_5;
    let e25: i32 = b_5;
    v_2 = (e24 | vec4<i32>(e25));
    let e28: vec4<i32> = a_5;
    let e29: i32 = b_5;
    v_2 = (e28 ^ vec4<i32>(e29));
    let e32: vec4<i32> = a_5;
    let e33: i32 = b_5;
    v_2 = (e32 >> vec4<u32>(u32(e33)));
    let e37: vec4<i32> = a_5;
    let e38: i32 = b_5;
    v_2 = (e37 << vec4<u32>(u32(e38)));
    return;
}

fn testBinOpIntIVec(a_6: i32, b_6: vec4<i32>) {
    var a_7: i32;
    var b_7: vec4<i32>;
    var v_3: vec4<i32>;

    a_7 = a_6;
    b_7 = b_6;
    let e5: i32 = a_7;
    let e6: vec4<i32> = b_7;
    v_3 = (e5 * e6);
    let e8: i32 = a_7;
    let e9: vec4<i32> = b_7;
    v_3 = (vec4<i32>(e8) + e9);
    let e12: i32 = a_7;
    let e13: vec4<i32> = b_7;
    v_3 = (vec4<i32>(e12) - e13);
    let e16: i32 = a_7;
    let e17: vec4<i32> = b_7;
    v_3 = (vec4<i32>(e16) & e17);
    let e20: i32 = a_7;
    let e21: vec4<i32> = b_7;
    v_3 = (vec4<i32>(e20) | e21);
    let e24: i32 = a_7;
    let e25: vec4<i32> = b_7;
    v_3 = (vec4<i32>(e24) ^ e25);
    return;
}

fn testBinOpUVecUint(a_8: vec4<u32>, b_8: u32) {
    var a_9: vec4<u32>;
    var b_9: u32;
    var v_4: vec4<u32>;

    a_9 = a_8;
    b_9 = b_8;
    let e5: vec4<u32> = a_9;
    let e6: u32 = b_9;
    v_4 = (e5 * e6);
    let e8: vec4<u32> = a_9;
    let e9: u32 = b_9;
    v_4 = (e8 / vec4<u32>(e9));
    let e12: vec4<u32> = a_9;
    let e13: u32 = b_9;
    v_4 = (e12 + vec4<u32>(e13));
    let e16: vec4<u32> = a_9;
    let e17: u32 = b_9;
    v_4 = (e16 - vec4<u32>(e17));
    let e20: vec4<u32> = a_9;
    let e21: u32 = b_9;
    v_4 = (e20 & vec4<u32>(e21));
    let e24: vec4<u32> = a_9;
    let e25: u32 = b_9;
    v_4 = (e24 | vec4<u32>(e25));
    let e28: vec4<u32> = a_9;
    let e29: u32 = b_9;
    v_4 = (e28 ^ vec4<u32>(e29));
    let e32: vec4<u32> = a_9;
    let e33: u32 = b_9;
    v_4 = (e32 >> vec4<u32>(e33));
    let e36: vec4<u32> = a_9;
    let e37: u32 = b_9;
    v_4 = (e36 << vec4<u32>(e37));
    return;
}

fn testBinOpUintUVec(a_10: u32, b_10: vec4<u32>) {
    var a_11: u32;
    var b_11: vec4<u32>;
    var v_5: vec4<u32>;

    a_11 = a_10;
    b_11 = b_10;
    let e5: u32 = a_11;
    let e6: vec4<u32> = b_11;
    v_5 = (e5 * e6);
    let e8: u32 = a_11;
    let e9: vec4<u32> = b_11;
    v_5 = (vec4<u32>(e8) + e9);
    let e12: u32 = a_11;
    let e13: vec4<u32> = b_11;
    v_5 = (vec4<u32>(e12) - e13);
    let e16: u32 = a_11;
    let e17: vec4<u32> = b_11;
    v_5 = (vec4<u32>(e16) & e17);
    let e20: u32 = a_11;
    let e21: vec4<u32> = b_11;
    v_5 = (vec4<u32>(e20) | e21);
    let e24: u32 = a_11;
    let e25: vec4<u32> = b_11;
    v_5 = (vec4<u32>(e24) ^ e25);
    return;
}

fn testStructConstructor() {
    var tree: BST = BST(1);

}

fn testNonScalarToScalarConstructor() {
    var f: f32 = 1.0;

}

fn testArrayConstructor() {
    var tree_1: array<f32,1u> = array<f32,1u>(0.0);

}

fn privatePointer(a_12: ptr<function, f32>) {
    return;
}

fn main_1() {
    var local: f32;

    let e3: f32 = global;
    local = e3;
    privatePointer((&local));
    let e5: f32 = local;
    global = e5;
    let e6: vec4<f32> = o_color;
    let e9: vec4<f32> = vec4<f32>(1.0);
    o_color.x = e9.x;
    o_color.y = e9.y;
    o_color.z = e9.z;
    o_color.w = e9.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main_1();
    return;
}
