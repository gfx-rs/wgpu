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
    let _e5: vec4<f32> = a_1;
    v = (_e5 * 2.0);
    let _e8: vec4<f32> = a_1;
    v = (_e8 / vec4<f32>(2.0));
    let _e12: vec4<f32> = a_1;
    v = (_e12 + vec4<f32>(2.0));
    let _e16: vec4<f32> = a_1;
    v = (_e16 - vec4<f32>(2.0));
    return;
}

fn testBinOpFloatVec(a_2: vec4<f32>, b_2: f32) {
    var a_3: vec4<f32>;
    var b_3: f32;
    var v_1: vec4<f32>;

    a_3 = a_2;
    b_3 = b_2;
    let _e5: vec4<f32> = a_3;
    let _e6: f32 = b_3;
    v_1 = (_e5 * _e6);
    let _e8: vec4<f32> = a_3;
    let _e9: f32 = b_3;
    v_1 = (_e8 / vec4<f32>(_e9));
    let _e12: vec4<f32> = a_3;
    let _e13: f32 = b_3;
    v_1 = (_e12 + vec4<f32>(_e13));
    let _e16: vec4<f32> = a_3;
    let _e17: f32 = b_3;
    v_1 = (_e16 - vec4<f32>(_e17));
    return;
}

fn testBinOpIVecInt(a_4: vec4<i32>, b_4: i32) {
    var a_5: vec4<i32>;
    var b_5: i32;
    var v_2: vec4<i32>;

    a_5 = a_4;
    b_5 = b_4;
    let _e5: vec4<i32> = a_5;
    let _e6: i32 = b_5;
    v_2 = (_e5 * _e6);
    let _e8: vec4<i32> = a_5;
    let _e9: i32 = b_5;
    v_2 = (_e8 / vec4<i32>(_e9));
    let _e12: vec4<i32> = a_5;
    let _e13: i32 = b_5;
    v_2 = (_e12 + vec4<i32>(_e13));
    let _e16: vec4<i32> = a_5;
    let _e17: i32 = b_5;
    v_2 = (_e16 - vec4<i32>(_e17));
    let _e20: vec4<i32> = a_5;
    let _e21: i32 = b_5;
    v_2 = (_e20 & vec4<i32>(_e21));
    let _e24: vec4<i32> = a_5;
    let _e25: i32 = b_5;
    v_2 = (_e24 | vec4<i32>(_e25));
    let _e28: vec4<i32> = a_5;
    let _e29: i32 = b_5;
    v_2 = (_e28 ^ vec4<i32>(_e29));
    let _e32: vec4<i32> = a_5;
    let _e33: i32 = b_5;
    v_2 = (_e32 >> vec4<u32>(u32(_e33)));
    let _e37: vec4<i32> = a_5;
    let _e38: i32 = b_5;
    v_2 = (_e37 << vec4<u32>(u32(_e38)));
    return;
}

fn testBinOpIntIVec(a_6: i32, b_6: vec4<i32>) {
    var a_7: i32;
    var b_7: vec4<i32>;
    var v_3: vec4<i32>;

    a_7 = a_6;
    b_7 = b_6;
    let _e5: i32 = a_7;
    let _e6: vec4<i32> = b_7;
    v_3 = (_e5 * _e6);
    let _e8: i32 = a_7;
    let _e9: vec4<i32> = b_7;
    v_3 = (vec4<i32>(_e8) + _e9);
    let _e12: i32 = a_7;
    let _e13: vec4<i32> = b_7;
    v_3 = (vec4<i32>(_e12) - _e13);
    let _e16: i32 = a_7;
    let _e17: vec4<i32> = b_7;
    v_3 = (vec4<i32>(_e16) & _e17);
    let _e20: i32 = a_7;
    let _e21: vec4<i32> = b_7;
    v_3 = (vec4<i32>(_e20) | _e21);
    let _e24: i32 = a_7;
    let _e25: vec4<i32> = b_7;
    v_3 = (vec4<i32>(_e24) ^ _e25);
    return;
}

fn testBinOpUVecUint(a_8: vec4<u32>, b_8: u32) {
    var a_9: vec4<u32>;
    var b_9: u32;
    var v_4: vec4<u32>;

    a_9 = a_8;
    b_9 = b_8;
    let _e5: vec4<u32> = a_9;
    let _e6: u32 = b_9;
    v_4 = (_e5 * _e6);
    let _e8: vec4<u32> = a_9;
    let _e9: u32 = b_9;
    v_4 = (_e8 / vec4<u32>(_e9));
    let _e12: vec4<u32> = a_9;
    let _e13: u32 = b_9;
    v_4 = (_e12 + vec4<u32>(_e13));
    let _e16: vec4<u32> = a_9;
    let _e17: u32 = b_9;
    v_4 = (_e16 - vec4<u32>(_e17));
    let _e20: vec4<u32> = a_9;
    let _e21: u32 = b_9;
    v_4 = (_e20 & vec4<u32>(_e21));
    let _e24: vec4<u32> = a_9;
    let _e25: u32 = b_9;
    v_4 = (_e24 | vec4<u32>(_e25));
    let _e28: vec4<u32> = a_9;
    let _e29: u32 = b_9;
    v_4 = (_e28 ^ vec4<u32>(_e29));
    let _e32: vec4<u32> = a_9;
    let _e33: u32 = b_9;
    v_4 = (_e32 >> vec4<u32>(_e33));
    let _e36: vec4<u32> = a_9;
    let _e37: u32 = b_9;
    v_4 = (_e36 << vec4<u32>(_e37));
    return;
}

fn testBinOpUintUVec(a_10: u32, b_10: vec4<u32>) {
    var a_11: u32;
    var b_11: vec4<u32>;
    var v_5: vec4<u32>;

    a_11 = a_10;
    b_11 = b_10;
    let _e5: u32 = a_11;
    let _e6: vec4<u32> = b_11;
    v_5 = (_e5 * _e6);
    let _e8: u32 = a_11;
    let _e9: vec4<u32> = b_11;
    v_5 = (vec4<u32>(_e8) + _e9);
    let _e12: u32 = a_11;
    let _e13: vec4<u32> = b_11;
    v_5 = (vec4<u32>(_e12) - _e13);
    let _e16: u32 = a_11;
    let _e17: vec4<u32> = b_11;
    v_5 = (vec4<u32>(_e16) & _e17);
    let _e20: u32 = a_11;
    let _e21: vec4<u32> = b_11;
    v_5 = (vec4<u32>(_e20) | _e21);
    let _e24: u32 = a_11;
    let _e25: vec4<u32> = b_11;
    v_5 = (vec4<u32>(_e24) ^ _e25);
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

    let _e3: f32 = global;
    local = _e3;
    privatePointer((&local));
    let _e5: f32 = local;
    global = _e5;
    let _e6: vec4<f32> = o_color;
    let _e9: vec4<f32> = vec4<f32>(1.0);
    o_color.x = _e9.x;
    o_color.y = _e9.y;
    o_color.z = _e9.z;
    o_color.w = _e9.w;
    return;
}

[[stage(fragment)]]
fn main() {
    main_1();
    return;
}
