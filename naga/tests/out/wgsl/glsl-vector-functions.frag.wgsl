fn ftest(a: vec4<f32>, b: vec4<f32>) {
    var a_1: vec4<f32>;
    var b_1: vec4<f32>;
    var c: vec4<bool>;
    var d: vec4<bool>;
    var e: vec4<bool>;
    var f: vec4<bool>;
    var g: vec4<bool>;
    var h: vec4<bool>;

    a_1 = a;
    b_1 = b;
    let _e4 = a_1;
    let _e5 = b_1;
    c = (_e4 < _e5);
    let _e8 = a_1;
    let _e9 = b_1;
    d = (_e8 <= _e9);
    let _e12 = a_1;
    let _e13 = b_1;
    e = (_e12 > _e13);
    let _e16 = a_1;
    let _e17 = b_1;
    f = (_e16 >= _e17);
    let _e20 = a_1;
    let _e21 = b_1;
    g = (_e20 == _e21);
    let _e24 = a_1;
    let _e25 = b_1;
    h = (_e24 != _e25);
    return;
}

fn dtest(a_2: vec4<f64>, b_2: vec4<f64>) {
    var a_3: vec4<f64>;
    var b_3: vec4<f64>;
    var c_1: vec4<bool>;
    var d_1: vec4<bool>;
    var e_1: vec4<bool>;
    var f_1: vec4<bool>;
    var g_1: vec4<bool>;
    var h_1: vec4<bool>;

    a_3 = a_2;
    b_3 = b_2;
    let _e4 = a_3;
    let _e5 = b_3;
    c_1 = (_e4 < _e5);
    let _e8 = a_3;
    let _e9 = b_3;
    d_1 = (_e8 <= _e9);
    let _e12 = a_3;
    let _e13 = b_3;
    e_1 = (_e12 > _e13);
    let _e16 = a_3;
    let _e17 = b_3;
    f_1 = (_e16 >= _e17);
    let _e20 = a_3;
    let _e21 = b_3;
    g_1 = (_e20 == _e21);
    let _e24 = a_3;
    let _e25 = b_3;
    h_1 = (_e24 != _e25);
    return;
}

fn itest(a_4: vec4<i32>, b_4: vec4<i32>) {
    var a_5: vec4<i32>;
    var b_5: vec4<i32>;
    var c_2: vec4<bool>;
    var d_2: vec4<bool>;
    var e_2: vec4<bool>;
    var f_2: vec4<bool>;
    var g_2: vec4<bool>;
    var h_2: vec4<bool>;

    a_5 = a_4;
    b_5 = b_4;
    let _e4 = a_5;
    let _e5 = b_5;
    c_2 = (_e4 < _e5);
    let _e8 = a_5;
    let _e9 = b_5;
    d_2 = (_e8 <= _e9);
    let _e12 = a_5;
    let _e13 = b_5;
    e_2 = (_e12 > _e13);
    let _e16 = a_5;
    let _e17 = b_5;
    f_2 = (_e16 >= _e17);
    let _e20 = a_5;
    let _e21 = b_5;
    g_2 = (_e20 == _e21);
    let _e24 = a_5;
    let _e25 = b_5;
    h_2 = (_e24 != _e25);
    return;
}

fn utest(a_6: vec4<u32>, b_6: vec4<u32>) {
    var a_7: vec4<u32>;
    var b_7: vec4<u32>;
    var c_3: vec4<bool>;
    var d_3: vec4<bool>;
    var e_3: vec4<bool>;
    var f_3: vec4<bool>;
    var g_3: vec4<bool>;
    var h_3: vec4<bool>;

    a_7 = a_6;
    b_7 = b_6;
    let _e4 = a_7;
    let _e5 = b_7;
    c_3 = (_e4 < _e5);
    let _e8 = a_7;
    let _e9 = b_7;
    d_3 = (_e8 <= _e9);
    let _e12 = a_7;
    let _e13 = b_7;
    e_3 = (_e12 > _e13);
    let _e16 = a_7;
    let _e17 = b_7;
    f_3 = (_e16 >= _e17);
    let _e20 = a_7;
    let _e21 = b_7;
    g_3 = (_e20 == _e21);
    let _e24 = a_7;
    let _e25 = b_7;
    h_3 = (_e24 != _e25);
    return;
}

fn btest(a_8: vec4<bool>, b_8: vec4<bool>) {
    var a_9: vec4<bool>;
    var b_9: vec4<bool>;
    var c_4: vec4<bool>;
    var d_4: vec4<bool>;
    var e_4: bool;
    var f_4: bool;
    var g_4: vec4<bool>;

    a_9 = a_8;
    b_9 = b_8;
    let _e4 = a_9;
    let _e5 = b_9;
    c_4 = (_e4 == _e5);
    let _e8 = a_9;
    let _e9 = b_9;
    d_4 = (_e8 != _e9);
    let _e12 = a_9;
    e_4 = any(_e12);
    let _e15 = a_9;
    f_4 = all(_e15);
    let _e18 = a_9;
    g_4 = !(_e18);
    return;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
