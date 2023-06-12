fn exact(a: f32) {
    var a_1: f32;

    a_1 = a;
    return;
}

fn exact_1(a_2: i32) {
    var a_3: i32;

    a_3 = a_2;
    return;
}

fn implicit(a_4: f32) {
    var a_5: f32;

    a_5 = a_4;
    return;
}

fn implicit_1(a_6: i32) {
    var a_7: i32;

    a_7 = a_6;
    return;
}

fn implicit_dims(v: f32) {
    var v_1: f32;

    v_1 = v;
    return;
}

fn implicit_dims_1(v_2: vec2<f32>) {
    var v_3: vec2<f32>;

    v_3 = v_2;
    return;
}

fn implicit_dims_2(v_4: vec3<f32>) {
    var v_5: vec3<f32>;

    v_5 = v_4;
    return;
}

fn implicit_dims_3(v_6: vec4<f32>) {
    var v_7: vec4<f32>;

    v_7 = v_6;
    return;
}

fn main_1() {
    exact_1(1);
    implicit(f32(1u));
    implicit_dims_2(vec3<f32>(vec3<i32>(1)));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
