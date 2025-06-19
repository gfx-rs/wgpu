fn exact(a: i32) {
    var a_1: i32;

    a_1 = a;
    return;
}

fn implicit(a_2: f32) {
    var a_3: f32;

    a_3 = a_2;
    return;
}

fn implicit_dims(v: vec3<f32>) {
    var v_1: vec3<f32>;

    v_1 = v;
    return;
}

fn main_1() {
    exact(1i);
    implicit(1f);
    implicit_dims(vec3(1f));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
