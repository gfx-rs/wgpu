fn exact(a: f32) {
    var a1: f32;

    a1 = a;
    return;
}

fn exact1(a2: i32) {
    var a3: i32;

    a3 = a2;
    return;
}

fn implicit(a4: f32) {
    var a5: f32;

    a5 = a4;
    return;
}

fn implicit1(a6: i32) {
    var a7: i32;

    a7 = a6;
    return;
}

fn implicit_dims(v: f32) {
    var v1: f32;

    v1 = v;
    return;
}

fn implicit_dims1(v2: vec2<f32>) {
    var v3: vec2<f32>;

    v3 = v2;
    return;
}

fn implicit_dims2(v4: vec3<f32>) {
    var v5: vec3<f32>;

    v5 = v4;
    return;
}

fn implicit_dims3(v6: vec4<f32>) {
    var v7: vec4<f32>;

    v7 = v6;
    return;
}

fn main1() {
    exact1(1);
    implicit(f32(1u));
    implicit_dims2(vec3<f32>(vec3<i32>(1)));
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}
