fn func_f(a: f32) {
    return;
}

fn func_i(a_1: i32) {
    return;
}

fn func_u(a_2: u32) {
    return;
}

fn func_vf(a_3: vec2<f32>) {
    return;
}

fn func_vi(a_4: vec2<i32>) {
    return;
}

fn func_vu(a_5: vec2<u32>) {
    return;
}

fn func_mf(a_6: mat2x2<f32>) {
    return;
}

fn func_af(a_7: array<f32, 2>) {
    return;
}

fn func_ai(a_8: array<i32, 2>) {
    return;
}

fn func_au(a_9: array<u32, 2>) {
    return;
}

fn func_f_i(a_10: f32, b: i32) {
    return;
}

fn main() {
    func_f(0f);
    func_f(0f);
    func_i(0i);
    func_u(0u);
    func_vf(vec2(0f));
    func_vf(vec2(0f));
    func_vi(vec2(0i));
    func_vu(vec2(0u));
    func_mf(mat2x2<f32>(vec2(0f), vec2(0f)));
    func_mf(mat2x2<f32>(vec2(0f), vec2(0f)));
    func_af(array<f32, 2>(0f, 0f));
    func_af(array<f32, 2>(0f, 0f));
    func_ai(array<i32, 2>(0i, 0i));
    func_au(array<u32, 2>(0u, 0u));
    func_f_i(0f, 0i);
    func_f_i(0f, 0i);
    return;
}

