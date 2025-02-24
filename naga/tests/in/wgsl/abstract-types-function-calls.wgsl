fn func_f(a: f32) {}
fn func_i(a: i32) {}
fn func_u(a: u32) {}

fn func_vf(a: vec2<f32>) {}
fn func_vi(a: vec2<i32>) {}
fn func_vu(a: vec2<u32>) {}

fn func_mf(a: mat2x2<f32>) {}

fn func_af(a: array<f32, 2>) {}
fn func_ai(a: array<i32, 2>) {}
fn func_au(a: array<u32, 2>) {}

fn func_f_i(a: f32, b: i32) {}

fn main() {
    func_f(0.0);
    func_f(0);
    func_i(0);
    func_u(0);

    func_vf(vec2(0.0));
    func_vf(vec2(0));
    func_vi(vec2(0));
    func_vu(vec2(0));

    func_mf(mat2x2(vec2(0.0), vec2(0.0)));
    func_mf(mat2x2(vec2(0), vec2(0)));

    func_af(array(0.0, 0.0));
    func_af(array(0, 0));
    func_ai(array(0, 0));
    func_au(array(0, 0));

    func_f_i(0.0, 0);
    func_f_i(0, 0);
}
