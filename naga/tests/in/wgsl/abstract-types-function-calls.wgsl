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

const const_af = 0;
const const_ai = 0;
const const_vec_af = vec2(0.0);
const const_vec_ai = vec2(0);
const const_mat_af = mat2x2(vec2(0.0), vec2(0.0));
const const_arr_af = array(0.0, 0.0);
const const_arr_ai = array(0, 0);

@compute @workgroup_size(1)
fn main() {
    func_f(0.0);
    func_f(0);
    func_i(0);
    func_u(0);

    func_f(const_af);
    func_f(const_ai);
    func_i(const_ai);
    func_u(const_ai);

    func_vf(vec2(0.0));
    func_vf(vec2(0));
    func_vi(vec2(0));
    func_vu(vec2(0));

    func_vf(const_vec_af);
    func_vf(const_vec_ai);
    func_vi(const_vec_ai);
    func_vu(const_vec_ai);

    func_mf(mat2x2(vec2(0.0), vec2(0.0)));
    func_mf(mat2x2(vec2(0), vec2(0)));

    func_mf(const_mat_af);

    func_af(array(0.0, 0.0));
    func_af(array(0, 0));
    func_ai(array(0, 0));
    func_au(array(0, 0));

    func_af(const_arr_af);
    func_af(const_arr_ai);
    func_ai(const_arr_ai);
    func_au(const_arr_ai);

    func_f_i(0.0, 0);
    func_f_i(0, 0);

    func_f_i(const_af, const_ai);
    func_f_i(const_ai, const_ai);
}
