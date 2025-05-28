#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void func_f(float a) {
    return;
}

void func_i(int a_1) {
    return;
}

void func_u(uint a_2) {
    return;
}

void func_vf(vec2 a_3) {
    return;
}

void func_vi(ivec2 a_4) {
    return;
}

void func_vu(uvec2 a_5) {
    return;
}

void func_mf(mat2x2 a_6) {
    return;
}

void func_af(float a_7[2]) {
    return;
}

void func_ai(int a_8[2]) {
    return;
}

void func_au(uint a_9[2]) {
    return;
}

void func_f_i(float a_10, int b) {
    return;
}

void main() {
    func_f(0.0);
    func_f(0.0);
    func_i(0);
    func_u(0u);
    func_f(0.0);
    func_f(0.0);
    func_i(0);
    func_u(0u);
    func_vf(vec2(0.0));
    func_vf(vec2(0.0));
    func_vi(ivec2(0));
    func_vu(uvec2(0u));
    func_vf(vec2(0.0));
    func_vf(vec2(0.0));
    func_vi(ivec2(0));
    func_vu(uvec2(0u));
    func_mf(mat2x2(vec2(0.0), vec2(0.0)));
    func_mf(mat2x2(vec2(0.0), vec2(0.0)));
    func_mf(mat2x2(vec2(0.0), vec2(0.0)));
    func_af(float[2](0.0, 0.0));
    func_af(float[2](0.0, 0.0));
    func_ai(int[2](0, 0));
    func_au(uint[2](0u, 0u));
    func_af(float[2](0.0, 0.0));
    func_af(float[2](0.0, 0.0));
    func_ai(int[2](0, 0));
    func_au(uint[2](0u, 0u));
    func_f_i(0.0, 0);
    func_f_i(0.0, 0);
    func_f_i(0.0, 0);
    func_f_i(0.0, 0);
    return;
}

