#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


int return_i32_ai() {
    return 1;
}

uint return_u32_ai() {
    return 1u;
}

float return_f32_ai() {
    return 1.0;
}

float return_f32_af() {
    return 1.0;
}

vec2 return_vec2f32_ai() {
    return vec2(1.0);
}

float[4] return_arrf32_ai() {
    return float[4](1.0, 1.0, 1.0, 1.0);
}

float return_const_f32_const_ai() {
    return 1.0;
}

vec2 return_vec2f32_const_ai() {
    return vec2(1.0);
}

void main() {
    return;
}

