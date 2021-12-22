#version 310 es
#extension GL_EXT_gpu_shader5 : require

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


vec2 test_fma() {
    vec2 a = vec2(2.0, 2.0);
    vec2 b = vec2(0.5, 0.5);
    vec2 c = vec2(0.5, 0.5);
    return fma(a, b, c);
}

void main() {
    vec2 _e0 = test_fma();
    return;
}

