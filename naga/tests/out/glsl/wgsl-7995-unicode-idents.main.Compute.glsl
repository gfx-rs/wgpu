#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) readonly buffer type_block_0Compute { float _group_0_binding_0_cs; };


float compute() {
    float _e1 = _group_0_binding_0_cs;
    float u03b8_2_ = (_e1 + 9001.0);
    return u03b8_2_;
}

void main() {
    float _e0 = compute();
    return;
}

