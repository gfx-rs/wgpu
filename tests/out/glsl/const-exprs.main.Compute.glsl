#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const int FOUR = 4;

layout(std430) buffer type_block_0Compute { ivec4 _group_0_binding_0_cs; };

layout(std430) buffer type_1_block_1Compute { int _group_0_binding_1_cs; };


void swizzle_of_compose() {
    ivec2 a = ivec2(1, 2);
    ivec2 b = ivec2(3, 4);
    _group_0_binding_0_cs = ivec4(4, 3, 2, 1);
    return;
}

void index_of_compose() {
    ivec2 a_1 = ivec2(1, 2);
    ivec2 b_1 = ivec2(3, 4);
    int _e7 = _group_0_binding_1_cs;
    _group_0_binding_1_cs = (_e7 + 2);
    return;
}

void compose_three_deep() {
    int _e2 = _group_0_binding_1_cs;
    _group_0_binding_1_cs = (_e2 + 6);
    return;
}

void non_constant_initializers() {
    int w = 30;
    int x = 0;
    int y = 0;
    int z = 70;
    int _e2 = w;
    x = _e2;
    int _e4 = x;
    y = _e4;
    int _e9 = w;
    int _e10 = x;
    int _e11 = y;
    int _e12 = z;
    ivec4 _e14 = _group_0_binding_0_cs;
    _group_0_binding_0_cs = (_e14 + ivec4(_e9, _e10, _e11, _e12));
    return;
}

void splat_of_constant() {
    _group_0_binding_0_cs = -(ivec4(FOUR));
    return;
}

void compose_of_constant() {
    _group_0_binding_0_cs = -(ivec4(FOUR, FOUR, FOUR, FOUR));
    return;
}

void main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    return;
}

