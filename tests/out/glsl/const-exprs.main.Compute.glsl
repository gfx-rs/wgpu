#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

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

void main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    return;
}

