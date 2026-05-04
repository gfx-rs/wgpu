#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Output {
    uvec2 sum;
    uvec2 carry;
    uvec2 diff;
    uvec2 borrow;
};
struct Input {
    uvec2 a;
    uvec2 b;
};
layout(std430) buffer Output_block_0Compute { Output _group_0_binding_1_cs; };

layout(std430) buffer Input_block_1Compute { Input _group_0_binding_0_cs; };


void main_1() {
    uvec2 c = uvec2(0u);
    uvec2 d = uvec2(0u);
    uvec2 _e5 = _group_0_binding_0_cs.a;
    uvec2 _e7 = _group_0_binding_0_cs.b;
    uvec2 _e8 = (_e5 + _e7);
    Input _e11 = Input(_e8, uvec2(lessThan(_e8, _e5)));
    c = _e11.b;
    _group_0_binding_1_cs.sum = _e11.a;
    uvec2 _e15 = c;
    _group_0_binding_1_cs.carry = _e15;
    uvec2 _e18 = _group_0_binding_0_cs.a;
    uvec2 _e20 = _group_0_binding_0_cs.b;
    Input _e24 = Input((_e18 - _e20), uvec2(lessThan(_e18, _e20)));
    d = _e24.b;
    _group_0_binding_1_cs.diff = _e24.a;
    uvec2 _e28 = d;
    _group_0_binding_1_cs.borrow = _e28;
    return;
}

void main() {
    main_1();
}

