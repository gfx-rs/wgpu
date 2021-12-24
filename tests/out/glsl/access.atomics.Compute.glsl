#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer Bar_block_0Compute {
    mat4x4 matrix;
    int atom;
    uvec2 arr[2];
    int data[];
} _group_0_binding_0_cs;


float read_from_private(inout float foo_2) {
    float _e2 = foo_2;
    return _e2;
}

void main() {
    int tmp = 0;
    int value = _group_0_binding_0_cs.atom;
    int _e6 = atomicAdd(_group_0_binding_0_cs.atom, 5);
    tmp = _e6;
    int _e9 = atomicAdd(_group_0_binding_0_cs.atom, -5);
    tmp = _e9;
    int _e12 = atomicAnd(_group_0_binding_0_cs.atom, 5);
    tmp = _e12;
    int _e15 = atomicOr(_group_0_binding_0_cs.atom, 5);
    tmp = _e15;
    int _e18 = atomicXor(_group_0_binding_0_cs.atom, 5);
    tmp = _e18;
    int _e21 = atomicMin(_group_0_binding_0_cs.atom, 5);
    tmp = _e21;
    int _e24 = atomicMax(_group_0_binding_0_cs.atom, 5);
    tmp = _e24;
    int _e27 = atomicExchange(_group_0_binding_0_cs.atom, 5);
    tmp = _e27;
    _group_0_binding_0_cs.atom = value;
    return;
}

