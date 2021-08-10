#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

buffer Bar_block_0Cs {
    mat4x4 matrix;
    int atom;
    uvec2 arr[2];
    int data[];
} _group_0_binding_0;


void main() {
    int tmp = 0;
    int value = _group_0_binding_0.atom;
    int _expr6 = atomicAdd(_group_0_binding_0.atom, 5);
    tmp = _expr6;
    int _expr9 = atomicAnd(_group_0_binding_0.atom, 5);
    tmp = _expr9;
    int _expr12 = atomicOr(_group_0_binding_0.atom, 5);
    tmp = _expr12;
    int _expr15 = atomicXor(_group_0_binding_0.atom, 5);
    tmp = _expr15;
    int _expr18 = atomicMin(_group_0_binding_0.atom, 5);
    tmp = _expr18;
    int _expr21 = atomicMax(_group_0_binding_0.atom, 5);
    tmp = _expr21;
    int _expr24 = atomicExchange(_group_0_binding_0.atom, 5);
    tmp = _expr24;
    _group_0_binding_0.atom = value;
    return;
}

