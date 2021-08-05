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
    tmp = atomicAdd(_group_0_binding_0.atom, 1);
    tmp = atomicAnd(_group_0_binding_0.atom, 1);
    tmp = atomicOr(_group_0_binding_0.atom, 1);
    tmp = atomicXor(_group_0_binding_0.atom, 1);
    tmp = atomicMin(_group_0_binding_0.atom, 1);
    tmp = atomicMax(_group_0_binding_0.atom, 1);
    _group_0_binding_0.atom = value;
    return;
}

