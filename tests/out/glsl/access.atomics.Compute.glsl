#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct AlignedWrapper {
    int value;
};
struct Baz {
    mat3x2 m;
};
layout(std430) buffer Bar_block_0Compute {
    mat4x3 _matrix;
    mat2x2 matrix_array[2];
    int atom;
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_cs;


float read_from_private(inout float foo_1) {
    float _e3 = foo_1;
    return _e3;
}

float test_arr_as_arg(float a[5][10]) {
    return a[4][9];
}

void main() {
    int tmp = 0;
    int value = _group_0_binding_0_cs.atom;
    int _e7 = atomicAdd(_group_0_binding_0_cs.atom, 5);
    tmp = _e7;
    int _e10 = atomicAdd(_group_0_binding_0_cs.atom, -5);
    tmp = _e10;
    int _e13 = atomicAnd(_group_0_binding_0_cs.atom, 5);
    tmp = _e13;
    int _e16 = atomicOr(_group_0_binding_0_cs.atom, 5);
    tmp = _e16;
    int _e19 = atomicXor(_group_0_binding_0_cs.atom, 5);
    tmp = _e19;
    int _e22 = atomicMin(_group_0_binding_0_cs.atom, 5);
    tmp = _e22;
    int _e25 = atomicMax(_group_0_binding_0_cs.atom, 5);
    tmp = _e25;
    int _e28 = atomicExchange(_group_0_binding_0_cs.atom, 5);
    tmp = _e28;
    _group_0_binding_0_cs.atom = value;
    return;
}

