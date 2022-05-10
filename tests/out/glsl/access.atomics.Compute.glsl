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
    float _e4 = foo_1;
    return _e4;
}

float test_arr_as_arg(float a[5][10]) {
    return a[4][9];
}

void assign_through_ptr_fn(inout uint p) {
    p = 42u;
    return;
}

void main() {
    int tmp = 0;
    int value = _group_0_binding_0_cs.atom;
    int _e8 = atomicAdd(_group_0_binding_0_cs.atom, 5);
    tmp = _e8;
    int _e11 = atomicAdd(_group_0_binding_0_cs.atom, -5);
    tmp = _e11;
    int _e14 = atomicAnd(_group_0_binding_0_cs.atom, 5);
    tmp = _e14;
    int _e17 = atomicOr(_group_0_binding_0_cs.atom, 5);
    tmp = _e17;
    int _e20 = atomicXor(_group_0_binding_0_cs.atom, 5);
    tmp = _e20;
    int _e23 = atomicMin(_group_0_binding_0_cs.atom, 5);
    tmp = _e23;
    int _e26 = atomicMax(_group_0_binding_0_cs.atom, 5);
    tmp = _e26;
    int _e29 = atomicExchange(_group_0_binding_0_cs.atom, 5);
    tmp = _e29;
    _group_0_binding_0_cs.atom = value;
    return;
}

