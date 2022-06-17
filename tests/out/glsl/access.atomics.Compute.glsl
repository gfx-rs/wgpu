#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct GlobalConst {
    uint a;
    uvec3 b;
    int c;
};
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
    float _e5 = foo_1;
    return _e5;
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
    int _e9 = atomicAdd(_group_0_binding_0_cs.atom, 5);
    tmp = _e9;
    int _e12 = atomicAdd(_group_0_binding_0_cs.atom, -5);
    tmp = _e12;
    int _e15 = atomicAnd(_group_0_binding_0_cs.atom, 5);
    tmp = _e15;
    int _e18 = atomicOr(_group_0_binding_0_cs.atom, 5);
    tmp = _e18;
    int _e21 = atomicXor(_group_0_binding_0_cs.atom, 5);
    tmp = _e21;
    int _e24 = atomicMin(_group_0_binding_0_cs.atom, 5);
    tmp = _e24;
    int _e27 = atomicMax(_group_0_binding_0_cs.atom, 5);
    tmp = _e27;
    int _e30 = atomicExchange(_group_0_binding_0_cs.atom, 5);
    tmp = _e30;
    _group_0_binding_0_cs.atom = value;
    return;
}

