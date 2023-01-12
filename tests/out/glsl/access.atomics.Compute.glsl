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
struct MatCx2InArray {
    mat4x2 am[2];
};
layout(std430) buffer Bar_block_0Compute {
    mat4x3 _matrix;
    mat2x2 matrix_array[2];
    int atom;
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_cs;


float read_from_private(inout float foo_1) {
    float _e1 = foo_1;
    return _e1;
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
    int _e7 = atomicAdd(_group_0_binding_0_cs.atom, 5);
    tmp = _e7;
    int _e11 = atomicAdd(_group_0_binding_0_cs.atom, -5);
    tmp = _e11;
    int _e15 = atomicAnd(_group_0_binding_0_cs.atom, 5);
    tmp = _e15;
    int _e19 = atomicOr(_group_0_binding_0_cs.atom, 5);
    tmp = _e19;
    int _e23 = atomicXor(_group_0_binding_0_cs.atom, 5);
    tmp = _e23;
    int _e27 = atomicMin(_group_0_binding_0_cs.atom, 5);
    tmp = _e27;
    int _e31 = atomicMax(_group_0_binding_0_cs.atom, 5);
    tmp = _e31;
    int _e35 = atomicExchange(_group_0_binding_0_cs.atom, 5);
    tmp = _e35;
    _group_0_binding_0_cs.atom = value;
    return;
}

