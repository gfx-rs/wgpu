#version 310 es

precision highp float;
precision highp int;

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
struct AssignToMember {
    uint x;
};

float read_from_private(inout float foo_2) {
    float _e1 = foo_2;
    return _e1;
}

float test_arr_as_arg(float a[5][10]) {
    return a[4][9];
}

void assign_through_ptr_fn(inout uint p) {
    p = 42u;
    return;
}

void assign_array_through_ptr_fn(inout vec4 foo_3[2]) {
    foo_3 = vec4[2](vec4(1.0), vec4(2.0));
    return;
}

int array_by_value(int a_1[5], int i) {
    return a_1[i];
}

uint fetch_arg_ptr_member(inout AssignToMember p_1) {
    uint _e2 = p_1.x;
    return _e2;
}

void assign_to_arg_ptr_member(inout AssignToMember p_2) {
    p_2.x = 10u;
    return;
}

uint fetch_arg_ptr_array_element(inout uint p_3[4]) {
    uint _e2 = p_3[1];
    return _e2;
}

void assign_to_arg_ptr_array_element(inout uint p_4[4]) {
    p_4[1] = 10u;
    return;
}

void main() {
    uint vi_1 = uint(gl_VertexID);
    int arr_1[5] = int[5](1, 2, 3, 4, 5);
    int value = arr_1[vi_1];
    gl_Position = vec4(ivec4(value));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

