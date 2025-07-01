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
struct AssignToMember {
    uint x;
};
struct S {
    int m;
};
struct Inner {
    int delicious;
};
struct Outer {
    Inner om_nom_nom;
    uint thing;
};

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

void assign_array_through_ptr_fn(inout vec4 foo_2[2]) {
    foo_2 = vec4[2](vec4(1.0), vec4(2.0));
    return;
}

void assign_through_ptr() {
    uint val = 33u;
    vec4 arr[2] = vec4[2](vec4(6.0), vec4(7.0));
    assign_through_ptr_fn(val);
    assign_array_through_ptr_fn(arr);
    return;
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

void assign_to_ptr_components() {
    AssignToMember s1_ = AssignToMember(0u);
    uint a1_[4] = uint[4](0u, 0u, 0u, 0u);
    assign_to_arg_ptr_member(s1_);
    uint _e1 = fetch_arg_ptr_member(s1_);
    assign_to_arg_ptr_array_element(a1_);
    uint _e3 = fetch_arg_ptr_array_element(a1_);
    return;
}

bool index_ptr(bool value) {
    bool a_1[1] = bool[1](false);
    a_1 = bool[1](value);
    bool _e4 = a_1[0];
    return _e4;
}

int member_ptr() {
    S s = S(42);
    int _e4 = s.m;
    return _e4;
}

int let_members_of_members() {
    Inner inner_1 = Outer(Inner(0), 0u).om_nom_nom;
    int delishus_1 = inner_1.delicious;
    if ((Outer(Inner(0), 0u).thing != uint(delishus_1))) {
    }
    return Outer(Inner(0), 0u).om_nom_nom.delicious;
}

int var_members_of_members() {
    Outer thing = Outer(Inner(0), 0u);
    Inner inner = Inner(0);
    int delishus = 0;
    Inner _e3 = thing.om_nom_nom;
    inner = _e3;
    int _e6 = inner.delicious;
    delishus = _e6;
    uint _e9 = thing.thing;
    int _e10 = delishus;
    if ((_e9 != uint(_e10))) {
    }
    int _e15 = thing.om_nom_nom.delicious;
    return _e15;
}

void main() {
    assign_through_ptr();
    assign_to_ptr_components();
    bool _e1 = index_ptr(true);
    int _e2 = member_ptr();
    int _e3 = let_members_of_members();
    int _e4 = var_members_of_members();
    return;
}

