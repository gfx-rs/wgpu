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
shared uint val;


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
    assign_through_ptr_fn(val);
    return;
}

