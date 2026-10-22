#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer Globals_block_0Compute {
    float a[4];
    vec4 v;
    mat3x4 m;
    float d[];
} _group_0_binding_0_cs;


float index_array_value() {
    float a_1[2] = float[2](1.0, 2.0);
    return a_1[1];
}

float index_vector_value() {
    vec2 v = vec2(1.0, 2.0);
    return v[1];
}

vec2 index_matrix_value() {
    mat2x2 m = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    return m[1];
}

float index_negative_value() {
    vec2 v_1 = vec2(1.0, 2.0);
    return v_1[1];
}

float index_local() {
    float a[2] = float[2](1.0, 2.0);
    a[1] = 3.0;
    float _e8 = a[1];
    return _e8;
}

float index_storage() {
    float _e4 = _group_0_binding_0_cs.a[3];
    float _e8 = _group_0_binding_0_cs.v[3];
    float _e14 = _group_0_binding_0_cs.m[2][3];
    return ((_e4 + _e8) + _e14);
}

void set_storage(float value) {
    _group_0_binding_0_cs.a[3] = value;
    _group_0_binding_0_cs.v[3] = value;
    _group_0_binding_0_cs.m[2][3] = value;
    return;
}

float index_runtime_sized_negative() {
    _group_0_binding_0_cs.d[0] = 1.0;
    float _e8 = _group_0_binding_0_cs.d[0];
    return _e8;
}

void main() {
    float _e0 = index_array_value();
    float _e1 = index_vector_value();
    vec2 _e3 = index_matrix_value();
    float _e6 = index_negative_value();
    float _e8 = index_local();
    float _e10 = index_storage();
    float _e12 = index_runtime_sized_negative();
    float sum = ((((((_e0 + _e1) + _e3.x) + _e6) + _e8) + _e10) + _e12);
    set_storage(sum);
    return;
}

