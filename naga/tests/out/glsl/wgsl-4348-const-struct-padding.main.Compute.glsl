#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct PackedVec3_ {
    float a;
    vec3 c;
    float d;
};
struct Matrix {
    int a;
    ivec4 c;
    mat3x2 d;
};
struct TrailingPad {
    mat2x2 m;
    float a;
};
const PackedVec3_ PACKED_VEC3_ = PackedVec3_(1.0, vec3(2.0), 3.0);
const Matrix MATRIX = Matrix(1, ivec4(3), mat3x2(vec2(1.0, 2.0), vec2(3.0, 4.0), vec2(5.0, 6.0)));
const TrailingPad TRAILING_PAD = TrailingPad(mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0)), 5.0);

layout(std430) buffer type_7_block_0Compute { float _group_0_binding_0_cs[]; };


float packed_vec3_(PackedVec3_ s) {
    return ((s.a + s.c.x) + s.d);
}

float matrix(Matrix s_1) {
    return ((float(s_1.a) + float(s_1.c.x)) + s_1.d[0].x);
}

float trailing_pad(TrailingPad s_2) {
    return (s_2.m[0].x + s_2.a);
}

void main() {
    float _e3 = packed_vec3_(PACKED_VEC3_);
    _group_0_binding_0_cs[0] = _e3;
    float _e7 = matrix(MATRIX);
    _group_0_binding_0_cs[1] = _e7;
    float _e11 = trailing_pad(TRAILING_PAD);
    _group_0_binding_0_cs[2] = _e11;
    return;
}

