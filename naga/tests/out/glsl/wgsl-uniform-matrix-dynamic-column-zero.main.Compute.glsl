#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct S {
    float a;
    mat3x2 m;
};
struct Aligned {
    mat3x2 m;
};
struct mat3x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
    vec2 c_2_;
};
struct S_std140_ {
    float a;
    vec2 m_0_;
    vec2 m_1_;
    vec2 m_2_;
};
struct Aligned_std140_ {
    vec2 m_0_;
    vec2 m_1_;
    vec2 m_2_;
};
layout(std140) uniform mat3x2f32_std140_block_0Compute { mat3x2f32_std140_ _group_0_binding_0_cs; };

layout(std140) uniform S_std140_block_1Compute { S_std140_ _group_0_binding_1_cs; };

layout(std140) uniform type_5_block_2Compute { Aligned_std140_ _group_0_binding_2_cs[2]; };

layout(std430) buffer type_3_block_3Compute { vec2 _group_0_binding_3_cs[8]; };


void main() {
    uint i = gl_LocalInvocationIndex;
    vec2 _e4 = _group_0_binding_0_cs.c_0_;
    vec2 _e6 = _group_0_binding_0_cs.c_1_;
    vec2 _e8 = _group_0_binding_0_cs.c_2_;
    _group_0_binding_3_cs[0] = mat3x2(_e4, _e6, _e8)[i];
    vec2 _e16 = _group_0_binding_1_cs.m_0_;
    vec2 _e18 = _group_0_binding_1_cs.m_1_;
    vec2 _e20 = _group_0_binding_1_cs.m_2_;
    _group_0_binding_3_cs[1] = mat3x2(_e16, _e18, _e20)[i];
    vec2 _e27 = _group_0_binding_0_cs.c_0_;
    vec2 _e29 = _group_0_binding_0_cs.c_1_;
    vec2 _e31 = _group_0_binding_0_cs.c_2_;
    vec2 _e37 = _group_0_binding_1_cs.m_0_;
    vec2 _e39 = _group_0_binding_1_cs.m_1_;
    vec2 _e41 = _group_0_binding_1_cs.m_2_;
    _group_0_binding_3_cs[2] = vec2(mat3x2(_e27, _e29, _e31)[i][i], mat3x2(_e37, _e39, _e41)[i][i]);
    vec2 _e51 = _group_0_binding_2_cs[i].m_0_;
    vec2 _e53 = _group_0_binding_2_cs[i].m_1_;
    vec2 _e55 = _group_0_binding_2_cs[i].m_2_;
    _group_0_binding_3_cs[3] = mat3x2(_e51, _e53, _e55)[i];
    vec2 _e63 = _group_0_binding_2_cs[i].m_0_;
    vec2 _e65 = _group_0_binding_2_cs[i].m_1_;
    vec2 _e67 = _group_0_binding_2_cs[i].m_2_;
    _group_0_binding_3_cs[4] = vec2(mat3x2(_e63, _e65, _e67)[i][i], 0.0);
    return;
}

