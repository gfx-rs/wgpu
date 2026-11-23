#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Aligned {
    mat3x2 m;
};
struct mat3x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
    vec2 c_2_;
};
struct Aligned_std140_ {
    vec2 m_0_;
    vec2 m_1_;
    vec2 m_2_;
};
layout(std140) uniform mat3x2f32_std140_block_0Compute { mat3x2f32_std140_ _group_0_binding_0_cs; };

layout(std140) uniform type_6_block_1Compute { Aligned_std140_ _group_0_binding_1_cs[2]; };

layout(std140) uniform type_2_block_2Compute { vec4 _group_0_binding_2_cs[2]; };

layout(std430) buffer type_4_block_3Compute { vec2 _group_0_binding_3_cs[4]; };


void main() {
    uint i = gl_LocalInvocationIndex;
    vec2 _e4 = _group_0_binding_0_cs.c_0_;
    vec2 _e6 = _group_0_binding_0_cs.c_1_;
    vec2 _e8 = _group_0_binding_0_cs.c_2_;
    _group_0_binding_3_cs[0] = mat3x2(_e4, _e6, _e8)[i];
    vec2 _e17 = _group_0_binding_1_cs[i].m_1_;
    _group_0_binding_3_cs[1] = _e17;
    vec2 _e23 = _group_0_binding_1_cs[i].m_0_;
    vec2 _e25 = _group_0_binding_1_cs[i].m_1_;
    vec2 _e27 = _group_0_binding_1_cs[i].m_2_;
    _group_0_binding_3_cs[2] = mat3x2(_e23, _e25, _e27)[i];
    vec4 _e34 = _group_0_binding_2_cs[i];
    _group_0_binding_3_cs[3] = _e34.xy;
    return;
}

