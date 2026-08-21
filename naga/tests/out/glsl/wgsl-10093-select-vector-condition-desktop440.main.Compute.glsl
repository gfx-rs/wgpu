#version 440 core
#extension GL_ARB_compute_shader : require
#extension GL_ARB_shader_storage_buffer_object : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Inputs {
    ivec4 ia;
    ivec4 ib;
    uvec4 ua;
    uvec4 ub;
    vec4 fa;
    vec4 fb;
};
layout(std140) uniform Inputs_block_0Compute { Inputs _group_0_binding_0_cs; };

layout(std430) buffer type_block_1Compute { ivec4 _group_0_binding_1_cs; };

layout(std430) buffer type_1_block_2Compute { uvec4 _group_0_binding_2_cs; };

layout(std430) buffer type_2_block_3Compute { vec4 _group_0_binding_3_cs; };

layout(std430) buffer type_block_4Compute { ivec4 _group_0_binding_4_cs; };


void main() {
    ivec4 _e2 = _group_0_binding_0_cs.ia;
    ivec4 _e5 = _group_0_binding_0_cs.ib;
    bvec4 icond = lessThan(_e2, _e5);
    ivec4 _e10 = _group_0_binding_0_cs.ia;
    ivec4 _e15 = _group_0_binding_0_cs.ib;
    _group_0_binding_1_cs = mix((_e10 * 2), (_e15 + ivec4(7)), icond);
    uvec4 _e23 = _group_0_binding_0_cs.ua;
    uvec4 _e26 = _group_0_binding_0_cs.ub;
    uvec4 _e29 = _group_0_binding_0_cs.ua;
    uvec4 _e32 = _group_0_binding_0_cs.ub;
    _group_0_binding_2_cs = mix(_e23, _e26, lessThan(_e29, _e32));
    vec4 _e38 = _group_0_binding_0_cs.fa;
    vec4 _e41 = _group_0_binding_0_cs.fb;
    vec4 _e44 = _group_0_binding_0_cs.fa;
    vec4 _e47 = _group_0_binding_0_cs.fb;
    _group_0_binding_3_cs = mix(_e38, _e41, lessThan(_e44, _e47));
    uvec4 _e52 = _group_0_binding_0_cs.ua;
    uvec4 _e55 = _group_0_binding_0_cs.ub;
    bvec4 picked = mix(icond, lessThan(_e52, _e55), icond);
    _group_0_binding_4_cs = mix(ivec4(0), ivec4(1), picked);
    return;
}

