#version 420 core
#extension GL_ARB_compute_shader : require
#extension GL_OES_shader_image_atomic : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

layout(r32ui) uniform uimage2DArray _group_0_binding_0_cs;

layout(r32i) uniform iimage2DArray _group_0_binding_1_cs;


void main() {
    uvec3 id = gl_LocalInvocationID;
    imageAtomicMax(_group_0_binding_0_cs, ivec3(ivec2(0, 0), 1), 1u);
    imageAtomicMin(_group_0_binding_1_cs, ivec3(ivec2(0, 0), 1u), 1);
    return;
}

