#version 430 core
#extension GL_ARB_compute_shader : require
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

uniform usampler2D _group_0_binding_0_cs;

uniform usampler2DMS _group_0_binding_3_cs;

layout(rgba8ui) readonly uniform uimage2D _group_0_binding_1_cs;

uniform usampler2DArray _group_0_binding_5_cs;

uniform usampler1D _group_0_binding_7_cs;

layout(r32ui) writeonly uniform uimage1D _group_0_binding_2_cs;


void main() {
    uvec3 local_id = gl_LocalInvocationID;
    uvec2 dim = uvec2(imageSize(_group_0_binding_1_cs).xy);
    ivec2 itc = (ivec2((dim * local_id.xy)) % ivec2(10, 20));
    uvec4 value1_ = texelFetch(_group_0_binding_0_cs, itc, int(local_id.z));
    uvec4 value2_ = texelFetch(_group_0_binding_3_cs, itc, int(local_id.z));
    uvec4 value4_ = imageLoad(_group_0_binding_1_cs, itc);
    uvec4 value5_ = texelFetch(_group_0_binding_5_cs, ivec3(itc, local_id.z), (int(local_id.z) + 1));
    uvec4 value6_ = texelFetch(_group_0_binding_5_cs, ivec3(itc, int(local_id.z)), (int(local_id.z) + 1));
    uvec4 value7_ = texelFetch(_group_0_binding_7_cs, int(local_id.x), int(local_id.z));
    uvec4 value1u = texelFetch(_group_0_binding_0_cs, ivec2(uvec2(itc)), int(local_id.z));
    uvec4 value2u = texelFetch(_group_0_binding_3_cs, ivec2(uvec2(itc)), int(local_id.z));
    uvec4 value4u = imageLoad(_group_0_binding_1_cs, ivec2(uvec2(itc)));
    uvec4 value5u = texelFetch(_group_0_binding_5_cs, ivec3(uvec2(itc), local_id.z), (int(local_id.z) + 1));
    uvec4 value6u = texelFetch(_group_0_binding_5_cs, ivec3(uvec2(itc), int(local_id.z)), (int(local_id.z) + 1));
    uvec4 value7u = texelFetch(_group_0_binding_7_cs, int(uint(local_id.x)), int(local_id.z));
    imageStore(_group_0_binding_2_cs, itc.x, ((((value1_ + value2_) + value4_) + value5_) + value6_));
    imageStore(_group_0_binding_2_cs, int(uint(itc.x)), ((((value1u + value2u) + value4u) + value5u) + value6u));
    return;
}

