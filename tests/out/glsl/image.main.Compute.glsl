#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

uniform highp usampler2D _group_0_binding_0_cs;

uniform highp usampler2DMS _group_0_binding_3_cs;

layout(rgba8ui) readonly uniform highp uimage2D _group_0_binding_1_cs;

uniform highp usampler2DArray _group_0_binding_5_cs;

uniform highp usampler2D _group_0_binding_7_cs;

layout(r32ui) writeonly uniform highp uimage2D _group_0_binding_2_cs;


void main() {
    uvec3 local_id = gl_LocalInvocationID;
    ivec2 dim = imageSize(_group_0_binding_1_cs).xy;
    ivec2 itc = ((dim * ivec2(local_id.xy)) % ivec2(10, 20));
    uvec4 value1_ = texelFetch(_group_0_binding_0_cs, itc, int(local_id.z));
    uvec4 value2_ = texelFetch(_group_0_binding_3_cs, itc, int(local_id.z));
    uvec4 value4_ = imageLoad(_group_0_binding_1_cs, itc);
    uvec4 value5_ = texelFetch(_group_0_binding_5_cs, ivec3(itc, int(local_id.z)), (int(local_id.z) + 1));
    uvec4 value6_ = texelFetch(_group_0_binding_7_cs, ivec2(int(local_id.x), 0), int(local_id.z));
    imageStore(_group_0_binding_2_cs, ivec2(itc.x, 0), ((((value1_ + value2_) + value4_) + value5_) + value6_));
    return;
}

