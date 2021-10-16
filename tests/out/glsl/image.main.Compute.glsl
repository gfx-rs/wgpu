#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

uniform highp usampler2D _group_0_binding_0;

uniform highp usampler2DMS _group_0_binding_3;

layout(rgba8ui) readonly uniform highp uimage2D _group_0_binding_1;

uniform highp usampler2DArray _group_0_binding_5;

layout(r32ui) writeonly uniform highp uimage2D _group_0_binding_2;


void main() {
    uvec3 local_id = gl_LocalInvocationID;
    ivec2 dim = imageSize(_group_0_binding_1).xy;
    ivec2 itc = ((dim * ivec2(local_id.xy)) % ivec2(10, 20));
    uvec4 value1 = texelFetch(_group_0_binding_0, itc, int(local_id.z));
    uvec4 value2 = texelFetch(_group_0_binding_3, itc, int(local_id.z));
    uvec4 value4 = imageLoad(_group_0_binding_1, itc);
    uvec4 value5 = texelFetch(_group_0_binding_5, ivec3(itc, int(local_id.z)), (int(local_id.z) + 1));
    imageStore(_group_0_binding_2, ivec2(itc.x, 0.0), (((value1 + value2) + value4) + value5));
    return;
}

