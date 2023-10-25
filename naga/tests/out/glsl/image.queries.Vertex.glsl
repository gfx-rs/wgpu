#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_vs;

uniform highp sampler2D _group_0_binding_1_vs;

uniform highp sampler2DArray _group_0_binding_4_vs;

uniform highp samplerCube _group_0_binding_5_vs;

uniform highp samplerCubeArray _group_0_binding_6_vs;

uniform highp sampler3D _group_0_binding_7_vs;

uniform highp sampler2DMS _group_0_binding_8_vs;


void main() {
    uint dim_1d = uint(textureSize(_group_0_binding_0_vs, 0).x);
    uint dim_1d_lod = uint(textureSize(_group_0_binding_0_vs, int(dim_1d)).x);
    uvec2 dim_2d = uvec2(textureSize(_group_0_binding_1_vs, 0).xy);
    uvec2 dim_2d_lod = uvec2(textureSize(_group_0_binding_1_vs, 1).xy);
    uvec2 dim_2d_array = uvec2(textureSize(_group_0_binding_4_vs, 0).xy);
    uvec2 dim_2d_array_lod = uvec2(textureSize(_group_0_binding_4_vs, 1).xy);
    uvec2 dim_cube = uvec2(textureSize(_group_0_binding_5_vs, 0).xy);
    uvec2 dim_cube_lod = uvec2(textureSize(_group_0_binding_5_vs, 1).xy);
    uvec2 dim_cube_array = uvec2(textureSize(_group_0_binding_6_vs, 0).xy);
    uvec2 dim_cube_array_lod = uvec2(textureSize(_group_0_binding_6_vs, 1).xy);
    uvec3 dim_3d = uvec3(textureSize(_group_0_binding_7_vs, 0).xyz);
    uvec3 dim_3d_lod = uvec3(textureSize(_group_0_binding_7_vs, 1).xyz);
    uvec2 dim_2s_ms = uvec2(textureSize(_group_0_binding_8_vs).xy);
    uint sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    gl_Position = vec4(float(sum));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

