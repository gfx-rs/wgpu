#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_vs;

uniform highp sampler2D _group_0_binding_1_vs;

uniform highp sampler2DArray _group_0_binding_2_vs;

uniform highp samplerCube _group_0_binding_3_vs;

uniform highp samplerCubeArray _group_0_binding_4_vs;

uniform highp sampler3D _group_0_binding_5_vs;


void main() {
    int dim_1d = textureSize(_group_0_binding_0_vs, 0).x;
    int dim_1d_lod = textureSize(_group_0_binding_0_vs, int(dim_1d)).x;
    ivec2 dim_2d = textureSize(_group_0_binding_1_vs, 0).xy;
    ivec2 dim_2d_lod = textureSize(_group_0_binding_1_vs, 1).xy;
    ivec2 dim_2d_array = textureSize(_group_0_binding_2_vs, 0).xy;
    ivec2 dim_2d_array_lod = textureSize(_group_0_binding_2_vs, 1).xy;
    ivec2 dim_cube = textureSize(_group_0_binding_3_vs, 0).xy;
    ivec2 dim_cube_lod = textureSize(_group_0_binding_3_vs, 1).xy;
    ivec2 dim_cube_array = textureSize(_group_0_binding_4_vs, 0).xy;
    ivec2 dim_cube_array_lod = textureSize(_group_0_binding_4_vs, 1).xy;
    ivec3 dim_3d = textureSize(_group_0_binding_5_vs, 0).xyz;
    ivec3 dim_3d_lod = textureSize(_group_0_binding_5_vs, 1).xyz;
    int sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    gl_Position = vec4(float(sum));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

