#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0;

uniform highp sampler2D _group_0_binding_1;

uniform highp sampler2DArray _group_0_binding_2;

uniform highp samplerCube _group_0_binding_3;

uniform highp samplerCubeArray _group_0_binding_4;

uniform highp sampler3D _group_0_binding_5;


void main() {
    int dim_1d = textureSize(_group_0_binding_0, 0).x;
    ivec2 dim_2d = textureSize(_group_0_binding_1, 0).xy;
    ivec2 dim_2d_lod = textureSize(_group_0_binding_1, 1).xy;
    ivec2 dim_2d_array = textureSize(_group_0_binding_2, 0).xy;
    ivec2 dim_2d_array_lod = textureSize(_group_0_binding_2, 1).xy;
    ivec2 dim_cube = textureSize(_group_0_binding_3, 0).xy;
    ivec2 dim_cube_lod = textureSize(_group_0_binding_3, 1).xy;
    ivec2 dim_cube_array = textureSize(_group_0_binding_4, 0).xy;
    ivec2 dim_cube_array_lod = textureSize(_group_0_binding_4, 1).xy;
    ivec3 dim_3d = textureSize(_group_0_binding_5, 0).xyz;
    ivec3 dim_3d_lod = textureSize(_group_0_binding_5, 1).xyz;
    int sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    gl_Position = vec4(float(sum));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

