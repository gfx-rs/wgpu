#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_1_fs;

uniform highp sampler2DShadow _group_1_binding_2_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec2 tc = vec2(0.5);
    vec4 s2d = textureGather(_group_0_binding_1_fs, vec2(tc), 1);
    vec4 s2d_offset = textureGatherOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 3);
    vec4 s2d_depth = textureGather(_group_1_binding_2_fs, vec2(tc), 0.5);
    vec4 s2d_depth_offset = textureGatherOffset(_group_1_binding_2_fs, vec2(tc), 0.5, ivec2(3, 1));
    _fs2p_location0 = (((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset);
    return;
}

