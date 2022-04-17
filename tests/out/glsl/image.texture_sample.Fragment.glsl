#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_fs;

uniform highp sampler2D _group_0_binding_1_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec2 tc = vec2(0.5);
    vec4 s1d = texture(_group_0_binding_0_fs, vec2(tc.x, 0.0));
    vec4 s2d = texture(_group_0_binding_1_fs, vec2(tc));
    vec4 s2d_offset = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1));
    vec4 s2d_level = textureLod(_group_0_binding_1_fs, vec2(tc), 2.299999952316284);
    vec4 s2d_level_offset = textureLodOffset(_group_0_binding_1_fs, vec2(tc), 2.299999952316284, ivec2(3, 1));
    vec4 s2d_bias_offset = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 2.0);
    _fs2p_location0 = ((((s1d + s2d) + s2d_offset) + s2d_level) + s2d_level_offset);
    return;
}

