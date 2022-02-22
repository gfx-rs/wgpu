#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2DShadow _group_1_binding_2_fs;

uniform highp samplerCubeShadow _group_1_binding_3_fs;

layout(location = 0) out float _fs2p_location0;

void main() {
    vec2 tc = vec2(0.5);
    float s2d_depth = texture(_group_1_binding_2_fs, vec3(tc, 0.5));
    float s2d_depth_level = textureLod(_group_1_binding_2_fs, vec3(tc, 0.5), 0.0);
    float scube_depth_level = textureGrad(_group_1_binding_3_fs, vec4(vec3(0.5), 0.5), vec3(0.0), vec3(0.0));
    _fs2p_location0 = (s2d_depth + s2d_depth_level);
    return;
}

