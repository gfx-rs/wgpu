#version 320 es
#extension GL_EXT_texture_cube_map_array : require
#extension GL_EXT_texture_shadow_lod : require

precision highp float;
precision highp int;

uniform highp samplerCubeArrayShadow _group_0_binding_0_fs;

layout(location = 0) out float _fs2p_location0;

void main() {
    vec3 pos = vec3(0.0);
    float _e6 = textureLod(_group_0_binding_0_fs, vec4(pos, 0), 0.0, 0.0);
    _fs2p_location0 = _e6;
    return;
}

