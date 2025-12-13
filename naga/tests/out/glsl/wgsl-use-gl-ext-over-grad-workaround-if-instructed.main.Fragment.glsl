#version 320 es
#extension GL_EXT_texture_shadow_lod : require

precision highp float;
precision highp int;

uniform highp sampler2DArrayShadow _group_0_binding_0_fs;

layout(location = 0) out float _fs2p_location0;

void main() {
    vec2 pos = vec2(0.0);
    float _e6 = textureLod(_group_0_binding_0_fs, vec4(pos, 0, 0.0), 0.0);
    _fs2p_location0 = _e6;
    return;
}

