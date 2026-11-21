#version 430 core
uniform sampler2DShadow _group_1_binding_2_fs;

uniform sampler2DArrayShadow _group_1_binding_3_fs;

uniform samplerCubeShadow _group_1_binding_4_fs;

layout(location = 0) out float _fs2p_location0;

void main() {
    float a = 0.0;
    vec2 tc = vec2(0.5);
    vec3 tc3_ = vec3(0.5);
    float _e6 = a;
    float _e9 = texture(_group_1_binding_2_fs, vec3(tc, 0.5));
    a = (_e6 + _e9);
    float _e11 = a;
    float _e15 = texture(_group_1_binding_3_fs, vec4(tc, 0u, 0.5));
    a = (_e11 + _e15);
    float _e17 = a;
    float _e21 = texture(_group_1_binding_3_fs, vec4(tc, 0, 0.5));
    a = (_e17 + _e21);
    float _e23 = a;
    float _e26 = texture(_group_1_binding_4_fs, vec4(tc3_, 0.5));
    a = (_e23 + _e26);
    float _e28 = a;
    float _e31 = textureLod(_group_1_binding_2_fs, vec3(tc, 0.5), 0.0);
    a = (_e28 + _e31);
    float _e33 = a;
    float _e37 = textureGrad(_group_1_binding_3_fs, vec4(tc, 0u, 0.5), vec2(0.0), vec2(0.0));
    a = (_e33 + _e37);
    float _e39 = a;
    float _e43 = textureGrad(_group_1_binding_3_fs, vec4(tc, 0, 0.5), vec2(0.0), vec2(0.0));
    a = (_e39 + _e43);
    float _e45 = a;
    float _e48 = textureGrad(_group_1_binding_4_fs, vec4(tc3_, 0.5), vec3(0.0), vec3(0.0));
    a = (_e45 + _e48);
    float _e50 = a;
    _fs2p_location0 = _e50;
    return;
}

