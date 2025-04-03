#version 430 core
uniform sampler2DShadow _group_1_binding_2_fs;

uniform sampler2DArrayShadow _group_1_binding_3_fs;

uniform samplerCubeShadow _group_1_binding_4_fs;

layout(location = 0) out float _fs2p_location0;

void main() {
    float a_1 = 0.0;
    vec2 tc = vec2(0.5);
    vec3 tc3_ = vec3(0.5);
    float _e8 = texture(_group_1_binding_2_fs, vec3(tc, 0.5));
    float _e9 = a_1;
    a_1 = (_e9 + _e8);
    float _e14 = texture(_group_1_binding_3_fs, vec4(tc, 0u, 0.5));
    float _e15 = a_1;
    a_1 = (_e15 + _e14);
    float _e20 = texture(_group_1_binding_3_fs, vec4(tc, 0, 0.5));
    float _e21 = a_1;
    a_1 = (_e21 + _e20);
    float _e25 = texture(_group_1_binding_4_fs, vec4(tc3_, 0.5));
    float _e26 = a_1;
    a_1 = (_e26 + _e25);
    float _e30 = textureLod(_group_1_binding_2_fs, vec3(tc, 0.5), 0.0);
    float _e31 = a_1;
    a_1 = (_e31 + _e30);
    float _e36 = textureGrad(_group_1_binding_3_fs, vec4(tc, 0u, 0.5), vec2(0.0), vec2(0.0));
    float _e37 = a_1;
    a_1 = (_e37 + _e36);
    float _e42 = textureGrad(_group_1_binding_3_fs, vec4(tc, 0, 0.5), vec2(0.0), vec2(0.0));
    float _e43 = a_1;
    a_1 = (_e43 + _e42);
    float _e47 = textureGrad(_group_1_binding_4_fs, vec4(tc3_, 0.5), vec3(0.0), vec3(0.0));
    float _e48 = a_1;
    a_1 = (_e48 + _e47);
    float _e50 = a_1;
    _fs2p_location0 = _e50;
    return;
}

