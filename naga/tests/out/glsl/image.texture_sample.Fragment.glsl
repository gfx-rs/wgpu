#version 430 core
uniform sampler1D _group_0_binding_0_fs;

uniform sampler2D _group_0_binding_1_fs;

uniform sampler2DArray _group_0_binding_4_fs;

uniform samplerCubeArray _group_0_binding_6_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 a = vec4(0.0);
    vec2 tc = vec2(0.5);
    vec3 tc3_ = vec3(0.5);
    ivec2 offset = ivec2(3, 1);
    vec4 _e11 = texture(_group_0_binding_0_fs, 0.5);
    vec4 _e12 = a;
    a = (_e12 + _e11);
    vec4 _e16 = texture(_group_0_binding_1_fs, vec2(tc));
    vec4 _e17 = a;
    a = (_e17 + _e16);
    vec4 _e24 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1));
    vec4 _e25 = a;
    a = (_e25 + _e24);
    vec4 _e29 = textureLod(_group_0_binding_1_fs, vec2(tc), 2.3);
    vec4 _e30 = a;
    a = (_e30 + _e29);
    vec4 _e34 = textureLodOffset(_group_0_binding_1_fs, vec2(tc), 2.3, ivec2(3, 1));
    vec4 _e35 = a;
    a = (_e35 + _e34);
    vec4 _e40 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 2.0);
    vec4 _e41 = a;
    a = (_e41 + _e40);
    vec4 _e46 = texture(_group_0_binding_4_fs, vec3(tc, 0u));
    vec4 _e47 = a;
    a = (_e47 + _e46);
    vec4 _e52 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1));
    vec4 _e53 = a;
    a = (_e53 + _e52);
    vec4 _e58 = textureLod(_group_0_binding_4_fs, vec3(tc, 0u), 2.3);
    vec4 _e59 = a;
    a = (_e59 + _e58);
    vec4 _e64 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0u), 2.3, ivec2(3, 1));
    vec4 _e65 = a;
    a = (_e65 + _e64);
    vec4 _e71 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1), 2.0);
    vec4 _e72 = a;
    a = (_e72 + _e71);
    vec4 _e77 = texture(_group_0_binding_4_fs, vec3(tc, 0));
    vec4 _e78 = a;
    a = (_e78 + _e77);
    vec4 _e83 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1));
    vec4 _e84 = a;
    a = (_e84 + _e83);
    vec4 _e89 = textureLod(_group_0_binding_4_fs, vec3(tc, 0), 2.3);
    vec4 _e90 = a;
    a = (_e90 + _e89);
    vec4 _e95 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0), 2.3, ivec2(3, 1));
    vec4 _e96 = a;
    a = (_e96 + _e95);
    vec4 _e102 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1), 2.0);
    vec4 _e103 = a;
    a = (_e103 + _e102);
    vec4 _e108 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u));
    vec4 _e109 = a;
    a = (_e109 + _e108);
    vec4 _e114 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.3);
    vec4 _e115 = a;
    a = (_e115 + _e114);
    vec4 _e121 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.0);
    vec4 _e122 = a;
    a = (_e122 + _e121);
    vec4 _e127 = texture(_group_0_binding_6_fs, vec4(tc3_, 0));
    vec4 _e128 = a;
    a = (_e128 + _e127);
    vec4 _e133 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0), 2.3);
    vec4 _e134 = a;
    a = (_e134 + _e133);
    vec4 _e140 = texture(_group_0_binding_6_fs, vec4(tc3_, 0), 2.0);
    vec4 _e141 = a;
    a = (_e141 + _e140);
    vec4 _e143 = a;
    _fs2p_location0 = _e143;
    return;
}

