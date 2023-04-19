#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp sampler2D _group_0_binding_0_fs;

uniform highp sampler2D _group_0_binding_1_fs;

uniform highp sampler2DArray _group_0_binding_4_fs;

uniform highp samplerCubeArray _group_0_binding_6_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 a = vec4(0.0);
    vec2 tc = vec2(0.5);
    vec3 tc3_ = vec3(0.5);
    vec4 _e8 = texture(_group_0_binding_0_fs, vec2(0.5, 0.0));
    vec4 _e9 = a;
    a = (_e9 + _e8);
    vec4 _e13 = texture(_group_0_binding_1_fs, vec2(tc));
    vec4 _e14 = a;
    a = (_e14 + _e13);
    vec4 _e18 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1));
    vec4 _e19 = a;
    a = (_e19 + _e18);
    vec4 _e23 = textureLod(_group_0_binding_1_fs, vec2(tc), 2.3);
    vec4 _e24 = a;
    a = (_e24 + _e23);
    vec4 _e28 = textureLodOffset(_group_0_binding_1_fs, vec2(tc), 2.3, ivec2(3, 1));
    vec4 _e29 = a;
    a = (_e29 + _e28);
    vec4 _e34 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 2.0);
    vec4 _e35 = a;
    a = (_e35 + _e34);
    vec4 _e40 = texture(_group_0_binding_4_fs, vec3(tc, 0u));
    vec4 _e41 = a;
    a = (_e41 + _e40);
    vec4 _e46 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1));
    vec4 _e47 = a;
    a = (_e47 + _e46);
    vec4 _e52 = textureLod(_group_0_binding_4_fs, vec3(tc, 0u), 2.3);
    vec4 _e53 = a;
    a = (_e53 + _e52);
    vec4 _e58 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0u), 2.3, ivec2(3, 1));
    vec4 _e59 = a;
    a = (_e59 + _e58);
    vec4 _e65 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1), 2.0);
    vec4 _e66 = a;
    a = (_e66 + _e65);
    vec4 _e71 = texture(_group_0_binding_4_fs, vec3(tc, 0));
    vec4 _e72 = a;
    a = (_e72 + _e71);
    vec4 _e77 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1));
    vec4 _e78 = a;
    a = (_e78 + _e77);
    vec4 _e83 = textureLod(_group_0_binding_4_fs, vec3(tc, 0), 2.3);
    vec4 _e84 = a;
    a = (_e84 + _e83);
    vec4 _e89 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0), 2.3, ivec2(3, 1));
    vec4 _e90 = a;
    a = (_e90 + _e89);
    vec4 _e96 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1), 2.0);
    vec4 _e97 = a;
    a = (_e97 + _e96);
    vec4 _e102 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u));
    vec4 _e103 = a;
    a = (_e103 + _e102);
    vec4 _e108 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.3);
    vec4 _e109 = a;
    a = (_e109 + _e108);
    vec4 _e115 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.0);
    vec4 _e116 = a;
    a = (_e116 + _e115);
    vec4 _e121 = texture(_group_0_binding_6_fs, vec4(tc3_, 0));
    vec4 _e122 = a;
    a = (_e122 + _e121);
    vec4 _e127 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0), 2.3);
    vec4 _e128 = a;
    a = (_e128 + _e127);
    vec4 _e134 = texture(_group_0_binding_6_fs, vec4(tc3_, 0), 2.0);
    vec4 _e135 = a;
    a = (_e135 + _e134);
    vec4 _e137 = a;
    _fs2p_location0 = _e137;
    return;
}

