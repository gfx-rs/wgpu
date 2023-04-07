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
    vec4 _e9 = texture(_group_0_binding_0_fs, vec2(tc.x, 0.0));
    vec4 _e10 = a;
    a = (_e10 + _e9);
    vec4 _e14 = texture(_group_0_binding_1_fs, vec2(tc));
    vec4 _e15 = a;
    a = (_e15 + _e14);
    vec4 _e19 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1));
    vec4 _e20 = a;
    a = (_e20 + _e19);
    vec4 _e24 = textureLod(_group_0_binding_1_fs, vec2(tc), 2.299999952316284);
    vec4 _e25 = a;
    a = (_e25 + _e24);
    vec4 _e29 = textureLodOffset(_group_0_binding_1_fs, vec2(tc), 2.299999952316284, ivec2(3, 1));
    vec4 _e30 = a;
    a = (_e30 + _e29);
    vec4 _e35 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 2.0);
    vec4 _e36 = a;
    a = (_e36 + _e35);
    vec4 _e41 = texture(_group_0_binding_4_fs, vec3(tc, 0u));
    vec4 _e42 = a;
    a = (_e42 + _e41);
    vec4 _e47 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1));
    vec4 _e48 = a;
    a = (_e48 + _e47);
    vec4 _e53 = textureLod(_group_0_binding_4_fs, vec3(tc, 0u), 2.299999952316284);
    vec4 _e54 = a;
    a = (_e54 + _e53);
    vec4 _e59 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0u), 2.299999952316284, ivec2(3, 1));
    vec4 _e60 = a;
    a = (_e60 + _e59);
    vec4 _e66 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1), 2.0);
    vec4 _e67 = a;
    a = (_e67 + _e66);
    vec4 _e72 = texture(_group_0_binding_4_fs, vec3(tc, 0));
    vec4 _e73 = a;
    a = (_e73 + _e72);
    vec4 _e78 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1));
    vec4 _e79 = a;
    a = (_e79 + _e78);
    vec4 _e84 = textureLod(_group_0_binding_4_fs, vec3(tc, 0), 2.299999952316284);
    vec4 _e85 = a;
    a = (_e85 + _e84);
    vec4 _e90 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0), 2.299999952316284, ivec2(3, 1));
    vec4 _e91 = a;
    a = (_e91 + _e90);
    vec4 _e97 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1), 2.0);
    vec4 _e98 = a;
    a = (_e98 + _e97);
    vec4 _e103 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u));
    vec4 _e104 = a;
    a = (_e104 + _e103);
    vec4 _e109 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.299999952316284);
    vec4 _e110 = a;
    a = (_e110 + _e109);
    vec4 _e116 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.0);
    vec4 _e117 = a;
    a = (_e117 + _e116);
    vec4 _e122 = texture(_group_0_binding_6_fs, vec4(tc3_, 0));
    vec4 _e123 = a;
    a = (_e123 + _e122);
    vec4 _e128 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0), 2.299999952316284);
    vec4 _e129 = a;
    a = (_e129 + _e128);
    vec4 _e135 = texture(_group_0_binding_6_fs, vec4(tc3_, 0), 2.0);
    vec4 _e136 = a;
    a = (_e136 + _e135);
    vec4 _e138 = a;
    _fs2p_location0 = _e138;
    return;
}

