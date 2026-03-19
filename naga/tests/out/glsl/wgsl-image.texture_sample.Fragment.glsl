#version 430 core
uniform sampler1D _group_0_binding_0_fs;

uniform sampler2D _group_0_binding_1_fs;

uniform sampler2DArray _group_0_binding_4_fs;

uniform samplerCubeArray _group_0_binding_6_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 a = vec4(0.0);
    vec2 _e1 = vec2(0.5);
    vec3 _e3 = vec3(0.5);
    ivec2 _e6 = ivec2(3, 1);
    vec4 _e9 = a;
    vec4 _e12 = texture(_group_0_binding_0_fs, 0.5);
    a = (_e9 + _e12);
    vec4 _e14 = a;
    vec4 _e17 = texture(_group_0_binding_1_fs, vec2(_e1));
    a = (_e14 + _e17);
    vec4 _e19 = a;
    vec4 _e25 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1));
    a = (_e19 + _e25);
    vec4 _e27 = a;
    vec4 _e30 = textureLod(_group_0_binding_1_fs, vec2(_e1), 2.3);
    a = (_e27 + _e30);
    vec4 _e32 = a;
    vec4 _e35 = textureLodOffset(_group_0_binding_1_fs, vec2(_e1), 2.3, ivec2(3, 1));
    a = (_e32 + _e35);
    vec4 _e37 = a;
    vec4 _e41 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1), 2.0);
    a = (_e37 + _e41);
    vec4 _e43 = a;
    vec4 _e46 = textureLod(_group_0_binding_1_fs, vec2(_e1), 0.0);
    a = (_e43 + _e46);
    vec4 _e48 = a;
    vec4 _e52 = texture(_group_0_binding_4_fs, vec3(_e1, 0u));
    a = (_e48 + _e52);
    vec4 _e54 = a;
    vec4 _e58 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1));
    a = (_e54 + _e58);
    vec4 _e60 = a;
    vec4 _e64 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3);
    a = (_e60 + _e64);
    vec4 _e66 = a;
    vec4 _e70 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3, ivec2(3, 1));
    a = (_e66 + _e70);
    vec4 _e72 = a;
    vec4 _e77 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1), 2.0);
    a = (_e72 + _e77);
    vec4 _e79 = a;
    vec4 _e83 = texture(_group_0_binding_4_fs, vec3(_e1, 0));
    a = (_e79 + _e83);
    vec4 _e85 = a;
    vec4 _e89 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1));
    a = (_e85 + _e89);
    vec4 _e91 = a;
    vec4 _e95 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0), 2.3);
    a = (_e91 + _e95);
    vec4 _e97 = a;
    vec4 _e101 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0), 2.3, ivec2(3, 1));
    a = (_e97 + _e101);
    vec4 _e103 = a;
    vec4 _e108 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1), 2.0);
    a = (_e103 + _e108);
    vec4 _e110 = a;
    vec4 _e114 = texture(_group_0_binding_6_fs, vec4(_e3, 0u));
    a = (_e110 + _e114);
    vec4 _e116 = a;
    vec4 _e120 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0u), 2.3);
    a = (_e116 + _e120);
    vec4 _e122 = a;
    vec4 _e127 = texture(_group_0_binding_6_fs, vec4(_e3, 0u), 2.0);
    a = (_e122 + _e127);
    vec4 _e129 = a;
    vec4 _e133 = texture(_group_0_binding_6_fs, vec4(_e3, 0));
    a = (_e129 + _e133);
    vec4 _e135 = a;
    vec4 _e139 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0), 2.3);
    a = (_e135 + _e139);
    vec4 _e141 = a;
    vec4 _e146 = texture(_group_0_binding_6_fs, vec4(_e3, 0), 2.0);
    a = (_e141 + _e146);
    vec4 _e148 = a;
    _fs2p_location0 = _e148;
    return;
}

