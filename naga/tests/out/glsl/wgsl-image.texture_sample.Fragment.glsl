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
    vec4 _e11 = texture(_group_0_binding_0_fs, 0.5);
    vec4 _e12 = a;
    a = (_e12 + _e11);
    vec4 _e16 = texture(_group_0_binding_1_fs, vec2(_e1));
    vec4 _e17 = a;
    a = (_e17 + _e16);
    vec4 _e24 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1));
    vec4 _e25 = a;
    a = (_e25 + _e24);
    vec4 _e29 = textureLod(_group_0_binding_1_fs, vec2(_e1), 2.3);
    vec4 _e30 = a;
    a = (_e30 + _e29);
    vec4 _e34 = textureLodOffset(_group_0_binding_1_fs, vec2(_e1), 2.3, ivec2(3, 1));
    vec4 _e35 = a;
    a = (_e35 + _e34);
    vec4 _e40 = textureOffset(_group_0_binding_1_fs, vec2(_e1), ivec2(3, 1), 2.0);
    vec4 _e41 = a;
    a = (_e41 + _e40);
    vec4 _e45 = textureLod(_group_0_binding_1_fs, vec2(_e1), 0.0);
    vec4 _e46 = a;
    a = (_e46 + _e45);
    vec4 _e51 = texture(_group_0_binding_4_fs, vec3(_e1, 0u));
    vec4 _e52 = a;
    a = (_e52 + _e51);
    vec4 _e57 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1));
    vec4 _e58 = a;
    a = (_e58 + _e57);
    vec4 _e63 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3);
    vec4 _e64 = a;
    a = (_e64 + _e63);
    vec4 _e69 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3, ivec2(3, 1));
    vec4 _e70 = a;
    a = (_e70 + _e69);
    vec4 _e76 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1), 2.0);
    vec4 _e77 = a;
    a = (_e77 + _e76);
    vec4 _e82 = texture(_group_0_binding_4_fs, vec3(_e1, 0));
    vec4 _e83 = a;
    a = (_e83 + _e82);
    vec4 _e88 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1));
    vec4 _e89 = a;
    a = (_e89 + _e88);
    vec4 _e94 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0), 2.3);
    vec4 _e95 = a;
    a = (_e95 + _e94);
    vec4 _e100 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0), 2.3, ivec2(3, 1));
    vec4 _e101 = a;
    a = (_e101 + _e100);
    vec4 _e107 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1), 2.0);
    vec4 _e108 = a;
    a = (_e108 + _e107);
    vec4 _e113 = texture(_group_0_binding_6_fs, vec4(_e3, 0u));
    vec4 _e114 = a;
    a = (_e114 + _e113);
    vec4 _e119 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0u), 2.3);
    vec4 _e120 = a;
    a = (_e120 + _e119);
    vec4 _e126 = texture(_group_0_binding_6_fs, vec4(_e3, 0u), 2.0);
    vec4 _e127 = a;
    a = (_e127 + _e126);
    vec4 _e132 = texture(_group_0_binding_6_fs, vec4(_e3, 0));
    vec4 _e133 = a;
    a = (_e133 + _e132);
    vec4 _e138 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0), 2.3);
    vec4 _e139 = a;
    a = (_e139 + _e138);
    vec4 _e145 = texture(_group_0_binding_6_fs, vec4(_e3, 0), 2.0);
    vec4 _e146 = a;
    a = (_e146 + _e145);
    vec4 _e148 = a;
    _fs2p_location0 = _e148;
    return;
}

