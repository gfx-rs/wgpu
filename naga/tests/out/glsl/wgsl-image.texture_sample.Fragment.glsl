#version 430 core
uniform sampler1D _group_0_binding_0_fs;

uniform sampler2D _group_0_binding_1_fs;

uniform sampler2DArray _group_0_binding_4_fs;

uniform samplerCube _group_0_binding_5_fs;

uniform samplerCubeArray _group_0_binding_6_fs;

uniform sampler3D _group_0_binding_7_fs;

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
    vec4 _e46 = textureGrad(_group_0_binding_1_fs, vec2(_e1), _e1, _e1);
    a = (_e43 + _e46);
    vec4 _e48 = a;
    vec4 _e51 = textureLod(_group_0_binding_1_fs, vec2(_e1), 0.0);
    a = (_e48 + _e51);
    vec4 _e53 = a;
    vec4 _e57 = texture(_group_0_binding_4_fs, vec3(_e1, 0u));
    a = (_e53 + _e57);
    vec4 _e59 = a;
    vec4 _e63 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1));
    a = (_e59 + _e63);
    vec4 _e65 = a;
    vec4 _e69 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3);
    a = (_e65 + _e69);
    vec4 _e71 = a;
    vec4 _e75 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0u), 2.3, ivec2(3, 1));
    a = (_e71 + _e75);
    vec4 _e77 = a;
    vec4 _e82 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0u), ivec2(3, 1), 2.0);
    a = (_e77 + _e82);
    vec4 _e84 = a;
    vec4 _e88 = texture(_group_0_binding_4_fs, vec3(_e1, 0));
    a = (_e84 + _e88);
    vec4 _e90 = a;
    vec4 _e94 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1));
    a = (_e90 + _e94);
    vec4 _e96 = a;
    vec4 _e100 = textureLod(_group_0_binding_4_fs, vec3(_e1, 0), 2.3);
    a = (_e96 + _e100);
    vec4 _e102 = a;
    vec4 _e106 = textureLodOffset(_group_0_binding_4_fs, vec3(_e1, 0), 2.3, ivec2(3, 1));
    a = (_e102 + _e106);
    vec4 _e108 = a;
    vec4 _e113 = textureOffset(_group_0_binding_4_fs, vec3(_e1, 0), ivec2(3, 1), 2.0);
    a = (_e108 + _e113);
    vec4 _e115 = a;
    vec4 _e118 = textureGrad(_group_0_binding_5_fs, vec3(_e3), _e3, _e3);
    a = (_e115 + _e118);
    vec4 _e120 = a;
    vec4 _e124 = texture(_group_0_binding_6_fs, vec4(_e3, 0u));
    a = (_e120 + _e124);
    vec4 _e126 = a;
    vec4 _e130 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0u), 2.3);
    a = (_e126 + _e130);
    vec4 _e132 = a;
    vec4 _e137 = texture(_group_0_binding_6_fs, vec4(_e3, 0u), 2.0);
    a = (_e132 + _e137);
    vec4 _e139 = a;
    vec4 _e143 = textureGrad(_group_0_binding_6_fs, vec4(_e3, 0u), _e3, _e3);
    a = (_e139 + _e143);
    vec4 _e145 = a;
    vec4 _e149 = texture(_group_0_binding_6_fs, vec4(_e3, 0));
    a = (_e145 + _e149);
    vec4 _e151 = a;
    vec4 _e155 = textureLod(_group_0_binding_6_fs, vec4(_e3, 0), 2.3);
    a = (_e151 + _e155);
    vec4 _e157 = a;
    vec4 _e162 = texture(_group_0_binding_6_fs, vec4(_e3, 0), 2.0);
    a = (_e157 + _e162);
    vec4 _e164 = a;
    vec4 _e168 = textureGrad(_group_0_binding_6_fs, vec4(_e3, 0), _e3, _e3);
    a = (_e164 + _e168);
    vec4 _e170 = a;
    vec4 _e173 = textureGrad(_group_0_binding_7_fs, vec3(_e3), _e3, _e3);
    a = (_e170 + _e173);
    vec4 _e175 = a;
    _fs2p_location0 = _e175;
    return;
}

