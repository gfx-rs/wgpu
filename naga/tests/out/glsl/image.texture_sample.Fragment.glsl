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
    vec4 _e9 = texture(_group_0_binding_0_fs, tc.x);
    vec4 _e10 = a;
    a = (_e10 + _e9);
    vec4 _e14 = texture(_group_0_binding_1_fs, vec2(tc));
    vec4 _e15 = a;
    a = (_e15 + _e14);
    vec4 _e22 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1));
    vec4 _e23 = a;
    a = (_e23 + _e22);
    vec4 _e27 = textureLod(_group_0_binding_1_fs, vec2(tc), 2.3);
    vec4 _e28 = a;
    a = (_e28 + _e27);
    vec4 _e35 = textureLodOffset(_group_0_binding_1_fs, vec2(tc), 2.3, ivec2(3, 1));
    vec4 _e36 = a;
    a = (_e36 + _e35);
    vec4 _e44 = textureOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 2.0);
    vec4 _e45 = a;
    a = (_e45 + _e44);
    vec4 _e50 = texture(_group_0_binding_4_fs, vec3(tc, 0u));
    vec4 _e51 = a;
    a = (_e51 + _e50);
    vec4 _e59 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1));
    vec4 _e60 = a;
    a = (_e60 + _e59);
    vec4 _e65 = textureLod(_group_0_binding_4_fs, vec3(tc, 0u), 2.3);
    vec4 _e66 = a;
    a = (_e66 + _e65);
    vec4 _e74 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0u), 2.3, ivec2(3, 1));
    vec4 _e75 = a;
    a = (_e75 + _e74);
    vec4 _e84 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0u), ivec2(3, 1), 2.0);
    vec4 _e85 = a;
    a = (_e85 + _e84);
    vec4 _e90 = texture(_group_0_binding_4_fs, vec3(tc, 0));
    vec4 _e91 = a;
    a = (_e91 + _e90);
    vec4 _e99 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1));
    vec4 _e100 = a;
    a = (_e100 + _e99);
    vec4 _e105 = textureLod(_group_0_binding_4_fs, vec3(tc, 0), 2.3);
    vec4 _e106 = a;
    a = (_e106 + _e105);
    vec4 _e114 = textureLodOffset(_group_0_binding_4_fs, vec3(tc, 0), 2.3, ivec2(3, 1));
    vec4 _e115 = a;
    a = (_e115 + _e114);
    vec4 _e124 = textureOffset(_group_0_binding_4_fs, vec3(tc, 0), ivec2(3, 1), 2.0);
    vec4 _e125 = a;
    a = (_e125 + _e124);
    vec4 _e130 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u));
    vec4 _e131 = a;
    a = (_e131 + _e130);
    vec4 _e136 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.3);
    vec4 _e137 = a;
    a = (_e137 + _e136);
    vec4 _e143 = texture(_group_0_binding_6_fs, vec4(tc3_, 0u), 2.0);
    vec4 _e144 = a;
    a = (_e144 + _e143);
    vec4 _e149 = texture(_group_0_binding_6_fs, vec4(tc3_, 0));
    vec4 _e150 = a;
    a = (_e150 + _e149);
    vec4 _e155 = textureLod(_group_0_binding_6_fs, vec4(tc3_, 0), 2.3);
    vec4 _e156 = a;
    a = (_e156 + _e155);
    vec4 _e162 = texture(_group_0_binding_6_fs, vec4(tc3_, 0), 2.0);
    vec4 _e163 = a;
    a = (_e163 + _e162);
    vec4 _e165 = a;
    _fs2p_location0 = _e165;
    return;
}

