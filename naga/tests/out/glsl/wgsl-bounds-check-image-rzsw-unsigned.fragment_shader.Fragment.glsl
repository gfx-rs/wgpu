#version 430 core
#extension GL_ARB_shader_texture_image_samples : require
uniform usampler2DArray _group_0_binding_0_fs;

uniform sampler3D _group_0_binding_1_fs;

uniform sampler2DMS _group_0_binding_2_fs;

layout(location = 0) out vec4 _fs2p_location0;

uvec4 test_textureLoad_2d_array_u32_level(ivec2 coords, int index, uint level) {
    uvec4 _e4 = (level < textureQueryLevels(_group_0_binding_0_fs) && all(lessThan(ivec3(coords, index), textureSize(_group_0_binding_0_fs, int(level)))) ? texelFetch(_group_0_binding_0_fs, ivec3(coords, index), int(level)) : uvec4(0u));
    return _e4;
}

vec4 test_textureLoad_3d_u32_level(ivec3 coords_1, uint level_1) {
    vec4 _e3 = (level_1 < textureQueryLevels(_group_0_binding_1_fs) && all(lessThan(coords_1, textureSize(_group_0_binding_1_fs, int(level_1)))) ? texelFetch(_group_0_binding_1_fs, coords_1, int(level_1)) : vec4(0.0));
    return _e3;
}

vec4 test_textureLoad_multisampled_2d_u32_sample(ivec2 coords_2, uint _sample) {
    vec4 _e3 = (_sample < textureSamples(_group_0_binding_2_fs) && all(lessThan(coords_2, textureSize(_group_0_binding_2_fs))) ? texelFetch(_group_0_binding_2_fs, coords_2, int(_sample)) : vec4(0.0));
    return _e3;
}

void main() {
    uvec4 _e3 = test_textureLoad_2d_array_u32_level(ivec2(0), 0, 0u);
    vec4 _e6 = test_textureLoad_3d_u32_level(ivec3(0), 0u);
    vec4 _e9 = test_textureLoad_multisampled_2d_u32_sample(ivec2(0), 0u);
    _fs2p_location0 = vec4(0.0, 0.0, 0.0, 0.0);
    return;
}

