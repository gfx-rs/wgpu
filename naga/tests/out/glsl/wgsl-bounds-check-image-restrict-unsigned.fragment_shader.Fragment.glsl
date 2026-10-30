#version 430 core
#extension GL_ARB_shader_texture_image_samples : require
uniform usampler2DArray _group_0_binding_0_fs;

uniform sampler3D _group_0_binding_1_fs;

uniform sampler2DMS _group_0_binding_2_fs;

layout(location = 0) out vec4 _fs2p_location0;

uvec4 test_textureLoad_2d_array_u32_level(ivec2 coords, int index, uint level) {
    int _e4_clamped_lod = clamp(int(level), 0, textureQueryLevels(_group_0_binding_0_fs) - 1);
    uvec4 _e4 = texelFetch(_group_0_binding_0_fs, clamp(ivec3(coords, index), ivec3(0), textureSize(_group_0_binding_0_fs, _e4_clamped_lod) - ivec3(1)), _e4_clamped_lod);
    return _e4;
}

vec4 test_textureLoad_3d_u32_level(ivec3 coords_1, uint level_1) {
    int _e3_clamped_lod = clamp(int(level_1), 0, textureQueryLevels(_group_0_binding_1_fs) - 1);
    vec4 _e3 = texelFetch(_group_0_binding_1_fs, clamp(coords_1, ivec3(0), textureSize(_group_0_binding_1_fs, _e3_clamped_lod) - ivec3(1)), _e3_clamped_lod);
    return _e3;
}

vec4 test_textureLoad_multisampled_2d_u32_sample(ivec2 coords_2, uint _sample) {
    vec4 _e3 = texelFetch(_group_0_binding_2_fs, clamp(coords_2, ivec2(0), textureSize(_group_0_binding_2_fs) - ivec2(1)), clamp(int(_sample), 0, textureSamples(_group_0_binding_2_fs) - 1)
);
    return _e3;
}

void main() {
    uvec4 _e3 = test_textureLoad_2d_array_u32_level(ivec2(0), 0, 0u);
    vec4 _e6 = test_textureLoad_3d_u32_level(ivec3(0), 0u);
    vec4 _e9 = test_textureLoad_multisampled_2d_u32_sample(ivec2(0), 0u);
    _fs2p_location0 = vec4(0.0, 0.0, 0.0, 0.0);
    return;
}

