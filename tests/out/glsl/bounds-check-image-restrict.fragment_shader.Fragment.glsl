#version 430 core
#extension GL_ARB_shader_texture_image_samples : require
uniform highp sampler1D _group_0_binding_0_fs;

uniform highp sampler2D _group_0_binding_1_fs;

uniform highp sampler2DArray _group_0_binding_2_fs;

uniform highp sampler3D _group_0_binding_3_fs;

uniform highp sampler2DMS _group_0_binding_4_fs;

layout(rgba8) writeonly uniform highp image1D _group_0_binding_8_fs;

layout(rgba8) writeonly uniform highp image2D _group_0_binding_9_fs;

layout(rgba8) writeonly uniform highp image2DArray _group_0_binding_10_fs;

layout(rgba8) writeonly uniform highp image3D _group_0_binding_11_fs;

layout(location = 0) out vec4 _fs2p_location0;

vec4 test_textureLoad_1d(int coords, int level) {
    int _e3_clamped_lod = clamp(level, 0, textureQueryLevels(_group_0_binding_0_fs) - 1);
    vec4 _e3 = texelFetch(_group_0_binding_0_fs, clamp(coords, 0, textureSize(_group_0_binding_0_fs, _e3_clamped_lod) - 1), _e3_clamped_lod);
    return _e3;
}

vec4 test_textureLoad_2d(ivec2 coords_1, int level_1) {
    int _e4_clamped_lod = clamp(level_1, 0, textureQueryLevels(_group_0_binding_1_fs) - 1);
    vec4 _e4 = texelFetch(_group_0_binding_1_fs, clamp(coords_1, ivec2(0), textureSize(_group_0_binding_1_fs, _e4_clamped_lod) - ivec2(1)), _e4_clamped_lod);
    return _e4;
}

vec4 test_textureLoad_2d_array(ivec2 coords_2, int index, int level_2) {
    int _e6_clamped_lod = clamp(level_2, 0, textureQueryLevels(_group_0_binding_2_fs) - 1);
    vec4 _e6 = texelFetch(_group_0_binding_2_fs, clamp(ivec3(coords_2, index), ivec3(0), textureSize(_group_0_binding_2_fs, _e6_clamped_lod) - ivec3(1)), _e6_clamped_lod);
    return _e6;
}

vec4 test_textureLoad_3d(ivec3 coords_3, int level_3) {
    int _e6_clamped_lod = clamp(level_3, 0, textureQueryLevels(_group_0_binding_3_fs) - 1);
    vec4 _e6 = texelFetch(_group_0_binding_3_fs, clamp(coords_3, ivec3(0), textureSize(_group_0_binding_3_fs, _e6_clamped_lod) - ivec3(1)), _e6_clamped_lod);
    return _e6;
}

vec4 test_textureLoad_multisampled_2d(ivec2 coords_4, int _sample) {
    vec4 _e7 = texelFetch(_group_0_binding_4_fs, clamp(coords_4, ivec2(0), textureSize(_group_0_binding_4_fs) - ivec2(1)), clamp(_sample, 0, textureSamples(_group_0_binding_4_fs) - 1)
);
    return _e7;
}

void test_textureStore_1d(int coords_8, vec4 value) {
    imageStore(_group_0_binding_8_fs, coords_8, value);
    return;
}

void test_textureStore_2d(ivec2 coords_9, vec4 value_1) {
    imageStore(_group_0_binding_9_fs, coords_9, value_1);
    return;
}

void test_textureStore_2d_array(ivec2 coords_10, int array_index, vec4 value_2) {
    imageStore(_group_0_binding_10_fs, ivec3(coords_10, array_index), value_2);
    return;
}

void test_textureStore_3d(ivec3 coords_11, vec4 value_3) {
    imageStore(_group_0_binding_11_fs, coords_11, value_3);
    return;
}

void main() {
    vec4 _e14 = test_textureLoad_1d(0, 0);
    vec4 _e17 = test_textureLoad_2d(ivec2(0, 0), 0);
    vec4 _e21 = test_textureLoad_2d_array(ivec2(0, 0), 0, 0);
    vec4 _e24 = test_textureLoad_3d(ivec3(0, 0, 0), 0);
    vec4 _e27 = test_textureLoad_multisampled_2d(ivec2(0, 0), 0);
    test_textureStore_1d(0, vec4(0.0, 0.0, 0.0, 0.0));
    test_textureStore_2d(ivec2(0, 0), vec4(0.0, 0.0, 0.0, 0.0));
    test_textureStore_2d_array(ivec2(0, 0), 0, vec4(0.0, 0.0, 0.0, 0.0));
    test_textureStore_3d(ivec3(0, 0, 0), vec4(0.0, 0.0, 0.0, 0.0));
    _fs2p_location0 = vec4(0.0, 0.0, 0.0, 0.0);
    return;
}

