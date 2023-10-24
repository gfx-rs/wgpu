#version 430 core
#extension GL_ARB_shader_texture_image_samples : require
uniform sampler1D _group_0_binding_0_fs;

uniform sampler2D _group_0_binding_1_fs;

uniform sampler2DArray _group_0_binding_2_fs;

uniform sampler3D _group_0_binding_3_fs;

uniform sampler2DMS _group_0_binding_4_fs;

layout(rgba8) writeonly uniform image1D _group_0_binding_8_fs;

layout(rgba8) writeonly uniform image2D _group_0_binding_9_fs;

layout(rgba8) writeonly uniform image2DArray _group_0_binding_10_fs;

layout(rgba8) writeonly uniform image3D _group_0_binding_11_fs;

layout(location = 0) out vec4 _fs2p_location0;

vec4 test_textureLoad_1d(int coords, int level) {
    vec4 _e3 = (level < textureQueryLevels(_group_0_binding_0_fs) && coords < textureSize(_group_0_binding_0_fs, level) ? texelFetch(_group_0_binding_0_fs, coords, level) : vec4(0.0));
    return _e3;
}

vec4 test_textureLoad_2d(ivec2 coords_1, int level_1) {
    vec4 _e3 = (level_1 < textureQueryLevels(_group_0_binding_1_fs) && all(lessThan(coords_1, textureSize(_group_0_binding_1_fs, level_1))) ? texelFetch(_group_0_binding_1_fs, coords_1, level_1) : vec4(0.0));
    return _e3;
}

vec4 test_textureLoad_2d_array_u(ivec2 coords_2, uint index, int level_2) {
    vec4 _e4 = (level_2 < textureQueryLevels(_group_0_binding_2_fs) && all(lessThan(ivec3(coords_2, index), textureSize(_group_0_binding_2_fs, level_2))) ? texelFetch(_group_0_binding_2_fs, ivec3(coords_2, index), level_2) : vec4(0.0));
    return _e4;
}

vec4 test_textureLoad_2d_array_s(ivec2 coords_3, int index_1, int level_3) {
    vec4 _e4 = (level_3 < textureQueryLevels(_group_0_binding_2_fs) && all(lessThan(ivec3(coords_3, index_1), textureSize(_group_0_binding_2_fs, level_3))) ? texelFetch(_group_0_binding_2_fs, ivec3(coords_3, index_1), level_3) : vec4(0.0));
    return _e4;
}

vec4 test_textureLoad_3d(ivec3 coords_4, int level_4) {
    vec4 _e3 = (level_4 < textureQueryLevels(_group_0_binding_3_fs) && all(lessThan(coords_4, textureSize(_group_0_binding_3_fs, level_4))) ? texelFetch(_group_0_binding_3_fs, coords_4, level_4) : vec4(0.0));
    return _e3;
}

vec4 test_textureLoad_multisampled_2d(ivec2 coords_5, int _sample) {
    vec4 _e3 = (_sample < textureSamples(_group_0_binding_4_fs) && all(lessThan(coords_5, textureSize(_group_0_binding_4_fs))) ? texelFetch(_group_0_binding_4_fs, coords_5, _sample) : vec4(0.0));
    return _e3;
}

void test_textureStore_1d(int coords_10, vec4 value) {
    imageStore(_group_0_binding_8_fs, coords_10, value);
    return;
}

void test_textureStore_2d(ivec2 coords_11, vec4 value_1) {
    imageStore(_group_0_binding_9_fs, coords_11, value_1);
    return;
}

void test_textureStore_2d_array_u(ivec2 coords_12, uint array_index, vec4 value_2) {
    imageStore(_group_0_binding_10_fs, ivec3(coords_12, array_index), value_2);
    return;
}

void test_textureStore_2d_array_s(ivec2 coords_13, int array_index_1, vec4 value_3) {
    imageStore(_group_0_binding_10_fs, ivec3(coords_13, array_index_1), value_3);
    return;
}

void test_textureStore_3d(ivec3 coords_14, vec4 value_4) {
    imageStore(_group_0_binding_11_fs, coords_14, value_4);
    return;
}

void main() {
    vec4 _e2 = test_textureLoad_1d(0, 0);
    vec4 _e5 = test_textureLoad_2d(ivec2(0), 0);
    vec4 _e9 = test_textureLoad_2d_array_u(ivec2(0), 0u, 0);
    vec4 _e13 = test_textureLoad_2d_array_s(ivec2(0), 0, 0);
    vec4 _e16 = test_textureLoad_3d(ivec3(0), 0);
    vec4 _e19 = test_textureLoad_multisampled_2d(ivec2(0), 0);
    test_textureStore_1d(0, vec4(0.0));
    test_textureStore_2d(ivec2(0), vec4(0.0));
    test_textureStore_2d_array_u(ivec2(0), 0u, vec4(0.0));
    test_textureStore_2d_array_s(ivec2(0), 0, vec4(0.0));
    test_textureStore_3d(ivec3(0), vec4(0.0));
    _fs2p_location0 = vec4(0.0, 0.0, 0.0, 0.0);
    return;
}

