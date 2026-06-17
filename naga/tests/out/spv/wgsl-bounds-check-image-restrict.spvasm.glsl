#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture1D image_1d;
layout(set = 0, binding = 1) uniform texture2D image_2d;
layout(set = 0, binding = 2) uniform texture2DArray image_2d_array;
layout(set = 0, binding = 3) uniform texture3D image_3d;
layout(set = 0, binding = 4) uniform texture2DMS image_multisampled_2d;
layout(set = 0, binding = 5, rgba8) uniform writeonly image1D image_storage_1d;
layout(set = 0, binding = 6, rgba8) uniform writeonly image2D image_storage_2d;
layout(set = 0, binding = 7, rgba8) uniform writeonly image2DArray image_storage_2d_array;
layout(set = 0, binding = 8, rgba8) uniform writeonly image3D image_storage_3d;

layout(location = 0) out vec4 _172;

vec4 test_textureLoad_1d(int coords, int level)
{
    int _47 = int(min(uint(level), uint(textureQueryLevels(image_1d) - 1)));
    return texelFetch(image_1d, int(min(uint(coords), uint(textureSize(image_1d, _47) - 1))), _47);
}

vec4 test_textureLoad_2d(ivec2 coords, int level)
{
    int _61 = int(min(uint(level), uint(textureQueryLevels(image_2d) - 1)));
    return texelFetch(image_2d, ivec2(min(uvec2(coords), uvec2(textureSize(image_2d, _61) - ivec2(1)))), _61);
}

vec4 test_textureLoad_2d_array_u(ivec2 coords, uint index, int level)
{
    int _79 = int(min(uint(level), uint(textureQueryLevels(image_2d_array) - 1)));
    return texelFetch(image_2d_array, ivec3(min(uvec3(ivec3(coords, int(index))), uvec3(textureSize(image_2d_array, _79) - ivec3(1)))), _79);
}

vec4 test_textureLoad_2d_array_s(ivec2 coords, int index, int level)
{
    int _96 = int(min(uint(level), uint(textureQueryLevels(image_2d_array) - 1)));
    return texelFetch(image_2d_array, ivec3(min(uvec3(ivec3(coords, index)), uvec3(textureSize(image_2d_array, _96) - ivec3(1)))), _96);
}

vec4 test_textureLoad_3d(ivec3 coords, int level)
{
    int _111 = int(min(uint(level), uint(textureQueryLevels(image_3d) - 1)));
    return texelFetch(image_3d, ivec3(min(uvec3(coords), uvec3(textureSize(image_3d, _111) - ivec3(1)))), _111);
}

vec4 test_textureLoad_multisampled_2d(ivec2 coords, int _sample)
{
    return texelFetch(image_multisampled_2d, ivec2(min(uvec2(coords), uvec2(textureSize(image_multisampled_2d) - ivec2(1)))), int(min(uint(_sample), uint(textureSamples(image_multisampled_2d) - 1))));
}

void test_textureStore_1d(int coords, vec4 value)
{
    imageStore(image_storage_1d, coords, value);
}

void test_textureStore_2d(ivec2 coords, vec4 value)
{
    imageStore(image_storage_2d, coords, value);
}

void test_textureStore_2d_array_u(ivec2 coords, uint array_index, vec4 value)
{
    imageStore(image_storage_2d_array, ivec3(coords, int(array_index)), value);
}

void test_textureStore_2d_array_s(ivec2 coords, int array_index, vec4 value)
{
    imageStore(image_storage_2d_array, ivec3(coords, array_index), value);
}

void test_textureStore_3d(ivec3 coords, vec4 value)
{
    imageStore(image_storage_3d, coords, value);
}

void main()
{
    test_textureStore_1d(0, vec4(0.0));
    test_textureStore_2d(ivec2(0), vec4(0.0));
    test_textureStore_2d_array_u(ivec2(0), 0u, vec4(0.0));
    test_textureStore_2d_array_s(ivec2(0), 0, vec4(0.0));
    test_textureStore_3d(ivec3(0), vec4(0.0));
    _172 = vec4(0.0);
}

