#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D image_depth_2d;
layout(set = 0, binding = 1) uniform texture2DArray image_depth_2d_array;
layout(set = 0, binding = 2) uniform texture2DMS image_depth_multisampled_2d;

layout(location = 0) out vec4 _89;

float test_textureLoad_depth_2d(ivec2 coords, int level)
{
    int _28 = int(min(uint(level), uint(textureQueryLevels(image_depth_2d) - 1)));
    return texelFetch(image_depth_2d, ivec2(min(uvec2(coords), uvec2(textureSize(image_depth_2d, _28) - ivec2(1)))), _28).x;
}

float test_textureLoad_depth_2d_array_u(ivec2 coords, uint index, int level)
{
    int _48 = int(min(uint(level), uint(textureQueryLevels(image_depth_2d_array) - 1)));
    return texelFetch(image_depth_2d_array, ivec3(min(uvec3(ivec3(coords, int(index))), uvec3(textureSize(image_depth_2d_array, _48) - ivec3(1)))), _48).x;
}

float test_textureLoad_depth_2d_array_s(ivec2 coords, int index, int level)
{
    int _66 = int(min(uint(level), uint(textureQueryLevels(image_depth_2d_array) - 1)));
    return texelFetch(image_depth_2d_array, ivec3(min(uvec3(ivec3(coords, index)), uvec3(textureSize(image_depth_2d_array, _66) - ivec3(1)))), _66).x;
}

float test_textureLoad_depth_multisampled_2d(ivec2 coords, int _sample)
{
    return texelFetch(image_depth_multisampled_2d, ivec2(min(uvec2(coords), uvec2(textureSize(image_depth_multisampled_2d) - ivec2(1)))), int(min(uint(_sample), uint(textureSamples(image_depth_multisampled_2d) - 1)))).x;
}

void main()
{
    _89 = vec4(0.0);
}

