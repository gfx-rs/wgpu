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

layout(location = 0) out vec4 _187;

vec4 test_textureLoad_1d(int coords, int level)
{
    vec4 _54;
    if (uint(level) < uint(textureQueryLevels(image_1d)))
    {
        if (uint(coords) < uint(textureSize(image_1d, level)))
        {
            _54 = texelFetch(image_1d, coords, level);
        }
        else
        {
            _54 = vec4(0.0);
        }
    }
    else
    {
        _54 = vec4(0.0);
    }
    return _54;
}

vec4 test_textureLoad_2d(ivec2 coords, int level)
{
    vec4 _72;
    if (uint(level) < uint(textureQueryLevels(image_2d)))
    {
        if (all(lessThan(uvec2(coords), uvec2(textureSize(image_2d, level)))))
        {
            _72 = texelFetch(image_2d, coords, level);
        }
        else
        {
            _72 = vec4(0.0);
        }
    }
    else
    {
        _72 = vec4(0.0);
    }
    return _72;
}

vec4 test_textureLoad_2d_array_u(ivec2 coords, uint index, int level)
{
    ivec3 _82 = ivec3(coords, int(index));
    vec4 _93;
    if (uint(level) < uint(textureQueryLevels(image_2d_array)))
    {
        if (all(lessThan(uvec3(_82), uvec3(textureSize(image_2d_array, level)))))
        {
            _93 = texelFetch(image_2d_array, _82, level);
        }
        else
        {
            _93 = vec4(0.0);
        }
    }
    else
    {
        _93 = vec4(0.0);
    }
    return _93;
}

vec4 test_textureLoad_2d_array_s(ivec2 coords, int index, int level)
{
    ivec3 _102 = ivec3(coords, index);
    vec4 _112;
    if (uint(level) < uint(textureQueryLevels(image_2d_array)))
    {
        if (all(lessThan(uvec3(_102), uvec3(textureSize(image_2d_array, level)))))
        {
            _112 = texelFetch(image_2d_array, _102, level);
        }
        else
        {
            _112 = vec4(0.0);
        }
    }
    else
    {
        _112 = vec4(0.0);
    }
    return _112;
}

vec4 test_textureLoad_3d(ivec3 coords, int level)
{
    vec4 _129;
    if (uint(level) < uint(textureQueryLevels(image_3d)))
    {
        if (all(lessThan(uvec3(coords), uvec3(textureSize(image_3d, level)))))
        {
            _129 = texelFetch(image_3d, coords, level);
        }
        else
        {
            _129 = vec4(0.0);
        }
    }
    else
    {
        _129 = vec4(0.0);
    }
    return _129;
}

vec4 test_textureLoad_multisampled_2d(ivec2 coords, int _sample)
{
    vec4 _145;
    if (uint(_sample) < uint(textureSamples(image_multisampled_2d)))
    {
        if (all(lessThan(uvec2(coords), uvec2(textureSize(image_multisampled_2d)))))
        {
            _145 = texelFetch(image_multisampled_2d, coords, _sample);
        }
        else
        {
            _145 = vec4(0.0);
        }
    }
    else
    {
        _145 = vec4(0.0);
    }
    return _145;
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
    _187 = vec4(0.0);
}

