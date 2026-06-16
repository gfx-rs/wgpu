#version 460
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D image_depth_2d;
layout(set = 0, binding = 1) uniform texture2DArray image_depth_2d_array;
layout(set = 0, binding = 2) uniform texture2DMS image_depth_multisampled_2d;

layout(location = 0) out vec4 _100;

float test_textureLoad_depth_2d(ivec2 coords, int level)
{
    vec4 _37;
    if (uint(level) < uint(textureQueryLevels(image_depth_2d)))
    {
        if (all(lessThan(uvec2(coords), uvec2(textureSize(image_depth_2d, level)))))
        {
            _37 = texelFetch(image_depth_2d, coords, level);
        }
        else
        {
            _37 = vec4(0.0);
        }
    }
    else
    {
        _37 = vec4(0.0);
    }
    return _37.x;
}

float test_textureLoad_depth_2d_array_u(ivec2 coords, uint index, int level)
{
    ivec3 _49 = ivec3(coords, int(index));
    vec4 _60;
    if (uint(level) < uint(textureQueryLevels(image_depth_2d_array)))
    {
        if (all(lessThan(uvec3(_49), uvec3(textureSize(image_depth_2d_array, level)))))
        {
            _60 = texelFetch(image_depth_2d_array, _49, level);
        }
        else
        {
            _60 = vec4(0.0);
        }
    }
    else
    {
        _60 = vec4(0.0);
    }
    return _60.x;
}

float test_textureLoad_depth_2d_array_s(ivec2 coords, int index, int level)
{
    ivec3 _70 = ivec3(coords, index);
    vec4 _80;
    if (uint(level) < uint(textureQueryLevels(image_depth_2d_array)))
    {
        if (all(lessThan(uvec3(_70), uvec3(textureSize(image_depth_2d_array, level)))))
        {
            _80 = texelFetch(image_depth_2d_array, _70, level);
        }
        else
        {
            _80 = vec4(0.0);
        }
    }
    else
    {
        _80 = vec4(0.0);
    }
    return _80.x;
}

float test_textureLoad_depth_multisampled_2d(ivec2 coords, int _sample)
{
    vec4 _97;
    if (uint(_sample) < uint(textureSamples(image_depth_multisampled_2d)))
    {
        if (all(lessThan(uvec2(coords), uvec2(textureSize(image_depth_multisampled_2d)))))
        {
            _97 = texelFetch(image_depth_multisampled_2d, coords, _sample);
        }
        else
        {
            _97 = vec4(0.0);
        }
    }
    else
    {
        _97 = vec4(0.0);
    }
    return _97.x;
}

void main()
{
    _100 = vec4(0.0);
}

