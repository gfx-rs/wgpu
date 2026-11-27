#version 460
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct InStorage
{
    vec4 a[10];
};

struct InUniform
{
    vec4 a[20];
};

const float _34[40] = float[](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const vec4 _52[2] = vec4[](vec4(0.7070000171661376953125, 0.0, 0.0, 1.0), vec4(0.0, 0.7070000171661376953125, 0.0, 1.0));

layout(set = 0, binding = 0, std430) readonly buffer in_storage
{
    InStorage _m0;
} in_storage_1;

layout(set = 0, binding = 1, std140) uniform in_uniform
{
    InUniform _m0;
} in_uniform_1;

layout(set = 0, binding = 2) uniform texture2DArray image_2d_array;

shared float in_workgroup[30];

vec4 mock_function(ivec2 c, int i, int l)
{
    ivec3 _66 = ivec3(c, i);
    vec4 _79;
    if (uint(l) < uint(textureQueryLevels(image_2d_array)))
    {
        if (all(lessThan(uvec3(_66), uvec3(textureSize(image_2d_array, l)))))
        {
            _79 = texelFetch(image_2d_array, _66, l);
        }
        else
        {
            _79 = vec4(0.0);
        }
    }
    else
    {
        _79 = vec4(0.0);
    }
    return ((((in_storage_1._m0.a[i] + in_uniform_1._m0.a[i]) + _79) + vec4(in_workgroup[min(uint(i), 29u)])) + vec4(_34[min(uint(i), 39u)])) + _52[min(uint(i), 1u)];
}

void main()
{
    uint zero_init_index = 0u;
    uvec2 loop_bound = uvec2(4294967295u);
    zero_init_index = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (zero_init_index >= 30u)
        {
            break;
        }
        in_workgroup[min(zero_init_index, 29u)] = 0.0;
        zero_init_index++;
        continue;
    }
    barrier();
}

