#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _14_13
{
    uint _m0[8];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    uvec4 _m0;
} _16;

void main()
{
    _13._m0[0u] = 0u;
    _13._m0[1u] = 7u;
    _13._m0[2u] = 0u;
    _13._m0[3u] = floatBitsToUint(0.0);
    _13._m0[4u] = floatBitsToUint(1.0);
    _13._m0[5u] = 0u;
    _13._m0[6u] = 9u;
    _13._m0[7u] = 7u;
    _16._m0 = uvec4(0u, 0u, 7u, 9u);
}

