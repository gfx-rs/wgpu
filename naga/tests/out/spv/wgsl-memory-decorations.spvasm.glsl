#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) coherent buffer _5_6
{
    uint _m0[];
} _6;

layout(set = 0, binding = 1, std430) buffer _5_8
{
    uint _m0[];
} _8;

layout(set = 0, binding = 2, std430) coherent buffer _5_9
{
    uint _m0[];
} _9;

layout(set = 0, binding = 3, std430) buffer _5_10
{
    uint _m0[];
} _10;

void main()
{
    _6._m0[0u] = _8._m0[0u];
    _9._m0[0u] = _10._m0[0u];
}

