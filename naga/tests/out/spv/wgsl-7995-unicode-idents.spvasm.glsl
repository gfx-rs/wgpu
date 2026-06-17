#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _5_4
{
    float _m0;
} _4;

float _8()
{
    return _4._m0 + 9001.0;
}

void main()
{
}

