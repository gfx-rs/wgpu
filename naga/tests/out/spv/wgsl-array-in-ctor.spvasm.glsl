#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    float _m0[2];
};

layout(set = 0, binding = 0, std430) readonly buffer _9_8
{
    _7 _m0;
} _8;

void main()
{
}

