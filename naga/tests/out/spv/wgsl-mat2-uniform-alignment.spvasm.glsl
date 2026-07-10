#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _7
{
    int _m0;
    mat2 _m1;
};

struct _8
{
    int _m0;
    vec2 _m1;
    vec2 _m2;
};

layout(set = 0, binding = 0, std140) uniform _10_9
{
    _8 _m0;
} _9;

_7 _12(_8 _14)
{
    return _7(_14._m0, mat2(_14._m1, _14._m2));
}

void main()
{
}

