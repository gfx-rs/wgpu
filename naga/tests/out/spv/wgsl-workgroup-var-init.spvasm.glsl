#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct WStruct
{
    uint arr[512];
    int atom;
    int atom_arr[8][8];
};

layout(set = 0, binding = 0, std430) buffer _output
{
    uint _m0[512];
} _output_1;

shared WStruct w_mem;

void main()
{
    for (uint _34 = gl_LocalInvocationIndex; _34 < 512u; _34++)
    {
        w_mem.arr[_34] = 0u;
    }
    for (uint _46 = gl_LocalInvocationIndex; _46 < 64u; _46++)
    {
        w_mem.atom_arr[_46 / 8u][_46 % 8u] = 0;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        w_mem.atom = 0;
    }
    barrier();
    _output_1._m0 = w_mem.arr;
}

