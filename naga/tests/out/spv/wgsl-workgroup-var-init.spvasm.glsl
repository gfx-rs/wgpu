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

uint naga_div(uint lhs, uint rhs)
{
    return lhs / ((rhs == 0u) ? 1u : rhs);
}

uint naga_mod(uint lhs, uint rhs)
{
    return lhs % ((rhs == 0u) ? 1u : rhs);
}

void main()
{
    uint zero_init_index = 0u;
    uvec2 loop_bound = uvec2(4294967295u);
    uvec2 loop_bound_1 = uvec2(4294967295u);
    zero_init_index = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (zero_init_index >= 512u)
        {
            break;
        }
        w_mem.arr[zero_init_index] = 0u;
        zero_init_index++;
        continue;
    }
    zero_init_index = gl_LocalInvocationIndex;
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound_1)))
        {
            break;
        }
        loop_bound_1 -= uvec2(uint(loop_bound_1.y == 0u), 1u);
        if (zero_init_index >= 64u)
        {
            break;
        }
        w_mem.atom_arr[naga_div(zero_init_index, 8u)][naga_mod(zero_init_index, 8u)] = 0;
        zero_init_index++;
        continue;
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        w_mem.atom = 0;
    }
    barrier();
    _output_1._m0 = w_mem.arr;
}

