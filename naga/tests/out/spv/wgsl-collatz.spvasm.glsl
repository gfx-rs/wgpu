#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer PrimeIndices
{
    uint data[];
} v_indices;

uint naga_mod(uint lhs, uint rhs)
{
    return lhs % ((rhs == 0u) ? 1u : rhs);
}

uint naga_div(uint lhs, uint rhs)
{
    return lhs / ((rhs == 0u) ? 1u : rhs);
}

uint collatz_iterations(uint n_base)
{
    uint n = 0u;
    uint i = 0u;
    uvec2 loop_bound = uvec2(4294967295u);
    n = n_base;
    for (;;)
    {
        uvec2 _51 = loop_bound;
        loop_bound = _51 - uvec2(uint(_51.y == 0u), 1u);
        if (all(equal(_51, uvec2(0u))))
        {
            break;
        }
        if (!(n > 1u))
        {
            break;
        }
        if (naga_mod(n, 2u) == 0u)
        {
            n = naga_div(n, 2u);
        }
        else
        {
            n = (3u * n) + 1u;
        }
        i++;
        continue;
    }
    return i;
}

void main()
{
    v_indices.data[gl_GlobalInvocationID.x] = collatz_iterations(v_indices.data[gl_GlobalInvocationID.x]);
}

