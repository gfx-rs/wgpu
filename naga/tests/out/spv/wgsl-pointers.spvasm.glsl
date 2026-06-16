#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer DynamicArray
{
    uint arr[];
} dynamic_array;

void f()
{
    mat2 v = mat2(vec2(0.0), vec2(0.0));
    v[0u] = vec2(10.0);
}

void index_unsized(int i, uint v)
{
    dynamic_array.arr[i] += v;
}

void index_dynamic_array(int i, uint v)
{
    dynamic_array.arr[i] += v;
}

void main()
{
    f();
    index_unsized(1, 1u);
    index_dynamic_array(1, 1u);
}

