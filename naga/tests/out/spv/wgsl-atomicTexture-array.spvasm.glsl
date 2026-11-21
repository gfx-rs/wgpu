#version 460
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r32ui) uniform uimage2DArray _9;
layout(set = 0, binding = 1, r32i) uniform iimage2DArray _11;

void main()
{
    uint _31 = imageAtomicMax(_9, ivec3(ivec2(0), 1), 1u);
    int _36 = imageAtomicMin(_11, ivec3(ivec2(0), int(1u)), 1);
}

