#version 460
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r32ui) uniform uimage2D _9;
layout(set = 0, binding = 1, r32i) uniform iimage2D _11;

void main()
{
    uint _29 = imageAtomicMax(_9, ivec2(0), 1u);
    uint _31 = imageAtomicMin(_9, ivec2(0), 1u);
    uint _33 = imageAtomicAdd(_9, ivec2(0), 1u);
    uint _35 = imageAtomicAnd(_9, ivec2(0), 1u);
    uint _37 = imageAtomicOr(_9, ivec2(0), 1u);
    uint _39 = imageAtomicXor(_9, ivec2(0), 1u);
    int _42 = imageAtomicMax(_11, ivec2(0), 1);
    int _44 = imageAtomicMin(_11, ivec2(0), 1);
    int _46 = imageAtomicAdd(_11, ivec2(0), 1);
    int _48 = imageAtomicAnd(_11, ivec2(0), 1);
    int _50 = imageAtomicOr(_11, ivec2(0), 1);
    int _52 = imageAtomicXor(_11, ivec2(0), 1);
}

