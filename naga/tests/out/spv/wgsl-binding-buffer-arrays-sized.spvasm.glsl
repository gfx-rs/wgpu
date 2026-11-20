#version 460

layout(set = 0, binding = 0, std430) readonly buffer _4_6
{
    uint _m0;
} _6[4];

layout(location = 0) out uint _11;

void main()
{
    _11 = _6[0u]._m0;
}

