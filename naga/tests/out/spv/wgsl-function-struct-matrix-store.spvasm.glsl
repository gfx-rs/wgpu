#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    float _m0;
    mat3x2 _m1;
};

layout(set = 0, binding = 0, std430) buffer _11_10
{
    float _m0[4];
} _10;

void main()
{
    _6 _29 = _6(0.0, mat3x2(vec2(0.0), vec2(0.0), vec2(0.0)));
    _29._m1 = mat3x2(vec2(1.0, 2.0), vec2(3.0, 4.0), vec2(5.0, 6.0));
    _10._m0[0u] = _29._m1[0u].x;
    _10._m0[1u] = _29._m1[2u].y;
}

