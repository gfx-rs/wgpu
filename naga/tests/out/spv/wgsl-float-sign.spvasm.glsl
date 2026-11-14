#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _11_10
{
    float _m0[4];
} _10;

layout(set = 0, binding = 1, std430) buffer _14_13
{
    int _m0[4];
} _13;

void main()
{
    _10._m0[0u] = sign(_10._m0[1u]);
    float _33 = _10._m0[2u];
    float _36 = _10._m0[3u];
    vec2 _38 = sign(vec2(_33, _36));
    _10._m0[2u] = _38.x;
    _10._m0[3u] = _38.y;
    _13._m0[0u] = sign(_13._m0[1u]);
}

