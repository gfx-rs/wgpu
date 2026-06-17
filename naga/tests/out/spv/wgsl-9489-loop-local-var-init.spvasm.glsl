#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _11_10
{
    float _m0[64];
} _10;

layout(set = 0, binding = 1, std430) buffer _14_13
{
    float _m0[8];
} _13;

void main()
{
    vec4 _31 = vec4(0.0);
    uint _29 = 0u;
    vec4 _34 = vec4(0.0);
    uvec2 _50 = uvec2(4294967295u);
    uvec2 _71 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _50)))
        {
            break;
        }
        _50 -= uvec2(uint(_50.y == 0u), 1u);
        if (!(_29 < 4u))
        {
            break;
        }
        _31 = vec4(0.0);
        _34 = vec4(0.0);
        uint _36 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _71)))
            {
                break;
            }
            _71 -= uvec2(uint(_71.y == 0u), 1u);
            if (!(_36 < 16u))
            {
                break;
            }
            vec4 _95 = vec4(_10._m0[(_29 * 16u) + _36]);
            _31 += _95;
            _34 += _95;
            _36++;
            continue;
        }
        _13._m0[_29 * 2u] = _31.x;
        _13._m0[(_29 * 2u) + 1u] = _34.x;
        _29++;
        continue;
    }
}

