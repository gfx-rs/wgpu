#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _12_11
{
    float _m0[64];
} _11;

layout(set = 0, binding = 1, std430) buffer _15_14
{
    float _m0[8];
} _14;

void main()
{
    uvec2 _42 = uvec2(4294967295u);
    vec4 _35 = vec4(0.0);
    uint _33 = 0u;
    uvec2 _44 = uvec2(4294967295u);
    vec4 _38 = vec4(0.0);
    for (;;)
    {
        uvec2 _52 = _42;
        _42 = _52 - uvec2(uint(_52.y == 0u), 1u);
        if (all(equal(_52, uvec2(0u))))
        {
            break;
        }
        if (!(_33 < 4u))
        {
            break;
        }
        _35 = vec4(0.0);
        _38 = vec4(0.0);
        _44 = uvec2(4294967295u);
        uint _40 = 0u;
        for (;;)
        {
            uvec2 _76 = _44;
            _44 = _76 - uvec2(uint(_76.y == 0u), 1u);
            if (all(equal(_76, uvec2(0u))))
            {
                break;
            }
            if (!(_40 < 16u))
            {
                break;
            }
            vec4 _99 = vec4(_11._m0[(_33 * 16u) + _40]);
            _35 += _99;
            _38 += _99;
            _40++;
            continue;
        }
        _14._m0[_33 * 2u] = _35.x;
        _14._m0[(_33 * 2u) + 1u] = _38.x;
        _33++;
        continue;
    }
}

