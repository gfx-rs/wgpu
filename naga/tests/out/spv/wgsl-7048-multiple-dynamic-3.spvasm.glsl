#version 460

struct _5
{
    float _m0;
    vec3 _m1;
};

const vec3 _95[12] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

_5 _13(vec3 _11[12], uint _12)
{
    uvec2 _31 = uvec2(0u);
    vec3 _25 = vec3(0.0);
    vec3 _29 = vec3(0.0);
    uint _23 = 0u;
    uvec2 _34 = uvec2(0u);
    uint _28 = 0u;
    _31 = uvec2(4294967295u);
    vec3 _61[12];
    for (;;)
    {
        if (all(equal(_31, uvec2(0u))))
        {
            break;
        }
        _31 -= uvec2(uint(_31.y == 0u), 1u);
        if (!(_23 < _12))
        {
            break;
        }
        _61 = _11;
        _25 = _61[_23];
        _23++;
        continue;
    }
    _34 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_34, uvec2(0u))))
        {
            break;
        }
        _34 -= uvec2(uint(_34.y == 0u), 1u);
        if (!(_28 < _12))
        {
            break;
        }
        _61 = _11;
        _29 = _61[_28];
        _28++;
        continue;
    }
    return _5(0.0, vec3(0.0));
}

void main()
{
}

