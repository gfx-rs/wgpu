#version 460

struct _5
{
    float _m0;
    vec3 _m1;
};

const vec3 _97[12] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

_5 _13(vec3 _11[12], uint _12)
{
    uvec2 _31 = uvec2(4294967295u);
    vec3 _25 = vec3(0.0);
    vec3 _29 = vec3(0.0);
    uint _23 = 0u;
    uvec2 _33 = uvec2(4294967295u);
    uint _28 = 0u;
    vec3 _61[12];
    for (;;)
    {
        uvec2 _41 = _31;
        _31 = _41 - uvec2(uint(_41.y == 0u), 1u);
        if (all(equal(_41, uvec2(0u))))
        {
            break;
        }
        if (!(_23 < _12))
        {
            break;
        }
        _61 = _11;
        _25 = _61[_23];
        _23++;
        continue;
    }
    for (;;)
    {
        uvec2 _72 = _33;
        _33 = _72 - uvec2(uint(_72.y == 0u), 1u);
        if (all(equal(_72, uvec2(0u))))
        {
            break;
        }
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

