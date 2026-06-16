#version 460

struct _5
{
    float _m0;
    vec3 _m1;
};

const vec3 _93[12] = vec3[](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));

_5 _12(vec3 _10[12], uint _11)
{
    vec3 _21 = vec3(0.0);
    vec3 _25 = vec3(0.0);
    uint _19 = 0u;
    uint _24 = 0u;
    uvec2 _39 = uvec2(4294967295u);
    uvec2 _67 = uvec2(4294967295u);
    vec3 _58[12];
    for (;;)
    {
        if (all(equal(uvec2(0u), _39)))
        {
            break;
        }
        _39 -= uvec2(uint(_39.y == 0u), 1u);
        if (!(_19 < _11))
        {
            break;
        }
        _58 = _10;
        _21 = _58[_19];
        _19++;
        continue;
    }
    for (;;)
    {
        if (all(equal(uvec2(0u), _67)))
        {
            break;
        }
        _67 -= uvec2(uint(_67.y == 0u), 1u);
        if (!(_24 < _11))
        {
            break;
        }
        _58 = _10;
        _25 = _58[_24];
        _24++;
        continue;
    }
    return _5(0.0, vec3(0.0));
}

void main()
{
}

