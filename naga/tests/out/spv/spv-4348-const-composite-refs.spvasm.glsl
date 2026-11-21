#version 460

struct _11
{
    vec3 _m0;
    float _m1;
};

layout(location = 0) out vec4 _38;
vec4 _21 = vec4(0.0);

void _25()
{
    _21 = vec4(mat2[](mat2(vec2(1.0, 2.0), vec2(1.0, 2.0)), mat2(vec2(1.0, 2.0), vec2(1.0, 2.0)))[1][0].y, _11[](_11(vec3(1.0), 2.0), _11(vec3(1.0), 2.0))[1]._m0.z, _11[](_11(vec3(1.0), 2.0), _11(vec3(1.0), 2.0))[0]._m1, 1.0);
}

void main()
{
    _25();
    _38 = _21;
}

