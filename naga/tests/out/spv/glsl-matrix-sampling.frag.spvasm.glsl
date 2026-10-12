#version 460

struct _9
{
    vec2 _m0;
};

struct _11
{
    vec4 _m0;
};

layout(location = 0) centroid in vec2 _57;
layout(location = 1) centroid in vec2 _60;
layout(location = 2) centroid in vec2 _62;
layout(location = 3) centroid in vec2 _64;
layout(location = 4) sample in vec2 _66;
layout(location = 5) sample in vec2 _68;
layout(location = 6) in vec2 _70;
layout(location = 7) centroid in vec2 _72;
layout(location = 8) sample in float _74;
layout(location = 0) out vec4 _77;
mat2 _12[2] = mat2[](mat2(vec2(0.0), vec2(0.0)), mat2(vec2(0.0), vec2(0.0)));
mat2 _15 = mat2(vec2(0.0), vec2(0.0));
_9 _18 = _9(vec2(0.0));
vec2 _21 = vec2(0.0);
float _24 = 0.0;
vec4 _27 = vec4(0.0);

void _31()
{
    _27 = vec4(((((_12[0u][0u] + _12[1u][1u]) + _15[0u]) + _18._m0) + _21) * _24, 0.0, 1.0);
}

void main()
{
    _12[0u][0u] = _57;
    _12[0u][1u] = _60;
    _12[1u][0u] = _62;
    _12[1u][1u] = _64;
    _15[0u] = _66;
    _15[1u] = _68;
    _18._m0 = _70;
    _21 = _72;
    _24 = _74;
    _31();
    _77 = _11(_27)._m0;
}

