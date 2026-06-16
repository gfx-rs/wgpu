#version 460

invariant gl_Position;

struct _5
{
    vec4 _m0;
    float _m1;
};

struct _7
{
    float _m0;
    uint _m1;
    float _m2;
};

struct _12
{
    uint _m0;
};

struct _13
{
    uint _m0;
};

layout(location = 10) in uint _20;
layout(location = 1) out float _24;

void main()
{
    gl_PointSize = 1.0;
    uint _32 = uint(gl_VertexIndex) + uint(gl_InstanceIndex);
    uint _33 = _32 + _20;
    float _34 = float(_33);
    _5 _35 = _5(vec4(1.0), _34);
    vec4 _36 = _35._m0;
    gl_Position = _36;
    _24 = _35._m1;
}

