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

void main()
{
    uint _31 = 2u;
    _12 _15 = _12(uint(gl_VertexIndex));
    _13 _19 = _13(uint(gl_InstanceIndex));
    gl_PointSize = 1.0;
    uint _34 = _15._m0;
    float _35 = float(_34);
    uint _36 = _19._m0;
    float _37 = float(_36);
    float _39 = float(_31);
    vec4 _40 = vec4(_35, _37, _39, 0.0);
    gl_Position = _40;
}

