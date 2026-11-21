#version 460

invariant gl_Position;

struct _6
{
    uint _m0;
};

struct _7
{
    uint _m0;
};

void main()
{
    uint _25 = 2u;
    _6 _9 = _6(uint(gl_VertexIndex));
    _7 _13 = _7(uint(gl_InstanceIndex));
    gl_PointSize = 1.0;
    uint _28 = _9._m0;
    float _29 = float(_28);
    uint _30 = _13._m0;
    float _31 = float(_30);
    float _33 = float(_25);
    vec4 _34 = vec4(_29, _31, _33, 0.0);
    gl_Position = _34;
}

