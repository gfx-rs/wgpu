#version 460

invariant gl_Position;

struct _5
{
    vec4 _m0;
    float _m1;
};

layout(location = 10) in uint _13;
layout(location = 1) out float _17;

void main()
{
    gl_PointSize = 1.0;
    uint _25 = uint(gl_VertexIndex) + uint(gl_InstanceIndex);
    uint _26 = _25 + _13;
    float _27 = float(_26);
    _5 _28 = _5(vec4(1.0), _27);
    vec4 _29 = _28._m0;
    gl_Position = _29;
    _17 = _28._m1;
}

