#version 460

struct _6
{
    uint _m0;
    vec2 _m1;
};

struct _8
{
    vec4 _m0;
    uint _m1;
};

layout(push_constant, std430) uniform _10_9
{
    _6 _m0;
} _9;

layout(location = 0) in vec4 _14;
layout(location = 0) out vec4 _20;

void main()
{
    _8 _13 = _8(_14, uint(gl_PrimitiveID));
    if (_13._m1 == _9._m0._m0)
    {
        _20 = _13._m0;
        return;
    }
    else
    {
        _20 = vec4(vec3(1.0) - _13._m0.xyz, _13._m0.w);
        return;
    }
}

