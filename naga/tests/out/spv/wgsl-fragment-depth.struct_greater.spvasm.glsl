#version 460
layout(depth_greater) out float gl_FragDepth;

struct _4
{
    float _m0;
};

void main()
{
    gl_FragDepth = _4(0.5)._m0;
    gl_FragDepth = clamp(gl_FragDepth, 0.0, 1.0);
}
