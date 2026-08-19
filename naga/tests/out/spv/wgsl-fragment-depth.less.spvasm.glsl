#version 460
layout(depth_less) out float gl_FragDepth;

struct _4
{
    float _m0;
};

void main()
{
    gl_FragDepth = 0.5;
    gl_FragDepth = clamp(gl_FragDepth, 0.0, 1.0);
}
