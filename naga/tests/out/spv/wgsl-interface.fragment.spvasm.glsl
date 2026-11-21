#version 460

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

layout(location = 1) in float _14;
layout(location = 0) out float _29;

void main()
{
    _7 _40 = _7(_5(gl_FragCoord, _14)._m1, uint(gl_SampleMaskIn) & (1u << uint(gl_SampleID)), float(gl_FrontFacing));
    gl_FragDepth = _40._m0;
    gl_FragDepth = clamp(gl_FragDepth, 0.0, 1.0);
    uint _44 = _40._m1;
    for (int i = 0; i < int(_0); i++)
    {
        gl_SampleMask[i] = int(_44[i]);
    }
    _29 = _40._m2;
}

