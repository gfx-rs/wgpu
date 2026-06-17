#version 460

layout(location = 0) out vec4 _15;

bool _7()
{
    return true;
}

void main()
{
    vec4 _19 = vec4(0.0);
    vec4 _22 = vec4(0.0);
    vec4 _24 = vec4(0.0);
    _19 = dFdxCoarse(gl_FragCoord);
    _22 = dFdyCoarse(gl_FragCoord);
    _24 = fwidthCoarse(gl_FragCoord);
    _19 = dFdxFine(gl_FragCoord);
    _22 = dFdyFine(gl_FragCoord);
    _24 = fwidthFine(gl_FragCoord);
    _19 = dFdx(gl_FragCoord);
    _22 = dFdy(gl_FragCoord);
    _24 = fwidth(gl_FragCoord);
    _15 = (_19 + _22) * _24;
}

