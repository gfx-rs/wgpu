#version 460
layout(early_fragment_tests) in;

float _6 = 0.0;

void _9()
{
    _6 = 0.5;
}

void main()
{
    _9();
    gl_FragDepth = _6;
    gl_FragDepth = clamp(gl_FragDepth, 0.0, 1.0);
}
