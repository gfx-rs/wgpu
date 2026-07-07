#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared uint _17[64];

void _20()
{
    uint _63 = 0u;
    uint _58 = 3u;
    int _52 = 0;
    float _47 = 0.0;
    float _42 = 3.0;
    float _38 = 3.0;
    int _33 = 43;
    int _67 = 0;
    uint _61 = 0u;
    int _56 = 0;
    int _51 = 3;
    float _45 = 0.0;
    float _41 = 3.0;
    float _37 = 3.0;
    float _31 = 42.0;
    int _65 = 0;
    uint _59 = 0u;
    int _54 = 0;
    float _49 = 0.0;
    float _43 = 0.0;
    float _39 = 0.0;
    uint _35 = 44u;
    _39 = 1.0 + _31;
    _43 = 1.0 + _31;
    _45 = _31 + 2.0;
    _47 = _31 + 2.0;
    _49 = _31 + _31;
    _52 = 1 + _33;
    _54 = _33 + 2;
    _56 = _33 + _33;
    _59 = 1u + _35;
    _61 = _35 + 2u;
    _63 = _35 + _35;
    _65 = 1 << int(_35);
    _67 = 1 << int(_35);
}

void _100()
{
}

void _105()
{
    int _107 = 1 - 1;
}

void main()
{
    if (gl_LocalInvocationIndex == 0u)
    {
        _17 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    barrier();
    _20();
    _100();
    _105();
}

