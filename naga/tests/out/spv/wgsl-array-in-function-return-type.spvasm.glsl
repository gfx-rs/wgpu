#version 460

layout(location = 0) out vec4 _26;

float[2] _11()
{
    return float[](1.0, 2.0);
}

float[3][2] _18()
{
    return float[][](_11(), _11(), _11());
}

void main()
{
    float _32[3][2] = _18();
    _26 = vec4(_32[0][0], _32[0][1], 0.0, 1.0);
}

