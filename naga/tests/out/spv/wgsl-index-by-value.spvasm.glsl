#version 460

const int _36[2][2] = int[][](int[](1, 2), int[](3, 4));
const int _65[5] = int[](1, 2, 3, 4, 5);

int _17(int _15[5], int _16)
{
    int _21[5] = _15;
    return _21[_16];
}

int _28(int _26, int _27)
{
    return _36[_26][_27];
}

float _45(int _43, int _44)
{
    mat2 _56 = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    return _56[_43][_44];
}

vec4 _62(uint _61)
{
    return vec4(ivec4(_65[_61]));
}

void main()
{
    gl_Position = _62(uint(gl_VertexIndex));
}

