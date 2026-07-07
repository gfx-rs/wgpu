#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) pervertexEXT in float _25[3];
layout(location = 0) out vec4 _28;
float _9[3] = float[](0.0, 0.0, 0.0);
vec4 _12 = vec4(0.0);

void _16()
{
    _12 = vec4(_9[0], _9[1], _9[2], 1.0);
}

void main()
{
    _9 = _25;
    _16();
    _28 = _12;
}

