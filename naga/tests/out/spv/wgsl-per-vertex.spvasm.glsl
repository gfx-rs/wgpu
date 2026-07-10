#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) pervertexEXT in float _9[3];
layout(location = 0) out vec4 _12;

void main()
{
    _12 = vec4(_9[0], _9[1], _9[2], 1.0);
}

