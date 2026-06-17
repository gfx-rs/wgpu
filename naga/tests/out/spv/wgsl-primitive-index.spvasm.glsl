#version 460

layout(location = 0) out vec4 _10;

void main()
{
    _10 = vec4(float(uint(gl_PrimitiveID)), 1.0, 1.0, 1.0);
}

