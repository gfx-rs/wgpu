#version 460

struct _4
{
    uint _m0;
};

void main()
{
    gl_Position = vec4(float(_4(uint(gl_DrawID))._m0), 1.0, 1.0, 1.0);
}

