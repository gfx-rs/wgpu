//////////////////////////////
// Entry point: "vs" (vert) //
//////////////////////////////
#version 460

struct _6
{
    vec4 _m0;
    vec2 _m1;
};

layout(location = 0) in vec2 _8;
layout(location = 0) out vec2 _13;

void main()
{
    _6 _19 = _6(vec4(0.0), vec2(0.0));
    _19._m0 = vec4(_8, 0.0, 1.0);
    gl_Position = _19._m0;
    _13 = _19._m1;
}


//////////////////////////////
// Entry point: "fs" (frag) //
//////////////////////////////
#version 460

struct _6
{
    vec4 _m0;
    vec2 _m1;
};

layout(location = 0) out vec4 _32;

void main()
{
    _32 = vec4(1.0, 0.0, 0.0, 1.0);
}

