#version 460

struct _5
{
    vec4 _m0;
};

struct _8
{
    uvec3 _m0;
    uint _m1;
};

struct _13
{
    _5 _m0[3];
    _8 _m1[1];
    uint _m2;
    uint _m3;
};

layout(location = 0) out vec4 _18;

void main()
{
    _18 = vec4(float(uint(gl_PrimitiveID)), 1.0, 1.0, 1.0);
}

