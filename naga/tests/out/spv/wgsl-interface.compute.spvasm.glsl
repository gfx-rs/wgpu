#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    vec4 _m0;
    float _m1;
};

struct _7
{
    float _m0;
    uint _m1;
    float _m2;
};

struct _12
{
    uint _m0;
};

struct _13
{
    uint _m0;
};

shared uint _14[1];

void main()
{
    for (uint _38 = gl_LocalInvocationIndex; _38 < 1u; _38++)
    {
        _14[_38] = 0u;
    }
    barrier();
    _14[0u] = (((gl_GlobalInvocationID.x + gl_LocalInvocationID.x) + gl_LocalInvocationIndex) + gl_WorkGroupID.x) + gl_NumWorkGroups.x;
}

