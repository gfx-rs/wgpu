#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared uint _7[1];

void main()
{
    _7[gl_LocalInvocationIndex] = 0u;
    barrier();
    _7[0u] = (((gl_GlobalInvocationID.x + gl_LocalInvocationID.x) + gl_LocalInvocationIndex) + gl_WorkGroupID.x) + gl_NumWorkGroups.x;
}

