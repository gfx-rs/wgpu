#version 460
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    subgroupMemoryBarrier();
    subgroupBarrier();
}

