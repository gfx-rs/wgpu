#version 460
#extension GL_EXT_debug_printf : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    debugPrintfEXT("debug id: %u %u %u", gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);
}

