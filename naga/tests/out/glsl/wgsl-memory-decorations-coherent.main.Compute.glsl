#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_ARB_shader_storage_buffer_object : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) coherent buffer Data_block_0Compute {
    uint values[];
} _group_0_binding_0_cs;

layout(std430) buffer Data_block_1Compute {
    uint values[];
} _group_0_binding_1_cs;


void main() {
    uint _e6 = _group_0_binding_1_cs.values[0];
    _group_0_binding_0_cs.values[0] = _e6;
    return;
}

