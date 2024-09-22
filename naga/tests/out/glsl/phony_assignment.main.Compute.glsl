#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uniform type_block_0Compute { float _group_0_binding_0_cs; };


void main() {
    uvec3 id = gl_GlobalInvocationID;
    float _phony_2 = _group_0_binding_0_cs;
    int _phony_3 = 5;
}

