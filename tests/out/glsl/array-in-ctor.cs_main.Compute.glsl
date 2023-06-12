#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Ah {
    float inner[2];
};
layout(std430) readonly buffer Ah_block_0Compute { Ah _group_0_binding_0_cs; };


void main() {
    Ah ah_1 = _group_0_binding_0_cs;
}

