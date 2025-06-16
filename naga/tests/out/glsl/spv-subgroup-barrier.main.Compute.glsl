#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


void function() {
    subgroupMemoryBarrier();
    subgroupMemoryBarrier();
    barrier();
    return;
}

void main() {
    function();
}

