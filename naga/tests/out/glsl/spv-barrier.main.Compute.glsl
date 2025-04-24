#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


void function() {
    memoryBarrierShared();
    barrier();
    memoryBarrierShared();
    barrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    barrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    barrier();
    memoryBarrierBuffer();
    memoryBarrierShared();
    memoryBarrierImage();
    barrier();
    memoryBarrierBuffer();
    memoryBarrierShared();
    memoryBarrierImage();
    barrier();
    return;
}

void main() {
    function();
}

