#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

shared int a;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        a = 0;
    }
    memoryBarrierShared();
    barrier();
    int _e2 = atomicAdd(a, - -1);
    return;
}

