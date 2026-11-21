#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

shared uint total;


void main() {
    uint local_invocation_index = gl_LocalInvocationIndex;
    if ((local_invocation_index == 0u)) {
        total = 0u;
    }
    memoryBarrierShared();
    barrier();
    uint phony = total;
    return;
}

