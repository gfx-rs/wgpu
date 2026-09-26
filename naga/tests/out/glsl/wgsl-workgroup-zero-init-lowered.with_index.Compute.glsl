#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Particle {
    vec4 pos;
    vec4 vel;
};
struct WithAtomics {
    uint count;
    uint bins[100];
};
struct Mixed {
    mat4x4 transform;
    uint flag;
    vec4 grid[18][18];
    WithAtomics cells[3];
};
struct Inputs {
    uvec3 group;
    uint index;
};
shared float small[16];

layout(std430) buffer type_14_block_0Compute { float _group_0_binding_0_cs[]; };


void main() {
    uint index = gl_LocalInvocationIndex;
    if ((index < 16u)) {
        small[index] = 0.0;
    }
    memoryBarrierShared();
    barrier();
    float _e7 = small[(index - 16u * (index / 16u))];
    _group_0_binding_0_cs[index] = _e7;
    return;
}

