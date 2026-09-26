#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

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
shared Particle particles[256];

shared int counter;

layout(std430) buffer type_14_block_0Compute { float _group_0_binding_0_cs[]; };


void main() {
    Inputs inputs = Inputs(gl_WorkGroupID, gl_LocalInvocationIndex);
    uint zero_init_index_1 = 0u;
    zero_init_index_1 = inputs.index;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e24 = zero_init_index_1;
            uint _e25 = (_e24 + 16u);
            zero_init_index_1 = _e25;
            if ((_e25 >= 256u)) {
                break;
            }
        }
        loop_init = false;
        uint _e18 = zero_init_index_1;
        particles[_e18] = Particle(vec4(0.0), vec4(0.0));
    }
    if ((inputs.index == 0u)) {
        atomicExchange(counter, 0);
    }
    memoryBarrierShared();
    barrier();
    float _e9 = particles[inputs.index].pos.x;
    int _e11 = atomicOr(counter, 0);
    _group_0_binding_0_cs[inputs.index] = (_e9 + float(_e11));
    return;
}

