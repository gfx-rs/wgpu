#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 8, local_size_y = 4, local_size_z = 1) in;

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
shared uint scalar;

shared float small[16];

shared Particle particles[256];

shared uvec2 single[1];

shared Mixed mixed;

shared int counter;

layout(std430) buffer type_14_block_0Compute { float _group_0_binding_0_cs[]; };


void main() {
    uvec3 id = gl_LocalInvocationID;
    uint local_invocation_index = gl_LocalInvocationIndex;
    uint zero_init_index = 0u;
    if ((local_invocation_index < 16u)) {
        small[local_invocation_index] = 0.0;
    }
    zero_init_index = local_invocation_index;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e92 = zero_init_index;
            uint _e93 = (_e92 + 32u);
            zero_init_index = _e93;
            if ((_e93 >= 256u)) {
                break;
            }
        }
        loop_init = false;
        uint _e86 = zero_init_index;
        particles[_e86] = Particle(vec4(0.0), vec4(0.0));
    }
    zero_init_index = local_invocation_index;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e121 = zero_init_index;
            uint _e122 = (_e121 + 32u);
            zero_init_index = _e122;
            if ((_e122 >= 324u)) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e110 = zero_init_index;
        mixed.grid[(_e110 / 18u)][(_e110 - 18u * (_e110 / 18u))] = vec4(0.0);
    }
    if ((local_invocation_index < 3u)) {
        atomicExchange(mixed.cells[local_invocation_index].count, 0u);
    }
    zero_init_index = local_invocation_index;
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            uint _e146 = zero_init_index;
            uint _e147 = (_e146 + 32u);
            zero_init_index = _e147;
            if ((_e147 >= 300u)) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e134 = zero_init_index;
        atomicExchange(mixed.cells[(_e134 / 100u)].bins[(_e134 - 100u * (_e134 / 100u))], 0u);
    }
    if ((local_invocation_index == 0u)) {
        scalar = 0u;
        single[0u] = uvec2(0u);
        mixed.transform = mat4x4(0.0);
        mixed.flag = 0u;
        atomicExchange(counter, 0);
    }
    memoryBarrierShared();
    barrier();
    uint base = ((id.x + (id.y * 8u)) * 4u);
    uint _e11 = scalar;
    float _e15 = small[3];
    float _e21 = particles[7].vel.y;
    uint _e26 = single[0].x;
    _group_0_binding_0_cs[base] = (((float(_e11) + _e15) + _e21) + float(_e26));
    float _e37 = mixed.transform[1][0];
    uint _e40 = mixed.flag;
    float _e48 = mixed.grid[2][5].z;
    _group_0_binding_0_cs[(base + 1u)] = ((_e37 + float(_e40)) + _e48);
    uint _e59 = atomicOr(mixed.cells[1].bins[42], 0u);
    uint _e64 = atomicOr(mixed.cells[2].count, 0u);
    _group_0_binding_0_cs[(base + 2u)] = float((_e59 + _e64));
    int _e72 = atomicOr(counter, 0);
    _group_0_binding_0_cs[(base + 3u)] = float(_e72);
    return;
}

