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
    uint local_invocation_index = gl_LocalInvocationIndex;
    uint zero_init_index = 0u;
    if ((local_invocation_index < 16u)) {
        small[local_invocation_index] = 0.0;
    }
    zero_init_index = local_invocation_index;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e78 = zero_init_index;
            uint _e79 = (_e78 + 32u);
            zero_init_index = _e79;
            if ((_e79 >= 256u)) {
                break;
            }
        }
        loop_init = false;
        uint _e72 = zero_init_index;
        particles[_e72] = Particle(vec4(0.0), vec4(0.0));
    }
    zero_init_index = local_invocation_index;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e107 = zero_init_index;
            uint _e108 = (_e107 + 32u);
            zero_init_index = _e108;
            if ((_e108 >= 324u)) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e96 = zero_init_index;
        mixed.grid[(_e96 / 18u)][(_e96 - 18u * (_e96 / 18u))] = vec4(0.0);
    }
    if ((local_invocation_index < 3u)) {
        atomicExchange(mixed.cells[local_invocation_index].count, 0u);
    }
    zero_init_index = local_invocation_index;
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            uint _e132 = zero_init_index;
            uint _e133 = (_e132 + 32u);
            zero_init_index = _e133;
            if ((_e133 >= 300u)) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e120 = zero_init_index;
        atomicExchange(mixed.cells[(_e120 / 100u)].bins[(_e120 - 100u * (_e120 / 100u))], 0u);
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
    uint _e3 = scalar;
    float _e7 = small[3];
    float _e13 = particles[7].vel.y;
    uint _e18 = single[0].x;
    _group_0_binding_0_cs[0] = (((float(_e3) + _e7) + _e13) + float(_e18));
    float _e27 = mixed.transform[1][0];
    uint _e30 = mixed.flag;
    float _e38 = mixed.grid[2][5].z;
    _group_0_binding_0_cs[1] = ((_e27 + float(_e30)) + _e38);
    uint _e47 = atomicOr(mixed.cells[1].bins[42], 0u);
    uint _e52 = atomicOr(mixed.cells[2].count, 0u);
    _group_0_binding_0_cs[2] = float((_e47 + _e52));
    int _e58 = atomicOr(counter, 0);
    _group_0_binding_0_cs[3] = float(_e58);
    return;
}

