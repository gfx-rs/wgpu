#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct WStruct {
    uint arr[512];
    int atom;
    int atom_arr[8][8];
};
shared WStruct w_mem;

layout(std430) buffer type_1_block_0Compute { uint _group_0_binding_0_cs[512]; };


void main() {
    uint local_invocation_index = gl_LocalInvocationIndex;
    uint zero_init_index = 0u;
    zero_init_index = local_invocation_index;
    while(true) {
        uint _e8 = zero_init_index;
        if ((_e8 >= 512u)) {
            break;
        }
        w_mem.arr[_e8] = 0u;
        zero_init_index = (_e8 + 1u);
    }
    zero_init_index = local_invocation_index;
    while(true) {
        uint _e18 = zero_init_index;
        if ((_e18 >= 64u)) {
            break;
        }
        atomicExchange(w_mem.atom_arr[(_e18 / 8u)][(_e18 - 8u * (_e18 / 8u))], 0);
        zero_init_index = (_e18 + 1u);
    }
    if ((local_invocation_index == 0u)) {
        atomicExchange(w_mem.atom, 0);
    }
    memoryBarrierShared();
    barrier();
    uint _e3[512] = w_mem.arr;
    _group_0_binding_0_cs = _e3;
    return;
}

