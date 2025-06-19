#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _atomic_compare_exchange_resultSint4_ {
    int old_value;
    bool exchanged;
};
struct _atomic_compare_exchange_resultUint4_ {
    uint old_value;
    bool exchanged;
};
const uint SIZE = 128u;

layout(std430) buffer type_4_block_0Compute { uint _group_0_binding_1_cs[128]; };


void main() {
    uint i_1 = 0u;
    uint old_1 = 0u;
    bool exchanged_1 = false;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e27 = i_1;
            i_1 = (_e27 + 1u);
        }
        loop_init = false;
        uint _e2 = i_1;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i_1;
            uint _e8 = _group_0_binding_1_cs[_e6];
            old_1 = _e8;
            exchanged_1 = false;
            while(true) {
                bool _e12 = exchanged_1;
                if (!(_e12)) {
                } else {
                    break;
                }
                {
                    uint _e14 = old_1;
                    uint new = floatBitsToUint((uintBitsToFloat(_e14) + 1.0));
                    uint _e20 = i_1;
                    uint _e22 = old_1;
                    _atomic_compare_exchange_resultUint4_ _e23; _e23.old_value = atomicCompSwap(_group_0_binding_1_cs[_e20], _e22, new);
                    _e23.exchanged = (_e23.old_value == _e22);
                    old_1 = _e23.old_value;
                    exchanged_1 = _e23.exchanged;
                }
            }
        }
    }
    return;
}

