#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _atomic_compare_exchange_result_Sint_4_ {
    int old_value;
    bool exchanged;
};
struct _atomic_compare_exchange_result_Uint_4_ {
    uint old_value;
    bool exchanged;
};
const uint SIZE = 128u;

layout(std430) buffer type_3_block_0Compute { uint _group_0_binding_1_cs[128]; };


void main() {
    uint i = 0u;
    uint old = 0u;
    bool exchanged = false;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e27 = i;
            i = (_e27 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < SIZE)) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            uint _e8 = atomicOr(_group_0_binding_1_cs[_e6], 0u);
            old = _e8;
            exchanged = false;
            while(true) {
                bool _e12 = exchanged;
                if (!(_e12)) {
                } else {
                    break;
                }
                {
                    uint _e14 = old;
                    uint new = floatBitsToUint((uintBitsToFloat(_e14) + 1.0));
                    uint _e20 = i;
                    uint _e22 = old;
                    _atomic_compare_exchange_result_Uint_4_ _e23; _e23.old_value = atomicCompSwap(_group_0_binding_1_cs[_e20], _e22, new);
                    _e23.exchanged = (_e23.old_value == _e22);
                    old = _e23.old_value;
                    exchanged = _e23.exchanged;
                }
            }
        }
    }
    return;
}

