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

layout(std430) buffer type_2_block_0Compute { int _group_0_binding_0_cs[128]; };


void main() {
    uint i = 0u;
    int old = 0;
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
            int _e8 = _group_0_binding_0_cs[_e6];
            old = _e8;
            exchanged = false;
            while(true) {
                bool _e12 = exchanged;
                if (!(_e12)) {
                } else {
                    break;
                }
                {
                    int _e14 = old;
                    int new = floatBitsToInt((intBitsToFloat(_e14) + 1.0));
                    uint _e20 = i;
                    int _e22 = old;
                    _atomic_compare_exchange_resultSint4_ _e23; _e23.old_value = atomicCompSwap(_group_0_binding_0_cs[_e20], _e22, new);
                    _e23.exchanged = (_e23.old_value == _e22);
                    old = _e23.old_value;
                    exchanged = _e23.exchanged;
                }
            }
        }
    }
    return;
}

