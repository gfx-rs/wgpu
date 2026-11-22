#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct S {
    uint a;
    uint b[4];
};
layout(std430) buffer S_block_0Compute { S _group_0_binding_0_cs; };

layout(std430) buffer type_2_block_1Compute { uint _group_0_binding_1_cs[]; };

shared uint wg[64];

uint private_arr[4] = uint[4](0u, 0u, 0u, 0u);


void bump(inout uint p) {
    uint count = 1u;
    uint _e3 = p;
    uint _e4 = count;
    p = (_e3 + _e4);
    uint _e6 = count;
    count = (_e6 + 1u);
    return;
}

void bump_twice(inout uint p_1) {
    bump(p_1);
    bump(p_1);
    return;
}

void set_private(inout uint p_2) {
    p_2 = 7u;
    return;
}

uint not_inlined(inout uint p_3) {
    uint _e1 = p_3;
    return _e1;
}

void main() {
    uint index = gl_LocalInvocationIndex;
    uint local[2] = uint[2](0u, 0u);
    uint i = 0u;
    uint whole = 3u;
    uint v = 0u;
    uint v_1 = 0u;
    uint v_2 = 0u;
    uint local_1 = 0u;
    uint v_3 = 0u;
    uint local_2 = 0u;
    uint needle = 0u;
    int i_1 = 0;
    int local_3 = 0;
    bool local_4 = false;
    uint local_5 = 0u;
    bool local_6 = false;
    bool c = false;
    uint zero_init_index = 0u;
    zero_init_index = index;
    while(true) {
        uint _e2 = zero_init_index;
        if ((_e2 >= 64u)) {
            break;
        }
        wg[_e2] = 0u;
        zero_init_index = (_e2 + 1u);
    }
    memoryBarrierShared();
    barrier();
    v = index;
    uint _e11 = v;
    _group_0_binding_0_cs.a = _e11;
    v_1 = 1u;
    uint _e16 = v_1;
    _group_0_binding_1_cs[index] = _e16;
    v_2 = 2u;
    uint _e21 = v_2;
    uint _e22 = _group_0_binding_0_cs.b[1];
    _group_0_binding_0_cs.b[1] = (_e22 + _e21);
    local_1 = _e22;
    uint _e25 = local_1;
    v_3 = 3u;
    uint _e30 = v_3;
    uint _e31 = _group_0_binding_0_cs.b[2];
    _group_0_binding_0_cs.b[2] = (_e31 + _e30);
    local_2 = _e31;
    uint _e34 = local_2;
    uint x = (_e25 + _e34);
    _group_0_binding_1_cs[0] = x;
    needle = index;
    uint _e41 = needle;
    do {
        i_1 = 0;
        local_4 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                int _e53 = i_1;
                i_1 = (_e53 + 1);
            }
            loop_init = false;
            int _e43 = i_1;
            if ((_e43 < 64)) {
            } else {
                break;
            }
            {
                int _e47 = i_1;
                uint _e49 = wg[_e47];
                if ((_e49 == _e41)) {
                    int _e51 = i_1;
                    local_3 = _e51;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e56 = local_4;
        if (_e56) {
            break;
        }
        local_3 = -1;
    } while(false);
    int _e58 = local_3;
    _group_0_binding_1_cs[1] = uint(_e58);
    do {
        local_6 = false;
        uint _e64 = _group_0_binding_1_cs[3];
        switch(_e64) {
            case 0u: {
                local_5 = 10u;
                local_6 = true;
                break;
            }
            case 1u: {
                _group_0_binding_1_cs[3] = 2u;
                break;
            }
            default: {
                local_5 = 30u;
                local_6 = true;
                break;
            }
        }
        bool _e66 = local_6;
        if (_e66) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e68 = local_5;
    _group_0_binding_1_cs[2] = _e68;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e77 = i;
            i = (_e77 + 1u);
        }
        loop_init_1 = false;
        uint _e70 = i;
        if ((_e70 < 2u)) {
        } else {
            break;
        }
        {
            uint _e74 = i;
            bump_twice(local[_e74]);
        }
    }
    set_private(private_arr[1]);
    c = (index == 0u);
    bool _e86 = c;
    do {
        if (_e86) {
            break;
        }
        _group_0_binding_1_cs[4] = 5u;
    } while(false);
    uint _e90 = local[0];
    uint _e93 = private_arr[1];
    uint _e95 = not_inlined(whole);
    _group_0_binding_1_cs[5] = ((_e90 + _e93) + _e95);
    return;
}

