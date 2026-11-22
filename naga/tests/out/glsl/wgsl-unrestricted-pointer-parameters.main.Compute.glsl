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
    if (gl_LocalInvocationID == uvec3(0u)) {
        wg = uint[64](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    memoryBarrierShared();
    barrier();
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
    v = index;
    uint _e3 = v;
    _group_0_binding_0_cs.a = _e3;
    v_1 = 1u;
    uint _e8 = v_1;
    _group_0_binding_1_cs[index] = _e8;
    v_2 = 2u;
    uint _e13 = v_2;
    uint _e14 = _group_0_binding_0_cs.b[1];
    _group_0_binding_0_cs.b[1] = (_e14 + _e13);
    local_1 = _e14;
    uint _e17 = local_1;
    v_3 = 3u;
    uint _e22 = v_3;
    uint _e23 = _group_0_binding_0_cs.b[2];
    _group_0_binding_0_cs.b[2] = (_e23 + _e22);
    local_2 = _e23;
    uint _e26 = local_2;
    uint x = (_e17 + _e26);
    _group_0_binding_1_cs[0] = x;
    needle = index;
    uint _e33 = needle;
    do {
        i_1 = 0;
        local_4 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                int _e45 = i_1;
                i_1 = (_e45 + 1);
            }
            loop_init = false;
            int _e35 = i_1;
            if ((_e35 < 64)) {
            } else {
                break;
            }
            {
                int _e39 = i_1;
                uint _e41 = wg[_e39];
                if ((_e41 == _e33)) {
                    int _e43 = i_1;
                    local_3 = _e43;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e48 = local_4;
        if (_e48) {
            break;
        }
        local_3 = -1;
    } while(false);
    int _e50 = local_3;
    _group_0_binding_1_cs[1] = uint(_e50);
    do {
        local_6 = false;
        uint _e56 = _group_0_binding_1_cs[3];
        switch(_e56) {
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
        bool _e58 = local_6;
        if (_e58) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e60 = local_5;
    _group_0_binding_1_cs[2] = _e60;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e69 = i;
            i = (_e69 + 1u);
        }
        loop_init_1 = false;
        uint _e62 = i;
        if ((_e62 < 2u)) {
        } else {
            break;
        }
        {
            uint _e66 = i;
            bump_twice(local[_e66]);
        }
    }
    set_private(private_arr[1]);
    c = (index == 0u);
    bool _e78 = c;
    do {
        if (_e78) {
            break;
        }
        _group_0_binding_1_cs[4] = 5u;
    } while(false);
    uint _e82 = local[0];
    uint _e85 = private_arr[1];
    uint _e87 = not_inlined(whole);
    _group_0_binding_1_cs[5] = ((_e82 + _e85) + _e87);
    return;
}

