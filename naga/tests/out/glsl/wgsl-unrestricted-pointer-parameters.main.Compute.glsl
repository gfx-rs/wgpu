#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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
    uint local_1 = 0u;
    uint local_2 = 0u;
    int i_1 = 0;
    int local_3 = 0;
    bool local_4 = false;
    uint local_5 = 0u;
    bool local_6 = false;
    _group_0_binding_0_cs.a = index;
    _group_0_binding_1_cs[index] = 1u;
    uint _e8 = _group_0_binding_0_cs.b[1];
    _group_0_binding_0_cs.b[1] = (_e8 + 2u);
    local_1 = _e8;
    uint _e12 = local_1;
    uint _e16 = _group_0_binding_0_cs.b[2];
    _group_0_binding_0_cs.b[2] = (_e16 + 3u);
    local_2 = _e16;
    uint _e20 = local_2;
    uint x = (_e12 + _e20);
    _group_0_binding_1_cs[0] = x;
    do {
        i_1 = 0;
        local_4 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                int _e37 = i_1;
                i_1 = (_e37 + 1);
            }
            loop_init = false;
            int _e27 = i_1;
            if ((_e27 < 64)) {
            } else {
                break;
            }
            {
                int _e31 = i_1;
                uint _e33 = wg[_e31];
                if ((_e33 == index)) {
                    int _e35 = i_1;
                    local_3 = _e35;
                    local_4 = true;
                    break;
                }
            }
        }
        bool _e40 = local_4;
        if (_e40) {
            break;
        }
        local_3 = -1;
    } while(false);
    int _e42 = local_3;
    _group_0_binding_1_cs[1] = uint(_e42);
    do {
        local_6 = false;
        uint _e48 = _group_0_binding_1_cs[3];
        switch(_e48) {
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
        bool _e50 = local_6;
        if (_e50) {
            break;
        }
        local_5 = 20u;
    } while(false);
    uint _e52 = local_5;
    _group_0_binding_1_cs[2] = _e52;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e61 = i;
            i = (_e61 + 1u);
        }
        loop_init_1 = false;
        uint _e54 = i;
        if ((_e54 < 2u)) {
        } else {
            break;
        }
        {
            uint _e58 = i;
            bump_twice(local[_e58]);
        }
    }
    set_private(private_arr[1]);
    do {
        if ((index == 0u)) {
            break;
        }
        _group_0_binding_1_cs[4] = 5u;
    } while(false);
    uint _e72 = local[0];
    uint _e75 = private_arr[1];
    uint _e77 = not_inlined(whole);
    _group_0_binding_1_cs[5] = ((_e72 + _e75) + _e77);
    return;
}

