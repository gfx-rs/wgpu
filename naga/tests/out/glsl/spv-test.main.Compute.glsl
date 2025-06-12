#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer RWStructuredBuffer_block_0Compute {
    uint member[];
} _group_0_binding_0_cs;


void main_1() {
    uint phi_19_ = 0u;
    phi_19_ = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            phi_19_ = (phi_19_ + 1u);
        }
        loop_init = false;
        uint _e7 = phi_19_;
        bool should_continue = false;
        do {
            if ((_e7 < 4u)) {
            } else {
                break;
            }
            uint _e11 = _group_0_binding_0_cs.member[_e7];
            if ((_e11 == 1u)) {
                break;
            }
            uint _e13 = _group_0_binding_0_cs.member[_e7];
            if ((_e13 == 2u)) {
                break;
            }
            _group_0_binding_0_cs.member[_e7] = (_e7 * 2u);
            break;
        } while(false);
        continue;
    }
    return;
}

void main() {
    main_1();
}

