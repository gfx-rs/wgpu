#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) buffer type_1_block_0Compute { uint _group_0_binding_0_cs[]; };


void main() {
    uint i = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e12 = i;
            i = (_e12 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < 4u)) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            uint _e8 = i;
            _group_0_binding_0_cs[_e6] = (_e8 * 2u);
        }
    }
    return;
}

