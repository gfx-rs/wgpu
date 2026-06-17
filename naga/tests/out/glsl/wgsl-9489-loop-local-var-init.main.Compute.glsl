#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_ARB_shader_storage_buffer_object : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(std430) readonly buffer type_1_block_0Compute { float _group_0_binding_0_cs[64]; };

layout(std430) buffer type_2_block_1Compute { float _group_0_binding_1_cs[8]; };


void main() {
    uint t = 0u;
    vec4 acc_noinit = vec4(0.0);
    vec4 acc_init = vec4(0.0);
    uint d = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e47 = t;
            t = (_e47 + 1u);
        }
        loop_init = false;
        uint _e2 = t;
        if ((_e2 < 4u)) {
        } else {
            break;
        }
        {
            acc_noinit = vec4(0.0);
            acc_init = vec4(0.0);
            d = 0u;
            bool loop_init_1 = true;
            while(true) {
                if (!loop_init_1) {
                    uint _e28 = d;
                    d = (_e28 + 1u);
                }
                loop_init_1 = false;
                uint _e11 = d;
                if ((_e11 < 16u)) {
                } else {
                    break;
                }
                {
                    uint _e15 = t;
                    uint _e18 = d;
                    float _e21 = _group_0_binding_0_cs[((_e15 * 16u) + _e18)];
                    vec4 v = vec4(_e21);
                    vec4 _e23 = acc_noinit;
                    acc_noinit = (_e23 + v);
                    vec4 _e25 = acc_init;
                    acc_init = (_e25 + v);
                }
            }
            uint _e31 = t;
            float _e36 = acc_noinit.x;
            _group_0_binding_1_cs[(_e31 * 2u)] = _e36;
            uint _e38 = t;
            float _e45 = acc_init.x;
            _group_0_binding_1_cs[((_e38 * 2u) + 1u)] = _e45;
        }
    }
    return;
}

