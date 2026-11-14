#version 310 es

precision highp float;
precision highp int;

struct S {
    uint a;
    uint b[4];
};
layout(std430) readonly buffer type_4_block_0Fragment { float _group_0_binding_2_fs[]; };

uniform highp sampler2D _group_0_binding_3_fs;

layout(location = 0) smooth in vec2 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

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
    vec2 uv = _vs2fs_location0;
    uint i_2 = 0u;
    float local_7 = 0.0;
    bool local_8 = false;
    do {
        i_2 = 0u;
        local_8 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                uint _e14 = i_2;
                i_2 = (_e14 + 1u);
            }
            loop_init = false;
            uint _e2 = i_2;
            if ((_e2 < uint(_group_0_binding_2_fs.length()))) {
            } else {
                break;
            }
            {
                uint _e5 = i_2;
                float _e7 = _group_0_binding_2_fs[_e5];
                if ((_e7 > 0.0)) {
                    uint _e10 = i_2;
                    float _e12 = _group_0_binding_2_fs[_e10];
                    local_7 = _e12;
                    local_8 = true;
                    break;
                }
            }
        }
        bool _e17 = local_8;
        if (_e17) {
            break;
        }
        local_7 = 0.0;
    } while(false);
    float v = local_7;
    vec4 _e23 = texture(_group_0_binding_3_fs, vec2(uv));
    _fs2p_location0 = (_e23 * v);
    return;
}

