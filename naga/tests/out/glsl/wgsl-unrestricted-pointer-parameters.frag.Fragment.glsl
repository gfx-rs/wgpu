#version 310 es

precision highp float;
precision highp int;

layout(std430) readonly buffer type_2_block_0Fragment { float _group_0_binding_2_fs[]; };

uniform highp sampler2D _group_0_binding_3_fs;

layout(location = 0) smooth in vec2 _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec2 uv = _vs2fs_location0;
    uint i = 0u;
    float local = 0.0;
    bool local_1 = false;
    do {
        i = 0u;
        local_1 = false;
        bool loop_init = true;
        while(true) {
            if (!loop_init) {
                uint _e14 = i;
                i = (_e14 + 1u);
            }
            loop_init = false;
            uint _e2 = i;
            if ((_e2 < uint(_group_0_binding_2_fs.length()))) {
            } else {
                break;
            }
            {
                uint _e5 = i;
                float _e7 = _group_0_binding_2_fs[_e5];
                if ((_e7 > 0.0)) {
                    uint _e10 = i;
                    float _e12 = _group_0_binding_2_fs[_e10];
                    local = _e12;
                    local_1 = true;
                    break;
                }
            }
        }
        bool _e17 = local_1;
        if (_e17) {
            break;
        }
        local = 0.0;
    } while(false);
    float v = local;
    vec4 _e23 = texture(_group_0_binding_3_fs, vec2(uv));
    _fs2p_location0 = (_e23 * v);
    return;
}

