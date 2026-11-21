#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct mat2x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
};
layout(std140) uniform type_7_block_0Compute { mat2x2f32_std140_ _group_0_binding_0_cs[64]; };

layout(std140) uniform type_9_block_1Compute { mat2x2f32_std140_ _group_0_binding_1_cs[20][2]; };

layout(std430) buffer type_4_block_2Compute { mat2x2 _group_0_binding_2_cs[4]; };


mat2x2[64] convert_std140_0_(mat2x2f32_std140_ v[64]) {
    mat2x2f32_std140_ s[64] = mat2x2f32_std140_[64](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0)));
    mat2x2 r[64] = mat2x2[64](mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0), mat2x2(0.0));
    uint i_1 = 0u;
    s = v;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e15 = i_1;
            i_1 = (_e15 + 1u);
        }
        loop_init = false;
        uint _e7 = i_1;
        if ((_e7 < 64u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e10 = s[_e7];
        r[_e7] = mat2x2(_e10.c_0_, _e10.c_1_);
    }
    mat2x2 _e17[64] = r;
    return _e17;
}

mat2x2[2] convert_std140_1_(mat2x2f32_std140_ v_1[2]) {
    return mat2x2[2](mat2x2(v_1[0].c_0_, v_1[0].c_1_), mat2x2(v_1[1].c_0_, v_1[1].c_1_));
}

mat2x2[20][2] convert_std140_2_(mat2x2f32_std140_ v_2[20][2]) {
    mat2x2f32_std140_ s_1[20][2] = mat2x2f32_std140_[20][2](mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))), mat2x2f32_std140_[2](mat2x2f32_std140_(vec2(0.0), vec2(0.0)), mat2x2f32_std140_(vec2(0.0), vec2(0.0))));
    mat2x2 r_1[20][2] = mat2x2[20][2](mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)), mat2x2[2](mat2x2(0.0), mat2x2(0.0)));
    uint i_2 = 0u;
    s_1 = v_2;
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            uint _e13 = i_2;
            i_2 = (_e13 + 1u);
        }
        loop_init_1 = false;
        uint _e7 = i_2;
        if ((_e7 < 20u)) {
        } else {
            break;
        }
        mat2x2f32_std140_ _e10[2] = s_1[_e7];
        mat2x2 _e11[2] = convert_std140_1_(_e10);
        r_1[_e7] = _e11;
    }
    mat2x2 _e15[20][2] = r_1;
    return _e15;
}

void main() {
    uint i = gl_LocalInvocationIndex;
    mat2x2f32_std140_ _e12[64] = _group_0_binding_0_cs;
    mat2x2 _e2[64] = convert_std140_0_(_e12);
    mat2x2f32_std140_ _e13[20][2] = _group_0_binding_1_cs;
    mat2x2 _e4[20][2] = convert_std140_2_(_e13);
    _group_0_binding_2_cs[0] = _e2[i];
    _group_0_binding_2_cs[1] = _e4[i][1];
    return;
}

