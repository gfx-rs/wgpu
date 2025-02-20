#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

struct NoPadding {
    vec3 v3_;
    float f3_;
};
struct NeedsPadding {
    float f3_forces_padding;
    vec3 v3_needs_padding;
    float f3_;
};
uniform NeedsPadding_block_0Compute { NeedsPadding _group_0_binding_2_cs; };

layout(std430) buffer NeedsPadding_block_1Compute { NeedsPadding _group_0_binding_3_cs; };


void main() {
    NeedsPadding x_1 = NeedsPadding(0.0, vec3(0.0), 0.0);
    NeedsPadding _e2 = _group_0_binding_2_cs;
    x_1 = _e2;
    NeedsPadding _e4 = _group_0_binding_3_cs;
    x_1 = _e4;
    return;
}

