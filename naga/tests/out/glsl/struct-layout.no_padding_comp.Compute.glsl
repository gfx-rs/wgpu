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
uniform NoPadding_block_0Compute { NoPadding _group_0_binding_0_cs; };

layout(std430) buffer NoPadding_block_1Compute { NoPadding _group_0_binding_1_cs; };


void main() {
    NoPadding x = NoPadding(vec3(0.0), 0.0);
    NoPadding _e2 = _group_0_binding_0_cs;
    x = _e2;
    NoPadding _e4 = _group_0_binding_1_cs;
    x = _e4;
    return;
}

