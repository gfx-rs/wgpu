#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Foo {
    vec3 v3_;
    float v1_;
};
shared float wg[10];

shared uint at_1;

layout(std430) buffer Foo_block_0Compute { Foo _group_0_binding_1_cs; };

layout(std430) readonly buffer type_6_block_1Compute { vec2 _group_0_binding_2_cs[]; };


void main() {
    float Foo_1 = 1.0;
    bool at = true;
    float _e9 = _group_0_binding_1_cs.v1_;
    wg[3] = _e9;
    float _e14 = _group_0_binding_1_cs.v3_.x;
    wg[2] = _e14;
    _group_0_binding_1_cs.v1_ = 4.0;
    wg[1] = float(uint(_group_0_binding_2_cs.length()));
    at_1 = 2u;
    return;
}

