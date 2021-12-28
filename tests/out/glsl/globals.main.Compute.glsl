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

layout(std430) readonly buffer Foo_block_0Compute { Foo _group_0_binding_1_cs; };


void main() {
    float Foo_1 = 1.0;
    bool at = true;
    float _e7 = _group_0_binding_1_cs.v1_;
    wg[3] = _e7;
    float _e12 = _group_0_binding_1_cs.v3_.x;
    wg[2] = _e12;
    at_1 = 2u;
    return;
}

