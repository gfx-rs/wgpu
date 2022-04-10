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
    vec3 unnamed = vec3(0.0);
    vec2 unnamed_1 = vec2(0.0);
    int idx = 1;
    float Foo_1 = 1.0;
    bool at = true;
    float _e9 = _group_0_binding_1_cs.v1_;
    wg[3] = _e9;
    float _e14 = _group_0_binding_1_cs.v3_.x;
    wg[2] = _e14;
    vec3 _e16 = _group_0_binding_1_cs.v3_;
    unnamed = _e16;
    vec3 _e19 = _group_0_binding_1_cs.v3_;
    unnamed_1 = _e19.zx;
    _group_0_binding_1_cs.v1_ = 4.0;
    wg[1] = float(uint(_group_0_binding_2_cs.length()));
    at_1 = 2u;
    _group_0_binding_1_cs.v3_ = vec3(1.0);
    _group_0_binding_1_cs.v3_.x = 1.0;
    _group_0_binding_1_cs.v3_.x = 2.0;
    int _e42 = idx;
    _group_0_binding_1_cs.v3_[_e42] = 3.0;
    vec3 _e47 = _group_0_binding_1_cs.v3_;
    vec3 unnamed_2 = (_e47 * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 _e50 = _group_0_binding_1_cs.v3_;
    vec3 unnamed_3 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * _e50);
}

