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

uniform type_8_block_2Compute { vec4 _group_0_binding_3_cs[20]; };

uniform type_7_block_3Compute { vec4 _group_0_binding_4_cs; };


void test_msl_packed_vec3_as_arg(vec3 arg) {
    return;
}

void test_msl_packed_vec3_() {
    int idx = 1;
    _group_0_binding_1_cs.v3_ = vec3(1.0);
    _group_0_binding_1_cs.v3_.x = 1.0;
    _group_0_binding_1_cs.v3_.x = 2.0;
    int _e20 = idx;
    _group_0_binding_1_cs.v3_[_e20] = 3.0;
    Foo data = _group_0_binding_1_cs;
    vec3 unnamed = data.v3_;
    vec2 unnamed_1 = data.v3_.zx;
    test_msl_packed_vec3_as_arg(data.v3_);
    vec3 unnamed_2 = (data.v3_ * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 unnamed_3 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * data.v3_);
    vec3 unnamed_4 = (data.v3_ * 2.0);
    vec3 unnamed_5 = (2.0 * data.v3_);
}

void main() {
    float Foo_1 = 1.0;
    bool at = true;
    test_msl_packed_vec3_();
    float _e10 = _group_0_binding_4_cs.x;
    wg[6] = _e10;
    float _e16 = _group_0_binding_2_cs[1].y;
    wg[5] = _e16;
    float _e22 = _group_0_binding_3_cs[0].w;
    wg[4] = _e22;
    float _e26 = _group_0_binding_1_cs.v1_;
    wg[3] = _e26;
    float _e31 = _group_0_binding_1_cs.v3_.x;
    wg[2] = _e31;
    _group_0_binding_1_cs.v1_ = 4.0;
    wg[1] = float(uint(_group_0_binding_2_cs.length()));
    at_1 = 2u;
    return;
}

