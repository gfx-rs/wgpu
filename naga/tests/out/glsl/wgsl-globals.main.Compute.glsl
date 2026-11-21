#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct FooStruct {
    vec3 v3_;
    float v1_;
};
struct mat3x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
    vec2 c_2_;
};
struct mat4x2f32_std140_ {
    vec2 c_0_;
    vec2 c_1_;
    vec2 c_2_;
    vec2 c_3_;
};
const bool Foo_1 = true;

shared float wg[10];

shared uint at_1;

layout(std430) buffer FooStruct_block_0Compute { FooStruct _group_0_binding_1_cs; };

layout(std430) readonly buffer type_6_block_1Compute { vec2 _group_0_binding_2_cs[]; };

layout(std140) uniform type_8_block_2Compute { vec4 _group_0_binding_3_cs[20]; };

layout(std140) uniform type_4_block_3Compute { vec3 _group_0_binding_4_cs; };

layout(std140) uniform mat3x2f32_std140_block_4Compute { mat3x2f32_std140_ _group_0_binding_5_cs; };

layout(std140) uniform type_12_block_5Compute { mat2x4 _group_0_binding_6_cs[2][2]; };

layout(std140) uniform type_17_block_6Compute { mat4x2f32_std140_ _group_0_binding_7_cs[2][2]; };


void test_msl_packed_vec3_as_arg(vec3 arg) {
    return;
}

void test_msl_packed_vec3_() {
    int idx = 1;
    _group_0_binding_1_cs.v3_ = vec3(1.0);
    _group_0_binding_1_cs.v3_.x = 1.0;
    _group_0_binding_1_cs.v3_.x = 2.0;
    int _e16 = idx;
    _group_0_binding_1_cs.v3_[_e16] = 3.0;
    FooStruct data = _group_0_binding_1_cs;
    vec3 l0_ = data.v3_;
    vec2 l1_ = data.v3_.zx;
    test_msl_packed_vec3_as_arg(data.v3_);
    vec3 mvm0_ = (data.v3_ * mat3x3(0.0));
    vec3 mvm1_ = (mat3x3(0.0) * data.v3_);
    vec3 svm0_ = (data.v3_ * 2.0);
    vec3 svm1_ = (2.0 * data.v3_);
    return;
}

void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        wg = float[10](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        at_1 = 0u;
    }
    memoryBarrierShared();
    barrier();
    float Foo = 1.0;
    bool at = true;
    test_msl_packed_vec3_();
    vec2 _e6 = _group_0_binding_7_cs[0][0].c_0_;
    vec2 _e8 = _group_0_binding_7_cs[0][0].c_1_;
    vec2 _e10 = _group_0_binding_7_cs[0][0].c_2_;
    vec2 _e12 = _group_0_binding_7_cs[0][0].c_3_;
    vec4 _e18 = _group_0_binding_6_cs[0][0][0];
    wg[7] = (mat4x2(_e6, _e8, _e10, _e12) * _e18).x;
    vec2 _e25 = _group_0_binding_5_cs.c_0_;
    vec2 _e27 = _group_0_binding_5_cs.c_1_;
    vec2 _e29 = _group_0_binding_5_cs.c_2_;
    vec3 _e32 = _group_0_binding_4_cs;
    wg[6] = (mat3x2(_e25, _e27, _e29) * _e32).x;
    float _e40 = _group_0_binding_2_cs[1].y;
    wg[5] = _e40;
    float _e46 = _group_0_binding_3_cs[0].w;
    wg[4] = _e46;
    float _e51 = _group_0_binding_1_cs.v1_;
    wg[3] = _e51;
    float _e57 = _group_0_binding_1_cs.v3_.x;
    wg[2] = _e57;
    _group_0_binding_1_cs.v1_ = 4.0;
    wg[1] = float(uint(_group_0_binding_2_cs.length()));
    atomicExchange(at_1, 2u);
    return;
}

