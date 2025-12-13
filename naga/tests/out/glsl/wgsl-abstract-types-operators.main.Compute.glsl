#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

const float plus_fafaf_1 = 3.0;
const float plus_fafai_1 = 3.0;
const float plus_faf_f_1 = 3.0;
const float plus_faiaf_1 = 3.0;
const float plus_faiai_1 = 3.0;
const float plus_fai_f_1 = 3.0;
const float plus_f_faf_1 = 3.0;
const float plus_f_fai_1 = 3.0;
const float plus_f_f_f_1 = 3.0;
const int plus_iaiai_1 = 3;
const int plus_iai_i_1 = 3;
const int plus_i_iai_1 = 3;
const int plus_i_i_i_1 = 3;
const uint plus_uaiai_1 = 3u;
const uint plus_uai_u_1 = 3u;
const uint plus_u_uai_1 = 3u;
const uint plus_u_u_u_1 = 3u;
const uint bitflip_u_u = 0u;
const uint bitflip_uai = 0u;
const int least_i32_ = -2147483648;
const float least_f32_ = -3.4028235e38;
const int shl_iaiai = 4;
const int shl_iai_u_1 = 4;
const uint shl_uaiai = 4u;
const uint shl_uai_u = 4u;
const int shr_iaiai = 0;
const int shr_iai_u_1 = 0;
const uint shr_uaiai = 0u;
const uint shr_uai_u = 0u;
const int wgpu_4492_ = -2147483648;

shared uint a[64];


void runtime_values() {
    float f = 42.0;
    int i = 43;
    uint u = 44u;
    float plus_fafaf = 3.0;
    float plus_fafai = 3.0;
    float plus_faf_f = 0.0;
    float plus_faiaf = 3.0;
    float plus_faiai = 3.0;
    float plus_fai_f = 0.0;
    float plus_f_faf = 0.0;
    float plus_f_fai = 0.0;
    float plus_f_f_f = 0.0;
    int plus_iaiai = 3;
    int plus_iai_i = 0;
    int plus_i_iai = 0;
    int plus_i_i_i = 0;
    uint plus_uaiai = 3u;
    uint plus_uai_u = 0u;
    uint plus_u_uai = 0u;
    uint plus_u_u_u = 0u;
    int shl_iai_u = 0;
    int shr_iai_u = 0;
    float _e8 = f;
    plus_faf_f = (1.0 + _e8);
    float _e14 = f;
    plus_fai_f = (1.0 + _e14);
    float _e18 = f;
    plus_f_faf = (_e18 + 2.0);
    float _e22 = f;
    plus_f_fai = (_e22 + 2.0);
    float _e26 = f;
    float _e27 = f;
    plus_f_f_f = (_e26 + _e27);
    int _e31 = i;
    plus_iai_i = (1 + _e31);
    int _e35 = i;
    plus_i_iai = (_e35 + 2);
    int _e39 = i;
    int _e40 = i;
    plus_i_i_i = (_e39 + _e40);
    uint _e44 = u;
    plus_uai_u = (1u + _e44);
    uint _e48 = u;
    plus_u_uai = (_e48 + 2u);
    uint _e52 = u;
    uint _e53 = u;
    plus_u_u_u = (_e52 + _e53);
    uint _e56 = u;
    shl_iai_u = (1 << _e56);
    uint _e60 = u;
    shr_iai_u = (1 << _e60);
    return;
}

void wgpu_4445_() {
    return;
}

void wgpu_4435_() {
    uint y = a[(1 - 1)];
    return;
}

void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        a = uint[64](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    memoryBarrierShared();
    barrier();
    runtime_values();
    wgpu_4445_();
    wgpu_4435_();
    return;
}

