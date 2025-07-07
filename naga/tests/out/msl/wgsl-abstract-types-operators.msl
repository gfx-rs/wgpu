// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_3 {
    uint inner[64];
};
constant float plus_fafaf_1 = 3.0;
constant float plus_fafai_1 = 3.0;
constant float plus_faf_f_1 = 3.0;
constant float plus_faiaf_1 = 3.0;
constant float plus_faiai_1 = 3.0;
constant float plus_fai_f_1 = 3.0;
constant float plus_f_faf_1 = 3.0;
constant float plus_f_fai_1 = 3.0;
constant float plus_f_f_f_1 = 3.0;
constant int plus_iaiai_1 = 3;
constant int plus_iai_i_1 = 3;
constant int plus_i_iai_1 = 3;
constant int plus_i_i_i_1 = 3;
constant uint plus_uaiai_1 = 3u;
constant uint plus_uai_u_1 = 3u;
constant uint plus_u_uai_1 = 3u;
constant uint plus_u_u_u_1 = 3u;
constant uint bitflip_u_u = 0u;
constant uint bitflip_uai = 0u;
constant int least_i32_ = (-2147483647 - 1);
constant float least_f32_ = -340282350000000000000000000000000000000.0;
constant int shl_iaiai = 4;
constant int shl_iai_u_1 = 4;
constant uint shl_uaiai = 4u;
constant uint shl_uai_u = 4u;
constant int shr_iaiai = 0;
constant int shr_iai_u_1 = 0;
constant uint shr_uaiai = 0u;
constant uint shr_uai_u = 0u;
constant int wgpu_4492_ = (-2147483647 - 1);

void runtime_values(
) {
    float f = 42.0;
    int i = 43;
    uint u = 44u;
    float plus_fafaf = 3.0;
    float plus_fafai = 3.0;
    float plus_faf_f = {};
    float plus_faiaf = 3.0;
    float plus_faiai = 3.0;
    float plus_fai_f = {};
    float plus_f_faf = {};
    float plus_f_fai = {};
    float plus_f_f_f = {};
    int plus_iaiai = 3;
    int plus_iai_i = {};
    int plus_i_iai = {};
    int plus_i_i_i = {};
    uint plus_uaiai = 3u;
    uint plus_uai_u = {};
    uint plus_u_uai = {};
    uint plus_u_u_u = {};
    int shl_iai_u = {};
    int shr_iai_u = {};
    float _e8 = f;
    plus_faf_f = 1.0 + _e8;
    float _e14 = f;
    plus_fai_f = 1.0 + _e14;
    float _e18 = f;
    plus_f_faf = _e18 + 2.0;
    float _e22 = f;
    plus_f_fai = _e22 + 2.0;
    float _e26 = f;
    float _e27 = f;
    plus_f_f_f = _e26 + _e27;
    int _e31 = i;
    plus_iai_i = as_type<int>(as_type<uint>(1) + as_type<uint>(_e31));
    int _e35 = i;
    plus_i_iai = as_type<int>(as_type<uint>(_e35) + as_type<uint>(2));
    int _e39 = i;
    int _e40 = i;
    plus_i_i_i = as_type<int>(as_type<uint>(_e39) + as_type<uint>(_e40));
    uint _e44 = u;
    plus_uai_u = 1u + _e44;
    uint _e48 = u;
    plus_u_uai = _e48 + 2u;
    uint _e52 = u;
    uint _e53 = u;
    plus_u_u_u = _e52 + _e53;
    uint _e56 = u;
    shl_iai_u = 1 << _e56;
    uint _e60 = u;
    shr_iai_u = 1 << _e60;
    return;
}

void wgpu_4445_(
) {
    return;
}

void wgpu_4435_(
    threadgroup type_3& a
) {
    uint y = a.inner[as_type<int>(as_type<uint>(1) - as_type<uint>(1))];
    return;
}

kernel void main_(
  metal::uint3 __local_invocation_id [[thread_position_in_threadgroup]]
, threadgroup type_3& a
) {
    if (metal::all(__local_invocation_id == metal::uint3(0u))) {
        a = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    runtime_values();
    wgpu_4445_();
    wgpu_4435_(a);
    return;
}
