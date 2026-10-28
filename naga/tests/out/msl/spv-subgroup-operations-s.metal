// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void function(
    thread uint& global_2,
    thread uint& global_3
) {
    uint _e5 = global_2;
    uint _e6 = global_3;
    metal::threadgroup_barrier(metal::mem_flags::mem_none);
    metal::uint4 _e9 = metal::uint4((uint64_t)metal::simd_ballot((_e6 & 1u) == 1u), 0, 0, 0);
    metal::uint4 _e10 = metal::uint4((uint64_t)metal::simd_ballot(true), 0, 0, 0);
    bool _e12 = metal::simd_all(_e6 != 0u);
    bool _e14 = metal::simd_any(_e6 == 0u);
    uint _e15 = metal::simd_sum(_e6);
    uint _e16 = metal::simd_product(_e6);
    uint _e17 = metal::simd_min(_e6);
    uint _e18 = metal::simd_max(_e6);
    uint _e19 = metal::simd_and(_e6);
    uint _e20 = metal::simd_or(_e6);
    uint _e21 = metal::simd_xor(_e6);
    uint _e22 = metal::simd_prefix_exclusive_sum(_e6);
    uint _e23 = metal::simd_prefix_exclusive_product(_e6);
    uint _e24 = metal::simd_prefix_inclusive_sum(_e6);
    uint _e25 = metal::simd_prefix_inclusive_product(_e6);
    uint _e26 = metal::simd_broadcast_first(_e6);
    uint _e27 = metal::simd_broadcast(_e6, 4u);
    uint _e30 = metal::simd_shuffle(_e6, (_e5 - 1u) - _e6);
    uint _e31 = metal::simd_shuffle_down(_e6, 1u);
    uint _e32 = metal::simd_shuffle_up(_e6, 1u);
    uint _e34 = metal::simd_shuffle_xor(_e6, _e5 - 1u);
    return;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  uint param [[simdgroups_per_threadgroup]]
, uint param_1 [[simdgroup_index_in_threadgroup]]
, uint param_2 [[threads_per_simdgroup]]
, uint param_3 [[thread_index_in_simdgroup]]
) {
    uint global = {};
    uint global_1 = {};
    uint global_2 = {};
    uint global_3 = {};
    global = param;
    global_1 = param_1;
    global_2 = param_2;
    global_3 = param_3;
    function(global_2, global_3);
}
