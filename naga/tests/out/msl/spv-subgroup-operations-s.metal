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
    metal::uint4 unnamed = metal::uint4((uint64_t)metal::simd_ballot((_e6 & 1u) == 1u), 0, 0, 0);
    metal::uint4 unnamed_1 = metal::uint4((uint64_t)metal::simd_ballot(true), 0, 0, 0);
    bool unnamed_2 = metal::simd_all(_e6 != 0u);
    bool unnamed_3 = metal::simd_any(_e6 == 0u);
    uint unnamed_4 = metal::simd_sum(_e6);
    uint unnamed_5 = metal::simd_product(_e6);
    uint unnamed_6 = metal::simd_min(_e6);
    uint unnamed_7 = metal::simd_max(_e6);
    uint unnamed_8 = metal::simd_and(_e6);
    uint unnamed_9 = metal::simd_or(_e6);
    uint unnamed_10 = metal::simd_xor(_e6);
    uint unnamed_11 = metal::simd_prefix_exclusive_sum(_e6);
    uint unnamed_12 = metal::simd_prefix_exclusive_product(_e6);
    uint unnamed_13 = metal::simd_prefix_inclusive_sum(_e6);
    uint unnamed_14 = metal::simd_prefix_inclusive_product(_e6);
    uint unnamed_15 = metal::simd_broadcast_first(_e6);
    uint unnamed_16 = metal::simd_broadcast(_e6, 4u);
    uint unnamed_17 = metal::simd_shuffle(_e6, (_e5 - 1u) - _e6);
    uint unnamed_18 = metal::simd_shuffle_down(_e6, 1u);
    uint unnamed_19 = metal::simd_shuffle_up(_e6, 1u);
    uint unnamed_20 = metal::simd_shuffle_xor(_e6, _e5 - 1u);
    return;
}

struct main_Input {
};
kernel void main_(
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
