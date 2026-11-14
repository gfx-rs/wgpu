// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    uint member;
};

void function(
    thread uint& global,
    device type_1& global_1
) {
    uint _e3 = global;
    metal::uint4 _e5 = metal::uint4((uint64_t)metal::simd_ballot(_e3 != 0u), 0, 0, 0);
    uint unnamed = _e5.x;
    uint unnamed_1 = _e5.y;
    uint unnamed_2 = _e5.z;
    uint unnamed_3 = _e5.w;
    uint _e6 = (unnamed != 0u ? (((metal::ctz(unnamed) + 1) % 33) - 1) : (unnamed_1 != 0u ? (((metal::ctz(unnamed_1) + 1) % 33) - 1) + 32u : (unnamed_2 != 0u ? (((metal::ctz(unnamed_2) + 1) % 33) - 1) + 64u : (((metal::ctz(unnamed_3) + 1) % 33) - 1) + 96u)));
    uint unnamed_4 = _e5.x;
    uint unnamed_5 = _e5.y;
    uint unnamed_6 = _e5.z;
    uint unnamed_7 = _e5.w;
    uint _e7 = (unnamed_7 != 0u ? metal::select(31 - metal::clz(unnamed_7), uint(-1), unnamed_7 == 0) + 96u : (unnamed_6 != 0u ? metal::select(31 - metal::clz(unnamed_6), uint(-1), unnamed_6 == 0) + 64u : (unnamed_5 != 0u ? metal::select(31 - metal::clz(unnamed_5), uint(-1), unnamed_5 == 0) + 32u : metal::select(31 - metal::clz(unnamed_4), uint(-1), unnamed_4 == 0))));
    global_1.member = _e6 + _e7;
    return;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  uint param [[thread_index_in_simdgroup]]
, device type_1& global_1 [[user(fake0)]]
) {
    uint global = {};
    global = param;
    function(global, global_1);
}
