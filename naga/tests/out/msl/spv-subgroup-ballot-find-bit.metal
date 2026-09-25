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
    metal::uint4 unnamed = metal::uint4((uint64_t)metal::simd_ballot(_e3 != 0u), 0, 0, 0);
    uint unnamed_1 = unnamed.x;
    uint unnamed_2 = unnamed.y;
    uint unnamed_3 = unnamed.z;
    uint unnamed_4 = unnamed.w;
    uint unnamed_5 = (unnamed_1 != 0u ? (((metal::ctz(unnamed_1) + 1) % 33) - 1) : (unnamed_2 != 0u ? (((metal::ctz(unnamed_2) + 1) % 33) - 1) + 32u : (unnamed_3 != 0u ? (((metal::ctz(unnamed_3) + 1) % 33) - 1) + 64u : (((metal::ctz(unnamed_4) + 1) % 33) - 1) + 96u)));
    uint unnamed_6 = unnamed.x;
    uint unnamed_7 = unnamed.y;
    uint unnamed_8 = unnamed.z;
    uint unnamed_9 = unnamed.w;
    uint unnamed_10 = (unnamed_9 != 0u ? metal::select(31 - metal::clz(unnamed_9), uint(-1), unnamed_9 == 0) + 96u : (unnamed_8 != 0u ? metal::select(31 - metal::clz(unnamed_8), uint(-1), unnamed_8 == 0) + 64u : (unnamed_7 != 0u ? metal::select(31 - metal::clz(unnamed_7), uint(-1), unnamed_7 == 0) + 32u : metal::select(31 - metal::clz(unnamed_6), uint(-1), unnamed_6 == 0))));
    global_1.member = unnamed_5 + unnamed_10;
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
