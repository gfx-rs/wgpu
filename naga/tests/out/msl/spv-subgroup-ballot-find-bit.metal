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
    uint unnamed_1 = (unnamed.x != 0u ? (((metal::ctz(unnamed.x) + 1) % 33) - 1) : (unnamed.y != 0u ? (((metal::ctz(unnamed.y) + 1) % 33) - 1) + 32u : (unnamed.z != 0u ? (((metal::ctz(unnamed.z) + 1) % 33) - 1) + 64u : (((metal::ctz(unnamed.w) + 1) % 33) - 1) + 96u)));
    uint unnamed_2 = (unnamed.w != 0u ? metal::select(31 - metal::clz(unnamed.w), uint(-1), unnamed.w == 0) + 96u : (unnamed.z != 0u ? metal::select(31 - metal::clz(unnamed.z), uint(-1), unnamed.z == 0) + 64u : (unnamed.y != 0u ? metal::select(31 - metal::clz(unnamed.y), uint(-1), unnamed.y == 0) + 32u : metal::select(31 - metal::clz(unnamed.x), uint(-1), unnamed.x == 0))));
    global_1.member = unnamed_1 + unnamed_2;
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
