// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    uint result;
};

void main_1(
    device Output& global,
    thread uint& gl_SubgroupInvocationID_1,
    thread uint& gl_SubgroupSize_1
) {
    metal::uint4 mask = {};
    uint lsb = {};
    uint msb = {};
    uint _e4 = gl_SubgroupInvocationID_1;
    uint _e5 = gl_SubgroupSize_1;
    mask = metal::uint4(_e4, _e5, 3u, 4u);
    metal::uint4 _e10 = mask;
    uint unnamed = _e10.x;
    uint unnamed_1 = _e10.y;
    uint unnamed_2 = _e10.z;
    uint unnamed_3 = _e10.w;
    uint unnamed_4 = (unnamed != 0u ? (((metal::ctz(unnamed) + 1) % 33) - 1) : (unnamed_1 != 0u ? (((metal::ctz(unnamed_1) + 1) % 33) - 1) + 32u : (unnamed_2 != 0u ? (((metal::ctz(unnamed_2) + 1) % 33) - 1) + 64u : (((metal::ctz(unnamed_3) + 1) % 33) - 1) + 96u)));
    lsb = unnamed_4;
    metal::uint4 _e13 = mask;
    uint unnamed_5 = _e13.x;
    uint unnamed_6 = _e13.y;
    uint unnamed_7 = _e13.z;
    uint unnamed_8 = _e13.w;
    uint unnamed_9 = (unnamed_8 != 0u ? metal::select(31 - metal::clz(unnamed_8), uint(-1), unnamed_8 == 0) + 96u : (unnamed_7 != 0u ? metal::select(31 - metal::clz(unnamed_7), uint(-1), unnamed_7 == 0) + 64u : (unnamed_6 != 0u ? metal::select(31 - metal::clz(unnamed_6), uint(-1), unnamed_6 == 0) + 32u : metal::select(31 - metal::clz(unnamed_5), uint(-1), unnamed_5 == 0))));
    msb = unnamed_9;
    uint _e16 = lsb;
    uint _e17 = msb;
    global.result = _e16 + _e17;
    return;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]]
, uint gl_SubgroupSize [[threads_per_simdgroup]]
, device Output& global [[user(fake0)]]
) {
    uint gl_SubgroupInvocationID_1 = {};
    uint gl_SubgroupSize_1 = {};
    gl_SubgroupInvocationID_1 = gl_SubgroupInvocationID;
    gl_SubgroupSize_1 = gl_SubgroupSize;
    main_1(global, gl_SubgroupInvocationID_1, gl_SubgroupSize_1);
    return;
}
