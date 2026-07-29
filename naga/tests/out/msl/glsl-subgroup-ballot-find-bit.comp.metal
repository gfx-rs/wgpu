// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    uint result;
};

void main_1(
    device Output& global,
    thread metal::uint3& gl_GlobalInvocationID_1
) {
    metal::uint4 mask = {};
    uint lsb = {};
    uint msb = {};
    metal::uint3 _e3 = gl_GlobalInvocationID_1;
    mask = metal::uint4(_e3.x, 2u, 3u, 4u);
    metal::uint4 _e10 = mask;
    uint unnamed = (_e10.x != 0u ? (((metal::ctz(_e10.x) + 1) % 33) - 1) : (_e10.y != 0u ? (((metal::ctz(_e10.y) + 1) % 33) - 1) + 32u : (_e10.z != 0u ? (((metal::ctz(_e10.z) + 1) % 33) - 1) + 64u : (((metal::ctz(_e10.w) + 1) % 33) - 1) + 96u)));
    lsb = unnamed;
    metal::uint4 _e13 = mask;
    uint unnamed_1 = (_e13.w != 0u ? metal::select(31 - metal::clz(_e13.w), uint(-1), _e13.w == 0) + 96u : (_e13.z != 0u ? metal::select(31 - metal::clz(_e13.z), uint(-1), _e13.z == 0) + 64u : (_e13.y != 0u ? metal::select(31 - metal::clz(_e13.y), uint(-1), _e13.y == 0) + 32u : metal::select(31 - metal::clz(_e13.x), uint(-1), _e13.x == 0))));
    msb = unnamed_1;
    uint _e16 = lsb;
    uint _e17 = msb;
    global.result = _e16 + _e17;
    return;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  metal::uint3 gl_GlobalInvocationID [[thread_position_in_grid]]
, device Output& global [[user(fake0)]]
) {
    metal::uint3 gl_GlobalInvocationID_1 = {};
    gl_GlobalInvocationID_1 = gl_GlobalInvocationID;
    main_1(global, gl_GlobalInvocationID_1);
    return;
}
