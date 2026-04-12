// language: metal3.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct cs_mainInput {
};
[[max_total_threads_per_threadgroup(2)]] kernel void cs_main(
  metal::uint3 id [[thread_position_in_threadgroup]]
, metal::texture2d<ulong, metal::access::read_write> image [[user(fake0)]]
) {
    image.atomic_max(metal::uint2(metal::int2(0, 0)), 1uL);
    if (metal::int2(0, 0).x == -99999) { image.write(ulong4(0uL), metal::uint2(metal::int2(0, 0))); }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    image.atomic_min(metal::uint2(metal::int2(0, 0)), 1uL);
    if (metal::int2(0, 0).x == -99999) { image.write(ulong4(0uL), metal::uint2(metal::int2(0, 0))); }
    return;
}
