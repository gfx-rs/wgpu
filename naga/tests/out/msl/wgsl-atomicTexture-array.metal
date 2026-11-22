// language: metal3.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct cs_mainInput {
};
[[max_total_threads_per_threadgroup(2)]] kernel void cs_main(
  metal::uint3 id [[thread_position_in_threadgroup]]
, metal::texture2d_array<uint, metal::access::read_write> image_u [[user(fake0)]]
, metal::texture2d_array<int, metal::access::read_write> image_s [[user(fake0)]]
) {
    image_u.atomic_fetch_max(metal::uint2(metal::int2(0, 0)), uint(1), 1u);
    if (metal::int2(0, 0).x == -99999) { image_u.write(uint4(0u), metal::uint2(metal::int2(0, 0)), 1); }
    image_s.atomic_fetch_min(metal::uint2(metal::int2(0, 0)), uint(1u), 1);
    if (metal::int2(0, 0).x == -99999) { image_s.write(int4(0), metal::uint2(metal::int2(0, 0)), 1u); }
    return;
}
