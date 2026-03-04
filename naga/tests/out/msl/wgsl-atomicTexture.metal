// language: metal3.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct cs_mainInput {
};
kernel void cs_main(
  metal::uint3 id [[thread_position_in_threadgroup]]
, metal::texture2d<uint, metal::access::read_write> image_u [[user(fake0)]]
, metal::texture2d<int, metal::access::read_write> image_s [[user(fake0)]]
) {
    image_u.atomic_fetch_max(metal::uint2(metal::int2(0, 0)), 1u);
    image_u.atomic_fetch_min(metal::uint2(metal::int2(0, 0)), 1u);
    image_u.atomic_fetch_add(metal::uint2(metal::int2(0, 0)), 1u);
    image_u.atomic_fetch_and(metal::uint2(metal::int2(0, 0)), 1u);
    image_u.atomic_fetch_or(metal::uint2(metal::int2(0, 0)), 1u);
    image_u.atomic_fetch_xor(metal::uint2(metal::int2(0, 0)), 1u);
    image_s.atomic_fetch_max(metal::uint2(metal::int2(0, 0)), 1);
    image_s.atomic_fetch_min(metal::uint2(metal::int2(0, 0)), 1);
    image_s.atomic_fetch_add(metal::uint2(metal::int2(0, 0)), 1);
    image_s.atomic_fetch_and(metal::uint2(metal::int2(0, 0)), 1);
    image_s.atomic_fetch_or(metal::uint2(metal::int2(0, 0)), 1);
    image_s.atomic_fetch_xor(metal::uint2(metal::int2(0, 0)), 1);
    return;
}
