// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void csLoad(
  metal::texture2d<float, metal::access::read> s_r_r [[user(fake0)]]
, metal::texture2d<float, metal::access::read> s_rg_r [[user(fake0)]]
, metal::texture2d<float, metal::access::read> s_rgba_r [[user(fake0)]]
) {
    metal::float4 phony = s_r_r.read(metal::uint2(metal::uint2(0u)));
    metal::float4 phony_1 = s_rg_r.read(metal::uint2(metal::uint2(0u)));
    metal::float4 phony_2 = s_rgba_r.read(metal::uint2(metal::uint2(0u)));
    return;
}


kernel void csStore(
  metal::texture2d<float, metal::access::write> s_r_w [[user(fake0)]]
, metal::texture2d<float, metal::access::write> s_rg_w [[user(fake0)]]
, metal::texture2d<float, metal::access::write> s_rgba_w [[user(fake0)]]
) {
    s_r_w.write(metal::float4(0.0), metal::uint2(metal::uint2(0u)));
    s_rg_w.write(metal::float4(0.0), metal::uint2(metal::uint2(0u)));
    s_rgba_w.write(metal::float4(0.0), metal::uint2(metal::uint2(0u)));
    return;
}
