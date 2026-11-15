// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct S {
    metal::packed_float3 v;
    float w;
};

kernel void main_(
  device metal::uint4& out [[user(fake0)]]
) {
    S s = S {metal::float3(1.0, 2.0, 3.0), 4.0};
    out = metal::uint4(as_type<metal::uint3>(metal::float3(s.v)), 0u);
    return;
}
