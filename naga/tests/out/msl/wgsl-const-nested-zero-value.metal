// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_3 {
    uint inner[8];
};
constant metal::uint4 composed = metal::uint4(metal::uint2(0u, 0u), 7u, 9u);

kernel void main_(
  device type_3& o [[user(fake0)]]
, device metal::uint4& v [[user(fake0)]]
) {
    o.inner[0] = 0u;
    o.inner[1] = 7u;
    o.inner[2] = 0u;
    o.inner[3] = as_type<uint>(0.0);
    o.inner[4] = as_type<uint>(1.0);
    o.inner[5] = 0u;
    o.inner[6] = 9u;
    o.inner[7] = 7u;
    v = composed;
    return;
}
