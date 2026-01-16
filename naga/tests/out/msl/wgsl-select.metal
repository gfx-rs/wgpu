// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    metal::int2 x0_ = metal::int2(1, 2);
    metal::float2 i1_ = {};
    int _e12 = x0_.x;
    int _e14 = x0_.y;
    i1_ = (_e12 < _e14) ? metal::float2(0.0, 1.0) : metal::float2(1.0, 0.0);
    return;
}
