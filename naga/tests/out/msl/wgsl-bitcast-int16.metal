// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    metal::short2 i2_ = metal::short2(static_cast<short>(1), static_cast<short>(2));
    metal::ushort2 u2_ = metal::ushort2(static_cast<ushort>(1), static_cast<ushort>(2));
    metal::short4 i4_ = metal::short4(static_cast<short>(1), static_cast<short>(2), static_cast<short>(3), static_cast<short>(4));
    metal::ushort4 u4_ = metal::ushort4(static_cast<ushort>(1), static_cast<ushort>(2), static_cast<ushort>(3), static_cast<ushort>(4));
    int i1_ = 0;
    uint u1_ = 0u;
    float f1_ = 0.0;
    metal::int2 iv2_ = metal::int2(0);
    metal::uint2 uv2_ = metal::uint2(0u);
    metal::float2 fv2_ = metal::float2(0.0);
    metal::short2 _e35 = i2_;
    i1_ = as_type<int>(_e35);
    metal::ushort2 _e37 = u2_;
    u1_ = as_type<uint>(_e37);
    metal::ushort2 _e39 = u2_;
    f1_ = as_type<float>(_e39);
    metal::short4 _e41 = i4_;
    uv2_ = as_type<metal::uint2>(_e41);
    metal::ushort4 _e43 = u4_;
    fv2_ = as_type<metal::float2>(_e43);
    uint _e45 = u1_;
    i2_ = as_type<metal::short2>(_e45);
    float _e47 = f1_;
    u2_ = as_type<metal::ushort2>(_e47);
    metal::float2 _e49 = fv2_;
    i4_ = as_type<metal::short4>(_e49);
    metal::int2 _e51 = iv2_;
    u4_ = as_type<metal::ushort4>(_e51);
    return;
}
