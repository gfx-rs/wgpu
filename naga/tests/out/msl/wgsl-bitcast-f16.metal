// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    metal::half2 h2_ = metal::half2(1.0h, 2.0h);
    metal::half4 h4_ = metal::half4(1.0h, 2.0h, 3.0h, 4.0h);
    int i1_ = 0;
    uint u1_ = 0u;
    float f1_ = 0.0;
    metal::int2 i2_ = metal::int2(0);
    metal::uint2 u2_ = metal::uint2(0u);
    metal::float2 f2_ = metal::float2(0.0);
    metal::half2 _e25 = h2_;
    i1_ = as_type<int>(_e25);
    metal::half2 _e27 = h2_;
    u1_ = as_type<uint>(_e27);
    metal::half2 _e29 = h2_;
    f1_ = as_type<float>(_e29);
    metal::half4 _e31 = h4_;
    i2_ = as_type<metal::int2>(_e31);
    metal::half4 _e33 = h4_;
    u2_ = as_type<metal::uint2>(_e33);
    metal::half4 _e35 = h4_;
    f2_ = as_type<metal::float2>(_e35);
    int _e37 = i1_;
    h2_ = as_type<metal::half2>(_e37);
    uint _e39 = u1_;
    h2_ = as_type<metal::half2>(_e39);
    float _e41 = f1_;
    h2_ = as_type<metal::half2>(_e41);
    metal::int2 _e43 = i2_;
    h4_ = as_type<metal::half4>(_e43);
    metal::uint2 _e45 = u2_;
    h4_ = as_type<metal::half4>(_e45);
    metal::float2 _e47 = f2_;
    h4_ = as_type<metal::half4>(_e47);
    metal::half2 _e49 = h2_;
    h2_ = as_type<metal::half2>(_e49);
    uint _e51 = u1_;
    u1_ = as_type<uint>(_e51);
    u1_ = as_type<uint>(4294967295u);
    f1_ = as_type<float>(1);
    i1_ = as_type<int>(1.0);
    h2_ = as_type<metal::half2>(1);
    h2_ = as_type<metal::half2>(metal::half2(1.0h, 2.0h));
    h4_ = as_type<metal::half4>(metal::int2(1, 2));
    u2_ = as_type<metal::uint2>(metal::uint2(4294967295u, 1u));
    return;
}
