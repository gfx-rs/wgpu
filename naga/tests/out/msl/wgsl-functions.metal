// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


metal::float2 test_fma(
) {
    metal::float2 a = metal::float2(2.0, 2.0);
    metal::float2 b = metal::float2(0.5, 0.5);
    metal::float2 c = metal::float2(0.5, 0.5);
    return metal::fma(a, b, c);
}

int naga_dot_int2(metal::int2 a, metal::int2 b) {
    return ( + a.x * b.x + a.y * b.y);
}

uint naga_dot_uint3(metal::uint3 a, metal::uint3 b) {
    return ( + a.x * b.x + a.y * b.y + a.z * b.z);
}

int test_integer_dot_product(
) {
    metal::int2 a_2_ = metal::int2(1);
    metal::int2 b_2_ = metal::int2(1);
    int c_2_ = naga_dot_int2(a_2_, b_2_);
    metal::uint3 a_3_ = metal::uint3(1u);
    metal::uint3 b_3_ = metal::uint3(1u);
    uint c_3_ = naga_dot_uint3(a_3_, b_3_);
    return 32;
}

uint test_packed_integer_dot_product(
) {
    int c_5_ = ( + (int(1u) << 24 >> 24) * (int(2u) << 24 >> 24) + (int(1u) << 16 >> 24) * (int(2u) << 16 >> 24) + (int(1u) << 8 >> 24) * (int(2u) << 8 >> 24) + (int(1u) >> 24) * (int(2u) >> 24));
    uint c_6_ = ( + ((3u) << 24 >> 24) * ((4u) << 24 >> 24) + ((3u) << 16 >> 24) * ((4u) << 16 >> 24) + ((3u) << 8 >> 24) * ((4u) << 8 >> 24) + ((3u) >> 24) * ((4u) >> 24));
    uint _e7 = 5u + c_6_;
    uint _e9 = 6u + c_6_;
    int c_7_ = ( + (int(_e7) << 24 >> 24) * (int(_e9) << 24 >> 24) + (int(_e7) << 16 >> 24) * (int(_e9) << 16 >> 24) + (int(_e7) << 8 >> 24) * (int(_e9) << 8 >> 24) + (int(_e7) >> 24) * (int(_e9) >> 24));
    uint _e12 = 7u + c_6_;
    uint _e14 = 8u + c_6_;
    uint c_8_ = ( + ((_e12) << 24 >> 24) * ((_e14) << 24 >> 24) + ((_e12) << 16 >> 24) * ((_e14) << 16 >> 24) + ((_e12) << 8 >> 24) * ((_e14) << 8 >> 24) + ((_e12) >> 24) * ((_e14) >> 24));
    return c_8_;
}

kernel void main_(
) {
    metal::float2 _e0 = test_fma();
    int _e1 = test_integer_dot_product();
    uint _e2 = test_packed_integer_dot_product();
    return;
}
