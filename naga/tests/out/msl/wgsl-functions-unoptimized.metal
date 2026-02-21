// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


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
    uint _e0 = test_packed_integer_dot_product();
    return;
}
