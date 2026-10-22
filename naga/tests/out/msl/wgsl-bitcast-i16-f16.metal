// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    half h = 1.0h;
    short i = static_cast<short>(1);
    ushort u = static_cast<ushort>(2);
    metal::half4 h4_ = metal::half4(1.0h, 2.0h, 3.0h, 4.0h);
    metal::ushort4 u4_ = metal::ushort4(static_cast<ushort>(1), static_cast<ushort>(2), static_cast<ushort>(3), static_cast<ushort>(4));
    half _e18 = h;
    i = as_type<short>(_e18);
    half _e20 = h;
    u = as_type<ushort>(_e20);
    short _e22 = i;
    h = as_type<half>(_e22);
    ushort _e24 = u;
    h = as_type<half>(_e24);
    metal::half4 _e26 = h4_;
    u4_ = as_type<metal::ushort4>(_e26);
    metal::ushort4 _e28 = u4_;
    h4_ = as_type<metal::half4>(_e28);
    return;
}
