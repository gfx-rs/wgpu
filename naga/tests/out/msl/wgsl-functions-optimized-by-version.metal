// language: metal2.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


uint test_packed_integer_dot_product(
) {
    packed_char4 reinterpreted_packed_char4_e0 = as_type<packed_char4>(1u);
    packed_char4 reinterpreted_packed_char4_e1 = as_type<packed_char4>(2u);
    int c_5_ = ( + reinterpreted_packed_char4_e0[0] * reinterpreted_packed_char4_e1[0] + reinterpreted_packed_char4_e0[1] * reinterpreted_packed_char4_e1[1] + reinterpreted_packed_char4_e0[2] * reinterpreted_packed_char4_e1[2] + reinterpreted_packed_char4_e0[3] * reinterpreted_packed_char4_e1[3]);
    packed_uchar4 reinterpreted_packed_uchar4_e3 = as_type<packed_uchar4>(3u);
    packed_uchar4 reinterpreted_packed_uchar4_e4 = as_type<packed_uchar4>(4u);
    uint c_6_ = ( + reinterpreted_packed_uchar4_e3[0] * reinterpreted_packed_uchar4_e4[0] + reinterpreted_packed_uchar4_e3[1] * reinterpreted_packed_uchar4_e4[1] + reinterpreted_packed_uchar4_e3[2] * reinterpreted_packed_uchar4_e4[2] + reinterpreted_packed_uchar4_e3[3] * reinterpreted_packed_uchar4_e4[3]);
    uint _e7 = 5u + c_6_;
    uint _e9 = 6u + c_6_;
    packed_char4 reinterpreted_packed_char4_e7 = as_type<packed_char4>(_e7);
    packed_char4 reinterpreted_packed_char4_e9 = as_type<packed_char4>(_e9);
    int c_7_ = ( + reinterpreted_packed_char4_e7[0] * reinterpreted_packed_char4_e9[0] + reinterpreted_packed_char4_e7[1] * reinterpreted_packed_char4_e9[1] + reinterpreted_packed_char4_e7[2] * reinterpreted_packed_char4_e9[2] + reinterpreted_packed_char4_e7[3] * reinterpreted_packed_char4_e9[3]);
    uint _e12 = 7u + c_6_;
    uint _e14 = 8u + c_6_;
    packed_uchar4 reinterpreted_packed_uchar4_e12 = as_type<packed_uchar4>(_e12);
    packed_uchar4 reinterpreted_packed_uchar4_e14 = as_type<packed_uchar4>(_e14);
    uint c_8_ = ( + reinterpreted_packed_uchar4_e12[0] * reinterpreted_packed_uchar4_e14[0] + reinterpreted_packed_uchar4_e12[1] * reinterpreted_packed_uchar4_e14[1] + reinterpreted_packed_uchar4_e12[2] * reinterpreted_packed_uchar4_e14[2] + reinterpreted_packed_uchar4_e12[3] * reinterpreted_packed_uchar4_e14[3]);
    return c_8_;
}

kernel void main_(
) {
    uint _e0 = test_packed_integer_dot_product();
    return;
}
