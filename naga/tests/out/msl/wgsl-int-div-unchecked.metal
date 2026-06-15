// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


kernel void main_(
) {
    int div_s = 5 / 2;
    int mod_s = 5 % 2;
    uint div_u = 5u / 2u;
    uint mod_u = 5u % 2u;
    return;
}
