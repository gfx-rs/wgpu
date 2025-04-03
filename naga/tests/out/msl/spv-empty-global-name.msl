// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    int member;
};

void function(
    device type_1& unnamed
) {
    int _e3 = unnamed.member;
    unnamed.member = as_type<int>(as_type<uint>(_e3) + as_type<uint>(1));
    return;
}

kernel void main_(
  device type_1& unnamed [[user(fake0)]]
) {
    function(unnamed);
}
