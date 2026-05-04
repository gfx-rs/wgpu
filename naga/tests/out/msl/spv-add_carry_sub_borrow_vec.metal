// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    metal::uint2 sum;
    metal::uint2 carry;
    metal::uint2 diff;
    metal::uint2 borrow;
};
struct Input {
    metal::uint2 a;
    metal::uint2 b;
};

void main_1(
    device Output& outp,
    device Input const& inp
) {
    metal::uint2 c = {};
    metal::uint2 d = {};
    metal::uint2 _e5 = inp.a;
    metal::uint2 _e7 = inp.b;
    metal::uint2 _e8 = _e5 + _e7;
    Input _e11 = Input {_e8, static_cast<metal::uint2>(_e8 < _e5)};
    c = _e11.b;
    outp.sum = _e11.a;
    metal::uint2 _e15 = c;
    outp.carry = _e15;
    metal::uint2 _e18 = inp.a;
    metal::uint2 _e20 = inp.b;
    Input _e24 = Input {_e18 - _e20, static_cast<metal::uint2>(_e18 < _e20)};
    d = _e24.b;
    outp.diff = _e24.a;
    metal::uint2 _e28 = d;
    outp.borrow = _e28;
    return;
}

kernel void main_(
  device Output& outp [[user(fake0)]]
, device Input const& inp [[user(fake0)]]
) {
    main_1(outp, inp);
}
