// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    uint sum;
    uint carry;
    uint diff;
    uint borrow;
};
struct Input {
    uint a;
    uint b;
};

void main_1(
    device Output& outp,
    device Input const& inp
) {
    uint c = {};
    uint d = {};
    uint _e5 = inp.a;
    uint _e7 = inp.b;
    uint _e8 = _e5 + _e7;
    Input _e11 = Input {_e8, static_cast<uint>(_e8 < _e5)};
    c = _e11.b;
    outp.sum = _e11.a;
    uint _e15 = c;
    outp.carry = _e15;
    uint _e18 = inp.a;
    uint _e20 = inp.b;
    Input _e24 = Input {_e18 - _e20, static_cast<uint>(_e18 < _e20)};
    d = _e24.b;
    outp.diff = _e24.a;
    uint _e28 = d;
    outp.borrow = _e28;
    return;
}

kernel void main_(
  device Output& outp [[user(fake0)]]
, device Input const& inp [[user(fake0)]]
) {
    main_1(outp, inp);
}
