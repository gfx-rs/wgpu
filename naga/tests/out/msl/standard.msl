// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


bool test_any_and_all_for_bool(
) {
    return true;
}

struct derivativesInput {
};
struct derivativesOutput {
    metal::float4 member [[color(0)]];
};
fragment derivativesOutput derivatives(
  metal::float4 foo [[position]]
) {
    metal::float4 x = {};
    metal::float4 y = {};
    metal::float4 z = {};
    metal::float4 _e1 = metal::dfdx(foo);
    x = _e1;
    metal::float4 _e3 = metal::dfdy(foo);
    y = _e3;
    metal::float4 _e5 = metal::fwidth(foo);
    z = _e5;
    metal::float4 _e7 = metal::dfdx(foo);
    x = _e7;
    metal::float4 _e8 = metal::dfdy(foo);
    y = _e8;
    metal::float4 _e9 = metal::fwidth(foo);
    z = _e9;
    metal::float4 _e10 = metal::dfdx(foo);
    x = _e10;
    metal::float4 _e11 = metal::dfdy(foo);
    y = _e11;
    metal::float4 _e12 = metal::fwidth(foo);
    z = _e12;
    bool _e13 = test_any_and_all_for_bool();
    metal::float4 _e14 = x;
    metal::float4 _e15 = y;
    metal::float4 _e17 = z;
    return derivativesOutput { (_e14 + _e15) * _e17 };
}
