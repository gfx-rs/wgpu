// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


float compute(
    device float const& asdf
) {
    float _e1 = asdf;
    float u03b8_2_ = _e1 + 9001.0;
    return u03b8_2_;
}

kernel void main_(
  device float const& asdf [[user(fake0)]]
) {
    float _e0 = compute(asdf);
    return;
}
