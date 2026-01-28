// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

constant bool has_point_light = false;
constant float specular_param = 2.3;
constant float gain = 1.1;
constant float width = 0.0;
constant float depth = 2.3;
constant float height = 4.6;
constant float inferred_f32_ = 2.718;
constant uint auto_conversion = 0u;

kernel void main_(
) {
    float gain_x_10_ = 11.0;
    float store_override = {};
    float t = 23.0;
    bool x = {};
    float gain_x_100_ = {};
    x = true;
    float _e9 = gain_x_10_;
    gain_x_100_ = _e9 * 10.0;
    store_override = gain;
    return;
}
