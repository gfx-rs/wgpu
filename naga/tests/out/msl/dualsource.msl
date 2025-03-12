// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct FragmentOutput {
    metal::float4 output0_;
    metal::float4 output1_;
};

struct main_Output {
    metal::float4 output0_ [[color(0) index(0)]];
    metal::float4 output1_ [[color(0) index(1)]];
};
fragment main_Output main_(
) {
    const auto _tmp = FragmentOutput {metal::float4(0.4, 0.3, 0.2, 0.1), metal::float4(0.9, 0.8, 0.7, 0.6)};
    return main_Output { _tmp.output0_, _tmp.output1_ };
}
