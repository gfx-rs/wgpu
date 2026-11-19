// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct ColorAndDepth {
    metal::float4 color;
    float depth;
    char _pad2[12];
};
struct OnlyDepth {
    float depth;
};

struct color_and_depthInput {
};
struct color_and_depthOutput {
    metal::float4 color [[color(0)]];
};
fragment color_and_depthOutput color_and_depth(
  metal::float4 pos [[position]]
) {
    const auto _tmp = ColorAndDepth {metal::float4(0.4, 0.3, 0.2, 0.1), pos.z};
    return color_and_depthOutput { _tmp.color };
}


struct only_depthInput {
};
fragment void only_depth(
  metal::float4 pos_1 [[position]]
) {
    return;
}


struct unrestrictedInput {
};
struct unrestrictedOutput {
    metal::float4 color [[color(0)]];
    float depth [[depth(any)]];
};
fragment unrestrictedOutput unrestricted(
  metal::float4 pos_2 [[position]]
) {
    const auto _tmp_1 = ColorAndDepth {metal::float4(0.1, 0.2, 0.3, 0.4), pos_2.z * 2.0};
    return unrestrictedOutput { _tmp_1.color, _tmp_1.depth };
}
