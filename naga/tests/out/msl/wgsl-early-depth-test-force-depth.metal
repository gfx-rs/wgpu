// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct ColorAndDepth {
    metal::float4 color;
    float depth;
    char _pad2[12];
};

struct only_depthInput {
};
[[early_fragment_tests]] fragment void only_depth(
  metal::float4 pos [[position]]
) {
    return;
}


struct color_and_depthInput {
};
struct color_and_depthOutput {
    metal::float4 color [[color(0)]];
};
[[early_fragment_tests]] fragment color_and_depthOutput color_and_depth(
  metal::float4 pos_1 [[position]]
) {
    const auto _tmp = ColorAndDepth {metal::float4(0.4, 0.3, 0.2, 0.1), pos_1.z};
    return color_and_depthOutput { _tmp.color };
}
