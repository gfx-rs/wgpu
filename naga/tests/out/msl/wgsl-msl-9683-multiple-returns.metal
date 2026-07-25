// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Outputs {
    metal::float4 color;
};

struct main_Output {
    metal::float4 color [[color(0)]];
};
fragment main_Output main_(
) {
    const auto _tmp = Outputs {metal::float4 {}};
    return main_Output { _tmp.color };
    const auto _tmp_1 = Outputs {metal::float4 {}};
    return main_Output { _tmp_1.color };
}
