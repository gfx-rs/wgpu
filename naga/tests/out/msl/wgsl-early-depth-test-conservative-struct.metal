// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Output {
    metal::float4 color;
    float depth;
    char _pad2[12];
};
struct DisagreeingOutput {
    metal::float4 color;
    float depth;
    char _pad2[12];
};

struct agree_firstInput {
};
struct agree_firstOutput {
    metal::float4 color [[color(0)]];
    float depth [[depth(less)]];
};
fragment agree_firstOutput agree_first(
  metal::float4 pos [[position]]
) {
    const auto _tmp = Output {metal::float4(0.4, 0.3, 0.2, 0.1), pos.z - 0.1};
    return agree_firstOutput { _tmp.color, _tmp.depth };
}


struct agree_secondInput {
};
struct agree_secondOutput {
    metal::float4 color [[color(0)]];
    float depth [[depth(less)]];
};
fragment agree_secondOutput agree_second(
  metal::float4 pos_1 [[position]]
) {
    const auto _tmp_1 = Output {metal::float4(0.1, 0.2, 0.3, 0.4), pos_1.z - 0.2};
    return agree_secondOutput { _tmp_1.color, _tmp_1.depth };
}


struct disagree_conservativeInput {
};
struct disagree_conservativeOutput {
    metal::float4 color [[color(0)]];
    float depth [[depth(greater)]];
};
fragment disagree_conservativeOutput disagree_conservative(
  metal::float4 pos_2 [[position]]
) {
    const auto _tmp_2 = DisagreeingOutput {metal::float4(0.4, 0.3, 0.2, 0.1), pos_2.z + 0.1};
    return disagree_conservativeOutput { _tmp_2.color, _tmp_2.depth };
}


struct disagree_unrestrictedInput {
};
struct disagree_unrestrictedOutput {
    metal::float4 color [[color(0)]];
    float depth [[depth(any)]];
};
fragment disagree_unrestrictedOutput disagree_unrestricted(
  metal::float4 pos_3 [[position]]
) {
    const auto _tmp_3 = DisagreeingOutput {metal::float4(0.1, 0.2, 0.3, 0.4), pos_3.z * 2.0};
    return disagree_unrestrictedOutput { _tmp_3.color, _tmp_3.depth };
}
