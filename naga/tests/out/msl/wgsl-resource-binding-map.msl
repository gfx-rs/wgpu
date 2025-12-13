// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};


struct entry_point_oneInput {
};
struct entry_point_oneOutput {
    metal::float4 member [[color(0)]];
};
fragment entry_point_oneOutput entry_point_one(
  metal::float4 pos [[position]]
, metal::texture2d<float, metal::access::sample> t1_ [[texture(0)]]
) {
    constexpr metal::sampler s1_(
        metal::s_address::clamp_to_edge,
        metal::t_address::clamp_to_edge,
        metal::r_address::clamp_to_edge,
        metal::mag_filter::linear,
        metal::min_filter::linear,
        metal::coord::normalized
    );
    metal::float4 _e4 = t1_.sample(s1_, pos.xy);
    return entry_point_oneOutput { _e4 };
}


struct entry_point_twoOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment entry_point_twoOutput entry_point_two(
  metal::texture2d<float, metal::access::sample> t1_ [[texture(0)]]
, metal::sampler s1_ [[sampler(0)]]
, constant metal::float2& uniformOne [[buffer(0)]]
) {
    metal::float2 _e3 = uniformOne;
    metal::float4 _e4 = t1_.sample(s1_, _e3);
    return entry_point_twoOutput { _e4 };
}


struct entry_point_threeOutput {
    metal::float4 member_2 [[color(0)]];
};
fragment entry_point_threeOutput entry_point_three(
  metal::texture2d<float, metal::access::sample> t1_ [[texture(0)]]
, metal::texture2d<float, metal::access::sample> t2_ [[texture(1)]]
, metal::sampler s2_ [[sampler(1)]]
, constant metal::float2& uniformOne [[buffer(0)]]
, constant metal::float2& uniformTwo [[buffer(1)]]
) {
    constexpr metal::sampler s1_(
        metal::s_address::clamp_to_edge,
        metal::t_address::clamp_to_edge,
        metal::r_address::clamp_to_edge,
        metal::mag_filter::linear,
        metal::min_filter::linear,
        metal::coord::normalized
    );
    metal::float2 _e3 = uniformTwo;
    metal::float2 _e5 = uniformOne;
    metal::float4 _e7 = t1_.sample(s1_, _e3 + _e5);
    metal::float2 _e11 = uniformOne;
    metal::float4 _e12 = t2_.sample(s2_, _e11);
    return entry_point_threeOutput { _e7 + _e12 };
}
