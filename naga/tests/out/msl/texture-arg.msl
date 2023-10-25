// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


metal::float4 test(
    metal::texture2d<float, metal::access::sample> Passed_Texture,
    metal::sampler Passed_Sampler
) {
    metal::float4 _e5 = Passed_Texture.sample(Passed_Sampler, metal::float2(0.0, 0.0));
    return _e5;
}

struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
  metal::texture2d<float, metal::access::sample> Texture [[user(fake0)]]
, metal::sampler Sampler [[user(fake0)]]
) {
    metal::float4 _e2 = test(Texture, Sampler);
    return main_Output { _e2 };
}
