// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

template <typename T>
struct NagaArgumentBufferWrapper {
    T inner;
};

struct main_Input {
    metal::float2 uv [[user(loc0), center_perspective]];
};
struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
  main_Input varyings [[stage_in]]
, constant NagaArgumentBufferWrapper<metal::texture2d<float, metal::access::sample>>* textures [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::sampler>* samplers [[user(fake0)]]
) {
    const auto uv = varyings.uv;
    uint i = {};
    metal::float2 uv_1 = {};
    metal::float4 local = {};
    metal::float2 uv_2 = {};
    uint i_1 = {};
    metal::float2 uv_3 = {};
    metal::float4 local_1 = {};
    uint i_2 = {};
    metal::float2 uv_4 = {};
    metal::float4 local_2 = {};
    metal::float4 local_3 = {};
    uint i_3 = {};
    metal::uint2 local_4 = {};
    i = 0u;
    uint _e1 = i;
    uv_1 = uv;
    metal::float2 _e3 = uv_1;
    metal::float4 _e8 = textures[_e1].inner.sample(samplers[_e1].inner, _e3);
    local = _e8;
    metal::float4 _e10 = local;
    uv_2 = uv;
    metal::float2 _e12 = uv_2;
    i_1 = uint {};
    uv_3 = metal::float2 {};
    local_1 = metal::float4 {};
    i_2 = uint {};
    uv_4 = metal::float2 {};
    local_2 = metal::float4 {};
    i_1 = 1u;
    uint _e14 = i_1;
    uv_3 = _e12;
    metal::float2 _e16 = uv_3;
    metal::float4 _e21 = textures[_e14].inner.sample(samplers[_e14].inner, _e16);
    local_1 = _e21;
    metal::float4 _e23 = local_1;
    i_2 = 2u;
    uint _e25 = i_2;
    uv_4 = _e12;
    metal::float2 _e27 = uv_4;
    metal::float4 _e30 = textures[_e25].inner.sample(samplers[_e25].inner, _e27);
    local_2 = _e30;
    metal::float4 _e32 = local_2;
    local_3 = _e23 + _e32;
    metal::float4 _e35 = local_3;
    metal::float4 v = _e10 + _e35;
    i_3 = 3u;
    uint _e38 = i_3;
    local_4 = metal::uint2(textures[_e38].inner.get_width(), textures[_e38].inner.get_height());
    metal::uint2 _e43 = local_4;
    return main_Output { v * static_cast<float>(_e43.x) };
}
