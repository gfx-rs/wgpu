// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};

struct UniformIndex {
    uint index;
};
template <typename T>
struct NagaArgumentBufferWrapper {
    T inner;
};
struct FragmentIn {
    uint index;
};

struct main_Input {
    uint index [[user(loc0), flat]];
};
struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
  main_Input varyings [[stage_in]]
, constant NagaArgumentBufferWrapper<metal::texture2d<float, metal::access::sample>>* texture_array_unbounded [[buffer(0)]]
, constant NagaArgumentBufferWrapper<metal::texture2d<float, metal::access::sample>>* texture_array_bounded [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::texture2d_array<float, metal::access::sample>>* texture_array_2darray [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::texture2d_ms<float, metal::access::read>>* texture_array_multisampled [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::depth2d<float, metal::access::sample>>* texture_array_depth [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::texture2d<float, metal::access::write>>* texture_array_storage [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::sampler>* samp [[user(fake0)]]
, constant NagaArgumentBufferWrapper<metal::sampler>* samp_comp [[user(fake0)]]
, constant UniformIndex& uni [[user(fake0)]]
) {
    const FragmentIn fragment_in = { varyings.index };
    uint u1_ = 0u;
    metal::uint2 u2_ = metal::uint2(0u);
    float v1_ = 0.0;
    metal::float4 v4_ = metal::float4(0.0);
    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    metal::float2 uv = metal::float2(0.0);
    metal::int2 pix = metal::int2(0);
    metal::uint2 _e19 = u2_;
    u2_ = _e19 + metal::uint2(texture_array_unbounded[0].inner.get_width(), texture_array_unbounded[0].inner.get_height());
    metal::uint2 _e24 = u2_;
    u2_ = _e24 + metal::uint2(texture_array_unbounded[uniform_index].inner.get_width(), texture_array_unbounded[uniform_index].inner.get_height());
    metal::uint2 _e29 = u2_;
    u2_ = _e29 + metal::uint2(texture_array_unbounded[non_uniform_index].inner.get_width(), texture_array_unbounded[non_uniform_index].inner.get_height());
    metal::float4 _e34 = v4_;
    metal::float4 _e39 = texture_array_bounded[0].inner.gather(samp[0].inner, uv);
    v4_ = _e34 + _e39;
    metal::float4 _e41 = v4_;
    metal::float4 _e46 = texture_array_bounded[uniform_index].inner.gather(samp[uniform_index].inner, uv);
    v4_ = _e41 + _e46;
    metal::float4 _e48 = v4_;
    metal::float4 _e53 = texture_array_bounded[non_uniform_index].inner.gather(samp[non_uniform_index].inner, uv);
    v4_ = _e48 + _e53;
    metal::float4 _e55 = v4_;
    metal::float4 _e61 = texture_array_depth[0].inner.gather_compare(samp_comp[0].inner, uv, 0.0);
    v4_ = _e55 + _e61;
    metal::float4 _e63 = v4_;
    metal::float4 _e69 = texture_array_depth[uniform_index].inner.gather_compare(samp_comp[uniform_index].inner, uv, 0.0);
    v4_ = _e63 + _e69;
    metal::float4 _e71 = v4_;
    metal::float4 _e77 = texture_array_depth[non_uniform_index].inner.gather_compare(samp_comp[non_uniform_index].inner, uv, 0.0);
    v4_ = _e71 + _e77;
    metal::float4 _e79 = v4_;
    metal::float4 _e83 = (uint(0) < texture_array_unbounded[0].inner.get_num_mip_levels() && metal::all(metal::uint2(pix) < metal::uint2(texture_array_unbounded[0].inner.get_width(0), texture_array_unbounded[0].inner.get_height(0))) ? texture_array_unbounded[0].inner.read(metal::uint2(pix), 0): DefaultConstructible());
    v4_ = _e79 + _e83;
    metal::float4 _e85 = v4_;
    metal::float4 _e89 = (uint(0) < texture_array_unbounded[uniform_index].inner.get_num_mip_levels() && metal::all(metal::uint2(pix) < metal::uint2(texture_array_unbounded[uniform_index].inner.get_width(0), texture_array_unbounded[uniform_index].inner.get_height(0))) ? texture_array_unbounded[uniform_index].inner.read(metal::uint2(pix), 0): DefaultConstructible());
    v4_ = _e85 + _e89;
    metal::float4 _e91 = v4_;
    metal::float4 _e95 = (uint(0) < texture_array_unbounded[non_uniform_index].inner.get_num_mip_levels() && metal::all(metal::uint2(pix) < metal::uint2(texture_array_unbounded[non_uniform_index].inner.get_width(0), texture_array_unbounded[non_uniform_index].inner.get_height(0))) ? texture_array_unbounded[non_uniform_index].inner.read(metal::uint2(pix), 0): DefaultConstructible());
    v4_ = _e91 + _e95;
    uint _e97 = u1_;
    u1_ = _e97 + texture_array_2darray[0].inner.get_array_size();
    uint _e102 = u1_;
    u1_ = _e102 + texture_array_2darray[uniform_index].inner.get_array_size();
    uint _e107 = u1_;
    u1_ = _e107 + texture_array_2darray[non_uniform_index].inner.get_array_size();
    uint _e112 = u1_;
    u1_ = _e112 + texture_array_bounded[0].inner.get_num_mip_levels();
    uint _e117 = u1_;
    u1_ = _e117 + texture_array_bounded[uniform_index].inner.get_num_mip_levels();
    uint _e122 = u1_;
    u1_ = _e122 + texture_array_bounded[non_uniform_index].inner.get_num_mip_levels();
    uint _e127 = u1_;
    u1_ = _e127 + texture_array_multisampled[0].inner.get_num_samples();
    uint _e132 = u1_;
    u1_ = _e132 + texture_array_multisampled[uniform_index].inner.get_num_samples();
    uint _e137 = u1_;
    u1_ = _e137 + texture_array_multisampled[non_uniform_index].inner.get_num_samples();
    metal::float4 _e142 = v4_;
    metal::float4 _e147 = texture_array_bounded[0].inner.sample(samp[0].inner, uv);
    v4_ = _e142 + _e147;
    metal::float4 _e149 = v4_;
    metal::float4 _e154 = texture_array_bounded[uniform_index].inner.sample(samp[uniform_index].inner, uv);
    v4_ = _e149 + _e154;
    metal::float4 _e156 = v4_;
    metal::float4 _e161 = texture_array_bounded[non_uniform_index].inner.sample(samp[non_uniform_index].inner, uv);
    v4_ = _e156 + _e161;
    metal::float4 _e163 = v4_;
    metal::float4 _e169 = texture_array_bounded[0].inner.sample(samp[0].inner, uv, metal::bias(0.0));
    v4_ = _e163 + _e169;
    metal::float4 _e171 = v4_;
    metal::float4 _e177 = texture_array_bounded[uniform_index].inner.sample(samp[uniform_index].inner, uv, metal::bias(0.0));
    v4_ = _e171 + _e177;
    metal::float4 _e179 = v4_;
    metal::float4 _e185 = texture_array_bounded[non_uniform_index].inner.sample(samp[non_uniform_index].inner, uv, metal::bias(0.0));
    v4_ = _e179 + _e185;
    float _e187 = v1_;
    float _e193 = texture_array_depth[0].inner.sample_compare(samp_comp[0].inner, uv, 0.0);
    v1_ = _e187 + _e193;
    float _e195 = v1_;
    float _e201 = texture_array_depth[uniform_index].inner.sample_compare(samp_comp[uniform_index].inner, uv, 0.0);
    v1_ = _e195 + _e201;
    float _e203 = v1_;
    float _e209 = texture_array_depth[non_uniform_index].inner.sample_compare(samp_comp[non_uniform_index].inner, uv, 0.0);
    v1_ = _e203 + _e209;
    float _e211 = v1_;
    float _e217 = texture_array_depth[0].inner.sample_compare(samp_comp[0].inner, uv, 0.0);
    v1_ = _e211 + _e217;
    float _e219 = v1_;
    float _e225 = texture_array_depth[uniform_index].inner.sample_compare(samp_comp[uniform_index].inner, uv, 0.0);
    v1_ = _e219 + _e225;
    float _e227 = v1_;
    float _e233 = texture_array_depth[non_uniform_index].inner.sample_compare(samp_comp[non_uniform_index].inner, uv, 0.0);
    v1_ = _e227 + _e233;
    metal::float4 _e235 = v4_;
    metal::float4 _e240 = texture_array_bounded[0].inner.sample(samp[0].inner, uv, metal::gradient2d(uv, uv));
    v4_ = _e235 + _e240;
    metal::float4 _e242 = v4_;
    metal::float4 _e247 = texture_array_bounded[uniform_index].inner.sample(samp[uniform_index].inner, uv, metal::gradient2d(uv, uv));
    v4_ = _e242 + _e247;
    metal::float4 _e249 = v4_;
    metal::float4 _e254 = texture_array_bounded[non_uniform_index].inner.sample(samp[non_uniform_index].inner, uv, metal::gradient2d(uv, uv));
    v4_ = _e249 + _e254;
    metal::float4 _e256 = v4_;
    metal::float4 _e262 = texture_array_bounded[0].inner.sample(samp[0].inner, uv, metal::level(0.0));
    v4_ = _e256 + _e262;
    metal::float4 _e264 = v4_;
    metal::float4 _e270 = texture_array_bounded[uniform_index].inner.sample(samp[uniform_index].inner, uv, metal::level(0.0));
    v4_ = _e264 + _e270;
    metal::float4 _e272 = v4_;
    metal::float4 _e278 = texture_array_bounded[non_uniform_index].inner.sample(samp[non_uniform_index].inner, uv, metal::level(0.0));
    v4_ = _e272 + _e278;
    metal::float4 _e282 = v4_;
    texture_array_storage[0].inner.write(_e282, metal::uint2(pix));
    metal::float4 _e285 = v4_;
    texture_array_storage[uniform_index].inner.write(_e285, metal::uint2(pix));
    metal::float4 _e288 = v4_;
    texture_array_storage[non_uniform_index].inner.write(_e288, metal::uint2(pix));
    metal::uint2 _e289 = u2_;
    uint _e290 = u1_;
    metal::float2 v2_ = static_cast<metal::float2>(_e289 + metal::uint2(_e290));
    metal::float4 _e294 = v4_;
    float _e301 = v1_;
    return main_Output { (_e294 + metal::float4(v2_.x, v2_.y, v2_.x, v2_.y)) + metal::float4(_e301) };
}
