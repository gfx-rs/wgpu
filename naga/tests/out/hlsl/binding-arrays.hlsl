struct UniformIndex {
    uint index;
};

struct FragmentIn {
    nointerpolation uint index : LOC0;
};

Texture2D<float4> texture_array_unbounded[10] : register(t0);
Texture2D<float4> texture_array_bounded[5] : register(t0, space1);
Texture2DArray<float4> texture_array_2darray[5] : register(t0, space2);
Texture2DMS<float4> texture_array_multisampled[5] : register(t0, space3);
Texture2D<float> texture_array_depth[5] : register(t0, space4);
RWTexture2D<float4> texture_array_storage[5] : register(u0, space5);
SamplerState samp[5] : register(s0, space6);
SamplerComparisonState samp_comp[5] : register(s0, space7);
cbuffer uni : register(b0, space8) { UniformIndex uni; }

struct FragmentInput_main {
    nointerpolation uint index : LOC0;
};

uint2 NagaDimensions2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.xy;
}

uint NagaNumLayers2DArray(Texture2DArray<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z, ret.w);
    return ret.w;
}

uint NagaNumLevels2D(Texture2D<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(0, ret.x, ret.y, ret.z);
    return ret.z;
}

uint NagaMSNumSamples2D(Texture2DMS<float4> tex)
{
    uint4 ret;
    tex.GetDimensions(ret.x, ret.y, ret.z);
    return ret.z;
}

float4 main(FragmentInput_main fragmentinput_main) : SV_Target0
{
    FragmentIn fragment_in = { fragmentinput_main.index };
    uint u1_ = 0u;
    uint2 u2_ = (0u).xx;
    float v1_ = 0.0;
    float4 v4_ = (0.0).xxxx;

    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    float2 uv = (0.0).xx;
    int2 pix = (0).xx;
    uint2 _e22 = u2_;
    u2_ = (_e22 + NagaDimensions2D(texture_array_unbounded[0]));
    uint2 _e27 = u2_;
    u2_ = (_e27 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    uint2 _e32 = u2_;
    u2_ = (_e32 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _e38 = texture_array_bounded[0].Gather(samp[0], uv);
    float4 _e39 = v4_;
    v4_ = (_e39 + _e38);
    float4 _e45 = texture_array_bounded[uniform_index].Gather(samp[uniform_index], uv);
    float4 _e46 = v4_;
    v4_ = (_e46 + _e45);
    float4 _e52 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _e53 = v4_;
    v4_ = (_e53 + _e52);
    float4 _e60 = texture_array_depth[0].GatherCmp(samp_comp[0], uv, 0.0);
    float4 _e61 = v4_;
    v4_ = (_e61 + _e60);
    float4 _e68 = texture_array_depth[uniform_index].GatherCmp(samp_comp[uniform_index], uv, 0.0);
    float4 _e69 = v4_;
    v4_ = (_e69 + _e68);
    float4 _e76 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _e77 = v4_;
    v4_ = (_e77 + _e76);
    float4 _e82 = texture_array_unbounded[0].Load(int3(pix, 0));
    float4 _e83 = v4_;
    v4_ = (_e83 + _e82);
    float4 _e88 = texture_array_unbounded[uniform_index].Load(int3(pix, 0));
    float4 _e89 = v4_;
    v4_ = (_e89 + _e88);
    float4 _e94 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, 0));
    float4 _e95 = v4_;
    v4_ = (_e95 + _e94);
    uint _e100 = u1_;
    u1_ = (_e100 + NagaNumLayers2DArray(texture_array_2darray[0]));
    uint _e105 = u1_;
    u1_ = (_e105 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    uint _e110 = u1_;
    u1_ = (_e110 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    uint _e115 = u1_;
    u1_ = (_e115 + NagaNumLevels2D(texture_array_bounded[0]));
    uint _e120 = u1_;
    u1_ = (_e120 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    uint _e125 = u1_;
    u1_ = (_e125 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    uint _e130 = u1_;
    u1_ = (_e130 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    uint _e135 = u1_;
    u1_ = (_e135 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    uint _e140 = u1_;
    u1_ = (_e140 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _e146 = texture_array_bounded[0].Sample(samp[0], uv);
    float4 _e147 = v4_;
    v4_ = (_e147 + _e146);
    float4 _e153 = texture_array_bounded[uniform_index].Sample(samp[uniform_index], uv);
    float4 _e154 = v4_;
    v4_ = (_e154 + _e153);
    float4 _e160 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _e161 = v4_;
    v4_ = (_e161 + _e160);
    float4 _e168 = texture_array_bounded[0].SampleBias(samp[0], uv, 0.0);
    float4 _e169 = v4_;
    v4_ = (_e169 + _e168);
    float4 _e176 = texture_array_bounded[uniform_index].SampleBias(samp[uniform_index], uv, 0.0);
    float4 _e177 = v4_;
    v4_ = (_e177 + _e176);
    float4 _e184 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _e185 = v4_;
    v4_ = (_e185 + _e184);
    float _e192 = texture_array_depth[0].SampleCmp(samp_comp[0], uv, 0.0);
    float _e193 = v1_;
    v1_ = (_e193 + _e192);
    float _e200 = texture_array_depth[uniform_index].SampleCmp(samp_comp[uniform_index], uv, 0.0);
    float _e201 = v1_;
    v1_ = (_e201 + _e200);
    float _e208 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _e209 = v1_;
    v1_ = (_e209 + _e208);
    float _e216 = texture_array_depth[0].SampleCmpLevelZero(samp_comp[0], uv, 0.0);
    float _e217 = v1_;
    v1_ = (_e217 + _e216);
    float _e224 = texture_array_depth[uniform_index].SampleCmpLevelZero(samp_comp[uniform_index], uv, 0.0);
    float _e225 = v1_;
    v1_ = (_e225 + _e224);
    float _e232 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _e233 = v1_;
    v1_ = (_e233 + _e232);
    float4 _e239 = texture_array_bounded[0].SampleGrad(samp[0], uv, uv, uv);
    float4 _e240 = v4_;
    v4_ = (_e240 + _e239);
    float4 _e246 = texture_array_bounded[uniform_index].SampleGrad(samp[uniform_index], uv, uv, uv);
    float4 _e247 = v4_;
    v4_ = (_e247 + _e246);
    float4 _e253 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(samp[NonUniformResourceIndex(non_uniform_index)], uv, uv, uv);
    float4 _e254 = v4_;
    v4_ = (_e254 + _e253);
    float4 _e261 = texture_array_bounded[0].SampleLevel(samp[0], uv, 0.0);
    float4 _e262 = v4_;
    v4_ = (_e262 + _e261);
    float4 _e269 = texture_array_bounded[uniform_index].SampleLevel(samp[uniform_index], uv, 0.0);
    float4 _e270 = v4_;
    v4_ = (_e270 + _e269);
    float4 _e277 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _e278 = v4_;
    v4_ = (_e278 + _e277);
    float4 _e282 = v4_;
    texture_array_storage[0][pix] = _e282;
    float4 _e285 = v4_;
    texture_array_storage[uniform_index][pix] = _e285;
    float4 _e288 = v4_;
    texture_array_storage[NonUniformResourceIndex(non_uniform_index)][pix] = _e288;
    uint2 _e289 = u2_;
    uint _e290 = u1_;
    float2 v2_ = float2((_e289 + (_e290).xx));
    float4 _e294 = v4_;
    float _e301 = v1_;
    return ((_e294 + float4(v2_.x, v2_.y, v2_.x, v2_.y)) + (_e301).xxxx);
}
