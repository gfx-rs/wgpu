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
    uint u1_ = (uint)0;
    uint2 u2_ = (uint2)0;
    float v1_ = (float)0;
    float4 v4_ = (float4)0;

    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    u1_ = 0u;
    u2_ = (0u).xx;
    v1_ = 0.0;
    v4_ = (0.0).xxxx;
    float2 uv = (0.0).xx;
    int2 pix = (0).xx;
    uint2 _expr22 = u2_;
    u2_ = (_expr22 + NagaDimensions2D(texture_array_unbounded[0]));
    uint2 _expr27 = u2_;
    u2_ = (_expr27 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    uint2 _expr32 = u2_;
    u2_ = (_expr32 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr38 = texture_array_bounded[0].Gather(samp[0], uv);
    float4 _expr39 = v4_;
    v4_ = (_expr39 + _expr38);
    float4 _expr45 = texture_array_bounded[uniform_index].Gather(samp[uniform_index], uv);
    float4 _expr46 = v4_;
    v4_ = (_expr46 + _expr45);
    float4 _expr52 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr53 = v4_;
    v4_ = (_expr53 + _expr52);
    float4 _expr60 = texture_array_depth[0].GatherCmp(samp_comp[0], uv, 0.0);
    float4 _expr61 = v4_;
    v4_ = (_expr61 + _expr60);
    float4 _expr68 = texture_array_depth[uniform_index].GatherCmp(samp_comp[uniform_index], uv, 0.0);
    float4 _expr69 = v4_;
    v4_ = (_expr69 + _expr68);
    float4 _expr76 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr77 = v4_;
    v4_ = (_expr77 + _expr76);
    float4 _expr82 = texture_array_unbounded[0].Load(int3(pix, 0));
    float4 _expr83 = v4_;
    v4_ = (_expr83 + _expr82);
    float4 _expr88 = texture_array_unbounded[uniform_index].Load(int3(pix, 0));
    float4 _expr89 = v4_;
    v4_ = (_expr89 + _expr88);
    float4 _expr94 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, 0));
    float4 _expr95 = v4_;
    v4_ = (_expr95 + _expr94);
    uint _expr100 = u1_;
    u1_ = (_expr100 + NagaNumLayers2DArray(texture_array_2darray[0]));
    uint _expr105 = u1_;
    u1_ = (_expr105 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    uint _expr110 = u1_;
    u1_ = (_expr110 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    uint _expr115 = u1_;
    u1_ = (_expr115 + NagaNumLevels2D(texture_array_bounded[0]));
    uint _expr120 = u1_;
    u1_ = (_expr120 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    uint _expr125 = u1_;
    u1_ = (_expr125 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    uint _expr130 = u1_;
    u1_ = (_expr130 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    uint _expr135 = u1_;
    u1_ = (_expr135 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    uint _expr140 = u1_;
    u1_ = (_expr140 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr146 = texture_array_bounded[0].Sample(samp[0], uv);
    float4 _expr147 = v4_;
    v4_ = (_expr147 + _expr146);
    float4 _expr153 = texture_array_bounded[uniform_index].Sample(samp[uniform_index], uv);
    float4 _expr154 = v4_;
    v4_ = (_expr154 + _expr153);
    float4 _expr160 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr161 = v4_;
    v4_ = (_expr161 + _expr160);
    float4 _expr168 = texture_array_bounded[0].SampleBias(samp[0], uv, 0.0);
    float4 _expr169 = v4_;
    v4_ = (_expr169 + _expr168);
    float4 _expr176 = texture_array_bounded[uniform_index].SampleBias(samp[uniform_index], uv, 0.0);
    float4 _expr177 = v4_;
    v4_ = (_expr177 + _expr176);
    float4 _expr184 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr185 = v4_;
    v4_ = (_expr185 + _expr184);
    float _expr192 = texture_array_depth[0].SampleCmp(samp_comp[0], uv, 0.0);
    float _expr193 = v1_;
    v1_ = (_expr193 + _expr192);
    float _expr200 = texture_array_depth[uniform_index].SampleCmp(samp_comp[uniform_index], uv, 0.0);
    float _expr201 = v1_;
    v1_ = (_expr201 + _expr200);
    float _expr208 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr209 = v1_;
    v1_ = (_expr209 + _expr208);
    float _expr216 = texture_array_depth[0].SampleCmpLevelZero(samp_comp[0], uv, 0.0);
    float _expr217 = v1_;
    v1_ = (_expr217 + _expr216);
    float _expr224 = texture_array_depth[uniform_index].SampleCmpLevelZero(samp_comp[uniform_index], uv, 0.0);
    float _expr225 = v1_;
    v1_ = (_expr225 + _expr224);
    float _expr232 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr233 = v1_;
    v1_ = (_expr233 + _expr232);
    float4 _expr239 = texture_array_bounded[0].SampleGrad(samp[0], uv, uv, uv);
    float4 _expr240 = v4_;
    v4_ = (_expr240 + _expr239);
    float4 _expr246 = texture_array_bounded[uniform_index].SampleGrad(samp[uniform_index], uv, uv, uv);
    float4 _expr247 = v4_;
    v4_ = (_expr247 + _expr246);
    float4 _expr253 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(samp[NonUniformResourceIndex(non_uniform_index)], uv, uv, uv);
    float4 _expr254 = v4_;
    v4_ = (_expr254 + _expr253);
    float4 _expr261 = texture_array_bounded[0].SampleLevel(samp[0], uv, 0.0);
    float4 _expr262 = v4_;
    v4_ = (_expr262 + _expr261);
    float4 _expr269 = texture_array_bounded[uniform_index].SampleLevel(samp[uniform_index], uv, 0.0);
    float4 _expr270 = v4_;
    v4_ = (_expr270 + _expr269);
    float4 _expr277 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr278 = v4_;
    v4_ = (_expr278 + _expr277);
    float4 _expr282 = v4_;
    texture_array_storage[0][pix] = _expr282;
    float4 _expr285 = v4_;
    texture_array_storage[uniform_index][pix] = _expr285;
    float4 _expr288 = v4_;
    texture_array_storage[NonUniformResourceIndex(non_uniform_index)][pix] = _expr288;
    uint2 _expr289 = u2_;
    uint _expr290 = u1_;
    float2 v2_ = float2((_expr289 + (_expr290).xx));
    float4 _expr294 = v4_;
    float _expr301 = v1_;
    return ((_expr294 + float4(v2_.x, v2_.y, v2_.x, v2_.y)) + (_expr301).xxxx);
}
