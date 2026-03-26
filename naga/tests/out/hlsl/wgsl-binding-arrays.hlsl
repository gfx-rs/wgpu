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
SamplerState nagaSamplerHeap[2048]: register(s0, space0);
SamplerComparisonState nagaComparisonSamplerHeap[2048]: register(s0, space1);
StructuredBuffer<uint> nagaGroup0SamplerIndexArray : register(t0, space255);
static const uint samp = 0;
static const uint samp_comp = 0;
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
    int2 pix = (int(0)).xx;
    uint2 _e19 = u2_;
    u2_ = (_e19 + NagaDimensions2D(texture_array_unbounded[0]));
    uint2 _e24 = u2_;
    u2_ = (_e24 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    uint2 _e29 = u2_;
    u2_ = (_e29 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _e34 = v4_;
    float4 _e39 = texture_array_bounded[0].Gather(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + 0]], uv);
    v4_ = (_e34 + _e39);
    float4 _e41 = v4_;
    float4 _e46 = texture_array_bounded[uniform_index].Gather(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + uniform_index]], uv);
    v4_ = (_e41 + _e46);
    float4 _e48 = v4_;
    float4 _e53 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp + non_uniform_index])], uv);
    v4_ = (_e48 + _e53);
    float4 _e55 = v4_;
    float4 _e61 = texture_array_depth[0].GatherCmp(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + 0]], uv, 0.0);
    v4_ = (_e55 + _e61);
    float4 _e63 = v4_;
    float4 _e69 = texture_array_depth[uniform_index].GatherCmp(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + uniform_index]], uv, 0.0);
    v4_ = (_e63 + _e69);
    float4 _e71 = v4_;
    float4 _e77 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(nagaComparisonSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp_comp + non_uniform_index])], uv, 0.0);
    v4_ = (_e71 + _e77);
    float4 _e79 = v4_;
    float4 _e83 = texture_array_unbounded[0].Load(int3(pix, int(0)));
    v4_ = (_e79 + _e83);
    float4 _e85 = v4_;
    float4 _e89 = texture_array_unbounded[uniform_index].Load(int3(pix, int(0)));
    v4_ = (_e85 + _e89);
    float4 _e91 = v4_;
    float4 _e95 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, int(0)));
    v4_ = (_e91 + _e95);
    uint _e97 = u1_;
    u1_ = (_e97 + NagaNumLayers2DArray(texture_array_2darray[0]));
    uint _e102 = u1_;
    u1_ = (_e102 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    uint _e107 = u1_;
    u1_ = (_e107 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    uint _e112 = u1_;
    u1_ = (_e112 + NagaNumLevels2D(texture_array_bounded[0]));
    uint _e117 = u1_;
    u1_ = (_e117 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    uint _e122 = u1_;
    u1_ = (_e122 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    uint _e127 = u1_;
    u1_ = (_e127 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    uint _e132 = u1_;
    u1_ = (_e132 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    uint _e137 = u1_;
    u1_ = (_e137 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _e142 = v4_;
    float4 _e147 = texture_array_bounded[0].Sample(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + 0]], uv);
    v4_ = (_e142 + _e147);
    float4 _e149 = v4_;
    float4 _e154 = texture_array_bounded[uniform_index].Sample(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + uniform_index]], uv);
    v4_ = (_e149 + _e154);
    float4 _e156 = v4_;
    float4 _e161 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp + non_uniform_index])], uv);
    v4_ = (_e156 + _e161);
    float4 _e163 = v4_;
    float4 _e169 = texture_array_bounded[0].SampleBias(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + 0]], uv, 0.0);
    v4_ = (_e163 + _e169);
    float4 _e171 = v4_;
    float4 _e177 = texture_array_bounded[uniform_index].SampleBias(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + uniform_index]], uv, 0.0);
    v4_ = (_e171 + _e177);
    float4 _e179 = v4_;
    float4 _e185 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp + non_uniform_index])], uv, 0.0);
    v4_ = (_e179 + _e185);
    float _e187 = v1_;
    float _e193 = texture_array_depth[0].SampleCmp(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + 0]], uv, 0.0);
    v1_ = (_e187 + _e193);
    float _e195 = v1_;
    float _e201 = texture_array_depth[uniform_index].SampleCmp(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + uniform_index]], uv, 0.0);
    v1_ = (_e195 + _e201);
    float _e203 = v1_;
    float _e209 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(nagaComparisonSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp_comp + non_uniform_index])], uv, 0.0);
    v1_ = (_e203 + _e209);
    float _e211 = v1_;
    float _e217 = texture_array_depth[0].SampleCmpLevelZero(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + 0]], uv, 0.0);
    v1_ = (_e211 + _e217);
    float _e219 = v1_;
    float _e225 = texture_array_depth[uniform_index].SampleCmpLevelZero(nagaComparisonSamplerHeap[nagaGroup0SamplerIndexArray[samp_comp + uniform_index]], uv, 0.0);
    v1_ = (_e219 + _e225);
    float _e227 = v1_;
    float _e233 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(nagaComparisonSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp_comp + non_uniform_index])], uv, 0.0);
    v1_ = (_e227 + _e233);
    float4 _e235 = v4_;
    float4 _e240 = texture_array_bounded[0].SampleGrad(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + 0]], uv, uv, uv);
    v4_ = (_e235 + _e240);
    float4 _e242 = v4_;
    float4 _e247 = texture_array_bounded[uniform_index].SampleGrad(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + uniform_index]], uv, uv, uv);
    v4_ = (_e242 + _e247);
    float4 _e249 = v4_;
    float4 _e254 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp + non_uniform_index])], uv, uv, uv);
    v4_ = (_e249 + _e254);
    float4 _e256 = v4_;
    float4 _e262 = texture_array_bounded[0].SampleLevel(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + 0]], uv, 0.0);
    v4_ = (_e256 + _e262);
    float4 _e264 = v4_;
    float4 _e270 = texture_array_bounded[uniform_index].SampleLevel(nagaSamplerHeap[nagaGroup0SamplerIndexArray[samp + uniform_index]], uv, 0.0);
    v4_ = (_e264 + _e270);
    float4 _e272 = v4_;
    float4 _e278 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(nagaSamplerHeap[NonUniformResourceIndex(nagaGroup0SamplerIndexArray[samp + non_uniform_index])], uv, 0.0);
    v4_ = (_e272 + _e278);
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
