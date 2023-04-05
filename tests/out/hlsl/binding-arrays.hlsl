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
    uint2 _expr23 = u2_;
    u2_ = (_expr23 + NagaDimensions2D(texture_array_unbounded[0]));
    uint2 _expr28 = u2_;
    u2_ = (_expr28 + NagaDimensions2D(texture_array_unbounded[uniform_index]));
    uint2 _expr33 = u2_;
    u2_ = (_expr33 + NagaDimensions2D(texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr42 = texture_array_bounded[0].Gather(samp[0], uv);
    float4 _expr43 = v4_;
    v4_ = (_expr43 + _expr42);
    float4 _expr50 = texture_array_bounded[uniform_index].Gather(samp[uniform_index], uv);
    float4 _expr51 = v4_;
    v4_ = (_expr51 + _expr50);
    float4 _expr58 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Gather(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr59 = v4_;
    v4_ = (_expr59 + _expr58);
    float4 _expr68 = texture_array_depth[0].GatherCmp(samp_comp[0], uv, 0.0);
    float4 _expr69 = v4_;
    v4_ = (_expr69 + _expr68);
    float4 _expr76 = texture_array_depth[uniform_index].GatherCmp(samp_comp[uniform_index], uv, 0.0);
    float4 _expr77 = v4_;
    v4_ = (_expr77 + _expr76);
    float4 _expr84 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].GatherCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr85 = v4_;
    v4_ = (_expr85 + _expr84);
    float4 _expr91 = texture_array_unbounded[0].Load(int3(pix, 0));
    float4 _expr92 = v4_;
    v4_ = (_expr92 + _expr91);
    float4 _expr97 = texture_array_unbounded[uniform_index].Load(int3(pix, 0));
    float4 _expr98 = v4_;
    v4_ = (_expr98 + _expr97);
    float4 _expr103 = texture_array_unbounded[NonUniformResourceIndex(non_uniform_index)].Load(int3(pix, 0));
    float4 _expr104 = v4_;
    v4_ = (_expr104 + _expr103);
    uint _expr110 = u1_;
    u1_ = (_expr110 + NagaNumLayers2DArray(texture_array_2darray[0]));
    uint _expr115 = u1_;
    u1_ = (_expr115 + NagaNumLayers2DArray(texture_array_2darray[uniform_index]));
    uint _expr120 = u1_;
    u1_ = (_expr120 + NagaNumLayers2DArray(texture_array_2darray[NonUniformResourceIndex(non_uniform_index)]));
    uint _expr126 = u1_;
    u1_ = (_expr126 + NagaNumLevels2D(texture_array_bounded[0]));
    uint _expr131 = u1_;
    u1_ = (_expr131 + NagaNumLevels2D(texture_array_bounded[uniform_index]));
    uint _expr136 = u1_;
    u1_ = (_expr136 + NagaNumLevels2D(texture_array_bounded[NonUniformResourceIndex(non_uniform_index)]));
    uint _expr142 = u1_;
    u1_ = (_expr142 + NagaMSNumSamples2D(texture_array_multisampled[0]));
    uint _expr147 = u1_;
    u1_ = (_expr147 + NagaMSNumSamples2D(texture_array_multisampled[uniform_index]));
    uint _expr152 = u1_;
    u1_ = (_expr152 + NagaMSNumSamples2D(texture_array_multisampled[NonUniformResourceIndex(non_uniform_index)]));
    float4 _expr160 = texture_array_bounded[0].Sample(samp[0], uv);
    float4 _expr161 = v4_;
    v4_ = (_expr161 + _expr160);
    float4 _expr167 = texture_array_bounded[uniform_index].Sample(samp[uniform_index], uv);
    float4 _expr168 = v4_;
    v4_ = (_expr168 + _expr167);
    float4 _expr174 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].Sample(samp[NonUniformResourceIndex(non_uniform_index)], uv);
    float4 _expr175 = v4_;
    v4_ = (_expr175 + _expr174);
    float4 _expr184 = texture_array_bounded[0].SampleBias(samp[0], uv, 0.0);
    float4 _expr185 = v4_;
    v4_ = (_expr185 + _expr184);
    float4 _expr192 = texture_array_bounded[uniform_index].SampleBias(samp[uniform_index], uv, 0.0);
    float4 _expr193 = v4_;
    v4_ = (_expr193 + _expr192);
    float4 _expr200 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleBias(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr201 = v4_;
    v4_ = (_expr201 + _expr200);
    float _expr210 = texture_array_depth[0].SampleCmp(samp_comp[0], uv, 0.0);
    float _expr211 = v1_;
    v1_ = (_expr211 + _expr210);
    float _expr218 = texture_array_depth[uniform_index].SampleCmp(samp_comp[uniform_index], uv, 0.0);
    float _expr219 = v1_;
    v1_ = (_expr219 + _expr218);
    float _expr226 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmp(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr227 = v1_;
    v1_ = (_expr227 + _expr226);
    float _expr236 = texture_array_depth[0].SampleCmpLevelZero(samp_comp[0], uv, 0.0);
    float _expr237 = v1_;
    v1_ = (_expr237 + _expr236);
    float _expr244 = texture_array_depth[uniform_index].SampleCmpLevelZero(samp_comp[uniform_index], uv, 0.0);
    float _expr245 = v1_;
    v1_ = (_expr245 + _expr244);
    float _expr252 = texture_array_depth[NonUniformResourceIndex(non_uniform_index)].SampleCmpLevelZero(samp_comp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float _expr253 = v1_;
    v1_ = (_expr253 + _expr252);
    float4 _expr261 = texture_array_bounded[0].SampleGrad(samp[0], uv, uv, uv);
    float4 _expr262 = v4_;
    v4_ = (_expr262 + _expr261);
    float4 _expr268 = texture_array_bounded[uniform_index].SampleGrad(samp[uniform_index], uv, uv, uv);
    float4 _expr269 = v4_;
    v4_ = (_expr269 + _expr268);
    float4 _expr275 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleGrad(samp[NonUniformResourceIndex(non_uniform_index)], uv, uv, uv);
    float4 _expr276 = v4_;
    v4_ = (_expr276 + _expr275);
    float4 _expr285 = texture_array_bounded[0].SampleLevel(samp[0], uv, 0.0);
    float4 _expr286 = v4_;
    v4_ = (_expr286 + _expr285);
    float4 _expr293 = texture_array_bounded[uniform_index].SampleLevel(samp[uniform_index], uv, 0.0);
    float4 _expr294 = v4_;
    v4_ = (_expr294 + _expr293);
    float4 _expr301 = texture_array_bounded[NonUniformResourceIndex(non_uniform_index)].SampleLevel(samp[NonUniformResourceIndex(non_uniform_index)], uv, 0.0);
    float4 _expr302 = v4_;
    v4_ = (_expr302 + _expr301);
    float4 _expr307 = v4_;
    texture_array_storage[0][pix] = _expr307;
    float4 _expr310 = v4_;
    texture_array_storage[uniform_index][pix] = _expr310;
    float4 _expr313 = v4_;
    texture_array_storage[NonUniformResourceIndex(non_uniform_index)][pix] = _expr313;
    uint2 _expr314 = u2_;
    uint _expr315 = u1_;
    float2 v2_ = float2((_expr314 + (_expr315).xx));
    float4 _expr319 = v4_;
    float _expr326 = v1_;
    return ((_expr319 + float4(v2_.x, v2_.y, v2_.x, v2_.y)) + (_expr326).xxxx);
}
